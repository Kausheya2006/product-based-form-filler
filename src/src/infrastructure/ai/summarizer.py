import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.utils import logging as transformers_logging
from ...domain.interfaces import ISummarizer
try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None
try:
    from huggingface_hub.utils import enable_progress_bars
except ImportError:  # pragma: no cover
    enable_progress_bars = None

logger = logging.getLogger(__name__)


def _enable_model_download_progress() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    transformers_logging.set_verbosity_info()
    if enable_progress_bars is not None:
        enable_progress_bars()


class LocalSummarizer(ISummarizer):
    """One-shot summarizer using distilbart-cnn-12-6 (seq2seq)."""

    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        # Load components manually
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    async def summarize(self, text: str) -> str:
        # Tokenize input with truncation (model has 1024 token limit)
        inputs = self.tokenizer(
            text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=130,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode and return
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


class GemmaSummarizer(ISummarizer):
    """Incremental summarizer using a fine-tuned functiongemma model.

    Processes the conversation line-by-line, maintaining a running summary.
    The prompt format matches the training data in train_summarizer.py.
    """

    def __init__(self, model_path: str, max_new_tokens: int = 256, device: str | None = None):
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading GemmaSummarizer from %s (device=%s)", model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Internal helpers (match training prompt format exactly)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_lines(text: str) -> list[str]:
        """Split full conversation text into individual lines."""
        return [ln for ln in text.split("\n") if ln.strip()]

    @staticmethod
    def _format_lines_before(lines: list[str]) -> str:
        return "\n".join(lines) if lines else "(none)"

    def _build_prompt(self, new_line: str, lines_before: list[str], current_summary: str) -> str:
        return (
            "<start_of_turn>user\n"
            "Update the conversation summary given the new line.\n\n"
            f"New line: {new_line}\n\n"
            f"Previous lines:\n{self._format_lines_before(lines_before)}\n\n"
            f"Current summary: {current_summary}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode only the new tokens (after the prompt)
        new_tokens = output_ids[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text

    # ------------------------------------------------------------------
    # ISummarizer interface
    # ------------------------------------------------------------------

    async def summarize(self, text: str) -> str:
        """Incrementally summarize a conversation given its full text.

        Splits the text into lines and processes each one, feeding the
        running summary back into the next step (matching the training
        data format).
        """
        lines = self._split_lines(text)
        if not lines:
            return ""

        current_summary = ""
        for i, line in enumerate(lines):
            lines_before = lines[max(0, i - 10):i]
            prompt = self._build_prompt(line, lines_before, current_summary)
            current_summary = self._generate(prompt)
            logger.info("[GemmaSummarizer] Line %d/%d: %s", i + 1, len(lines), line[:80])
            logger.info("[GemmaSummarizer] Summary:   %s", current_summary[:200])

        return current_summary


class QwenSummarizer(ISummarizer):
    """One-shot summarizer using the base Qwen instruct model without LoRA."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 160):
        _enable_model_download_progress()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading QwenSummarizer from %s (device=%s)", self.model_name, self.device)
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        logger.info("Loading base model...")
        self.model = self._load_model()
        self.model.eval()
        logger.info("Base model loaded")
        logger.info("Model ready")

    def _load_model(self):
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        attempts = []
        if self.device == "cuda" and BitsAndBytesConfig is not None:
            try:
                attempts.append({
                    "device_map": "auto",
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,
                    ),
                    "torch_dtype": dtype,
                })
            except Exception:  # pragma: no cover
                pass
        if self.device == "cuda":
            attempts.append({"device_map": "auto", "torch_dtype": dtype})
        attempts.append({"torch_dtype": torch.float32})

        last_error = None
        for kwargs in attempts:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    **kwargs,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning("QwenSummarizer load attempt failed for %s: %s", self.model_name, exc)
        raise RuntimeError(f"Unable to load summarizer model {self.model_name}") from last_error

    async def summarize(self, text: str) -> str:
        conversation = text.strip()
        if not conversation:
            return ""

        messages = [
            {
                "role": "system",
                "content": "You summarize conversations concisely. Return only the summary text.",
            },
            {
                "role": "user",
                "content": f"Summarize this conversation in a short paragraph.\n\n{conversation}",
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        target_device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = {
            key: value.to(target_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
