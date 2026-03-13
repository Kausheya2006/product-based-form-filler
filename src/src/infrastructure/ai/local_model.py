import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any
from pathlib import Path
from transformers import pipeline
from ...domain.interfaces import IExtractionModel
from ...domain.domain import ExtractionRequest
from ..config import settings

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class LocalHuggingFaceModel(IExtractionModel):
    def __init__(self):
        model_name = settings.MODEL_NAME
        print(f"Loading local model: {model_name}...")
        self.pipeline = pipeline("question-answering", model=model_name)
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def extract_batch(self, requests: List[ExtractionRequest]) -> List[Any]:
        """
        Extracts multiple fields in one go. 
        HuggingFace pipelines are optimized for batching.
        """
        if not requests:
            return []

        # Prepare inputs for the pipeline
        # Passing separate lists for questions and contexts to avoid DeprecationWarning
        # system_prompt = "You are retrieving information from the above conversation to retrieve the data of a particular field. Answer N/A if the information is not available. Retrieve the information which matches this given field name the closest: "
        questions = [req.instruction for req in requests]
        contexts = [req.context for req in requests]
        
        loop = asyncio.get_running_loop()
        try:
            # We run the batch inference call in a separate thread
            # pipeline(question=..., context=...) handles batching natively
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.pipeline(question=questions, context=contexts)
            )
            # results will be [{'score':.., 'start':.., 'end':.., 'answer':..}, ...]
            # Use confidence threshold: QA models always return *something*, so we
            # gate on score to avoid filling unrelated fields with spurious spans.
            CONFIDENCE_THRESHOLD = 0.1
            return [
                r['answer'] if (r and r.get('score', 0) >= CONFIDENCE_THRESHOLD) else None
                for r in results
            ]
        except Exception as e:
            print(f"Batch extraction error: {e}")
            return [None] * len(requests)

class GemmaFunctionalModel(IExtractionModel):
    def __init__(self, max_input_tokens=512, max_new_tokens=256, temperature=0.0, checkpoint_path="data_generation/models/checkpoint-200"):
        print(f"Loading local model: Gemma Functional ")
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        resolved_checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_checkpoint_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_checkpoint_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        candidate = Path(checkpoint_path).expanduser()
        file_dir = Path(__file__).resolve()
        app_root = file_dir.parents[3]

        candidates = [
            candidate,
            Path.cwd() / candidate,
            app_root / candidate,
            app_root / "data_generation/models/checkpoint-200",
            Path("/app/data_generation/models/checkpoint-200"),
            Path("/app/src/data_generation/models/checkpoint-200"),
        ]

        for path in candidates:
            if path.is_dir() and (path / "config.json").is_file():
                return str(path)

        checked = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Checkpoint directory not found or incomplete for '{checkpoint_path}'. "
            f"Tried: {checked}"
        )

    async def process_extraction_request(self, input_str: str) -> List[Any]:
        full_input = f"<start_of_turn>user\n{input_str}<end_of_turn>\n<start_of_turn>model\n"

        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        def parse_model_json(text):
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        prediction = parse_model_json(generated)

        return list(prediction.values())


class GemmaFormStateModel(IExtractionModel):
    """Incremental form-state extraction using the fine-tuned functiongemma model.

    Processes a conversation line-by-line, carrying forward the form state at
    each step.  The prompt format matches train_form_state.py exactly.
    """

    def __init__(self, model_path: str, max_new_tokens: int = 512, device: str | None = None):
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading GemmaFormStateModel from %s (device=%s)", model_path, self.device)
        resolved = self._resolve_path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else (
                torch.float16 if self.device == "cuda" else torch.float32
            ),
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()

    @staticmethod
    def _resolve_path(model_path: str) -> str:
        candidates = [
            Path(model_path).expanduser(),
            Path("/app/data_generation/models/form_state/merged"),
        ]
        for p in candidates:
            if p.is_dir() and (p / "config.json").is_file():
                return str(p)
        raise FileNotFoundError(
            f"Form-state model not found. Tried: {', '.join(str(c) for c in candidates)}"
        )

    # ---- helpers ----

    @staticmethod
    def _split_lines(text: str) -> list[str]:
        return [ln for ln in text.split("\n") if ln.strip()]

    @staticmethod
    def _format_lines_before(lines: list[str]) -> str:
        return "\n".join(lines) if lines else "(none)"

    def _build_prompt(self, new_line: str, lines_before: list[str], current_form_state_json: str) -> str:
        return (
            "<start_of_turn>user\n"
            "Update the form state given the new conversation line.\n\n"
            f"New line: {new_line}\n\n"
            f"Previous lines:\n{self._format_lines_before(lines_before)}\n\n"
            f"Current form state:\n{current_form_state_json}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_form_state(text: str) -> dict:
        """Extract JSON form state from model output (direct JSON, no prefix)."""
        text = text.strip()
        # Try parsing the whole text as JSON first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fall back to extracting first JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _flatten_state(state: dict) -> dict:
        """Flatten nested form state to {field_key: value}."""
        flat = {}
        for section_key, section_val in state.items():
            if isinstance(section_val, dict):
                for k, v in section_val.items():
                    flat[k] = v
            else:
                flat[section_key] = section_val
        return flat

    @staticmethod
    def _build_initial_form_state(field_keys: list[str]) -> dict:
        """Build the initial form state structure matching training data format."""
        return {
            "Initial fields": {k: "N/A" for k in field_keys},
        }

    # ---- IExtractionModel interface ----

    async def extract_batch(self, requests: List[ExtractionRequest]) -> List[Any]:
        """Not used for this model — delegates to process_extraction_request."""
        raise NotImplementedError("GemmaFormStateModel uses process_extraction_request, not extract_batch")

    async def process_extraction_request(self, input_str: str, field_keys: list[str] | None = None) -> List[Any]:
        """Run incremental form-state extraction over the conversation.

        *input_str* follows the format used by the pipeline:
            Extract info from conversation to fill form.
            Conversation: ...
            Form: ...
            Fields: {"field1": "N/A", ...}

        We parse the conversation and fields from it, then process line-by-line.
        """
        # Parse conversation text from the pipeline's input_str
        convo_text = ""
        fields_json_str = "{}"
        if "Conversation:" in input_str:
            after_convo = input_str.split("Conversation:", 1)[1]
            if "Form:" in after_convo:
                convo_text = after_convo.split("Form:", 1)[0].strip()
            else:
                convo_text = after_convo.strip()
        if "Fields:" in input_str:
            fields_json_str = input_str.split("Fields:", 1)[1].strip()

        # Figure out field keys from the pipeline-provided fields or the arg
        try:
            fields_dict = json.loads(fields_json_str)
            parsed_keys = list(fields_dict.keys())
        except json.JSONDecodeError:
            parsed_keys = []

        if field_keys:
            all_keys = field_keys
        elif parsed_keys:
            all_keys = parsed_keys
        else:
            all_keys = []

        # Build initial form state
        form_state = self._build_initial_form_state(all_keys)
        form_state_json = json.dumps(form_state, indent=2)

        # Process each line incrementally
        lines = self._split_lines(convo_text)
        for i, line in enumerate(lines):
            lines_before = lines[max(0, i - 10):i]
            prompt = self._build_prompt(line, lines_before, form_state_json)
            output = self._generate(prompt)

            logger.info("[GemmaFormState] Line %d/%d: %s", i + 1, len(lines), line)
            logger.info("[GemmaFormState] Output: %s", output)

            new_state = self._parse_form_state(output)
            if new_state:
                form_state = new_state
                form_state_json = json.dumps(form_state, indent=2)

        # Flatten the final form state and return values in field_keys order
        flat = self._flatten_state(form_state)
        return [flat.get(k, "N/A") for k in all_keys]