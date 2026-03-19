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
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging
try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None
try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None
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

class LocalHuggingFaceModel(IExtractionModel):
    def __init__(self):
        _enable_model_download_progress()
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
        _enable_model_download_progress()
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
    """Incremental form-state extraction using the trained mono-model adapter.

    Processes a conversation line-by-line, carrying forward form state and
    summary at each step.
    """

    TOOL_SCHEMA = [{
        "type": "function",
        "function": {
            "name": "update_form_state",
            "description": "Updates the form state and summary based on the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {"type": "string"},
                    "next_form_state": {"type": "object"},
                    "new_summary_state": {"type": "string"},
                },
                "required": ["thinking", "next_form_state", "new_summary_state"],
            },
        },
    }]

    def __init__(self, model_path: str, max_new_tokens: int = 256, device: str | None = None):
        _enable_model_download_progress()
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading incremental form-state model from %s (device=%s)", model_path, self.device)
        resolved = self._resolve_path(model_path)
        self.adapter_path = resolved
        self.base_model_id = self._resolve_base_model_id(resolved)
        tokenizer_source = resolved if (Path(resolved) / "tokenizer.json").is_file() else self.base_model_id
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.model = self._load_model()
        self.model.eval()
        logger.info("Model ready")

    @staticmethod
    def _resolve_path(model_path: str) -> str:
        app_root = Path(__file__).resolve().parents[3]
        candidates = [
            Path(model_path).expanduser(),
            Path.cwd() / model_path,
            app_root / model_path,
            app_root / "data_generation/monomodel/model",
            Path("/app/data_generation/models/form_state/merged"),
            Path("/app/data_generation/monomodel/model"),
        ]
        for p in candidates:
            if p.is_dir() and ((p / "config.json").is_file() or (p / "adapter_config.json").is_file()):
                return str(p)
        raise FileNotFoundError(
            f"Form-state model not found. Tried: {', '.join(str(c) for c in candidates)}"
        )

    @staticmethod
    def _resolve_base_model_id(adapter_path: str) -> str:
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        fallback = "Qwen/Qwen2.5-1.5B-Instruct"
        if not adapter_config_path.is_file():
            return adapter_path
        try:
            with adapter_config_path.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
            configured_path = config.get("base_model_name_or_path") or fallback
        except Exception:
            return fallback
        candidate = Path(str(configured_path)).expanduser()
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return str(candidate)
        return fallback

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
                logger.info("Loading base model...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    trust_remote_code=True,
                    **kwargs,
                )
                logger.info("Base model loaded")
                if (Path(self.adapter_path) / "adapter_config.json").is_file():
                    if PeftModel is None:
                        raise ImportError("peft is required to load the monomodel adapter")
                    logger.info("Attaching LoRA adapter...")
                    model = PeftModel.from_pretrained(base_model, self.adapter_path)
                    logger.info("Adapter attached")
                    return model
                return base_model
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning("Incremental model load attempt failed for %s: %s", self.base_model_id, exc)
        raise RuntimeError(f"Unable to load form-state model from {self.adapter_path}") from last_error

    # ---- helpers ----

    @staticmethod
    def _split_lines(text: str) -> list[str]:
        return [ln for ln in text.split("\n") if ln.strip()]

    @staticmethod
    def _format_lines_before(lines: list[str]) -> str:
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _normalize_state_value(value: Any) -> str:
        text = "" if value is None else str(value).strip()
        return text if text else "N/A"

    @staticmethod
    def _build_form_state(field_values: dict[str, Any]) -> dict:
        return {
            "Initial fields": {
                key: GemmaFormStateModel._normalize_state_value(value)
                for key, value in field_values.items()
            }
        }

    def _build_messages(
        self,
        form_name: str,
        current_summary: str,
        current_form_state: dict[str, Any],
        lines_before: list[str],
        new_line: str,
    ) -> list[dict[str, str]]:
        system_content = (
            "You are a conversational form-filling assistant. Analyze the conversation "
            "and update the form state and summary accordingly."
        )
        user_content = (
            f"Form: {form_name}\n"
            "Description: Fill the provided form fields from the conversation.\n\n"
            f"Current Summary: {current_summary or 'No summary yet.'}\n\n"
            "Current Form State:\n"
            f"{json.dumps(current_form_state, ensure_ascii=True)}\n\n"
            "Conversation Context:\n"
            f"{self._format_lines_before(lines_before)}\n\n"
            "New Line:\n"
            f"{new_line}"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _generate(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=self.TOOL_SCHEMA,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            prompt = "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

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
            )

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_tool_payload(text: str) -> dict:
        match = re.search(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()
        for raw in (candidate, re.search(r"\{[\s\S]*\}", candidate).group(0) if re.search(r"\{[\s\S]*\}", candidate) else None):
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("arguments"), str):
                try:
                    payload["arguments"] = json.loads(payload["arguments"])
                except json.JSONDecodeError:
                    pass
            return payload if isinstance(payload, dict) else {}
        return {}

    @staticmethod
    def _extract_fields_from_text(text: str) -> dict[str, Any] | None:
        clean_text = text.replace('\\"', '"')
        raw_objects: list[str] = []

        next_form_state_match = re.search(r'"next_form_state"\s*:\s*\{', clean_text, re.IGNORECASE)
        if next_form_state_match:
            brace_start = clean_text.find("{", next_form_state_match.start())
            extracted = GemmaFormStateModel._extract_balanced_braces(clean_text, brace_start)
            if extracted:
                raw_objects.append(extracted)

        for pattern in (
            r'"Initial fields"\s*:\s*(\{[\s\S]*\})',
            r'"initial_fields"\s*:\s*(\{[\s\S]*\})',
        ):
            match = re.search(pattern, clean_text, re.DOTALL | re.IGNORECASE)
            if match:
                raw_objects.append(match.group(1))

        for raw_obj in raw_objects:
            logger.info("Raw field object from string: %s", raw_obj)
            for candidate in (
                raw_obj,
                re.sub(r",\s*}", "}", raw_obj),
                re.sub(r",?\s*\.\.\.\s*", "", raw_obj),
                re.sub(r",?\s*\.\.\.\s*", "", re.sub(r",\s*}", "}", raw_obj)),
            ):
                logger.info("Candidate field object from string: %s", candidate)
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                logger.info("Parsed field object from string: %s", parsed)
                if "Initial fields" in parsed or "initial_fields" in parsed or "New fields" in parsed or "new_fields" in parsed:
                    return parsed
                return {"Initial fields": parsed}

        extracted_pairs = GemmaFormStateModel._extract_field_pairs(clean_text)
        if extracted_pairs:
            logger.info("Recovered field pairs from malformed payload: %s", extracted_pairs)
            return {"Initial fields": extracted_pairs}
        return None

    @staticmethod
    def _extract_balanced_braces(text: str, start_index: int) -> str | None:
        if start_index < 0 or start_index >= len(text) or text[start_index] != "{":
            return None

        depth = 0
        in_string = False
        escape = False
        for index in range(start_index, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start_index:index + 1]
        return None

    @staticmethod
    def _extract_field_pairs(text: str) -> dict[str, Any]:
        field_block = ""
        for marker in ('"initial_fields"', '"Initial fields"'):
            match = re.search(rf'{re.escape(marker)}\s*:\s*\{{', text)
            if not match:
                continue
            brace_start = text.find("{", match.start())
            extracted = GemmaFormStateModel._extract_balanced_braces(text, brace_start)
            if extracted:
                field_block = extracted
                break

        if not field_block:
            field_block = text

        cleaned_block = re.sub(r",?\s*\.\.\.\s*", "", field_block)
        pairs: dict[str, Any] = {}
        pair_pattern = r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
        for key, value_token in re.findall(pair_pattern, cleaned_block):
            if value_token.startswith('"') and value_token.endswith('"'):
                try:
                    pairs[key] = json.loads(value_token)
                except json.JSONDecodeError:
                    pairs[key] = value_token[1:-1]
            elif value_token == "true":
                pairs[key] = True
            elif value_token == "false":
                pairs[key] = False
            elif value_token == "null":
                pairs[key] = None
            else:
                try:
                    pairs[key] = int(value_token) if "." not in value_token else float(value_token)
                except ValueError:
                    pairs[key] = value_token
        return pairs

    @staticmethod
    def _extract_summary_from_text(text: str) -> str:
        clean_text = text.replace('\\"', '"')
        match = re.search(r'"new_summary_state"\s*:\s*"([\s\S]*?)"', clean_text, re.DOTALL | re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _extract_updated_state(payload: dict) -> tuple[dict[str, Any] | None, str]:
        raw_arguments = payload.get("arguments")
        summary_from_text = ""

        if isinstance(raw_arguments, str):
            summary_from_text = GemmaFormStateModel._extract_summary_from_text(raw_arguments)
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                next_form_state = GemmaFormStateModel._extract_fields_from_text(raw_arguments)
                logger.info(f"Next Form State From String: {next_form_state}")
                return next_form_state, summary_from_text
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = payload

        if not isinstance(arguments, dict):
            return None, ""
        next_form_state = arguments.get("next_form_state")
        if isinstance(next_form_state, str):
            try:
                next_form_state = json.loads(next_form_state)
            except json.JSONDecodeError:
                next_form_state = GemmaFormStateModel._extract_fields_from_text(next_form_state)
        new_summary_state = arguments.get("new_summary_state")
        if not isinstance(new_summary_state, str):
            new_summary_state = summary_from_text
        logger.info(f"Next Form State: {next_form_state}")
        if not isinstance(next_form_state, dict):
            return None, new_summary_state if isinstance(new_summary_state, str) else ""
        return next_form_state, new_summary_state if isinstance(new_summary_state, str) else ""

    @staticmethod
    def _normalize_form_state_shape(state: dict[str, Any], field_keys: list[str]) -> dict[str, Any]:
        if not isinstance(state, dict):
            return {"Initial fields": {}}

        if any(key in state for key in ("Initial fields", "New fields", "initial_fields", "new_fields")):
            normalized = dict(state)
            if "initial_fields" in normalized and "Initial fields" not in normalized:
                normalized["Initial fields"] = normalized.pop("initial_fields")
            if "new_fields" in normalized and "New fields" not in normalized:
                normalized["New fields"] = normalized.pop("new_fields")
            if not isinstance(normalized.get("Initial fields"), dict):
                normalized["Initial fields"] = {}
            if "New fields" in normalized and not isinstance(normalized.get("New fields"), dict):
                normalized["New fields"] = {}
            return normalized

        # Treat direct field/value maps as updates to the main field-state bucket.
        normalized_fields = {
            key: value
            for key, value in state.items()
            if isinstance(key, str) and (not field_keys or key in field_keys)
        }
        return {"Initial fields": normalized_fields}

    @staticmethod
    def _merge_form_state(current_state: dict[str, Any], update_state: dict[str, Any], field_keys: list[str]) -> dict[str, Any]:
        merged = GemmaFormStateModel._normalize_form_state_shape(current_state, field_keys)
        normalized_update = GemmaFormStateModel._normalize_form_state_shape(update_state, field_keys)

        for section_name, section_values in normalized_update.items():
            if not isinstance(section_values, dict):
                continue
            target_section = merged.setdefault(section_name, {})
            if not isinstance(target_section, dict):
                target_section = {}
                merged[section_name] = target_section
            for key, value in section_values.items():
                if not isinstance(key, str):
                    continue
                normalized_value = GemmaFormStateModel._normalize_state_value(value)
                if normalized_value == "N/A" and key in target_section:
                    continue
                target_section[key] = normalized_value
        return merged

    @staticmethod
    def _parse_input_request(input_str: str) -> tuple[str, str, dict[str, Any]]:
        convo_text = ""
        form_name = "Form"
        fields_json_str = "{}"

        if "Conversation:" in input_str:
            after_convo = input_str.split("Conversation:", 1)[1]
            if "Form:" in after_convo:
                convo_text = after_convo.split("Form:", 1)[0].strip()
            else:
                convo_text = after_convo.strip()
        if "Form:" in input_str:
            after_form = input_str.split("Form:", 1)[1]
            form_name = after_form.splitlines()[0].strip() or "Form"
        if "Fields:" in input_str:
            fields_json_str = input_str.split("Fields:", 1)[1].strip()

        try:
            fields = json.loads(fields_json_str)
        except json.JSONDecodeError:
            fields = {}
        if not isinstance(fields, dict):
            fields = {}
        return convo_text, form_name, fields

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
        convo_text, form_name, fields_dict = self._parse_input_request(input_str)
        parsed_keys = list(fields_dict.keys())

        if field_keys:
            all_keys = field_keys
        elif parsed_keys:
            all_keys = parsed_keys
        else:
            all_keys = []

        seeded_fields = {
            key: self._normalize_state_value(fields_dict.get(key, "N/A"))
            for key in all_keys
        }
        form_state = self._build_form_state(seeded_fields)
        summary_state = ""

        logger.info("[IncrementalFormState] form_name=%s", form_name)
        logger.info("[IncrementalFormState] requested field keys=%s", all_keys)
        logger.info("[IncrementalFormState] parsed input fields=%s", fields_dict)
        logger.info("[IncrementalFormState] initial form state=%s", form_state)
        logger.info("[IncrementalFormState] conversation text=%s", convo_text)

        lines = self._split_lines(convo_text)
        for i, line in enumerate(lines):
            lines_before = lines[max(0, i - 5):i]
            messages = self._build_messages(form_name, summary_state, form_state, lines_before, line)
            logger.info("[IncrementalFormState] Prompt messages for line %d=%s", i + 1, messages)
            output = self._generate(messages)

            logger.info("[IncrementalFormState] Line %d/%d: %s", i + 1, len(lines), line)
            logger.info("[IncrementalFormState] Output: %s", output)

            payload = self._parse_tool_payload(output)
            if not payload:
                extracted_fields = self._extract_fields_from_text(output)
                if extracted_fields is not None:
                    payload = {"arguments": {"next_form_state": extracted_fields}}
            logger.info("[IncrementalFormState] Parsed payload: %s", payload)
            new_state, new_summary = self._extract_updated_state(payload)
            logger.info("[IncrementalFormState] Extracted new_state=%s", new_state)
            logger.info("[IncrementalFormState] Extracted new_summary=%s", new_summary)
            if new_state:
                form_state = self._merge_form_state(form_state, new_state, all_keys)
                logger.info("[IncrementalFormState] Merged form state: %s", form_state)
            if new_summary:
                summary_state = new_summary

        flat = self._flatten_state(form_state)
        logger.info("[IncrementalFormState] Flattened state=%s", flat)
        final_answers = [self._normalize_state_value(flat.get(k, "N/A")) for k in all_keys]
        logger.info("[IncrementalFormState] Final answers=%s", final_answers)
        return final_answers
