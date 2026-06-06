import json
import logging
from typing import Any, Dict, List

import httpx

from ...domain.domain import ExtractionRequest
from ...domain.interfaces import IExtractionModel, ISummarizer

logger = logging.getLogger(__name__)


class OllamaFormStateModel(IExtractionModel):
    """Ollama implementation for form extraction."""

    def __init__(self, model_name: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        logger.info("Initialized Ollama extractor model=%s base_url=%s", self.model_name, self.base_url)

    async def extract_batch(self, requests: List[ExtractionRequest]) -> List[Any]:
        raise NotImplementedError("Ollama extractor uses full_process mode in this project.")

    @staticmethod
    def _extract_field_keys_and_seeded(input_str: str) -> tuple[list[str], Dict[str, Any]]:
        if "Fields:" not in input_str:
            return [], {}
        raw = input_str.split("Fields:", 1)[1].strip()
        try:
            seeded = json.loads(raw)
            if isinstance(seeded, dict):
                return list(seeded.keys()), seeded
        except json.JSONDecodeError:
            pass
        return [], {}

    @staticmethod
    def _get_value_from_output(extracted_data: Dict[str, Any], dotted_key: str) -> Any:
        if dotted_key in extracted_data:
            return extracted_data[dotted_key]

        parts = dotted_key.split(".")
        current: Any = extracted_data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return "N/A"
            current = current[part]
        return current

    @staticmethod
    def _flatten_dict(data: Dict[str, Any], prefix: str = "", out: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if out is None:
            out = {}
        for key, value in (data or {}).items():
            dotted_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                OllamaFormStateModel._flatten_dict(value, dotted_key, out)
            else:
                out[dotted_key] = value
        return out

    async def process_extraction_request(self, input_str: str, field_keys: list[str] | None = None) -> List[Any]:
        keys_from_input, seeded = self._extract_field_keys_and_seeded(input_str)
        keys = field_keys or keys_from_input

        payload = {
            "model": self.model_name,
            "system": (
                "You are a form-filling extraction assistant. "
                "Return ONLY valid JSON matching the provided field schema. "
                "For missing values, use 'N/A'."
            ),
            "prompt": input_str,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
            },
        }

        timeout = httpx.Timeout(120.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                result_text = str(data.get("response", "{}")).strip()
                extracted_data = json.loads(result_text) if result_text else {}
                if not isinstance(extracted_data, dict):
                    extracted_data = {}
            except Exception as exc:
                logger.error("Ollama extraction error: %s", exc)
                return [str(seeded.get(k, "N/A")) for k in keys]

        if not keys:
            return list(extracted_data.values())
        return [self._get_value_from_output(extracted_data, k) for k in keys]

    async def process_live_update(
        self,
        *,
        conversation_text: str,
        form_name: str,
        current_field_state: dict[str, Any],
        field_keys: list[str],
        accepted_new_fields: dict[str, Any] | None = None,
    ) -> Any:
        seeded = {k: (str(current_field_state.get(k, "N/A")) or "N/A") for k in field_keys}
        accepted_new_fields = {
            key: (str(value).strip() or "N/A")
            for key, value in (accepted_new_fields or {}).items()
            if isinstance(key, str) and key.strip()
        }
        is_qwen = "qwen" in self.model_name.lower()

        if not is_qwen:
            input_str = (
                "Extract info from conversation to fill form.\n"
                f"Conversation: {conversation_text}\n"
                f"Form: {form_name}\n"
                f"Fields: {json.dumps(seeded)}"
            )
            return await self.process_extraction_request(input_str, field_keys=field_keys)

        input_str = (
            "Extract information from the conversation to fill the form.\n"
            f"Conversation: {conversation_text}\n"
            f"Form: {form_name}\n"
            f"Schema fields: {json.dumps(seeded)}\n"
            f"Accepted new fields: {json.dumps(accepted_new_fields)}\n"
            "Return only JSON with exactly two top-level keys:\n"
            "- \"filled_data\": an object containing only schema field keys\n"
            "- \"suggested_new_fields\": an object containing flat out-of-schema field/value suggestions\n"
            "Do not repeat accepted new fields inside suggested_new_fields unless the value changed.\n"
            "Use 'N/A' for missing schema values."
        )

        payload = {
            "model": self.model_name,
            "system": (
                "You are a form-filling extraction assistant. "
                "Return ONLY valid JSON following the required shape."
            ),
            "prompt": input_str,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "filled_data": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                    "suggested_new_fields": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "required": ["filled_data", "suggested_new_fields"],
            },
            "options": {
                "temperature": 0.0,
            },
        }

        timeout = httpx.Timeout(120.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                result_text = str(data.get("response", "{}")).strip()
                parsed = json.loads(result_text) if result_text else {}
                if not isinstance(parsed, dict):
                    parsed = {}
            except Exception as exc:
                logger.error("Ollama live extraction error: %s", exc)
                return {
                    "filled_data": dict(seeded),
                    "suggested_new_fields": {},
                }

        filled_candidate = parsed.get("filled_data", {})
        suggested_candidate = parsed.get("suggested_new_fields", {})
        filled_flat = self._flatten_dict(filled_candidate if isinstance(filled_candidate, dict) else {})
        suggested_flat = self._flatten_dict(suggested_candidate if isinstance(suggested_candidate, dict) else {})

        filled_data = {key: filled_flat.get(key, seeded.get(key, "N/A")) for key in field_keys}
        suggested_new_fields: Dict[str, Any] = {}
        for key, value in suggested_flat.items():
            normalized_key = str(key).strip()
            if not normalized_key or normalized_key in filled_data:
                continue
            normalized_value = str(value).strip() if value is not None else ""
            if not normalized_value:
                continue
            if accepted_new_fields.get(normalized_key) == normalized_value:
                continue
            suggested_new_fields[normalized_key] = normalized_value

        return {
            "filled_data": filled_data,
            "suggested_new_fields": suggested_new_fields,
        }


class OllamaSummarizer(ISummarizer):
    """Ollama implementation for conversation summarization."""

    def __init__(self, model_name: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        logger.info("Initialized Ollama summarizer model=%s base_url=%s", self.model_name, self.base_url)

    async def summarize(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        payload = {
            "model": self.model_name,
            "system": "Summarize the conversation in one concise paragraph. Output only the summary text.",
            "prompt": text.strip(),
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                return str(data.get("response", "")).strip()
            except Exception as exc:
                logger.error("Ollama summarization error: %s", exc)
                return ""
