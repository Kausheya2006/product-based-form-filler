import asyncio
import logging
from typing import Any, List

import httpx

from ...domain.domain import ExtractionRequest
from ...domain.interfaces import IExtractionModel, ISummarizer
from .mock_models import MockExtractionModel, MockSummarizer

logger = logging.getLogger(__name__)


class ModalExtractionModel(IExtractionModel):
    """Modal-backed extraction model.

    Supports either:
    1) SDK mode: call Modal function by app/function name (no endpoint in code)
    2) HTTP mode: call a configured URL (for web endpoint deployments)
    """

    def __init__(
        self,
        *,
        use_sdk: bool,
        base_url: str = "",
        app_name: str = "",
        function_name: str = "",
    ):
        self.use_sdk = use_sdk
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.function_name = function_name
        self.mock_model = MockExtractionModel()
        self._modal_function = None

        if self.use_sdk:
            if not self.app_name or not self.function_name:
                raise ValueError("Modal SDK mode requires app_name and function_name.")
            logger.info(
                "Initialized Modal extractor in SDK mode app=%s function=%s",
                self.app_name,
                self.function_name,
            )
        else:
            if not self.base_url:
                raise ValueError("Modal HTTP mode requires base_url.")
            logger.info("Initialized Modal extractor in HTTP mode base_url=%s", self.base_url)

    async def extract_batch(self, requests: List[ExtractionRequest]) -> List[Any]:
        raise NotImplementedError("Modal extractor uses process_extraction_request in this project.")

    async def _get_modal_function(self):
        if self._modal_function is not None:
            return self._modal_function
        try:
            import modal  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Modal SDK not installed. Install `modal` or switch to HTTP mode.") from exc
        self._modal_function = modal.Function.from_name(self.app_name, self.function_name)
        return self._modal_function

    async def _call_sdk(self, payload: dict[str, Any]) -> dict[str, Any]:
        fn = await self._get_modal_function()
        result = await asyncio.to_thread(fn.remote, payload)
        if not isinstance(result, dict):
            return {}
        return result

    async def _call_http(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = httpx.Timeout(120.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/{route.lstrip('/')}", json=payload)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}

    async def process_extraction_request(self, input_str: str, field_keys: list[str] | None = None) -> List[Any]:
        payload = {
            "mode": "extract",
            "input_str": input_str,
            "field_keys": field_keys or [],
        }
        try:
            if self.use_sdk:
                data = await self._call_sdk(payload)
            else:
                data = await self._call_http("/extract", payload)

            answers = data.get("answers", [])
            if isinstance(answers, list):
                return answers
            result = data.get("result")
            if isinstance(result, dict) and field_keys:
                filled = result.get("filled_data", {})
                if isinstance(filled, dict):
                    return [filled.get(key, "N/A") for key in field_keys]
            return []
        except Exception as exc:
            logger.warning("Modal extraction unavailable, falling back to mock model: %s", exc)
            return await self.mock_model.process_extraction_request(input_str)

    async def process_live_update(
        self,
        *,
        conversation_text: str,
        form_name: str,
        current_field_state: dict[str, Any],
        field_keys: list[str],
        accepted_new_fields: dict[str, Any] | None = None,
    ) -> Any:
        payload = {
            "mode": "live_extract",
            "conversation_text": conversation_text,
            "form_name": form_name,
            "current_field_state": current_field_state,
            "field_keys": field_keys,
            "accepted_new_fields": accepted_new_fields or {},
        }
        try:
            if self.use_sdk:
                data = await self._call_sdk(payload)
            else:
                data = await self._call_http("/live-extract", payload)
            if "result" in data:
                return data.get("result", {})
            return list(data.get("answers", []))
        except Exception as exc:
            logger.warning("Modal live extraction unavailable, falling back to mock model: %s", exc)
            return await self.mock_model.process_live_update(
                conversation_text=conversation_text,
                form_name=form_name,
                current_field_state=current_field_state,
                field_keys=field_keys,
                accepted_new_fields=accepted_new_fields or {},
            )


class ModalSummarizer(ISummarizer):
    def __init__(
        self,
        *,
        use_sdk: bool,
        base_url: str = "",
        app_name: str = "",
        function_name: str = "",
    ):
        self.use_sdk = use_sdk
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.function_name = function_name
        self.mock_summarizer = MockSummarizer()
        self._modal_function = None

    async def _get_modal_function(self):
        if self._modal_function is not None:
            return self._modal_function
        try:
            import modal  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Modal SDK not installed. Install `modal` or switch to HTTP mode.") from exc
        self._modal_function = modal.Function.from_name(self.app_name, self.function_name)
        return self._modal_function

    async def summarize(self, text: str) -> str:
        payload = {"mode": "summarize", "text": text}
        try:
            if self.use_sdk:
                fn = await self._get_modal_function()
                data = await asyncio.to_thread(fn.remote, payload)
                if not isinstance(data, dict):
                    return ""
                return str(data.get("summary", ""))

            timeout = httpx.Timeout(60.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}/summarize", json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    return ""
                return str(data.get("summary", ""))
        except Exception as exc:
            logger.warning("Modal summarization unavailable, falling back to mock summarizer: %s", exc)
            return await self.mock_summarizer.summarize(text)
