import json
import logging
from typing import Any, List

import httpx

from ...domain.domain import ExtractionRequest
from ...domain.interfaces import IExtractionModel, ISummarizer
from .mock_models import MockExtractionModel, MockSummarizer

logger = logging.getLogger(__name__)


class RemoteModelServiceExtractionModel(IExtractionModel):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.mock_model = MockExtractionModel()

    async def extract_batch(self, requests: List[ExtractionRequest]) -> List[Any]:
        raise NotImplementedError("Remote extraction uses process_extraction_request in this project.")

    async def process_extraction_request(self, input_str: str, field_keys: list[str] | None = None) -> List[Any]:
        payload = {
            "input_str": input_str,
            "field_keys": field_keys or [],
        }
        timeout = httpx.Timeout(120.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/extract", json=payload)
                response.raise_for_status()
                data = response.json()
                return list(data.get("answers", []))
            except Exception as exc:
                logger.warning("Model service extraction unavailable, falling back to mock model: %s", exc)
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
            "conversation_text": conversation_text,
            "form_name": form_name,
            "current_field_state": current_field_state,
            "field_keys": field_keys,
            "accepted_new_fields": accepted_new_fields or {},
        }
        timeout = httpx.Timeout(120.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/live-extract", json=payload)
                response.raise_for_status()
                data = response.json()
                if "result" in data:
                    return data.get("result", {})
                return list(data.get("answers", []))
            except Exception as exc:
                logger.warning("Model service live extraction unavailable, falling back to mock model: %s", exc)
                return await self.mock_model.process_live_update(
                    conversation_text=conversation_text,
                    form_name=form_name,
                    current_field_state=current_field_state,
                    field_keys=field_keys,
                    accepted_new_fields=accepted_new_fields or {},
                )


class RemoteModelServiceSummarizer(ISummarizer):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.mock_summarizer = MockSummarizer()

    async def summarize(self, text: str) -> str:
        payload = {
            "text": text,
        }
        timeout = httpx.Timeout(60.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/summarize", json=payload)
                response.raise_for_status()
                data = response.json()
                return str(data.get("summary", ""))
            except Exception as exc:
                logger.warning("Model service summarization unavailable, falling back to mock summarizer: %s", exc)
                return await self.mock_summarizer.summarize(text)
