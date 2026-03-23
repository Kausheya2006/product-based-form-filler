"""
src/infrastructure/ai/mock_models.py

Mock implementations of IExtractionModel and ISummarizer.
Used when MOCK_MODELS=true so the app starts without any ML models or GPU.
"""
from typing import List, Any
from ...domain.interfaces import IExtractionModel, ISummarizer
from ...domain.domain import ExtractionRequest


class MockExtractionModel(IExtractionModel):
    """Returns 'N/A' for every field — no model loaded."""

    async def extract_batch(self, requests: List[ExtractionRequest]) -> List[Any]:
        return ["N/A"] * len(requests)

    async def process_extraction_request(self, input_str: str) -> List[Any]:
        import json, re
        try:
            match = re.search(r"\{.*\}", input_str, re.DOTALL)
            fields = json.loads(match.group()) if match else {}
            return ["N/A"] * len(fields)
        except Exception:
            return ["N/A"] * 20


class MockSummarizer(ISummarizer):
    """Returns a static placeholder — no model loaded."""

    async def summarize(self, text: str) -> str:
        return "[Mock mode — summary unavailable without GPU]"