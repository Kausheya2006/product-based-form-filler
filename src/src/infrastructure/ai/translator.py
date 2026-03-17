import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from transformers import pipeline


class LocalTranslator:
    """Small local translation helper used by ASR-assisted static extraction."""

    _MODEL_BY_LANG: Dict[str, str] = {
        "es": "Helsinki-NLP/opus-mt-es-en",
        "fr": "Helsinki-NLP/opus-mt-fr-en",
        "de": "Helsinki-NLP/opus-mt-de-en",
        "it": "Helsinki-NLP/opus-mt-it-en",
        "pt": "Helsinki-NLP/opus-mt-pt-en",
    }

    def __init__(self):
        self._pipelines: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _get_pipeline(self, source_lang: str):
        model_name = self._MODEL_BY_LANG.get(source_lang)
        if not model_name:
            return None

        if source_lang not in self._pipelines:
            self._pipelines[source_lang] = pipeline("translation", model=model_name)
        return self._pipelines[source_lang]

    async def translate_to_english(self, text: str, source_lang: str) -> str:
        source_lang = (source_lang or "en").strip().lower()
        if source_lang == "en" or not text.strip():
            return text

        translator = self._get_pipeline(source_lang)
        if translator is None:
            # Unsupported language: fail soft and continue with original text.
            return text

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: translator(text)
        )

        if isinstance(result, list) and result:
            return result[0].get("translation_text", text)
        return text
