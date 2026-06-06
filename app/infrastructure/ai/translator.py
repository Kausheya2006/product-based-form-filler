import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model(self, source_lang: str):
        model_name = self._MODEL_BY_LANG.get(source_lang)
        if not model_name:
            return None, None
        if source_lang not in self._models:
            self._tokenizers[source_lang] = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._models[source_lang] = model.to(self._device)
        return self._models[source_lang], self._tokenizers[source_lang]

    def _translate_sync(self, text: str, source_lang: str) -> str:
        model, tokenizer = self._ensure_model(source_lang)
        if model is None or tokenizer is None:
            return text

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256)
        return str(tokenizer.decode(generated[0], skip_special_tokens=True)).strip() or text

    async def translate_to_english(self, text: str, source_lang: str) -> str:
        source_lang = (source_lang or "en").strip().lower()
        if source_lang == "en" or not text.strip():
            return text

        model, tokenizer = self._ensure_model(source_lang)
        if model is None or tokenizer is None:
            # Unsupported language: fail soft and continue with original text.
            return text

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._translate_sync(text, source_lang),
        )
