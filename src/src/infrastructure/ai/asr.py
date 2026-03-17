import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from transformers import pipeline


class LocalASRTranscriber:
    """Local Whisper-based transcriber for uploaded/recorded audio."""

    def __init__(self, model_name: str = "openai/whisper-small"):
        self.model_name = model_name
        self._asr_pipeline: Any | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _get_pipeline(self):
        if self._asr_pipeline is None:
            self._asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
            )
        return self._asr_pipeline

    async def transcribe_to_text(self, audio_bytes: bytes, filename: str = "", input_language: str = "en") -> str:
        if not audio_bytes:
            return ""

        suffix = Path(filename).suffix or ".wav"
        tmp_path = ""

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            asr = self._get_pipeline()
            language = (input_language or "").strip().lower()
            generate_kwargs: Dict[str, Any] = {"task": "transcribe"}
            if language and language != "en":
                generate_kwargs["language"] = language

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: asr(tmp_path, generate_kwargs=generate_kwargs),
            )

            if isinstance(result, dict):
                return str(result.get("text", "")).strip()
            return str(result).strip()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
