import asyncio
import logging
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class LocalASRTranscriber:
    """Local Whisper-based transcriber for uploaded/recorded audio."""

    def __init__(self, model_name: str = "openai/whisper-small"):
        self.model_name = model_name
        self._asr_pipeline: Any | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

    def _ensure_model(self) -> None:
        if self._asr_pipeline is None:
            device = 0 if self._device == "cuda" else -1
            self._asr_pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model_name,
                torch_dtype=self._dtype,
                device=device,
            )

    def _transcribe_sync(self, audio_path: str, input_language: str) -> list[dict[str, Any]]:
        self._ensure_model()
        lang = (input_language or "").strip().lower() or "en"

        result: dict[str, Any]
        try:
            result = self._asr_pipeline(
                audio_path,
                return_timestamps=True,
                generate_kwargs={"language": lang, "task": "transcribe"},
            )
        except Exception as exc:
            logger.warning(
                "ASR native audio path failed; falling back to librosa-loaded waveform input. Error: %s",
                exc,
            )
            import librosa

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

            result = self._asr_pipeline(
                {"array": waveform, "sampling_rate": sample_rate},
                return_timestamps=True,
                generate_kwargs={"language": lang, "task": "transcribe"},
            )

        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        normalized: list[dict[str, Any]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            ts = chunk.get("timestamp")
            if not isinstance(ts, (tuple, list)) or len(ts) != 2:
                continue
            normalized.append(
                {
                    "timestamp": (ts[0], ts[1]),
                    "text": text,
                }
            )
        return normalized

    async def transcribe_to_text(self, audio_bytes: bytes, filename: str = "", input_language: str = "en") -> str:
        if not audio_bytes:
            return ""

        suffix = Path(filename).suffix or ".wav"
        tmp_path = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(
                self._executor,
                lambda: self._transcribe_sync(tmp_path, input_language),
            )
            return " ".join(str(chunk.get("text", "")).strip() for chunk in chunks if chunk.get("text", "")).strip()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
