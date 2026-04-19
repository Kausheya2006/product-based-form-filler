import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


class LocalASRTranscriber:
    """Local Whisper-based transcriber for uploaded/recorded audio."""

    def __init__(self, model_name: str = "openai/whisper-small"):
        self.model_name = model_name
        self._processor: Any | None = None
        self._model: Any | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

    def _ensure_model(self) -> None:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
        if self._model is None:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self._dtype,
            )
            self._model = model.to(self._device)

    def _transcribe_sync(self, audio_path: str, input_language: str) -> str:
        self._ensure_model()
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

        inputs = self._processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self._device)
        if self._device == "cuda":
            input_features = input_features.to(self._dtype)

        lang = (input_language or "").strip().lower() or "en"
        forced_decoder_ids = None
        try:
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=lang,
                task="transcribe",
            )
        except Exception:
            # Fall back to model defaults when language prompt cannot be derived.
            forced_decoder_ids = None

        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=256,
            )

        text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        return str(text).strip()

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
            return await loop.run_in_executor(
                self._executor,
                lambda: self._transcribe_sync(tmp_path, input_language),
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
