import asyncio
import os
import re
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import speech_recognition as sr
import soundfile as sf
# NOTE: Whisper path is intentionally commented for now (kept for future work).
# from faster_whisper import WhisperModel


class LocalSpeechToText:
    """Local STT pipeline: normalize -> VAD -> denoise -> SpeechRecognition -> ITN."""

    def __init__(self, model_size: str = "tiny.en"):
        self.model_size = model_size
        self._model = None
        self._recognizer = sr.Recognizer()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()

    def _get_model(self):
        # NOTE: Whisper path is intentionally disabled for now.
        # if self._model is None:
        #     self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        # return self._model
        raise RuntimeError("Whisper STT is temporarily disabled.")

    @staticmethod
    def _to_google_lang(input_language: str) -> str:
        code = (input_language or "en").strip().lower()
        mapping = {
            "en": "en-US",
            "es": "es-ES",
            "fr": "fr-FR",
            "de": "de-DE",
            "it": "it-IT",
            "pt": "pt-PT",
        }
        return mapping.get(code, "en-US")

    @staticmethod
    def _load_normalized_audio(audio_path: str) -> tuple[np.ndarray, int]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as exc:
            raise RuntimeError(f"STT audio decode failed: {exc}") from exc
        if y.size == 0:
            return y, 16000
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        return y.astype(np.float32), 16000

    @staticmethod
    def _vad_energy_chunks(y: np.ndarray, sr: int, frame_ms: int = 30, min_speech_ms: int = 240) -> list[np.ndarray]:
        if y.size == 0:
            return []

        frame_len = max(1, int(sr * frame_ms / 1000))
        min_frames = max(1, int(min_speech_ms / frame_ms))
        frames = [y[i:i + frame_len] for i in range(0, len(y), frame_len)]
        if not frames:
            return []

        energies = np.array([float(np.sqrt(np.mean(np.square(f))) + 1e-12) for f in frames])
        threshold = max(float(np.median(energies) * 1.8), 0.01)
        speech_mask = energies > threshold

        chunks: list[np.ndarray] = []
        start = None
        for i, is_speech in enumerate(speech_mask):
            if is_speech and start is None:
                start = i
            if not is_speech and start is not None:
                if i - start >= min_frames:
                    chunks.append(y[start * frame_len:i * frame_len])
                start = None

        if start is not None:
            end = len(speech_mask)
            if end - start >= min_frames:
                chunks.append(y[start * frame_len:end * frame_len])

        if not chunks:
            return [y]
        return chunks

    @staticmethod
    def _denoise_chunks(chunks: list[np.ndarray], sr: int) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=np.float32)

        cleaned: list[np.ndarray] = []
        for chunk in chunks:
            if chunk.size < sr // 10:
                continue
            reduced = nr.reduce_noise(y=chunk, sr=sr, prop_decrease=0.75)
            cleaned.append(reduced.astype(np.float32))
        if not cleaned:
            return np.concatenate(chunks).astype(np.float32)
        return np.concatenate(cleaned).astype(np.float32)

    @staticmethod
    def _itn_basic(text: str) -> str:
        token_to_digit = {
            "zero": "0",
            "oh": "0",
            "o": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
        }

        def repl(match: re.Match[str]) -> str:
            phrase = match.group(0).lower()
            words = phrase.split()
            out = []
            i = 0
            while i < len(words):
                w = words[i]
                if w in ("double", "triple") and i + 1 < len(words) and words[i + 1] in token_to_digit:
                    repeat = 2 if w == "double" else 3
                    out.extend([token_to_digit[words[i + 1]]] * repeat)
                    i += 2
                    continue
                if w in token_to_digit:
                    out.append(token_to_digit[w])
                    i += 1
                    continue
                return phrase
            return "".join(out) if len(out) >= 2 else phrase

        pattern = r"\b(?:double|triple|zero|oh|o|one|two|three|four|five|six|seven|eight|nine)(?:\s+(?:double|triple|zero|oh|o|one|two|three|four|five|six|seven|eight|nine))+\b"
        normalized = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _transcribe_sync(self, audio_bytes: bytes, filename: str, language: str) -> str:
        suffix = Path(filename).suffix or ".wav"
        in_path = ""
        out_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
                src.write(audio_bytes)
                in_path = src.name

            try:
                y, sample_rate = self._load_normalized_audio(in_path)
                if y.size == 0:
                    return ""

                chunks = self._vad_energy_chunks(y, sample_rate)
                cleaned = self._denoise_chunks(chunks, sample_rate)
                if cleaned.size == 0:
                    cleaned = y

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
                    out_path = dst.name
                sf.write(out_path, cleaned, sample_rate)
            except Exception as exc:
                raise RuntimeError(f"STT audio preprocessing failed: {exc}") from exc

            # NOTE: Whisper transcription path is intentionally commented for now.
            # model = self._get_model()
            # lang = (language or "en").strip().lower() or "en"
            # task_lang = "en" if lang == "en" else lang
            # segments, _ = model.transcribe(
            #     out_path,
            #     language=task_lang,
            #     beam_size=3,
            #     vad_filter=False,
            #     temperature=0.0,
            # )
            # text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
            # return self._itn_basic(text)

            google_lang = self._to_google_lang(language)
            try:
                with sr.AudioFile(out_path) as source:
                    audio_data = self._recognizer.record(source)
                text = self._recognizer.recognize_google(audio_data, language=google_lang)
            except sr.UnknownValueError:
                return ""
            except sr.RequestError as exc:
                raise RuntimeError(f"SpeechRecognition backend request failed: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"SpeechRecognition decode failed: {exc}") from exc

            return self._itn_basic(text)
        finally:
            if in_path and os.path.exists(in_path):
                os.remove(in_path)
            if out_path and os.path.exists(out_path):
                os.remove(out_path)

    async def transcribe_to_text(self, audio_bytes: bytes, filename: str = "", input_language: str = "en") -> str:
        if not audio_bytes:
            return ""
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self._transcribe_sync(audio_bytes, filename, input_language),
            )