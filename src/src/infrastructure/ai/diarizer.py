"""
Speaker Diarization Pipeline
=============================
Implements the approach described in:
  "Speaker Diarization using Whisper, pyannote & Agglomerative Clustering"

Pipeline steps:
  1. Write audio bytes to a temporary WAV file (converting via soundfile/librosa if needed).
  2. Run Whisper ASR with return_timestamps=True to get time-stamped text segments.
  3. For each segment, extract a speaker embedding from the pre-trained
     speechbrain/spkrec-ecapa-voxceleb model via pyannote.
  4. Cluster embeddings with AgglomerativeClustering (n_clusters = num_speakers).
  5. Return a merged, speaker-labelled transcript list.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers — kept outside the class so they are module-level
# singletons once loaded, but the import only happens on first use so the
# container still starts up even if pyannote is not yet installed.
# ---------------------------------------------------------------------------

_embedding_model: Any = None
_audio_helper: Any = None


def _get_embedding_model(device_str: str = "cpu"):
    """Lazy-load the pyannote PretrainedSpeakerEmbedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            import torch
            from pyannote.audio.pipelines.speaker_verification import (
                PretrainedSpeakerEmbedding,
            )

            device = torch.device(device_str)
            _embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb", device=device
            )
            logger.info("Speaker embedding model loaded (device=%s).", device_str)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load speaker embedding model: {exc}. "
                "Ensure pyannote.audio is installed."
            ) from exc
    return _embedding_model


def _get_audio_helper():
    """Lazy-load pyannote Audio helper."""
    global _audio_helper
    if _audio_helper is None:
        try:
            from pyannote.audio import Audio

            _audio_helper = Audio()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pyannote Audio helper: {exc}."
            ) from exc
    return _audio_helper


# ---------------------------------------------------------------------------
# Helper: convert arbitrary audio bytes to WAV (16 kHz mono)
# ---------------------------------------------------------------------------

def _bytes_to_wav(audio_bytes: bytes, src_suffix: str) -> str:
    """
    Write *audio_bytes* to a temp file, convert to 16 kHz mono WAV using
    librosa + soundfile (both already in requirements.txt).
    Returns the path to the resulting WAV tempfile (caller must delete).
    """
    import librosa
    import soundfile as sf
    import warnings

    src_path = ""
    wav_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=src_suffix) as src:
            src.write(audio_bytes)
            src_path = src.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
            wav_path = dst.name

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(src_path, sr=16000, mono=True)

        sf.write(wav_path, y, sr)
        return wav_path
    finally:
        if src_path and os.path.exists(src_path):
            os.remove(src_path)


# ---------------------------------------------------------------------------
# Helper: get WAV duration in seconds
# ---------------------------------------------------------------------------

def _wav_duration(wav_path: str) -> float:
    with contextlib.closing(wave.open(wav_path, "r")) as f:
        return f.getnframes() / float(f.getframerate())


# ---------------------------------------------------------------------------
# Core synchronous diarization
# ---------------------------------------------------------------------------

def _diarize_sync(
    audio_bytes: bytes,
    filename: str,
    num_speakers: int,
    input_language: str,
) -> list[dict]:
    """
    Blocking implementation — meant to be run in a ThreadPoolExecutor.

    Returns a list of dicts: [{"speaker": "SPEAKER 1", "text": "..."}, ...]
    where consecutive same-speaker entries are already merged.
    """
    from transformers import pipeline as hf_pipeline
    from sklearn.cluster import AgglomerativeClustering
    from pyannote.core import Segment

    suffix = Path(filename).suffix or ".wav"
    wav_path = _bytes_to_wav(audio_bytes, src_suffix=suffix)

    try:
        # ------------------------------------------------------------------
        # 1. ASR — get time-stamped word/chunk segments
        # ------------------------------------------------------------------
        asr = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
        )
        lang = (input_language or "en").strip().lower()
        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if lang and lang != "en":
            generate_kwargs["language"] = lang

        result = asr(
            wav_path,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

        # Extract chunk-level segments (each has {"timestamp": (start, end), "text": ...})
        segments: list[dict] = []
        if isinstance(result, dict):
            raw_chunks = result.get("chunks") or []
            for chunk in raw_chunks:
                ts = chunk.get("timestamp") or (None, None)
                start, end = ts if (ts and len(ts) == 2) else (None, None)
                text = str(chunk.get("text", "")).strip()
                if text and start is not None and end is not None:
                    segments.append({"start": float(start), "end": float(end), "text": text})

        if not segments:
            # Fallback: single segment covering the whole file
            duration = _wav_duration(wav_path)
            flat_text = str(result.get("text", "")).strip() if isinstance(result, dict) else str(result).strip()
            if flat_text:
                segments = [{"start": 0.0, "end": duration, "text": flat_text}]
            else:
                return []

        if len(segments) == 1 or num_speakers <= 1:
            # No diarization needed
            return [{"speaker": "SPEAKER 1", "text": " ".join(s["text"] for s in segments)}]

        # ------------------------------------------------------------------
        # 2. Speaker embeddings
        # ------------------------------------------------------------------
        duration = _wav_duration(wav_path)
        embedding_model = _get_embedding_model("cpu")
        audio_helper = _get_audio_helper()

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, seg in enumerate(segments):
            start = seg["start"]
            end = min(duration, seg["end"])
            if end <= start:
                end = min(duration, start + 0.5)
            try:
                clip = Segment(start, end)
                waveform, _sr = audio_helper.crop(wav_path, clip)
                emb = embedding_model(waveform[None])
                embeddings[i] = emb.detach().cpu().numpy().squeeze()
            except Exception as exc:
                logger.warning("Embedding failed for segment %d: %s", i, exc)

        embeddings = np.nan_to_num(embeddings)

        # ------------------------------------------------------------------
        # 3. Cluster
        # ------------------------------------------------------------------
        n_clusters = min(num_speakers, len(segments))
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
        labels = clustering.labels_

        for i, seg in enumerate(segments):
            seg["speaker"] = f"SPEAKER {int(labels[i]) + 1}"

        # ------------------------------------------------------------------
        # 4. Merge consecutive same-speaker segments
        # ------------------------------------------------------------------
        merged: list[dict] = []
        for seg in segments:
            if merged and merged[-1]["speaker"] == seg["speaker"]:
                merged[-1]["text"] = merged[-1]["text"] + " " + seg["text"]
            else:
                merged.append({"speaker": seg["speaker"], "text": seg["text"]})

        return merged

    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


# ---------------------------------------------------------------------------
# Public async class
# ---------------------------------------------------------------------------

class LocalSpeakerDiarizer:
    """
    Async-friendly speaker diarization using:
      - Whisper (via HuggingFace transformers) for timestamped ASR
      - speechbrain/spkrec-ecapa-voxceleb for speaker embeddings
      - scikit-learn AgglomerativeClustering

    Usage::

        diarizer = LocalSpeakerDiarizer()
        turns = await diarizer.diarize(audio_bytes, "recording.webm", num_speakers=2)
        # turns = [{"speaker": "SPEAKER 1", "text": "..."}, ...]
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()

    async def diarize(
        self,
        audio_bytes: bytes,
        filename: str = "",
        num_speakers: int = 2,
        input_language: str = "en",
    ) -> list[dict]:
        """
        Returns a list of speaker-labelled turns:
            [{"speaker": "SPEAKER 1", "text": "Hello there."}, ...]

        Consecutive turns from the same speaker are merged into one entry.
        """
        if not audio_bytes:
            return []

        num_speakers = max(1, int(num_speakers))

        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: _diarize_sync(
                    audio_bytes,
                    filename or "audio.wav",
                    num_speakers,
                    input_language,
                ),
            )
