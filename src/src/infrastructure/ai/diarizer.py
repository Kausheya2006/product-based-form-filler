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
import inspect
import logging
import os
import re
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from .asr import LocalASRTranscriber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers — kept outside the class so they are module-level
# singletons once loaded, but the import only happens on first use so the
# container still starts up even if pyannote is not yet installed.
# ---------------------------------------------------------------------------

_embedding_model: Any = None
_audio_helper: Any = None
_asr_transcriber: Any = None


def _patch_speechbrain_token_kwarg() -> None:
    """Patch older speechbrain versions by dropping unsupported init kwargs."""
    from speechbrain.inference.interfaces import Pretrained

    if getattr(Pretrained, "_pl_token_kwarg_patch", False):
        return

    init_sig = inspect.signature(Pretrained.__init__)
    # If this speechbrain build already accepts common HF kwargs, no patch needed.
    if (
        "token" in init_sig.parameters
        or "use_auth_token" in init_sig.parameters
        or "huggingface_cache_dir" in init_sig.parameters
    ):
        Pretrained._pl_token_kwarg_patch = True
        return

    original_init = Pretrained.__init__
    accepted = set(init_sig.parameters.keys())

    def _patched_init(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return original_init(self, *args, **filtered_kwargs)

    Pretrained.__init__ = _patched_init
    Pretrained._pl_token_kwarg_patch = True


def _get_embedding_model(device_str: str = "cpu"):
    """Lazy-load the pyannote PretrainedSpeakerEmbedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from pyannote.audio.pipelines.speaker_verification import (
                PretrainedSpeakerEmbedding,
            )

            _patch_speechbrain_token_kwarg()

            # pyannote/speechbrain compatibility: prefer plain string device.
            try:
                _embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb", device=device_str
                )
            except TypeError:
                # Some builds infer device internally and reject the explicit kwarg.
                _embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb"
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


def _get_asr_transcriber():
    """Lazy-load ASR transcriber that avoids transformers pipeline/torchcodec path."""
    global _asr_transcriber
    if _asr_transcriber is None:
        _asr_transcriber = LocalASRTranscriber(model_name="openai/whisper-small")
    return _asr_transcriber


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


def _load_wav_array(wav_path: str):
    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    return y, sr


def _build_time_windows(duration: float, window_sec: float = 2.0, hop_sec: float = 1.5) -> list[dict]:
    if duration <= 0:
        return []
    if duration <= window_sec:
        return [{"start": 0.0, "end": duration}]

    windows: list[dict] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_sec)
        windows.append({"start": start, "end": end})
        if end >= duration:
            break
        start += hop_sec
    return windows


def _split_transcript_into_chunks(transcript: str, chunk_count: int) -> list[str]:
    text = " ".join((transcript or "").split()).strip()
    if not text:
        return []
    if chunk_count <= 1:
        return [text]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    buckets = ["" for _ in range(chunk_count)]
    for i, sentence in enumerate(sentences):
        idx = min(int(i * chunk_count / max(1, len(sentences))), chunk_count - 1)
        buckets[idx] = (buckets[idx] + " " + sentence).strip()

    if all(not b for b in buckets):
        return [text]
    return [b for b in buckets if b]


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
    from sklearn.cluster import AgglomerativeClustering
    from pyannote.core import Segment

    suffix = Path(filename).suffix or ".wav"
    wav_path = _bytes_to_wav(audio_bytes, src_suffix=suffix)

    try:
        # ------------------------------------------------------------------
        # 1. ASR — torchcodec-free transcription
        # ------------------------------------------------------------------
        asr = _get_asr_transcriber()
        transcript_text = asr._transcribe_sync(wav_path, input_language)
        if not transcript_text:
            return []

        duration = _wav_duration(wav_path)
        if num_speakers <= 1:
            return [{"speaker": "SPEAKER 1", "text": transcript_text}]

        segments = _build_time_windows(duration)
        if len(segments) <= 1:
            return [{"speaker": "SPEAKER 1", "text": transcript_text}]

        # ------------------------------------------------------------------
        # 2. Speaker embeddings
        # ------------------------------------------------------------------
        embedding_model = _get_embedding_model("cpu")
        audio_helper = _get_audio_helper()
        full_waveform, _wave_sr = _load_wav_array(wav_path)

        import torch

        pyannote_audio = {
            "waveform": torch.from_numpy(full_waveform).unsqueeze(0),
            "sample_rate": 16000,
        }

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, seg in enumerate(segments):
            start = seg["start"]
            end = min(duration, seg["end"])
            if end <= start:
                end = min(duration, start + 0.5)
            try:
                clip = Segment(start, end)
                waveform, _sr = audio_helper.crop(pyannote_audio, clip)
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

        text_chunks = _split_transcript_into_chunks(transcript_text, len(segments))
        if not text_chunks:
            text_chunks = [transcript_text]

        for i, seg in enumerate(segments):
            seg["text"] = text_chunks[i] if i < len(text_chunks) else ""

        # ------------------------------------------------------------------
        # 4. Merge consecutive same-speaker segments
        # ------------------------------------------------------------------
        merged: list[dict] = []
        for seg in segments:
            if not seg.get("text", "").strip():
                continue
            if merged and merged[-1]["speaker"] == seg["speaker"]:
                merged[-1]["text"] = merged[-1]["text"] + " " + seg["text"]
            else:
                merged.append({"speaker": seg["speaker"], "text": seg["text"]})

        if not merged:
            return [{"speaker": "SPEAKER 1", "text": transcript_text}]
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
