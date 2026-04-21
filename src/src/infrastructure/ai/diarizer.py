"""
Speaker Diarization Pipeline
=============================
Uses pyannote/speaker-diarization@2.1 for speaker turn detection and
Whisper (via LocalASRTranscriber) for timestamped transcription.

Pipeline steps:
  1. Convert audio bytes to a temporary 16 kHz mono WAV file.
  2. Run pyannote/speaker-diarization@2.1 to get speaker-labelled time intervals.
  3. Run Whisper ASR (return_timestamps=True) to get text chunks with timestamps.
  4. Assign each Whisper chunk the dominant speaker from pyannote output by overlap.
  5. Merge consecutive same-speaker chunks and return speaker-labelled transcript.

Prerequisites:
  - HUGGINGFACE_ACCESS_TOKEN env var (access must be granted to both
    hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation)
  - ffmpeg installed on the system (required by pyannote's audio backend)
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .asr import LocalASRTranscriber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — loaded lazily on first use so the container
# starts up even if pyannote or the HF token are not yet configured.
# ---------------------------------------------------------------------------

_diarization_pipeline: Any = None
_asr_transcriber: Any = None


def _get_diarization_pipeline():
    """Lazy-load the pyannote/speaker-diarization@2.1 pipeline."""
    global _diarization_pipeline
    if _diarization_pipeline is None:
        try:
            from pyannote.audio import Pipeline
            import torch

            token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN", "").strip()
            if not token:
                raise RuntimeError(
                    "HUGGINGFACE_ACCESS_TOKEN is not set. "
                    "Create a token at hf.co/settings/tokens, accept the model "
                    "conditions at hf.co/pyannote/speaker-diarization and "
                    "hf.co/pyannote/segmentation, then add the token to .env."
                )

            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=token,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _diarization_pipeline.to(device)
            logger.info(
                "pyannote speaker-diarization@2.1 pipeline loaded (device=%s).", device
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pyannote diarization pipeline: {exc}. "
                "Ensure pyannote.audio is installed and HUGGINGFACE_ACCESS_TOKEN is set."
            ) from exc
    return _diarization_pipeline


def _get_asr_transcriber():
    """Lazy-load the Whisper ASR transcriber."""
    global _asr_transcriber
    if _asr_transcriber is None:
        _asr_transcriber = LocalASRTranscriber(model_name="openai/whisper-small")
    return _asr_transcriber


# ---------------------------------------------------------------------------
# Helper: convert arbitrary audio bytes to 16 kHz mono WAV
# ---------------------------------------------------------------------------

def _bytes_to_wav(audio_bytes: bytes, src_suffix: str) -> str:
    """
    Write *audio_bytes* to a temp file and convert to a 16 kHz mono WAV using
    librosa + soundfile.  Returns the path to the WAV file (caller must delete).
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
# Helper: map a time window to a pyannote speaker label
# ---------------------------------------------------------------------------

def _assign_speaker(diarization: Any, start: float, end: float) -> str:
    """
    Given a pyannote ``Annotation`` and a time window [start, end], return the
    speaker label with the greatest cumulative overlap in that window.
    Falls back to ``"SPEAKER_00"`` if no diarized turn overlaps at all.
    """
    speaker_times: dict[str, float] = {}
    for seg, _, label in diarization.itertracks(yield_label=True):
        overlap = min(seg.end, end) - max(seg.start, start)
        if overlap > 0.0:
            speaker_times[label] = speaker_times.get(label, 0.0) + overlap
    if not speaker_times:
        return "SPEAKER_00"
    return max(speaker_times, key=speaker_times.__getitem__)


# ---------------------------------------------------------------------------
# Core synchronous diarization (runs in a ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _diarize_sync(
    audio_bytes: bytes,
    filename: str,
    num_speakers: int,
    input_language: str,
) -> list[dict]:
    """
    Blocking implementation — meant to be run inside a ThreadPoolExecutor.

    Returns a list of dicts: [{"speaker": "SPEAKER_00", "text": "..."}, ...]
    where consecutive same-speaker entries are already merged.
    """
    suffix = Path(filename).suffix or ".wav"
    wav_path = _bytes_to_wav(audio_bytes, src_suffix=suffix)

    try:
        # ------------------------------------------------------------------
        # 1. Run pyannote speaker diarization → time-labelled speaker turns
        # ------------------------------------------------------------------
        pipeline = _get_diarization_pipeline()

        pipeline_kwargs: dict[str, Any] = {}
        if num_speakers > 1:
            pipeline_kwargs["num_speakers"] = num_speakers

        logger.info("Running pyannote diarization (num_speakers=%d)...", num_speakers)
        diarization = pipeline(wav_path, **pipeline_kwargs)

        # ------------------------------------------------------------------
        # 2. Run Whisper ASR → timestamped text chunks
        # ------------------------------------------------------------------
        asr = _get_asr_transcriber()
        asr_chunks = asr._transcribe_sync(wav_path, input_language)

        if not asr_chunks:
            logger.warning("ASR returned no chunks; returning empty result.")
            return []

        # ------------------------------------------------------------------
        # 3. Assign each Whisper chunk a speaker from the pyannote output
        # ------------------------------------------------------------------
        segments: list[dict] = []
        for chunk in asr_chunks:
            if not isinstance(chunk, dict):
                continue
            timestamp = chunk.get("timestamp")
            if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
                continue
            start, end = timestamp
            if start is None or end is None:
                continue
            try:
                start_f = float(start)
                end_f = float(end)
            except (TypeError, ValueError):
                continue
            if end_f <= start_f:
                continue
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue

            speaker = _assign_speaker(diarization, start_f, end_f)
            segments.append({"speaker": speaker, "text": text})

        if not segments:
            return []

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
      - ``pyannote/speaker-diarization@2.1`` for speaker turn detection
      - ``openai/whisper-small`` for timestamped transcription

    Requires ``HUGGINGFACE_ACCESS_TOKEN`` in the environment, with access
    granted to ``pyannote/speaker-diarization`` and ``pyannote/segmentation``
    on huggingface.co.

    Usage::

        diarizer = LocalSpeakerDiarizer()
        turns = await diarizer.diarize(audio_bytes, "recording.webm", num_speakers=2)
        # turns = [{"speaker": "SPEAKER_00", "text": "Hello there."}, ...]
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
        Returns a list of speaker-labelled turns::

            [{"speaker": "SPEAKER_00", "text": "Hello there."}, ...]

        Consecutive turns from the same speaker are merged into one entry.

        Args:
            audio_bytes:    Raw audio data (any format supported by librosa/ffmpeg).
            filename:       Original filename — used only to infer the file suffix.
            num_speakers:   Expected number of speakers (passed to pyannote pipeline).
                            Set to 1 to skip diarization and return a single turn.
            input_language: ISO 639-1 language code for Whisper (default: "en").
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
