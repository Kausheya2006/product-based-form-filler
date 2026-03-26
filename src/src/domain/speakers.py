from __future__ import annotations

import re
from collections.abc import Mapping


_SPEAKER_TIMESTAMP_SUFFIX_RE = re.compile(r"^(?P<speaker>.+?)\s+(?P<timestamp>\d{6,})$")


def strip_speaker_timestamp_suffix(raw_speaker: str) -> str:
    speaker = (raw_speaker or "").strip()
    match = _SPEAKER_TIMESTAMP_SUFFIX_RE.match(speaker)
    if not match:
        return speaker
    stripped = match.group("speaker").strip()
    return stripped or speaker


def render_history_for_model(history: Mapping[str, str]) -> str:
    return "\n".join(
        f"{strip_speaker_timestamp_suffix(speaker)}: {text}"
        for speaker, text in history.items()
    )
