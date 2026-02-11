import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Optional, Tuple
from transformers import pipeline
from ...domain.interfaces import IExtractionModel
from ...domain.domain import ExtractionRequest
from ..config import settings

LINE_PATTERN = re.compile(r"^\s*(.+?)\s+(\d+)\s*:\s*(.*)\s*$")

def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())

def parse_line(line: str) -> Optional[Tuple[str, str, str]]:
    """
    Parses: <speaker> <timestamp>: <message>
    Returns (speaker, timestamp, message) or None if not match.
    Speaker can contain spaces.
    """
    m = LINE_PATTERN.match(line)
    if not m:
        return None
    speaker = m.group(1).strip()
    ts = m.group(2).strip()
    msg = m.group(3).strip()
    return speaker, ts, msg

def smart_turn_chunks_generic(
    text: str,
    max_words: int = 100, # default
    carry_incomplete_last_line: bool = True
) -> List[str]:
    """
    Splits conversation into chunks based on WORD COUNT, ensuring:
      - each chunk ends with the "second speaker"
      - message length determines how many pairs fit in a chunk
      - only splits a pair at word boundaries if that single pair > max_words
    """

    if not text:
        return []

    ends_with_newline = text.endswith("\n")
    raw_lines = text.split("\n")

    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()

    if carry_incomplete_last_line and (not ends_with_newline) and raw_lines:
        raw_lines = raw_lines[:-1]

    raw_lines = [ln.strip() for ln in raw_lines if ln.strip()]

    lines: List[Tuple[str, str, str, str, int]] = []
    for ln in raw_lines:
        parsed = parse_line(ln)
        if parsed is None:
            continue
        speaker, ts, msg = parsed
        lines.append((ln, speaker, ts, msg, count_words(ln)))

    if len(lines) < 2:
        return []

    second_speaker = lines[1][1]

    def is_s2(item) -> bool:
        return item[1] == second_speaker

    # ── Step 1: group lines into pairs (each pair ends with S2) ──
    pairs: List[List[Tuple]] = []
    current_pair: List[Tuple] = []
    for item in lines:
        current_pair.append(item)
        if is_s2(item):
            pairs.append(current_pair)
            current_pair = []
    # incomplete pair (no S2 at end) is discarded

    if not pairs:
        return []

    # ── Step 2: build chunks from pairs, respecting max_words ──
    chunks: List[str] = []
    chunk_items: List[Tuple] = []
    chunk_words = 0

    for pair in pairs:
        pair_words = sum(item[4] for item in pair)

        # Would adding this pair overflow?
        if chunk_items and chunk_words + pair_words > max_words:
            # Emit current chunk (it already ends with S2)
            chunks.append("\n".join(item[0] for item in chunk_items))
            chunk_items = []
            chunk_words = 0

        # Single pair exceeds max_words → split at word boundaries
        if pair_words > max_words and not chunk_items:
            pair_text = "\n".join(item[0] for item in pair)
            words = pair_text.split()
            sub: List[str] = []
            for w in words:
                sub.append(w)
                if len(sub) >= max_words:
                    chunks.append(" ".join(sub))
                    sub = []
            if sub:
                chunks.append(" ".join(sub))
        else:
            chunk_items.extend(pair)
            chunk_words += pair_words

    # Flush remaining chunk (guaranteed to end with S2)
    if chunk_items:
        chunks.append("\n".join(item[0] for item in chunk_items))

    return chunks

class LocalHuggingFaceModel(IExtractionModel):
    def __init__(self):
        model_name = settings.MODEL_NAME
        print(f"Loading local model: {model_name}...")
        self.pipeline = pipeline("question-answering", model=model_name)
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def extract_batch(self, requests: List[ExtractionRequest]) -> List[Any]:
        """
        Extracts multiple fields in one go. 
        HuggingFace pipelines are optimized for batching.
        """
        if not requests:
            return []

        # Prepare inputs for the pipeline
        # Passing separate lists for questions and contexts to avoid DeprecationWarning
        # system_prompt = "You are retrieving information from the above conversation to retrieve the data of a particular field. Answer N/A if the information is not available. Retrieve the information which matches this given field name the closest: "
        questions = [req.instruction for req in requests]
        contexts = [req.context for req in requests]

        loop = asyncio.get_running_loop()
        try:
            # We run the batch inference call in a separate thread
            # pipeline(question=..., context=...) handles batching natively
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.pipeline(question=questions, context=contexts)
            )
            # results will be [{'score':.., 'start':.., 'end':.., 'answer':..}, ...]
            return [r['answer'] if r else None for r in results]
        except Exception as e:
            print(f"Batch extraction error: {e}")
            return [None] * len(requests)
