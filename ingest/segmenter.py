"""Paragraph-level text segmentation.

Splits raw text into paragraphs (separated by blank lines). Paragraphs longer
than ``max_length`` are split further at sentence boundaries. Every segment
keeps ``start``/``end`` offsets into the ORIGINAL text, so
``text[seg.start:seg.end] == seg.text`` always holds. Cleaning and language
detection happen after segmentation, so ``lang`` is left empty here.
"""

from __future__ import annotations

import re

from src.schemas import Segment

DEFAULT_MAX_LENGTH = 1000

_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t\r]*\n(?:[ \t\r]*\n)*")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])[\"'»”)\]]*\s+")


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) so it excludes surrounding whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_long(text: str, start: int, end: int, max_length: int) -> list[tuple[int, int]]:
    """Split a long span into chunks <= max_length, preferring sentence ends."""
    if end - start <= max_length:
        return [(start, end)]

    # Sentence boundaries inside the span, as (sentence_start, sentence_end).
    pieces: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_END_RE.finditer(text, start, end):
        pieces.append((cursor, match.start() + len(match.group().rstrip())))
        cursor = match.end()
    pieces.append((cursor, end))

    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = start
    for p_start, p_end in pieces:
        # A single sentence longer than the limit is hard-split on whitespace.
        for s, e in _hard_split(text, p_start, p_end, max_length):
            if cur_start is None:
                cur_start, cur_end = s, e
            elif e - cur_start <= max_length:
                cur_end = e
            else:
                spans.append((cur_start, cur_end))
                cur_start, cur_end = s, e
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return spans


def _hard_split(text: str, start: int, end: int, max_length: int) -> list[tuple[int, int]]:
    """Split on the last whitespace before max_length (or mid-word if none)."""
    spans = []
    while end - start > max_length:
        cut = text.rfind(" ", start, start + max_length + 1)
        if cut <= start:
            cut = start + max_length
        s, e = _strip_span(text, start, cut)
        if e > s:
            spans.append((s, e))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    s, e = _strip_span(text, start, end)
    if e > s:
        spans.append((s, e))
    return spans


def segment_text(text: str, max_length: int = DEFAULT_MAX_LENGTH) -> list[Segment]:
    """Split ``text`` into paragraph-sized segments with original offsets."""
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    segments: list[Segment] = []
    bounds = [0]
    for match in _PARAGRAPH_BREAK_RE.finditer(text):
        bounds.extend((match.start(), match.end()))
    bounds.append(len(text))

    for i in range(0, len(bounds), 2):
        start, end = _strip_span(text, bounds[i], bounds[i + 1])
        if end <= start:
            continue
        for s, e in _split_long(text, start, end, max_length):
            segments.append(
                Segment(id=len(segments), text=text[s:e], lang="", start=s, end=e)
            )
    return segments
