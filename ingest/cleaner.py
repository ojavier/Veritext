"""Text cleaning applied to each segment after segmentation.

Cleaning is deliberately conservative: it must not alter the wording, since
the downstream signals (perplexity, embeddings, matching) depend on it.
Only invisible or layout noise is removed.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata

from src.schemas import Segment

# Zero-width chars, BOM, soft hyphen.
_INVISIBLE_RE = re.compile("[​‌‍⁠﻿­]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES_RE = re.compile(r"[ \t  -   　]+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalize Unicode, remove invisible/control chars, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)  # re-join words split at line end
    text = _SPACES_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    # Single newlines inside a paragraph are just line wrapping.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def clean_segment(segment: Segment) -> Segment:
    """Return a copy with cleaned text; start/end still point to the original."""
    return dataclasses.replace(segment, text=clean_text(segment.text))


def clean_segments(segments: list[Segment]) -> list[Segment]:
    """Clean segments and drop those left empty (ids are kept as-is)."""
    cleaned = (clean_segment(s) for s in segments)
    return [s for s in cleaned if s.text]
