"""Lightweight language identification for the languages supported by Veritext.

The project intentionally has no runtime dependencies, so this module uses
accent markers and small stop-word vocabularies instead of a statistical model.
It is intended for short text segments, not for linguistic classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUPPORTED_LANGUAGES = ("es", "en")
UNKNOWN_LANGUAGE = "unknown"

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_SPANISH_WORDS = frozenset(
    {
        "a",
        "al",
        "como",
        "con",
        "de",
        "del",
        "el",
        "en",
        "es",
        "esta",
        "este",
        "la",
        "las",
        "lo",
        "los",
        "para",
        "por",
        "que",
        "se",
        "su",
        "una",
        "un",
        "y",
    }
)
_ENGLISH_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
_SPANISH_MARKERS = frozenset("áéíóúüñ¿¡".casefold())
_SPANISH_SUFFIXES = ("ción", "mente", "ando", "iendo", "ados", "idos")
_ENGLISH_SUFFIXES = ("tion", "ment", "ing", "ed", "ness", "ly")


@dataclass(frozen=True)
class LanguageResult:
    """Language prediction and its confidence, both in the range [0, 1]."""

    language: str
    confidence: float


def _validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.strip()


def _words(text: str) -> list[str]:
    return [word.casefold() for word in _WORD_RE.findall(text)]


def _score(
    words: list[str],
    text: str,
    vocabulary: frozenset[str],
    suffixes: tuple[str, ...],
    *,
    accented: bool = False,
) -> float:
    score = sum(word in vocabulary for word in words)
    score += sum(any(word.endswith(suffix) for suffix in suffixes) for word in words)
    if accented:
        score += sum(character in _SPANISH_MARKERS for character in text)
    return float(score)


def identify_language(text: str) -> str:
    """Return ``es``, ``en``, or ``unknown`` for *text*.

    Empty or equally ambiguous text is reported as ``unknown`` rather than
    guessing a language.
    """

    return detect_language_with_confidence(text).language


def detect_language(text: str) -> str:
    """Compatibility alias for :func:`identify_language`."""

    return identify_language(text)


def detect_language_with_confidence(text: str) -> LanguageResult:
    """Return the most likely supported language and a confidence score."""

    text = _validate_text(text)
    words = _words(text)
    if not words:
        return LanguageResult(UNKNOWN_LANGUAGE, 0.0)

    spanish = _score(
        words, text, _SPANISH_WORDS, _SPANISH_SUFFIXES, accented=True
    )
    english = _score(words, text, _ENGLISH_WORDS, _ENGLISH_SUFFIXES)
    total = spanish + english

    if total == 0 or spanish == english:
        return LanguageResult(UNKNOWN_LANGUAGE, 0.0)

    language = "es" if spanish > english else "en"
    confidence = abs(spanish - english) / total
    return LanguageResult(language, min(confidence, 1.0))


__all__ = [
    "LanguageResult",
    "SUPPORTED_LANGUAGES",
    "UNKNOWN_LANGUAGE",
    "detect_language",
    "detect_language_with_confidence",
    "identify_language",
]