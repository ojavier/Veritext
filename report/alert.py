"""Build a human-review report (alerts with distilled evidence) from alerts.

The wording is deliberately "alert for human review": nothing here claims that
a text was written by AI.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from src.schemas import Alert

NOTE = "Alerts for human review; not a verdict of authorship."
LEVELS = ("high", "uncertain", "low")
EXCERPT_CHARS = 200


def _excerpt(text: Any, limit: int = EXCERPT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _top_neighbour(evidence: dict, key: str) -> dict | None:
    section = evidence.get(key)
    if not isinstance(section, dict):
        return None
    nearest = section.get("nearest")
    if not nearest:
        return None
    return nearest[0]


def build_evidence(alert: Alert) -> dict[str, Any]:
    """Distil the raw signal evidence into short, reviewable items."""
    raw = alert.signals.evidence or {}
    out: dict[str, Any] = {}

    source = _top_neighbour(raw, "similarity")
    if source is not None:
        out["closest_source"] = {
            "id": source.get("id"),
            "title": source.get("title"),
            "source": source.get("source"),
            "similarity": source.get("score"),
            "excerpt": _excerpt(source.get("text")),
        }

    known = _top_neighbour(raw, "matching")
    if known is not None:
        out["closest_known_generation"] = {
            "id": known.get("id"),
            "generator": known.get("generator"),
            "score": known.get("score"),
            "shared_text": known.get("shared_text"),
            "excerpt": _excerpt(known.get("text")),
        }

    zeroshot = raw.get("zeroshot")
    if isinstance(zeroshot, dict):
        out["zeroshot"] = {
            k: zeroshot[k] for k in ("binoculars", "threshold", "n_tokens", "warning") if k in zeroshot
        }

    for key in ("fusion",):
        if key in raw:
            out["fusion"] = raw[key]
    return out


def build_alert(alert: Alert) -> dict[str, Any]:
    """One alert as a plain dict ready for rendering."""
    return {
        "segment": {
            "id": alert.segment.id,
            "lang": alert.segment.lang,
            "start": alert.segment.start,
            "end": alert.segment.end,
            "text": alert.segment.text,
        },
        "level": alert.level,
        "score": round(float(alert.score), 4),
        "signals": {
            "similarity": round(float(alert.signals.similarity), 4),
            "matching": round(float(alert.signals.matching), 4),
            "ai_prob": round(float(alert.signals.ai_prob), 4),
        },
        "evidence": build_evidence(alert),
    }


def build_report(alerts: Iterable[Alert]) -> dict[str, Any]:
    """Whole-document report: summary counts plus alerts in text order."""
    items = [build_alert(a) for a in alerts]
    counts = Counter(i["level"] for i in items)
    return {
        "note": NOTE,
        "summary": {
            "segments": len(items),
            "levels": {level: counts.get(level, 0) for level in LEVELS},
            "needs_review": [i["segment"]["id"] for i in items if i["level"] in ("high", "uncertain")],
        },
        "alerts": items,
    }
