"""Choose score thresholds on a calibration split.

Scores are probabilities (or any other quantity where larger means "more
likely LLM").  The high threshold is selected from human examples so that the
empirical false-positive rate is at most the requested target.  A lower
threshold, calibrated at a larger FPR, defines the uncertainty zone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_TARGET_FPR = 0.01
DEFAULT_UNCERTAINTY_FPR = 0.05


def _label_is_llm(label: int | bool | str) -> bool:
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized == "human":
            return False
        if normalized == "llm":
            return True
        raise ValueError(f"unsupported label {label!r}; expected 'human' or 'llm'")
    if isinstance(label, bool):
        return label
    if isinstance(label, int) and label in (0, 1):
        return bool(label)
    raise ValueError(
        f"unsupported label {label!r}; expected 0, 1, True, False, human or llm"
    )


def _validate_fpr(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value between 0 and 1")
    return value


def _validated_scores(scores: Sequence[float]) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        raise ValueError("scores must not be empty")
    if not all(math.isfinite(score) for score in values):
        raise ValueError("scores must contain only finite values")
    return values


def _threshold_for_fpr(human_scores: Sequence[float], target_fpr: float) -> float:
    """Return the smallest useful threshold whose human FPR is at most target."""
    ordered = sorted(set(human_scores))
    n_human = len(human_scores)
    for threshold in ordered:
        false_positives = sum(score >= threshold for score in human_scores)
        if false_positives / n_human <= target_fpr:
            return threshold
    # No observed score can meet FPR=0 (or a smaller target); move just above
    # the maximum while keeping the result finite and reproducible.
    return math.nextafter(max(human_scores), math.inf)


def threshold_at_fpr(
    scores: Sequence[float],
    labels: Sequence[int | bool | str],
    target_fpr: float = DEFAULT_TARGET_FPR,
) -> float:
    """Choose a threshold with empirical human false-positive rate <= ``target_fpr``."""
    values = _validated_scores(scores)
    if len(values) != len(labels):
        raise ValueError("scores and labels must have the same length")
    target_fpr = _validate_fpr(target_fpr, "target_fpr")
    human_scores = [
        score for score, label in zip(values, labels) if not _label_is_llm(label)
    ]
    if not human_scores:
        raise ValueError("at least one human example is required")
    return _threshold_for_fpr(human_scores, target_fpr)


@dataclass(frozen=True)
class Calibration:
    """Calibrated high and uncertainty thresholds for a score."""

    threshold: float
    uncertainty_threshold: float
    target_fpr: float = DEFAULT_TARGET_FPR
    uncertainty_fpr: float = DEFAULT_UNCERTAINTY_FPR

    def level(self, score: float) -> str:
        """Return ``low``, ``uncertain`` or ``high`` for a calibrated score."""
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        if score >= self.threshold:
            return "high"
        if score >= self.uncertainty_threshold:
            return "uncertain"
        return "low"

    def classify(self, score: float) -> str:
        """Alias for :meth:`level`."""
        return self.level(score)


def calibrate(
    scores: Sequence[float],
    labels: Sequence[int | bool | str],
    target_fpr: float = DEFAULT_TARGET_FPR,
    uncertainty_fpr: float = DEFAULT_UNCERTAINTY_FPR,
) -> Calibration:
    """Calibrate alert thresholds from scores and labels.

    The uncertainty boundary uses the larger FPR, so it is never above the
    high-alert threshold.  Labels may be ``human``/``llm`` or 0/1 values.
    """
    target_fpr = _validate_fpr(target_fpr, "target_fpr")
    uncertainty_fpr = _validate_fpr(uncertainty_fpr, "uncertainty_fpr")
    if uncertainty_fpr < target_fpr:
        raise ValueError("uncertainty_fpr must be greater than or equal to target_fpr")
    high = threshold_at_fpr(scores, labels, target_fpr)
    values = _validated_scores(scores)
    human_scores = [
        score for score, label in zip(values, labels) if not _label_is_llm(label)
    ]
    uncertain = _threshold_for_fpr(human_scores, uncertainty_fpr)
    return Calibration(high, uncertain, target_fpr, uncertainty_fpr)


def calibrate_threshold(
    scores: Sequence[float],
    labels: Sequence[int | bool | str],
    target_fpr: float = DEFAULT_TARGET_FPR,
) -> float:
    """Compatibility helper returning only the high-alert threshold."""
    return threshold_at_fpr(scores, labels, target_fpr)


__all__ = [
    "Calibration",
    "DEFAULT_TARGET_FPR",
    "DEFAULT_UNCERTAINTY_FPR",
    "calibrate",
    "calibrate_threshold",
    "threshold_at_fpr",
]
