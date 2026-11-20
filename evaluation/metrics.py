"""Binary classification metrics, including per-domain evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _label_is_llm(label: int | bool | float | str) -> bool:
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized == "human":
            return False
        if normalized == "llm":
            return True
        raise ValueError(f"unsupported label {label!r}; expected 'human' or 'llm'")
    if isinstance(label, bool):
        return label
    if isinstance(label, (int, float)) and label in (0, 1):
        return bool(label)
    raise ValueError(
        f"unsupported label {label!r}; expected 0, 1, True, False, human or llm"
    )


def _validated_scores_labels(
    scores: Sequence[float],
    labels: Sequence[int | bool | float | str],
) -> tuple[list[float], list[bool]]:
    values = [float(score) for score in scores]
    if not values:
        raise ValueError("scores must not be empty")
    if len(values) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not all(math.isfinite(score) for score in values):
        raise ValueError("scores must contain only finite values")

    binary_labels = [_label_is_llm(label) for label in labels]
    if not any(binary_labels):
        raise ValueError("at least one llm example is required")
    if all(binary_labels):
        raise ValueError("at least one human example is required")
    return values, binary_labels


def auroc(
    scores: Sequence[float],
    labels: Sequence[int | bool | float | str],
) -> float:
    """Compute area under the ROC curve (pairwise Mann-Whitney formulation)."""
    values, is_llm = _validated_scores_labels(scores, labels)
    positives = [value for value, positive in zip(values, is_llm) if positive]
    negatives = [value for value, positive in zip(values, is_llm) if not positive]

    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                favorable += 1.0
            elif positive == negative:
                favorable += 0.5
    return favorable / (len(positives) * len(negatives))


def tpr_at_fpr(
    scores: Sequence[float],
    labels: Sequence[int | bool | float | str],
    target_fpr: float = 0.01,
) -> float:
    """Return the TPR at the largest observed threshold with FPR <= target."""
    values, is_llm = _validated_scores_labels(scores, labels)
    target_fpr = float(target_fpr)
    if not math.isfinite(target_fpr) or not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be a finite value between 0 and 1")

    human_count = sum(not positive for positive in is_llm)
    llm_count = sum(is_llm)
    thresholds = sorted(set(values), reverse=True)
    best_tpr = 0.0
    threshold_reached = False

    for threshold in thresholds:
        false_positives = sum(
            score >= threshold and not positive
            for score, positive in zip(values, is_llm)
        )
        fpr = false_positives / human_count
        if fpr <= target_fpr:
            threshold_reached = True
            true_positives = sum(
                score >= threshold and positive
                for score, positive in zip(values, is_llm)
            )
            best_tpr = max(best_tpr, true_positives / llm_count)

    if threshold_reached:
        return best_tpr

    # The lowest threshold predicts every example as LLM.  It is the only
    # possible fallback when no threshold exists under a target below 1/n_human.
    true_positives = llm_count
    return true_positives / llm_count


def _extract_score(
    score_entry: float | Mapping[str, Any],
    record_id: str,
) -> float:
    if isinstance(score_entry, Mapping):
        for name in ("score", "fusion_score", "llm_score", "probability", "ai_prob"):
            if name in score_entry:
                score_entry = score_entry[name]
                break
        else:
            raise ValueError(
                f"{record_id}: score entry must contain a scalar score "
                "(score, fusion_score, llm_score, probability or ai_prob)"
            )
    try:
        value = float(score_entry)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record_id}: score must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{record_id}: score must be finite")
    return value


def metrics_by_domain(
    records: Iterable[Mapping[str, Any]],
    scores: Mapping[str, float | Mapping[str, Any]],
    target_fpr: float = 0.01,
) -> dict[str, dict[str, int | float]]:
    """Calculate AUROC and TPR@FPR for every domain represented in records."""
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, list[int | bool | float | str]] = defaultdict(list)

    record_count = 0
    missing: list[str] = []
    for record in records:
        record_count += 1
        domain = str(record.get("domain", "unknown"))
        record_id = str(record["id"])
        if record_id not in scores:
            missing.append(record_id)
            continue
        grouped_scores[domain].append(_extract_score(scores[record_id], record_id))
        grouped_labels[domain].append(record["label"])

    if missing:
        raise KeyError(f"missing scores for {len(missing)} records: {missing[:5]}")
    if record_count == 0:
        raise ValueError("records must not be empty")

    out: dict[str, dict[str, int | float]] = {}
    for domain in sorted(grouped_scores):
        domain_scores = grouped_scores[domain]
        domain_labels = grouped_labels[domain]
        binary = [_label_is_llm(label) for label in domain_labels]
        human_count = sum(not positive for positive in binary)
        llm_count = sum(binary)
        result: dict[str, int | float] = {
            "n": len(domain_scores),
            "n_human": human_count,
            "n_llm": llm_count,
        }
        if human_count and llm_count:
            result["auroc"] = auroc(domain_scores, domain_labels)
            result["tpr_at_fpr"] = tpr_at_fpr(
                domain_scores,
                domain_labels,
                target_fpr=target_fpr,
            )
        else:
            result["auroc"] = float("nan")
            result["tpr_at_fpr"] = float("nan")
        out[domain] = result
    return out


# Descriptive aliases for callers that prefer an evaluation-style name.
compute_auroc = auroc
domain_metrics = metrics_by_domain

__all__ = [
    "auroc",
    "compute_auroc",
    "domain_metrics",
    "metrics_by_domain",
    "tpr_at_fpr",
]