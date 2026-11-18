"""Fusion of the three signals into one score.

``LogisticCombiner`` fits a logistic regression on ``(similarity, matching,
ai_prob)`` and returns the probability that a segment is LLM-generated, which
is the ``score`` of an alert. Thresholds and the uncertainty zone are chosen
afterwards on the calibration split (``fusion/calibration.py``).

Two parameter-free baselines, ``MeanCombiner`` and ``MaxCombiner``, share the
same interface for the ablation (each signal alone, mean, max, regression).

The fitted model is stored as JSON (coefficients only), so loading needs no
pickle and no scikit-learn.

``numpy`` and ``scikit-learn`` are imported lazily.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.schemas import Signals

if TYPE_CHECKING:
    import numpy as np

FEATURES = ("similarity", "matching", "ai_prob")

Row = Signals | Sequence[float]


def features_of(row: Row) -> list[float]:
    """Feature vector in ``FEATURES`` order from a ``Signals`` or a sequence."""
    if isinstance(row, Signals):
        return [float(getattr(row, name)) for name in FEATURES]
    values = [float(v) for v in row]
    if len(values) != len(FEATURES):
        raise ValueError(f"expected {len(FEATURES)} features {FEATURES}, got {len(values)}")
    return values


def _matrix(rows: Sequence[Row]) -> np.ndarray:
    import numpy as np

    return np.asarray([features_of(r) for r in rows], dtype=float).reshape(-1, len(FEATURES))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


class LogisticCombiner:
    """Logistic regression over the signal scores; ``label 1`` = LLM text."""

    name = "logistic"

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced", seed: int = 0):
        self.C = C
        self.class_weight = class_weight
        self.seed = seed
        self.coef: list[float] | None = None
        self.intercept: float | None = None

    @property
    def fitted(self) -> bool:
        return self.coef is not None

    def fit(self, rows: Sequence[Row], labels: Sequence[int | bool | str]) -> LogisticCombiner:
        """Fit on signal rows and labels (``1``/``True``/``"llm"`` = LLM)."""
        from sklearn.linear_model import LogisticRegression

        y = [int(label == "llm" if isinstance(label, str) else label) for label in labels]
        if len(y) != len(rows):
            raise ValueError("rows and labels must have the same length")
        if len(set(y)) != 2:
            raise ValueError("fitting needs both human (0) and LLM (1) examples")
        model = LogisticRegression(
            C=self.C, class_weight=self.class_weight, random_state=self.seed, max_iter=1000
        )
        model.fit(_matrix(rows), y)
        self.coef = [float(c) for c in model.coef_[0]]
        self.intercept = float(model.intercept_[0])
        return self

    def score(self, row: Row) -> float:
        """Probability in [0, 1] that the segment is LLM-generated."""
        if not self.fitted:
            raise RuntimeError("combiner is not fitted; call fit() or load()")
        z = self.intercept + sum(w * x for w, x in zip(self.coef, features_of(row)))
        return _sigmoid(z)

    def score_many(self, rows: Sequence[Row]) -> list[float]:
        return [self.score(r) for r in rows]

    def weights(self) -> dict[str, float]:
        """Coefficient per signal (plus ``intercept``), for the README/ablation."""
        if not self.fitted:
            raise RuntimeError("combiner is not fitted")
        return {**dict(zip(FEATURES, self.coef)), "intercept": self.intercept}

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("combiner is not fitted")
        return {
            "type": self.name,
            "features": list(FEATURES),
            "coef": self.coef,
            "intercept": self.intercept,
            "C": self.C,
            "class_weight": self.class_weight,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LogisticCombiner:
        if tuple(payload["features"]) != FEATURES:
            raise ValueError(f"feature mismatch: {payload['features']} vs {list(FEATURES)}")
        combiner = cls(payload.get("C", 1.0), payload.get("class_weight"))
        combiner.coef = [float(c) for c in payload["coef"]]
        combiner.intercept = float(payload["intercept"])
        return combiner

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LogisticCombiner:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class _Baseline:
    """Parameter-free combiner with the same interface (``fit`` is a no-op)."""

    name: str

    def fit(self, rows: Sequence[Row] = (), labels: Sequence[Any] = ()) -> _Baseline:
        return self

    def score_many(self, rows: Sequence[Row]) -> list[float]:
        return [self.score(r) for r in rows]


class MeanCombiner(_Baseline):
    name = "mean"

    def score(self, row: Row) -> float:
        values = features_of(row)
        return sum(values) / len(values)


class MaxCombiner(_Baseline):
    name = "max"

    def score(self, row: Row) -> float:
        return max(features_of(row))


__all__ = [
    "FEATURES",
    "LogisticCombiner",
    "MaxCombiner",
    "MeanCombiner",
    "features_of",
]
