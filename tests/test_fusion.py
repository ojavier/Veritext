"""Combiner checks: fit, score range, persistence, baselines."""

import random

import pytest

pytest.importorskip("sklearn")

from fusion.combiner import (  # noqa: E402
    FEATURES,
    LogisticCombiner,
    MaxCombiner,
    MeanCombiner,
    features_of,
)
from src.schemas import Signals  # noqa: E402


def _data(n=200, seed=0):
    """LLM rows have a high ai_prob; similarity is pure noise."""
    rng = random.Random(seed)
    rows, labels = [], []
    for i in range(n):
        llm = i % 2
        rows.append([rng.random(), rng.random() * 0.3, 0.7 * llm + 0.3 * rng.random()])
        labels.append(llm)
    return rows, labels


def test_fit_learns_the_informative_signal():
    rows, labels = _data()
    combiner = LogisticCombiner().fit(rows, labels)
    w = combiner.weights()
    assert w["ai_prob"] > 0 and w["ai_prob"] > abs(w["similarity"])
    scores = combiner.score_many(rows)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert sum(s for s, y in zip(scores, labels) if y) / 100 > 0.7
    assert sum(s for s, y in zip(scores, labels) if not y) / 100 < 0.3


def test_accepts_signals_and_string_labels():
    rows, labels = _data(40)
    signals = [Signals(r[0], r[1], r[2], {}) for r in rows]
    combiner = LogisticCombiner().fit(signals, ["llm" if y else "human" for y in labels])
    assert combiner.score(signals[0]) == pytest.approx(combiner.score(rows[0]))


def test_save_and_load_roundtrip(tmp_path):
    rows, labels = _data()
    combiner = LogisticCombiner().fit(rows, labels)
    combiner.save(tmp_path / "fusion.json")
    loaded = LogisticCombiner.load(tmp_path / "fusion.json")
    assert loaded.score(rows[3]) == pytest.approx(combiner.score(rows[3]))


def test_errors():
    with pytest.raises(RuntimeError):
        LogisticCombiner().score([0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        LogisticCombiner().fit([[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]], [1, 1])
    with pytest.raises(ValueError):
        features_of([0.1, 0.2])


def test_extreme_inputs_do_not_overflow():
    combiner = LogisticCombiner()
    combiner.coef, combiner.intercept = [1e4, 1e4, 1e4], -1e4
    assert 0.0 <= combiner.score([0, 0, 0]) <= 1.0
    assert 0.0 <= combiner.score([1, 1, 1]) <= 1.0


def test_baselines():
    assert MeanCombiner().score([0.2, 0.4, 0.9]) == pytest.approx(0.5)
    assert MaxCombiner().score([0.2, 0.4, 0.9]) == 0.9
    assert MaxCombiner().fit() is not None
    assert len(FEATURES) == 3
