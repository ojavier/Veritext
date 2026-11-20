import math

import pytest

from evaluation.metrics import (
    auroc,
    metrics_by_domain,
    tpr_at_fpr,
)


def test_auroc_perfect_and_tied():
    assert auroc([0.9, 0.8, 0.2, 0.1], ["llm", "llm", "human", "human"]) == 1.0
    assert auroc([0.5, 0.5], ["human", "llm"]) == 0.5


def test_tpr_at_fpr():
    scores = [0.9, 0.8, 0.7, 0.1]
    labels = ["llm", "llm", "human", "human"]
    assert tpr_at_fpr(scores, labels, target_fpr=0.5) == 1.0
    assert tpr_at_fpr([0.1, 0.2], ["human", "llm"], target_fpr=0.0) == 1.0


def test_metrics_by_domain():
    records = [
        {"id": "h-news", "domain": "news", "label": "human"},
        {"id": "l-news", "domain": "news", "label": "llm"},
        {"id": "h-wiki", "domain": "wiki", "label": "human"},
    ]
    scores = {
        "h-news": {"ai_prob": 0.1},
        "l-news": {"fusion_score": 0.9},
        "h-wiki": 0.2,
    }

    result = metrics_by_domain(records, scores)
    assert result["news"]["auroc"] == 1.0
    assert result["news"]["n_human"] == 1
    assert result["news"]["n_llm"] == 1
    assert math.isnan(result["wiki"]["auroc"])


def test_metric_validation():
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], ["human", "human"])
    with pytest.raises(ValueError):
        auroc([0.1], ["llm", "human"])
    with pytest.raises(ValueError):
        tpr_at_fpr([0.1], ["llm"], target_fpr=1.1)
