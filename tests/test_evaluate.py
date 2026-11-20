import pytest

pytest.importorskip("sklearn")

from evaluation.splits import make_split
from scripts.evaluate import evaluate_ablations


def test_evaluate_ablations_per_domain():
    records = []
    scores = {}
    for i in range(12):
        for label in ("human", "llm"):
            record_id = f"{label}-{i}"
            domain = "news" if i < 6 else "wiki"
            records.append({
                "id": record_id,
                "domain": domain,
                "label": label,
                "text": "x" * 250,
            })
            if label == "llm":
                row = {"similarity": 0.8, "matching": 0.7, "ai_prob": 0.9}
            else:
                row = {"similarity": 0.2, "matching": 0.3, "ai_prob": 0.1}
            scores[record_id] = row

    split = make_split(records)
    results = evaluate_ablations(split, scores)

    assert set(results) == {
        "similarity",
        "matching",
        "ai_prob",
        "mean",
        "max",
        "logistic",
    }
    assert results["logistic"]["n"] == len(split.test)
    assert set(results["logistic"]["by_domain"]) == {"news", "wiki"}
