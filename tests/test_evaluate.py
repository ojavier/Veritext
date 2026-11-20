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


def test_evaluate_with_attack_uses_attacked_llm_records():
    from scripts.evaluate import attacked_test_set

    records, scores = [], {}
    for i in range(12):
        for label in ("human", "llm"):
            rid = f"{label}-{i}"
            records.append({"id": rid, "domain": "news" if i < 6 else "wiki", "label": label,
                            "text": "Además el proceso es importante. " * 10})
            scores[rid] = ({"similarity": 0.8, "matching": 0.7, "ai_prob": 0.9} if label == "llm"
                           else {"similarity": 0.2, "matching": 0.3, "ai_prob": 0.1})
    split = make_split(records)
    test = attacked_test_set(split, "synonyms")
    attacked = [r for r in test if r.get("attack")]
    assert attacked and all(r["label"] == "llm" and r["source_id"] in {x["id"] for x in split.test} for r in attacked)
    assert not any(r["label"] == "human" and r.get("attack") for r in test)
    for r in attacked:
        scores[r["id"]] = {"similarity": 0.6, "matching": 0.5, "ai_prob": 0.8}
    results = evaluate_ablations(split, scores, test_records=test)
    assert results["logistic"]["n"] == len(test)
