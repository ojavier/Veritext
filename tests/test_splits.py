"""Split checks: no overlap, no held-out leakage, determinism."""

import pytest

from evaluation.splits import (
    check_no_leakage,
    load_split,
    make_split,
    save_split,
    shift_of,
)
from src.dataset import load_dataset


def _rec(i, label, domain, generator=None):
    r = {"id": f"{label}-{domain}-{generator}-{i}", "label": label, "domain": domain,
         "text": "x"}
    if generator:
        r["generator"] = generator
    return r


RECORDS = [
    *(_rec(i, "human", d) for d in ("a", "b", "c") for i in range(20)),
    *(_rec(i, "llm", d, g) for d in ("a", "b", "c") for g in ("g1", "g2") for i in range(20)),
]


def test_plain_split_covers_everything_once():
    split = make_split(RECORDS)
    ids = [r["id"] for name in ("train", "calibration", "test") for r in split[name]]
    assert sorted(ids) == sorted(r["id"] for r in RECORDS)
    assert len(split.train) > len(split.calibration) > 0
    assert split.test


def test_holdout_never_reaches_train_or_calibration():
    split = make_split(RECORDS, holdout_domains=["c"], holdout_generators=["g2"])
    for name in ("train", "calibration"):
        assert all(r["domain"] != "c" and r.get("generator") != "g2" for r in split[name])
    kinds = {split.shift(r) for r in split.test}
    assert kinds == {"in_distribution", "unseen_domain", "unseen_generator", "unseen_both"}
    # every record of a held-out value is in test
    assert all(r in split.test for r in RECORDS if r["domain"] == "c")


def test_deterministic_and_seed_changes_result():
    assert make_split(RECORDS, seed=1).ids() == make_split(RECORDS, seed=1).ids()
    assert make_split(RECORDS, seed=1).ids() != make_split(RECORDS, seed=2).ids()


def test_validation_errors():
    with pytest.raises(ValueError):
        make_split(RECORDS, holdout_domains=["zzz"])
    with pytest.raises(ValueError):
        make_split(RECORDS, fractions=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):  # no human text left to train on
        make_split(RECORDS, holdout_domains=["a", "b", "c"])


def test_leakage_check_detects_problems():
    split = make_split(RECORDS, holdout_domains=["c"])
    split.train.append(next(r for r in RECORDS if r["domain"] == "c"))
    with pytest.raises(AssertionError):
        check_no_leakage(split)


def test_shift_of():
    assert shift_of(_rec(0, "llm", "a", "g1"), ["a"], ["g1"]) == "unseen_both"
    assert shift_of(_rec(0, "human", "a"), [], ["g1"]) == "in_distribution"


def test_save_and_load_roundtrip(tmp_path):
    split = make_split(RECORDS, holdout_generators=["g1"])
    save_split(split, tmp_path / "split.json")
    assert load_split(tmp_path / "split.json", RECORDS).ids() == split.ids()


def test_real_dataset_split():
    split = make_split(load_dataset(), ["noticias"], ["claude"])
    check_no_leakage(split)
    assert {r["label"] for r in split.test} == {"human", "llm"}
