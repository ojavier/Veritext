"""Attacks are deterministic, change the text and keep the record shape."""

import pytest

from evaluation.attacks import ATTACKS, apply_attack, attack_records, paraphrase_with, synonyms, typos

TEXT = (
    "Además, el proceso es importante porque permite un gran desarrollo. "
    "Sin embargo, hoy en día muchos países utilizan otro método."
)


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_attack_changes_text_and_is_deterministic(name):
    a = apply_attack(TEXT, name, seed=1)
    assert a == apply_attack(TEXT, name, seed=1)
    assert a != TEXT


def test_typos_rate_zero_is_identity_and_seed_changes_output():
    assert typos(TEXT, rate=0.0) == TEXT
    assert typos(TEXT, rate=0.2, seed=1) != typos(TEXT, rate=0.2, seed=2)


def test_synonyms_keep_capitalisation_and_rate_zero_is_identity():
    assert synonyms(TEXT, rate=0.0) == TEXT
    out = synonyms("Además es importante.", rate=1.0, seed=0)
    assert out[0].isupper() and "importante" not in out


def test_unknown_attack_raises():
    with pytest.raises(ValueError):
        apply_attack(TEXT, "nope")


def test_attack_records_tags_copies_and_filters_labels():
    records = [
        {"id": "h1", "label": "human", "text": TEXT},
        {"id": "g1", "label": "llm", "text": TEXT, "domain": "x"},
    ]
    out = attack_records(records, "synonyms", labels=["llm"], rate=1.0)
    assert [r["id"] for r in out] == ["g1~synonyms"]
    assert out[0]["attack"] == "synonyms" and out[0]["domain"] == "x"
    assert records[1]["text"] == TEXT  # originals untouched


def test_pluggable_paraphraser():
    assert paraphrase_with(lambda t: t.upper())("hola", seed=3) == "HOLA"
