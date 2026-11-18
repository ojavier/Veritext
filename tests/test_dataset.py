"""Checks that the data files follow the shared schema and keep the splits clean."""

import pytest

from src.dataset import load_dataset, normalize, validate

MIN_CHARS, MAX_CHARS = 250, 1000


def test_normalize_renames_language():
    assert normalize({"language": "es"})["lang"] == "es"


def test_validate_flags_problems():
    assert validate({"id": "x"})  # missing fields
    ok = {"id": "1", "text": "hola", "domain": "d", "label": "human", "lang": "es", "source": "s"}
    assert validate(ok) == []
    assert validate({**ok, "label": "llm"})  # llm needs generator/model
    assert validate({**ok, "label": "robot"})


def test_all_files_match_schema():
    records = load_dataset(strict=True)
    assert records, "no data found under data/"
    for r in records:
        assert MIN_CHARS <= len(r["text"]) <= MAX_CHARS, r["id"]
        assert r["lang"] == "es", r["id"]
        # human files hold only human text, generated files only LLM text
        assert (r["kind"] == "generated") == (r["label"] == "llm"), r["id"]


def test_ids_unique():
    ids = [r["id"] for r in load_dataset()]
    assert len(ids) == len(set(ids))


def test_no_duplicate_texts_across_sets():
    texts = [r["text"] for r in load_dataset()]
    assert len(texts) == len(set(texts))


def test_reference_and_eval_share_no_source_article():
    ref = {(r["source"], r["title"]) for r in load_dataset(["reference"])}
    ev = {(r["source"], r["title"]) for r in load_dataset(["eval"])}
    assert not (ref & ev)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        load_dataset(["nope"])
