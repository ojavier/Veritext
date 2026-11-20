"""calculate_signals must not let a record find itself in the indexes."""

import pytest

np = pytest.importorskip("numpy")

from scripts import train_fusion  # noqa: E402
from signals import matching as matching_mod  # noqa: E402
from signals import similarity as similarity_mod  # noqa: E402
from tests.test_similarity import FakeEmbedder  # noqa: E402

TEXTS = {
    "a": "la luz estelar se polariza por el polvo interestelar de la galaxia",
    "b": "el imperio romano cayo en el siglo quinto despues de cristo en occidente",
    "c": "las ballenas azules migran miles de kilometros cada ano buscando alimento",
}


def test_records_do_not_match_themselves(monkeypatch):
    records = [{"id": k, "text": v, "domain": "x", "generator": "g"} for k, v in TEXTS.items()]
    embedder = FakeEmbedder()

    monkeypatch.setattr(
        similarity_mod.SimilaritySignal,
        "from_config",
        classmethod(
            lambda cls, config: cls(similarity_mod.ReferenceIndex.build(records, embedder), embedder)
        ),
    )
    monkeypatch.setattr(
        matching_mod.MatchingSignal,
        "from_config",
        classmethod(lambda cls, config, data_dir: cls(matching_mod.GeneratedIndex(records, n=3), min_ngrams=1)),
    )

    scores = train_fusion.calculate_signals(records, config={})
    for record_id, row in scores.items():
        assert row["similarity"] < 0.99, record_id
        assert row["matching"] < 0.99, record_id


def test_attacked_record_hides_its_source(monkeypatch):
    records = [{"id": k, "text": v, "domain": "x", "generator": "g"} for k, v in TEXTS.items()]
    embedder = FakeEmbedder()
    monkeypatch.setattr(similarity_mod.SimilaritySignal, "from_config", classmethod(
        lambda cls, config: cls(similarity_mod.ReferenceIndex.build(records, embedder), embedder)))
    monkeypatch.setattr(matching_mod.MatchingSignal, "from_config", classmethod(
        lambda cls, config, data_dir: cls(matching_mod.GeneratedIndex(records, n=3), min_ngrams=1)))
    attacked = [{"id": "a~typos", "source_id": "a", "text": TEXTS["a"], "domain": "x"}]
    row = train_fusion.calculate_signals(attacked, config={})["a~typos"]
    assert row["similarity"] < 0.99 and row["matching"] < 0.99


def test_hide_holdout_generators_merges_with_config():
    from types import SimpleNamespace

    split = SimpleNamespace(holdout_generators=("g2",))
    cfg = {"matching": {"exclude_generators": ["g1"], "top_k": 3}}
    out = train_fusion.hide_holdout_generators(cfg, split)
    assert out["matching"]["exclude_generators"] == ["g1", "g2"] and out["matching"]["top_k"] == 3
    assert cfg["matching"]["exclude_generators"] == ["g1"]  # input untouched
    assert train_fusion.hide_holdout_generators({}, split)["matching"]["exclude_generators"] == ["g2"]
