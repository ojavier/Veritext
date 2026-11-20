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
