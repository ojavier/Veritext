"""Similarity signal checks with a fake embedder (no model download)."""

import hashlib

import pytest

np = pytest.importorskip("numpy")

from signals.similarity import ReferenceIndex, SimilaritySignal  # noqa: E402
from src.schemas import Segment  # noqa: E402


class FakeEmbedder:
    """Hashed bag-of-words: texts sharing words get a high cosine."""

    def encode(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


RECORDS = [
    {"id": "a", "domain": "ciencia", "source": "wiki", "text": "la luz estelar se polariza por el polvo interestelar"},
    {"id": "b", "domain": "historia", "source": "wiki", "text": "el imperio romano cayo en el siglo quinto despues de cristo"},
]


def _signal(**kwargs):
    embedder = FakeEmbedder()
    return SimilaritySignal(ReferenceIndex.build(RECORDS, embedder), embedder, **kwargs)


def _segment(text):
    return Segment(id=0, text=text, lang="es", start=0, end=len(text))


def test_identical_text_scores_one_and_names_its_source():
    signals = _signal().analyze(_segment(RECORDS[0]["text"]))
    assert signals.similarity == pytest.approx(1.0, abs=1e-5)
    nearest = signals.evidence["similarity"]["nearest"]
    assert nearest[0]["id"] == "a"
    assert nearest[0]["score"] >= nearest[1]["score"]


def test_unrelated_text_scores_lower_and_stays_in_range():
    related = _signal().analyze(_segment(RECORDS[1]["text"]))
    unrelated = _signal().analyze(_segment("xyz qwerty zzz"))
    assert 0.0 <= unrelated.similarity <= 1.0
    assert unrelated.similarity < related.similarity


def test_exclude_ids_hides_the_own_record():
    signals = _signal(exclude_ids={"a"}).analyze(_segment(RECORDS[0]["text"]))
    assert all(s["id"] != "a" for s in signals.evidence["similarity"]["nearest"])
    assert signals.similarity < 0.99


def test_index_round_trip(tmp_path):
    embedder = FakeEmbedder()
    index = ReferenceIndex.build(RECORDS, embedder, model="fake")
    index.save(tmp_path)
    loaded = ReferenceIndex.load(tmp_path)
    assert len(loaded) == 2 and loaded.model == "fake"
    assert np.allclose(loaded.embeddings, index.embeddings)
    assert loaded.records[0]["id"] == "a"


def test_missing_index_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        ReferenceIndex.load(tmp_path / "nope")
