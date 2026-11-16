"""Similarity-to-sources signal: multilingual embeddings + cosine search.

Every paragraph of ``data/reference`` (known human text) is embedded once and
stored in a :class:`ReferenceIndex`. A segment's ``similarity`` is the cosine
similarity to its nearest reference paragraph, clipped to [0, 1]; the closest
sources are returned as evidence.

The index is a plain normalized matrix searched by dot product. With a few
thousand paragraphs this is instant, so no FAISS is needed.

``numpy`` and ``sentence-transformers`` are imported lazily, so importing this
module is free and tests can inject a fake embedder.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from signals.base import BaseSignal
from src.schemas import Segment, Signals

if TYPE_CHECKING:
    import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INDEX_DIR = (
    Path(__file__).resolve().parent.parent / "data" / ".cache" / "reference_index"
)
EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"
SNIPPET_CHARS = 300
# Fields copied from each record into the index metadata (shown as evidence).
META_FIELDS = ("id", "domain", "source", "title", "lang")


class Embedder(Protocol):
    """Anything that maps texts to L2-normalized float32 row vectors."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Sentence-transformers model, loaded once on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int = 256,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        # The MiniLM default is 128 tokens, which would cut most paragraphs.
        self.max_seq_length = max_seq_length
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "The similarity signal requires 'sentence-transformers': "
                    "pip install sentence-transformers"
                ) from error
            model = SentenceTransformer(self.model_name, device=self.device)
            model.max_seq_length = self.max_seq_length
            self._model = model
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import numpy as np

        vectors = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class Match:
    """One nearest-neighbour hit."""

    position: int
    score: float


class ReferenceIndex:
    """Normalized embeddings plus per-paragraph metadata, searched by cosine."""

    def __init__(self, embeddings: np.ndarray, records: list[dict[str, Any]], model: str):
        if len(records) != len(embeddings):
            raise ValueError("embeddings and records must have the same length")
        self.embeddings = embeddings
        self.records = records
        self.model = model

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def build(
        cls,
        records: Sequence[dict[str, Any]],
        embedder: Embedder,
        model: str = DEFAULT_MODEL,
    ) -> ReferenceIndex:
        """Embed ``records`` (dataset records with ``id`` and ``text``)."""
        if not records:
            raise ValueError("cannot build an index from zero records")
        embeddings = embedder.encode([r["text"] for r in records])
        meta = [
            {
                **{k: r[k] for k in META_FIELDS if k in r},
                "text": r["text"][:SNIPPET_CHARS],
            }
            for r in records
        ]
        return cls(embeddings, meta, model)

    def save(self, directory: Path = DEFAULT_INDEX_DIR) -> Path:
        import numpy as np

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMBEDDINGS_FILE, self.embeddings)
        payload = {"model": self.model, "records": self.records}
        (directory / META_FILE).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path = DEFAULT_INDEX_DIR) -> ReferenceIndex:
        import numpy as np

        directory = Path(directory)
        try:
            embeddings = np.load(directory / EMBEDDINGS_FILE)
            payload = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"no reference index in {directory}; run scripts/build_index.py"
            ) from error
        return cls(embeddings, payload["records"], payload["model"])

    def search(
        self,
        vectors: np.ndarray,
        k: int = 3,
        exclude_ids: Collection[str] = (),
    ) -> list[list[Match]]:
        """Top-``k`` matches per query row, best first.

        ``exclude_ids`` hides reference paragraphs (e.g. the query's own record
        when scoring the reference set itself).
        """
        import numpy as np

        scores = np.atleast_2d(vectors) @ self.embeddings.T
        if exclude_ids:
            hidden = [i for i, r in enumerate(self.records) if r["id"] in exclude_ids]
            scores[:, hidden] = -np.inf
        k = min(k, scores.shape[1])
        top = np.argsort(-scores, axis=1)[:, :k]
        return [
            [Match(int(i), float(row[i])) for i in idx if np.isfinite(row[i])]
            for row, idx in zip(scores, top)
        ]


class SimilaritySignal(BaseSignal):
    """``similarity`` = cosine to the closest known reference paragraph."""

    name: ClassVar[str] = "similarity"

    def __init__(
        self,
        index: ReferenceIndex,
        embedder: Embedder,
        top_k: int = 3,
        exclude_ids: Collection[str] = (),
    ):
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        self.exclude_ids = frozenset(exclude_ids)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SimilaritySignal:
        """Build from the ``similarity`` section of the YAML config."""
        cfg = config.get("similarity", {})
        index_dir = Path(cfg.get("index_dir") or DEFAULT_INDEX_DIR)
        embedder = SentenceTransformerEmbedder(
            cfg.get("model", DEFAULT_MODEL),
            device=cfg.get("device", "cpu"),
            batch_size=int(cfg.get("batch_size", 32)),
            max_seq_length=int(cfg.get("max_seq_length", 256)),
        )
        return cls(ReferenceIndex.load(index_dir), embedder, int(cfg.get("top_k", 3)))

    def analyze(self, segment: Segment) -> Signals:
        matches = self.index.search(
            self.embedder.encode([segment.text]), self.top_k, self.exclude_ids
        )[0]
        sources = [
            {**self.index.records[m.position], "score": round(m.score, 4)}
            for m in matches
        ]
        best = matches[0].score if matches else 0.0
        similarity = min(max(best, 0.0), 1.0)
        return Signals(
            similarity=similarity,
            matching=0.0,
            ai_prob=0.0,
            evidence={"similarity": {"model": self.index.model, "nearest": sources}},
        )


__all__ = [
    "DEFAULT_INDEX_DIR",
    "DEFAULT_MODEL",
    "Embedder",
    "Match",
    "ReferenceIndex",
    "SentenceTransformerEmbedder",
    "SimilaritySignal",
]
