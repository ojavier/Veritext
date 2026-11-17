from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from signals.base import BaseSignal
from src.dataset import DATA_DIR, load_dataset
from src.schemas import Segment, Signals

SNIPPET_CHARS = 300
META_FIELDS = ("id", "generator", "model", "domain", "title", "source")

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _WORD_RE.findall(plain)


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def longest_common_span(a: Sequence[str], b: Sequence[str]) -> tuple[int, int]:
    """Length and start (in ``a``) of the longest run of words shared with ``b``."""
    best_len, best_start = 0, 0
    previous = [0] * (len(b) + 1)
    for i, word in enumerate(a, 1):
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b, 1):
            if word == other:
                current[j] = previous[j - 1] + 1
                if current[j] > best_len:
                    best_len, best_start = current[j], i - current[j]
        previous = current
    return best_len, best_start


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_length = (sum(self.lengths) / len(documents)) if documents else 0.0
        doc_freq: Counter[str] = Counter()
        for freqs in self.term_freqs:
            doc_freq.update(freqs.keys())
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        # Inverted index: only documents sharing a term with the query are scored.
        self.postings: dict[str, list[int]] = {}
        for position, freqs in enumerate(self.term_freqs):
            for term in freqs:
                self.postings.setdefault(term, []).append(position)

    def top(self, query: Sequence[str], k: int, allowed: Collection[int] | None = None) -> list[tuple[int, float]]:
        """Best ``k`` (position, score) pairs, highest first."""
        scores: dict[int, float] = {}
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for position in self.postings[term]:
                if allowed is not None and position not in allowed:
                    continue
                tf = self.term_freqs[position][term]
                norm = self.k1 * (1 - self.b + self.b * self.lengths[position] / self.avg_length)
                scores[position] = scores.get(position, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


@dataclass
class Match:
    """One re-scored candidate."""

    position: int
    containment: float
    bm25: float
    shared_ngrams: int
    span_length: int
    span_text: str


class GeneratedIndex:
    """BM25 index plus n-gram sets for a list of generated records."""

    def __init__(self, records: Sequence[dict[str, Any]], n: int = 3):
        if not records:
            raise ValueError("cannot build an index from zero records")
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.records = list(records)
        self.tokens = [tokenize(r["text"]) for r in self.records]
        self.ngram_sets = [ngrams(t, n) for t in self.tokens]
        self.bm25 = BM25(self.tokens)
        self._positions_by_id = {r["id"]: i for i, r in enumerate(self.records)}

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def from_dataset(cls, data_dir: Path = DATA_DIR, n: int = 3) -> GeneratedIndex:
        """Index every LLM record in ``data/generated``."""
        records = load_dataset(["generated"], labels=["llm"], data_dir=data_dir)
        if not records:
            raise FileNotFoundError(f"no records in {Path(data_dir) / 'generated'}")
        return cls(records, n)

    def allowed_positions(
        self, exclude_ids: Collection[str] = (), exclude_generators: Collection[str] = ()
    ) -> set[int] | None:
        if not exclude_ids and not exclude_generators:
            return None
        return {
            i
            for i, r in enumerate(self.records)
            if r["id"] not in exclude_ids and r.get("generator") not in exclude_generators
        }

    def search(
        self,
        text: str,
        top_k: int = 3,
        candidates: int = 20,
        allowed: Collection[int] | None = None,
    ) -> tuple[list[Match], int]:
        """Top-``top_k`` matches by containment, plus the query's n-gram count."""
        query = tokenize(text)
        query_ngrams = ngrams(query, self.n)
        if not query_ngrams:
            return [], 0
        matches = []
        for position, bm25_score in self.bm25.top(query, candidates, allowed):
            shared = len(query_ngrams & self.ngram_sets[position])
            if shared == 0:
                continue
            span_len, span_start = longest_common_span(query, self.tokens[position])
            matches.append(
                Match(
                    position=position,
                    containment=shared / len(query_ngrams),
                    bm25=bm25_score,
                    shared_ngrams=shared,
                    span_length=span_len,
                    span_text=" ".join(query[span_start : span_start + span_len]),
                )
            )
        matches.sort(key=lambda m: (-m.containment, -m.bm25))
        return matches[:top_k], len(query_ngrams)


class MatchingSignal(BaseSignal):
    """``matching`` = share of the segment's n-grams found in one known generation."""

    name: ClassVar[str] = "matching"

    def __init__(
        self,
        index: GeneratedIndex,
        top_k: int = 3,
        candidates: int = 20,
        min_ngrams: int = 5,
        exclude_ids: Iterable[str] = (),
        exclude_generators: Iterable[str] = (),
    ):
        self.index = index
        self.top_k = top_k
        self.candidates = max(candidates, top_k)
        self.min_ngrams = min_ngrams
        self.allowed = index.allowed_positions(frozenset(exclude_ids), frozenset(exclude_generators))

    @classmethod
    def from_config(cls, config: dict[str, Any], data_dir: Path = DATA_DIR) -> MatchingSignal:
        """Build from the ``matching`` section of the YAML config."""
        cfg = config.get("matching", {})
        index = GeneratedIndex.from_dataset(data_dir, n=int(cfg.get("ngram", 3)))
        return cls(
            index,
            top_k=int(cfg.get("top_k", 3)),
            candidates=int(cfg.get("candidates", 20)),
            min_ngrams=int(cfg.get("min_ngrams", 5)),
            exclude_generators=cfg.get("exclude_generators") or (),
        )

    def analyze(self, segment: Segment) -> Signals:
        matches, n_query = self.index.search(
            segment.text, self.top_k, self.candidates, self.allowed
        )
        info: dict[str, Any] = {"ngram": self.index.n, "query_ngrams": n_query}
        if n_query < self.min_ngrams:
            info["warning"] = f"only {n_query} {self.index.n}-grams; the score is unreliable"
        info["nearest"] = [
            {
                **{k: self.index.records[m.position][k] for k in META_FIELDS if k in self.index.records[m.position]},
                "score": round(m.containment, 4),
                "bm25": round(m.bm25, 3),
                "shared_ngrams": m.shared_ngrams,
                "longest_shared_words": m.span_length,
                "shared_text": m.span_text,
                "text": self.index.records[m.position]["text"][:SNIPPET_CHARS],
            }
            for m in matches
        ]
        matching = matches[0].containment if matches else 0.0
        return Signals(
            similarity=0.0,
            matching=min(max(matching, 0.0), 1.0),
            ai_prob=0.0,
            evidence={"matching": info},
        )


__all__ = [
    "BM25",
    "GeneratedIndex",
    "Match",
    "MatchingSignal",
    "longest_common_span",
    "ngrams",
    "tokenize",
]
