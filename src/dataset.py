"""Single loader and schema for every dataset file under ``data/``.

Human files (``data/reference``, ``data/eval``) and LLM files
(``data/generated``) share the same record format, so evaluation, indexing and
fusion code can read them with one function.

Record fields::

    id, text, domain, label ("human" | "llm"), lang, source, title
    + for LLM records: generator, model, topic
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KINDS = ("reference", "eval", "generated")
LABELS = ("human", "llm")
REQUIRED_FIELDS = ("id", "text", "domain", "label", "lang", "source")
LLM_FIELDS = ("generator", "model")

Record = dict[str, Any]


def normalize(record: Record) -> Record:
    """Return a copy using the canonical field names (``language`` -> ``lang``)."""
    out = dict(record)
    if "lang" not in out and "language" in out:
        out["lang"] = out.pop("language")
    return out


def validate(record: Record) -> list[str]:
    """Return a list of problems with ``record`` (empty if it is valid)."""
    problems = [f"missing field '{f}'" for f in REQUIRED_FIELDS if f not in record]
    if record.get("label") not in LABELS:
        problems.append(f"label must be one of {LABELS}, got {record.get('label')!r}")
    if record.get("label") == "llm":
        problems += [f"llm record missing '{f}'" for f in LLM_FIELDS if f not in record]
    if not str(record.get("text", "")).strip():
        problems.append("empty text")
    return problems


def read_jsonl(path: Path) -> list[Record]:
    """Read one JSONL file, ignoring a possibly half-written last line."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(normalize(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return records


def load_dataset(
    kinds: Iterable[str] = KINDS,
    *,
    domains: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    data_dir: Path = DATA_DIR,
    strict: bool = True,
) -> list[Record]:
    """Load records from ``data/<kind>/*.jsonl``, optionally filtered.

    Each record gets a ``kind`` field ("reference", "eval" or "generated") so
    callers know which folder it came from. With ``strict`` an invalid record
    raises ``ValueError``.
    """
    domain_set = set(domains) if domains is not None else None
    label_set = set(labels) if labels is not None else None
    out: list[Record] = []
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
        for path in sorted((Path(data_dir) / kind).glob("*.jsonl")):
            for record in read_jsonl(path):
                problems = validate(record)
                if problems and strict:
                    raise ValueError(f"{path.name} {record.get('id')}: {'; '.join(problems)}")
                if problems:
                    continue
                if domain_set is not None and record["domain"] not in domain_set:
                    continue
                if label_set is not None and record["label"] not in label_set:
                    continue
                out.append({**record, "kind": kind})
    return out
