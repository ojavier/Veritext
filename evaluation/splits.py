"""Train / calibration / test splits that keep unseen domains and generators out.

The fusion model is fit on ``train``, its FPR threshold is chosen on
``calibration`` and everything is reported on ``test``. To measure
generalization, some domains and/or generators are *held out*: records that
touch them never appear in ``train`` or ``calibration``.

How records are assigned:

* A record is **unseen** if its domain is in ``holdout_domains`` or (for LLM
  records) its generator is in ``holdout_generators``. All unseen records go to
  ``test``.
* The remaining **seen** records are split by stratified random sampling over
  (label, domain, generator), so every group is spread over the three splits.
  Seen human text also lands in ``test``; it supplies the negatives needed to
  evaluate an unseen generator.

``shift_of`` tags each record (``in_distribution``, ``unseen_domain``,
``unseen_generator``, ``unseen_both``) so metrics can be sliced.

Signals that search ``data/reference`` or ``data/generated`` (similarity,
matching) find a record's own copy; exclude the record id (``exclude_ids``)
when scoring it.

CLI summary: ``python -m evaluation.splits --holdout-domain noticias``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset import Record, load_dataset

SPLIT_NAMES = ("train", "calibration", "test")
DEFAULT_FRACTIONS = (0.6, 0.2, 0.2)
SHIFTS = ("in_distribution", "unseen_domain", "unseen_generator", "unseen_both")


@dataclass
class Split:
    train: list[Record] = field(default_factory=list)
    calibration: list[Record] = field(default_factory=list)
    test: list[Record] = field(default_factory=list)
    holdout_domains: tuple[str, ...] = ()
    holdout_generators: tuple[str, ...] = ()

    def __getitem__(self, name: str) -> list[Record]:
        if name not in SPLIT_NAMES:
            raise KeyError(name)
        return getattr(self, name)

    def ids(self) -> dict[str, list[str]]:
        return {name: [r["id"] for r in self[name]] for name in SPLIT_NAMES}

    def shift(self, record: Record) -> str:
        return shift_of(record, self.holdout_domains, self.holdout_generators)

    def summary(self) -> dict[str, dict[str, int]]:
        """Counts per split by label (and total)."""
        out = {}
        for name in SPLIT_NAMES:
            counts = Counter(r["label"] for r in self[name])
            out[name] = {"total": len(self[name]), **counts}
        return out


def shift_of(
    record: Record,
    holdout_domains: Iterable[str] = (),
    holdout_generators: Iterable[str] = (),
) -> str:
    domain = record["domain"] in set(holdout_domains)
    generator = record.get("generator") in set(holdout_generators)
    if domain and generator:
        return "unseen_both"
    if domain:
        return "unseen_domain"
    if generator:
        return "unseen_generator"
    return "in_distribution"


def _allocate(n: int, fractions: Sequence[float]) -> tuple[int, int, int]:
    """Sizes of (train, calibration, test) for a group of ``n`` records."""
    if n < 3:
        return n, 0, 0  # too small to spread; keep it in train
    cal = max(1, round(n * fractions[1]))
    test = max(1, round(n * fractions[2]))
    return n - cal - test, cal, test


def make_split(
    records: Iterable[Record],
    holdout_domains: Iterable[str] = (),
    holdout_generators: Iterable[str] = (),
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> Split:
    """Split ``records`` as described in the module docstring (deterministic)."""
    records = list(records)
    holdout_domains = tuple(holdout_domains)
    holdout_generators = tuple(holdout_generators)
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-6 or min(fractions) < 0:
        raise ValueError("fractions must be three non-negative numbers summing to 1")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("record ids must be unique")
    domains = {r["domain"] for r in records}
    generators = {r["generator"] for r in records if r.get("generator")}
    for name, held, known in (
        ("domain", holdout_domains, domains),
        ("generator", holdout_generators, generators),
    ):
        unknown = set(held) - known
        if unknown:
            raise ValueError(f"unknown {name}(s): {sorted(unknown)}; have {sorted(known)}")

    split = Split(holdout_domains=holdout_domains, holdout_generators=holdout_generators)
    groups: dict[tuple, list[Record]] = defaultdict(list)
    for record in records:
        if split.shift(record) != "in_distribution":
            split.test.append(record)
        else:
            key = (record["label"], record["domain"], record.get("generator") or "")
            groups[key].append(record)

    rng = random.Random(seed)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["id"])
        rng.shuffle(group)
        n_train, n_cal, _ = _allocate(len(group), fractions)
        split.train += group[:n_train]
        split.calibration += group[n_train : n_train + n_cal]
        split.test += group[n_train + n_cal :]

    for name in SPLIT_NAMES:
        split[name].sort(key=lambda r: r["id"])
    for name in ("train", "calibration"):
        labels = {r["label"] for r in split[name]}
        if labels != {"human", "llm"}:
            raise ValueError(f"{name} split needs both labels, has {sorted(labels)}")
    check_no_leakage(split)
    return split


def check_no_leakage(split: Split) -> None:
    """Raise if splits overlap or train/calibration touch a held-out value."""
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        current = {r["id"] for r in split[name]}
        if current & seen:
            raise AssertionError(f"records appear in more than one split ({name})")
        seen |= current
    for name in ("train", "calibration"):
        for r in split[name]:
            if split.shift(r) != "in_distribution":
                raise AssertionError(f"{r['id']} is held out but lands in {name}")


def save_split(split: Split, path: Path) -> None:
    """Write ids and hold-out settings as JSON, so runs can reuse the split."""
    payload = {
        "holdout_domains": list(split.holdout_domains),
        "holdout_generators": list(split.holdout_generators),
        **split.ids(),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path: Path, records: Iterable[Record]) -> Split:
    """Rebuild a saved split from ``records`` (every saved id must exist)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in records}
    split = Split(
        holdout_domains=tuple(payload["holdout_domains"]),
        holdout_generators=tuple(payload["holdout_generators"]),
    )
    for name in SPLIT_NAMES:
        missing = [i for i in payload[name] if i not in by_id]
        if missing:
            raise KeyError(f"{len(missing)} ids in {name} not found, e.g. {missing[0]}")
        split[name].extend(by_id[i] for i in payload[name])
    check_no_leakage(split)
    return split


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show or save a data split.")
    parser.add_argument("--holdout-domain", action="append", default=[])
    parser.add_argument("--holdout-generator", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="save the split ids as JSON")
    args = parser.parse_args(argv)

    split = make_split(
        load_dataset(),
        args.holdout_domain,
        args.holdout_generator,
        seed=args.seed,
    )
    for name, counts in split.summary().items():
        print(f"{name:12} {counts}")
    shifts = Counter(split.shift(r) for r in split.test)
    print("test shifts:", {s: shifts[s] for s in SHIFTS if shifts[s]})
    if args.out:
        save_split(split, args.out)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
