"""Train and calibrate the Veritext fusion model.

Typical usage:

    py scripts/train_fusion.py --scores data/.cache/fusion_scores.json

The scores file may contain an object keyed by record id or a JSON array/JSONL
of objects with ``id`` and ``similarity``, ``matching`` and ``ai_prob`` fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.splits import load_split, make_split, save_split  # noqa: E402
from fusion.calibration import Calibration, calibrate  # noqa: E402
from fusion.combiner import LogisticCombiner  # noqa: E402
from src.dataset import DATA_DIR, Record, load_dataset  # noqa: E402
from src.schemas import Segment, Signals  # noqa: E402

DEFAULT_MODEL_OUT = ROOT / "data" / ".cache" / "fusion.json"
DEFAULT_CALIBRATION_OUT = ROOT / "data" / ".cache" / "fusion_calibration.json"


def load_scores(path: str | Path) -> dict[str, Mapping[str, float]]:
    """Load signal scores indexed by record id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        entries: list[Any] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(json.loads(line))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "scores" in payload and isinstance(payload["scores"], dict):
                payload = payload["scores"]
            elif "scores" in payload and isinstance(payload["scores"], list):
                entries = payload["scores"]
                payload = None
            elif all(
                isinstance(row, (Mapping, list, tuple)) for row in payload.values()
            ):
                scores: dict[str, Mapping[str, float]] = {}
                for record_id, row in payload.items():
                    if isinstance(row, (list, tuple)) and len(row) == 3:
                        row = dict(zip(("similarity", "matching", "ai_prob"), row))
                    if not isinstance(row, Mapping):
                        raise ValueError(
                            f"{record_id}: score entry must be an object or 3-list"
                        )
                    if not all(
                        name in row for name in ("similarity", "matching", "ai_prob")
                    ):
                        raise ValueError(
                            f"{record_id}: expected similarity, matching and ai_prob"
                        )
                    scores[str(record_id)] = {
                        name: float(row[name])
                        for name in ("similarity", "matching", "ai_prob")
                    }
                return scores
            else:
                entries = [payload]
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ValueError("scores JSON must contain an object or array")

    scores: dict[str, Mapping[str, float]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each scores entry must be an object")
        record_id = entry.get("id")
        if record_id is None:
            raise ValueError("scores entry missing 'id'")
        row = entry.get("signals", entry)
        if not all(name in row for name in ("similarity", "matching", "ai_prob")):
            raise ValueError(f"{record_id}: expected similarity, matching and ai_prob")
        scores[str(record_id)] = {
            name: float(row[name]) for name in ("similarity", "matching", "ai_prob")
        }

    return scores


def signals_from_score(row: Mapping[str, float], record_id: str = "") -> Signals:
    values = [float(row[name]) for name in ("similarity", "matching", "ai_prob")]
    if not all(0.0 <= value <= 1.0 for value in values):
        prefix = f"{record_id}: " if record_id else ""
        raise ValueError(f"{prefix}signal values must be between 0 and 1")
    return Signals(
        similarity=values[0],
        matching=values[1],
        ai_prob=values[2],
        evidence={},
    )


def rows_for_records(
    records: Sequence[Record],
    scores: Mapping[str, Mapping[str, float]],
) -> tuple[list[Signals], list[str]]:
    rows: list[Signals] = []
    labels: list[str] = []
    missing: list[str] = []
    for record in records:
        record_id = str(record["id"])
        if record_id not in scores:
            missing.append(record_id)
            continue
        rows.append(signals_from_score(scores[record_id], record_id))
        labels.append(str(record["label"]))
    if missing:
        raise KeyError(f"missing scores for {len(missing)} records: {missing[:5]}")
    return rows, labels


def calculate_signals(
    records: Sequence[Record],
    *,
    config: Mapping[str, Any],
    data_dir: Path = DATA_DIR,
    use_zeroshot: bool = False,
    device: str = "cpu",
) -> dict[str, Mapping[str, float]]:
    """Calculate available signals for records.

    Zeroshot remains opt-in because it loads two language models. Similarity and
    matching use the on-disk indexes and are always calculated when possible.
    """
    from signals.matching import MatchingSignal
    from signals.similarity import SimilaritySignal

    similarity = SimilaritySignal.from_config(dict(config))
    matching = MatchingSignal.from_config(dict(config), data_dir)

    zeroshot = None
    if use_zeroshot:
        from src.pipeline import HFBinocularsScorer, ZeroShotSignal

        zs = config["zeroshot"]
        scorer = HFBinocularsScorer(
            zs["observer_model"],
            zs["performer_model"],
            device=device,
            max_tokens=int(zs["max_tokens"]),
        )
        zeroshot = ZeroShotSignal(
            scorer,
            threshold=float(zs["threshold"]),
            temperature=float(zs["temperature"]),
            min_tokens=int(zs["min_tokens"]),
            supported_languages=config["input"]["supported_languages"],
        )

    out: dict[str, Mapping[str, float]] = {}
    for position, record in enumerate(records):
        segment = Segment(
            id=position,
            text=str(record["text"]),
            lang=str(record.get("lang", "unknown")),
            start=0,
            end=len(str(record["text"])),
        )
        sim = similarity.analyze(segment)
        match = matching.analyze(segment)
        ai_prob = zeroshot.analyze(segment).ai_prob if zeroshot is not None else 0.0
        out[str(record["id"])] = {
            "similarity": sim.similarity,
            "matching": match.matching,
            "ai_prob": ai_prob,
        }
    return out


def train_fusion(
    train_records: Sequence[Record],
    calibration_records: Sequence[Record],
    scores: Mapping[str, Mapping[str, float]],
    *,
    C: float = 1.0,
    seed: int = 0,
    target_fpr: float = 0.01,
    uncertainty_fpr: float = 0.05,
) -> tuple[LogisticCombiner, Calibration]:
    """Fit the combiner on train records and calibrate on calibration records."""
    train_rows, train_labels = rows_for_records(train_records, scores)
    calibration_rows, calibration_labels = rows_for_records(calibration_records, scores)
    combiner = LogisticCombiner(C=C, seed=seed).fit(train_rows, train_labels)
    calibration_scores = combiner.score_many(calibration_rows)
    calibration = calibrate(
        calibration_scores,
        calibration_labels,
        target_fpr=target_fpr,
        uncertainty_fpr=uncertainty_fpr,
    )
    return combiner, calibration


def calibration_to_dict(calibration: Calibration) -> dict[str, float]:
    return {
        "threshold": calibration.threshold,
        "uncertainty_threshold": calibration.uncertainty_threshold,
        "target_fpr": calibration.target_fpr,
        "uncertainty_fpr": calibration.uncertainty_fpr,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scores", type=Path, help="precomputed signal scores JSON/JSONL"
    )
    parser.add_argument("--split", type=Path, help="saved split JSON to reuse")
    parser.add_argument("--holdout-domain", action="append", default=[])
    parser.add_argument("--holdout-generator", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--uncertainty-fpr", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=DEFAULT_CALIBRATION_OUT,
    )
    parser.add_argument("--split-out", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--calculate",
        action="store_true",
        help="calculate similarity and matching from on-disk indexes",
    )
    parser.add_argument(
        "--zeroshot",
        action="store_true",
        help="also calculate the zero-shot model signal",
    )
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        records = load_dataset(data_dir=args.data_dir, strict=False)
        if not records:
            print("error: no dataset records found", file=sys.stderr)
            return 1

        if args.split:
            split = load_split(args.split, records)
        else:
            split = make_split(
                records,
                holdout_domains=args.holdout_domain,
                holdout_generators=args.holdout_generator,
                seed=args.seed,
            )

        if args.scores:
            scores = load_scores(args.scores)
        elif args.calculate:
            from src.pipeline import load_config

            config = load_config(args.config) if args.config else load_config()
            scores = calculate_signals(
                records,
                config=config,
                data_dir=args.data_dir,
                use_zeroshot=args.zeroshot,
                device=args.device,
            )
        else:
            print(
                "error: provide --scores or --calculate",
                file=sys.stderr,
            )
            return 1

        combiner, calibration = train_fusion(
            split.train,
            split.calibration,
            scores,
            C=args.C,
            seed=args.seed,
            target_fpr=args.target_fpr,
            uncertainty_fpr=args.uncertainty_fpr,
        )
        combiner.save(args.out)

        args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_out.write_text(
            json.dumps(
                {
                    "model": str(args.out),
                    "split": split.summary(),
                    **calibration_to_dict(calibration),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if args.split_out:
            save_split(split, args.split_out)

        print(
            f"Saved fusion model to {args.out} "
            f"(high threshold={calibration.threshold:.6f}, "
            f"uncertain threshold={calibration.uncertainty_threshold:.6f})"
        )
        print(f"Saved calibration to {args.calibration_out}")
        return 0
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
