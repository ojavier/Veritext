"""Evaluate Veritext with per-domain metrics and signal ablations.

Typical usage:

    py scripts/evaluate.py --scores data/.cache/fusion_scores.json
    py scripts/evaluate.py --calculate --device cpu

Ablations are:
* Each signal alone: similarity, matching, ai_prob
* Simple mean of the three signals
* Maximum of the three signals
* Logistic regression fitted on the train split

Metrics are calculated on the test split and split again by record domain.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import auroc, metrics_by_domain, tpr_at_fpr  # noqa: E402
from evaluation.splits import load_split, make_split, save_split  # noqa: E402
from fusion.combiner import (  # noqa: E402
    FEATURES,
    LogisticCombiner,
    MaxCombiner,
    MeanCombiner,
)
from scripts.train_fusion import calculate_signals, load_scores  # noqa: E402
from src.dataset import DATA_DIR, Record, load_dataset  # noqa: E402
from src.schemas import Signals  # noqa: E402

ABLATION_NAMES = (*FEATURES, "mean", "max", "logistic")


def records_to_rows(
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
        row = scores[record_id]
        try:
            values = [float(row[name]) for name in FEATURES]
        except KeyError as exc:
            raise KeyError(f"{record_id}: missing signal {exc.args[0]}") from exc
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{record_id}: signal values must be between 0 and 1")
        rows.append(
            Signals(
                similarity=values[0],
                matching=values[1],
                ai_prob=values[2],
                evidence={},
            )
        )
        labels.append(str(record["label"]))
    if missing:
        raise KeyError(f"missing scores for {len(missing)} records: {missing[:5]}")
    return rows, labels


def ablation_scores(
    name: str,
    records: Sequence[Record],
    rows: Sequence[Signals],
    train_rows: Sequence[Signals],
    train_labels: Sequence[str],
    *,
    C: float,
    seed: int,
) -> list[float]:
    if name in FEATURES:
        return [float(getattr(row, name)) for row in rows]
    if name == "mean":
        return MeanCombiner().score_many(rows)
    if name == "max":
        return MaxCombiner().score_many(rows)
    if name == "logistic":
        combiner = LogisticCombiner(C=C, seed=seed).fit(train_rows, train_labels)
        return combiner.score_many(rows)
    raise ValueError(f"unknown ablation {name!r}")


def metrics_payload(
    scores: Sequence[float],
    labels: Sequence[str],
    records: Sequence[Record],
    target_fpr: float,
) -> dict[str, Any]:
    domain_map = metrics_by_domain(records, {
        str(record["id"]): float(score)
        for record, score in zip(records, scores)
    }, target_fpr=target_fpr)
    return {
        "n": len(scores),
        "auroc": auroc(scores, labels),
        "tpr_at_fpr": tpr_at_fpr(scores, labels, target_fpr),
        "target_fpr": target_fpr,
        "by_domain": domain_map,
    }


def evaluate_ablations(
    split: Any,
    scores: Mapping[str, Mapping[str, float]],
    *,
    target_fpr: float = 0.01,
    C: float = 1.0,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    train_rows, train_labels = records_to_rows(split.train, scores)
    test_rows, test_labels = records_to_rows(split.test, scores)
    test_records = split.test

    out: dict[str, dict[str, Any]] = {}
    for name in ABLATION_NAMES:
        ablation_scores_for_test = ablation_scores(
            name,
            test_records,
            test_rows,
            train_rows,
            train_labels,
            C=C,
            seed=seed,
        )
        out[name] = metrics_payload(
            ablation_scores_for_test,
            test_labels,
            test_records,
            target_fpr,
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scores",
        type=Path,
        help="precomputed signal scores JSON/JSONL",
    )
    parser.add_argument("--split", type=Path, help="saved split JSON to reuse")
    parser.add_argument("--holdout-domain", action="append", default=[])
    parser.add_argument("--holdout-generator", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--out", type=Path)
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
    parser.add_argument(
        "--scores-out",
        type=Path,
        help="save calculated signal scores JSON",
    )
    parser.add_argument("--split-out", type=Path)
    parser.add_argument("--config", type=Path)
    return parser


def _print_table(results: Mapping[str, Mapping[str, Any]]) -> None:
    print(f"{'ablation':12} {'n':>6} {'AUROC':>9} {'TPR@FPR':>9}")
    print("-" * 40)
    for name, result in results.items():
        auroc_value = float(result["auroc"])
        tpr_value = float(result["tpr_at_fpr"])
        print(
            f"{name:12} {int(result['n']):6d} "
            f"{auroc_value:9.4f} {tpr_value:9.4f}"
        )


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
            print("error: provide --scores or --calculate", file=sys.stderr)
            return 1

        if args.scores_out:
            args.scores_out.parent.mkdir(parents=True, exist_ok=True)
            args.scores_out.write_text(
                json.dumps(
                    {
                        "ai_prob_status": (
                            "calculated" if args.zeroshot else "not_calculated"
                        ),
                        "scores": scores,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved precomputed signals to {args.scores_out}")

        if args.split_out:
            save_split(split, args.split_out)

        results = evaluate_ablations(
            split,
            scores,
            target_fpr=args.target_fpr,
            C=args.C,
            seed=args.seed,
        )
        _print_table(results)

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {
                        "target_fpr": args.target_fpr,
                        "split": split.summary(),
                        "holdout_domains": list(split.holdout_domains),
                        "holdout_generators": list(split.holdout_generators),
                        "ablations": results,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved evaluation to {args.out}")
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
    raise SystemExit(main())