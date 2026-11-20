"""Minimal end-to-end Veritext pipeline and a basic CLI.


Only the zero-shot signal exists in Phase 1, so ``similarity`` and
``matching`` are reported as 0.0 with a note in the evidence, and the alert
score is the zero-shot ``ai_prob`` on its own. Phase 4 replaces that with the
calibrated fusion of all three signals.

"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from ingest.cleaner import clean_segments
from ingest.langid import SUPPORTED_LANGUAGES, detect_language_with_confidence
from ingest.segmenter import DEFAULT_MAX_LENGTH, segment_text
from signals.base import BaseSignal
from signals.zeroshot import perplexity
from src.schemas import Alert, Segment, Signals

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "input": {"supported_languages": list(SUPPORTED_LANGUAGES), "max_segment_length": DEFAULT_MAX_LENGTH},
    "alerts": {"levels": {"low": 0.40, "uncertain": 0.70, "high": 0.85}},
    "zeroshot": {
        "observer_model": "Qwen/Qwen2.5-0.5B",
        "performer_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": "cpu",
        "max_tokens": 512,
        "min_tokens": 16,
        # Placeholder
        "threshold": 0.90,
        "temperature": 0.05,
    },
    "fusion": {
        "model": "data/.cache/fusion.json",
        "calibration": "data/.cache/fusion_calibration.json",
    },
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config on top of the built-in defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return DEFAULT_CONFIG
    try:
        import yaml
    except ImportError: 
        print("warning: PyYAML is not installed; using the default configuration", file=sys.stderr)
        return DEFAULT_CONFIG
    with path.open(encoding="utf-8") as handle:
        return _deep_merge(DEFAULT_CONFIG, yaml.safe_load(handle) or {})


# --------------------------------------------------------------------------- #
# Zero-shot signal (Binoculars, Hans et al.)
# --------------------------------------------------------------------------- #
class BinocularsScorer(Protocol):
    """Anything that returns (binoculars_score, n_tokens) for a text."""

    def score(self, text: str) -> tuple[float, int]: ...


class HFBinocularsScorer:
    """Binoculars score with two Hugging Face causal LMs sharing a tokenizer.

    score = log PPL_observer(text) / log X-PPL(observer, performer)(text)

    Models are loaded lazily, once, on first use.
    """

    def __init__(self, observer: str, performer: str, device: str = "cpu", max_tokens: int = 512):
        self.observer_name = observer
        self.performer_name = performer
        self.device = device
        self.max_tokens = max_tokens
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "The zero-shot signal requires 'torch' and 'transformers': pip install torch transformers"
            ) from error
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        self.observer = AutoModelForCausalLM.from_pretrained(self.observer_name).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(self.performer_name).to(self.device).eval()
        self._loaded = True

    def score(self, text: str) -> tuple[float, int]:
        self._load()
        torch = self._torch
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens
        ).to(self.device)
        input_ids = encoded["input_ids"]
        n_tokens = int(input_ids.shape[1]) - 1  # tokens that get predicted
        if n_tokens < 1:
            raise ValueError("segment is too short to score")

        with torch.no_grad():
            obs_logits = self.observer(**encoded).logits[:, :-1].float()
            perf_logits = self.performer(**encoded).logits[:, :-1].float()
        targets = input_ids[:, 1:]

        obs_log_probs = torch.log_softmax(obs_logits, dim=-1)
        token_log_probs = obs_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        log_ppl = math.log(perplexity(token_log_probs.tolist()))

        # Cross-perplexity: performer's next-token distribution scored by the observer.
        perf_probs = torch.softmax(perf_logits, dim=-1)
        x_log_ppl = float(-(perf_probs * obs_log_probs).sum(-1).mean())
        return log_ppl / x_log_ppl, n_tokens


class ZeroShotSignal(BaseSignal):
    """Turns a Binoculars score into ``ai_prob`` in [0, 1]."""

    name = "zeroshot"

    def __init__(self, scorer: BinocularsScorer, threshold: float, temperature: float,
                 min_tokens: int, supported_languages: Iterable[str]):
        self.scorer = scorer
        self.threshold = threshold
        self.temperature = temperature
        self.min_tokens = min_tokens
        self.supported_languages = set(supported_languages)

    def _to_probability(self, raw: float) -> float:
        # Lower Binoculars score => more machine-like => higher ai_prob.
        z = (self.threshold - raw) / self.temperature
        return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))

    def analyze(self, segment: Segment) -> Signals:
        evidence: dict[str, Any] = {"zeroshot": {}}
        info = evidence["zeroshot"]
        if segment.lang not in self.supported_languages:
            info["warning"] = f"language '{segment.lang}' is unsupported; the score may be unreliable"
        try:
            raw, n_tokens = self.scorer.score(segment.text)
        except ValueError as error:
            info["error"] = str(error)
            ai_prob = 0.0
        else:
            ai_prob = self._to_probability(raw)
            info.update(binoculars=round(raw, 4), threshold=self.threshold, n_tokens=n_tokens)
            if n_tokens < self.min_tokens:
                info["warning"] = f"only {n_tokens} tokens; the score may be unreliable"
        evidence["similarity"] = {"status": "not implemented (Phase 3)"}
        evidence["matching"] = {"status": "not implemented (Phase 3)"}
        return Signals(similarity=0.0, matching=0.0, ai_prob=ai_prob, evidence=evidence)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    """Orchestrates ingest -> signals -> calibrated fusion -> alerts."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        scorer: BinocularsScorer | None = None,
        max_workers: int = 1,
        combiner: Any = None,
        calibration: Any = None,
        fusion_model_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
    ):
        self.config = config or load_config()
        fusion_disabled = combiner is False
        if fusion_disabled:
            combiner = None
        zs = self.config["zeroshot"]
        if scorer is None:
            scorer = HFBinocularsScorer(zs["observer_model"], zs["performer_model"],
                                        device=zs["device"], max_tokens=zs["max_tokens"])
        self.signal = ZeroShotSignal(
            scorer,
            threshold=float(zs["threshold"]),
            temperature=float(zs["temperature"]),
            min_tokens=int(zs["min_tokens"]),
            supported_languages=self.config["input"]["supported_languages"],
        )
        levels = self.config["alerts"]["levels"]
        self.uncertain_cutoff = float(levels["uncertain"])
        self.high_cutoff = float(levels["high"])
        self.max_workers = max(1, max_workers)
        self.combiner = combiner
        self.calibration = calibration
        self.similarity_signal = None
        self.matching_signal = None
        if not fusion_disabled:
            self._load_fusion(fusion_model_path, calibration_path)
            self._load_fusion_signals()

    def _resolve_path(self, path: str | Path) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(__file__).resolve().parent.parent / resolved
        return resolved

    def _load_fusion(
        self,
        fusion_model_path: str | Path | None,
        calibration_path: str | Path | None,
    ) -> None:
        from fusion.calibration import Calibration
        from fusion.combiner import LogisticCombiner

        fusion_config = self.config.get("fusion", {})
        model_path = fusion_model_path or fusion_config.get("model")
        cal_path = calibration_path or fusion_config.get("calibration")

        if self.combiner is None and model_path:
            path = self._resolve_path(model_path)
            if path.exists():
                self.combiner = LogisticCombiner.load(path)

        if self.calibration is None and cal_path:
            path = self._resolve_path(cal_path)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.calibration = Calibration(
                    threshold=float(payload["threshold"]),
                    uncertainty_threshold=float(payload["uncertainty_threshold"]),
                    target_fpr=float(payload.get("target_fpr", 0.01)),
                    uncertainty_fpr=float(payload.get("uncertainty_fpr", 0.05)),
                )
        elif isinstance(self.calibration, dict):
            payload = self.calibration
            self.calibration = Calibration(
                threshold=float(payload["threshold"]),
                uncertainty_threshold=float(payload["uncertainty_threshold"]),
                target_fpr=float(payload.get("target_fpr", 0.01)),
                uncertainty_fpr=float(payload.get("uncertainty_fpr", 0.05)),
            )
        elif isinstance(self.calibration, (str, Path)):
            path = self._resolve_path(self.calibration)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.calibration = Calibration(
                threshold=float(payload["threshold"]),
                uncertainty_threshold=float(payload["uncertainty_threshold"]),
                target_fpr=float(payload.get("target_fpr", 0.01)),
                uncertainty_fpr=float(payload.get("uncertainty_fpr", 0.05)),
            )

    def _load_fusion_signals(self) -> None:
        if self.combiner is None:
            return
        try:
            from signals.matching import MatchingSignal
            from signals.similarity import SimilaritySignal

            self.similarity_signal = SimilaritySignal.from_config(dict(self.config))
            self.matching_signal = MatchingSignal.from_config(
                dict(self.config), Path(__file__).resolve().parent.parent / "data"
            )
        except (FileNotFoundError, ImportError, RuntimeError) as error:
            self._fusion_signals_warning = str(error)
        else:
            self._fusion_signals_warning = None

    def ingest(self, text: str) -> list[Segment]:
        max_length = int(self.config["input"]["max_segment_length"])
        segments = clean_segments(segment_text(text, max_length=max_length))
        return [dataclasses.replace(s, lang=detect_language_with_confidence(s.text).language)
                for s in segments]

    def level(self, score: float) -> str:
        if hasattr(self.calibration, "level"):
            return self.calibration.level(score)
        if score >= self.high_cutoff:
            return "high"
        if score >= self.uncertain_cutoff:
            return "uncertain"
        return "low"

    def _alert(self, segment: Segment) -> Alert:
        signals = self.signal.analyze(segment)
        score = signals.ai_prob
        evidence = dict(signals.evidence)

        if self.combiner is not None and self.similarity_signal is not None:
            try:
                similarity = self.similarity_signal.analyze(segment)
                evidence.update(similarity.evidence)
                signals.similarity = similarity.similarity
            except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
                evidence["similarity"] = {"status": "error", "error": str(error)}

        if self.combiner is not None and self.matching_signal is not None:
            try:
                matching = self.matching_signal.analyze(segment)
                evidence.update(matching.evidence)
                signals.matching = matching.matching
            except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
                evidence["matching"] = {"status": "error", "error": str(error)}

        if (
            self.combiner is not None
            and self.similarity_signal is not None
            and self.matching_signal is not None
        ):
            try:
                score = self.combiner.score(signals)
            except (RuntimeError, ValueError) as error:
                evidence["fusion"] = {"status": "error", "error": str(error)}

        if getattr(self, "_fusion_signals_warning", None):
            evidence["fusion"] = {
                "status": "unavailable",
                "error": self._fusion_signals_warning,
            }
        signals.evidence = evidence
        return Alert(segment=segment, score=score, level=self.level(score), signals=signals)

    def run(self, text: str) -> list[Alert]:
        segments = self.ingest(text)
        if self.max_workers == 1:
            return [self._alert(s) for s in segments]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._alert, segments))


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def alerts_to_json(alerts: list[Alert]) -> str:
    payload = {
        "note": "Alerts for human review.",
        "alerts": [dataclasses.asdict(a) for a in alerts],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def alerts_to_text(alerts: list[Alert]) -> str:
    lines = ["Alerts for human review (not a verdict of authorship).", ""]
    for a in alerts:
        preview = a.segment.text if len(a.segment.text) <= 80 else a.segment.text[:77] + "..."
        zs = a.signals.evidence.get("zeroshot", {})
        extra = f" binoculars={zs['binoculars']}" if "binoculars" in zs else ""
        warning = f"  [{zs['warning']}]" if "warning" in zs else ""
        lines.append(f"#{a.segment.id} [{a.level.upper():8}] score={a.score:.3f}{extra} "
                     f"lang={a.segment.lang} chars={a.segment.start}-{a.segment.end}{warning}")
        lines.append(f"    {preview}")
    if not alerts:
        lines.append("(no segments)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _read_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.input in (None, "-"):
        return sys.stdin.read()
    return Path(args.input).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veritext", description="Analyze text by paragraph and issue alerts for human review.")
    parser.add_argument("input", nargs="?", help="text file (or '-' for stdin)")
    parser.add_argument("--text", help="texto a analizar directamente")
    parser.add_argument("--config", help=f"YAML path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--device", help="cpu, cuda, mps... (overrides the config)")
    parser.add_argument("--threshold", type=float, help="Binoculars threshold (overrides the config)")
    parser.add_argument("--workers", type=int, default=1, help="threads for processing segments")
    parser.add_argument("--fusion-model", help="trained fusion model JSON")
    parser.add_argument("--fusion-calibration", help="calibration JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.text is None and args.input is None and sys.stdin.isatty():
        build_parser().print_usage(sys.stderr)
        return 2

    config = load_config(args.config)
    if args.device:
        config = _deep_merge(config, {"zeroshot": {"device": args.device}})
    if args.threshold is not None:
        config = _deep_merge(config, {"zeroshot": {"threshold": args.threshold}})

    try:
        alerts = Pipeline(
            config,
            max_workers=args.workers,
            fusion_model_path=args.fusion_model,
            calibration_path=args.fusion_calibration,
        ).run(_read_input(args))
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(alerts_to_json(alerts) if args.format == "json" else alerts_to_text(alerts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
