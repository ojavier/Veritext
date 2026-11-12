"""Generate a reproducible local corpus for Veritext evaluation.

The project currently has no LLM API dependency.  The ``llm`` records are
therefore local template fixtures, not outputs claimed to come from a remote
model.  Replace the template generator when a real provider is introduced.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Final

DEFAULT_OUTPUT: Final = Path("data/generated/veritext_dataset.jsonl")
DEFAULT_AI_PER_COMBINATION: Final = 60
DEFAULT_HUMAN_COUNT: Final = 1000
DATASET_VERSION: Final = "0.1.0"

DOMAINS: Final = {
    "educacion": (
        "Una estrategia educativa puede mejorar {topic} cuando combina "
        "objetivos claros, práctica constante y retroalimentación útil."
    ),
    "tecnologia": (
        "La tecnología aplicada a {topic} requiere evaluar sus beneficios, "
        "sus límites y la forma en que afecta a las personas."
    ),
    "salud": (
        "En el ámbito de la salud, {topic} debe abordarse con información "
        "verificable, atención a los riesgos y decisiones responsables."
    ),
    "medio_ambiente": (
        "Las decisiones relacionadas con {topic} tienen consecuencias "
        "ambientales que deben medirse antes de elegir una solución."
    ),
}

TOPICS: Final = (
    "la planificación de proyectos",
    "el uso responsable de los datos",
    "la colaboración entre equipos",
    "la evaluación de resultados",
    "la toma de decisiones",
)

PURPOSES: Final = (
    "explicacion",
    "resumen",
    "recomendacion",
)

GENERATORS: Final = ("local_llm_a", "local_llm_b", "local_llm_c")


def _text(domain: str, generator: str, index: int, *, human: bool) -> str:
    topic = TOPICS[index % len(TOPICS)]
    base = DOMAINS[domain].format(topic=topic)
    if human:
        endings = (
            " En la práctica, el resultado depende también del contexto.",
            " Conviene comparar esta idea con la experiencia disponible.",
            " Por eso es importante revisar el caso concreto antes de actuar.",
        )
        return base + endings[index % len(endings)]

    styles = {
        "local_llm_a": " El proceso debe ser gradual y revisarse con criterios definidos.",
        "local_llm_b": " Además, una implementación ordenada facilita observar sus resultados.",
        "local_llm_c": " En consecuencia, conviene equilibrar las oportunidades con las posibles dificultades.",
    }
    return base + styles[generator]


def _record(
    *,
    record_id: str,
    text: str,
    label: str,
    domain: str,
    generator: str,
    purpose: str,
    index: int,
) -> dict[str, object]:
    return {
        "id": record_id,
        "text": text,
        "label": label,
        "language": "es",
        "domain": domain,
        "generator": generator,
        "purpose": purpose,
        "source": "local_template",
        "split": "all",
        "dataset_version": DATASET_VERSION,
        "seed_index": index,
    }


def generate_records(
    *,
    ai_per_combination: int = DEFAULT_AI_PER_COMBINATION,
    human_count: int = DEFAULT_HUMAN_COUNT,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Return local LLM-style and human-reference records."""

    if ai_per_combination < 0 or human_count < 0:
        raise ValueError("record counts must be non-negative")

    records: list[dict[str, object]] = []
    index = 0
    for domain in DOMAINS:
        for generator in GENERATORS:
            for item in range(ai_per_combination):
                purpose = PURPOSES[item % len(PURPOSES)]
                records.append(
                    _record(
                        record_id=f"llm-{domain}-{generator}-{item:04d}",
                        text=_text(domain, generator, item, human=False),
                        label="llm",
                        domain=domain,
                        generator=generator,
                        purpose=purpose,
                        index=index,
                    )
                )
                index += 1

    human_rng = random.Random(seed)
    for item in range(human_count):
        domain = list(DOMAINS)[human_rng.randrange(len(DOMAINS))]
        purpose = PURPOSES[human_rng.randrange(len(PURPOSES))]
        records.append(
            _record(
                record_id=f"human-{item:04d}",
                text=_text(domain, "human_reference", item, human=True),
                label="human",
                domain=domain,
                generator="human_reference",
                purpose=purpose,
                index=index,
            )
        )
        index += 1

    return records


def write_jsonl(records: list[dict[str, object]], output: Path) -> None:
    """Write records as UTF-8 JSON Lines, creating the parent directory."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ai-per-combination", type=int, default=DEFAULT_AI_PER_COMBINATION)
    parser.add_argument("--human-count", type=int, default=DEFAULT_HUMAN_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = generate_records(
        ai_per_combination=args.ai_per_combination,
        human_count=args.human_count,
        seed=args.seed,
    )
    write_jsonl(records, args.output)
    print(f"Generated {len(records)} records in {args.output}")


if __name__ == "__main__":
    main()