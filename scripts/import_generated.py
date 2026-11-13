"""Import LLM paragraphs written in a separate chat into data/generated/.

Why: a model that has seen this project (the human data, the topic lists, the
detector) may write differently. To avoid that bias, generate the text in a
FRESH chat with no project context and import it here.

Workflow (from the repo root):

1. Print a ready-to-paste prompt for one domain:
       py scripts/import_generated.py --print-prompt ciencia --per-topic 2
2. Paste it in a new chat, save the answer as a text file (JSON lines).
3. Import it:
       py scripts/import_generated.py --input answer.txt \
           --generator claude-sonnet-5-5 --model "<model name used in the chat>"

Each input line is a JSON object: {"domain": ..., "topic": ..., "text": ...}.
Lines that are not JSON (code fences, comments) are ignored. Paragraphs that are
too short/long, not Spanish, multi-line or duplicated are rejected and counted.
The output format matches the rest of data/generated/ (see src/dataset.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_data import DOMAINS, MAX_CHARS, MIN_CHARS  # noqa: E402

from ingest.cleaner import clean_text  # noqa: E402
from ingest.langid import detect_language  # noqa: E402
from src.dataset import load_dataset  # noqa: E402

PROMPT = """Voy a pedirte varios párrafos en español. Responde ÚNICAMENTE con líneas JSON \
(una por párrafo), sin texto adicional, sin introducción y sin bloques de código. Formato de cada línea:
{{"domain": "{domain}", "topic": "<tema>", "text": "<párrafo>"}}

Reglas:
- Cada párrafo es un solo bloque de texto, sin saltos de línea, sin títulos, sin listas y sin comillas envolventes.
- Longitud: entre 300 y 900 caracteres.
- Escribe cada párrafo como lo harías normalmente si un usuario te lo pidiera, sin imitar ningún estilo en particular.
- Para cada tema escribe {n} párrafos distintos (enfoque y redacción diferentes entre sí).

Instrucción para cada párrafo: «{template}»
Temas ({count}):
{topics}
"""


def build_prompt(domain: str, per_topic: int) -> str:
    template, topics = DOMAINS[domain]
    shown = template.replace("{topic}", "<tema>")
    return PROMPT.format(domain=domain, n=per_topic, template=shown, count=len(topics),
                         topics="\n".join(f"- {t}" for t in topics))


def parse_lines(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and {"domain", "text"} <= obj.keys():
            rows.append(obj)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-prompt", choices=sorted(DOMAINS), metavar="DOMAIN",
                    help="print the prompt for a domain and exit; one of: " + ", ".join(sorted(DOMAINS)))
    ap.add_argument("--per-topic", type=int, default=2, help="paragraphs per topic in the printed prompt")
    ap.add_argument("--input", type=Path, help="file with the chat answer (JSON lines)")
    ap.add_argument("--generator", help="short name, e.g. claude-sonnet-5-5 (output file name)")
    ap.add_argument("--model", help="model name as reported in the chat")
    ap.add_argument("--source", default="claude-chat", help="how the text was produced")
    args = ap.parse_args()

    if args.print_prompt:
        print(build_prompt(args.print_prompt, args.per_topic))
        return
    if not (args.input and args.generator and args.model):
        ap.error("--input, --generator and --model are required to import")

    out_path = ROOT / "data" / "generated" / f"{args.generator}.jsonl"
    existing = [r for r in load_dataset(["generated"]) if r["generator"] == args.generator]
    seen = {r["text"] for r in load_dataset(["generated", "reference", "eval"])}
    counters: dict[str, int] = {}
    for r in existing:
        counters[r["domain"]] = counters.get(r["domain"], 0) + 1

    rejected: dict[str, int] = {}
    new = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for row in parse_lines(args.input.read_text(encoding="utf-8")):
            domain = row["domain"]
            text = clean_text(str(row["text"])).strip("\"“”«» ")
            reason = None
            if domain not in DOMAINS:
                reason = "unknown domain"
            elif not (MIN_CHARS <= len(text) <= MAX_CHARS):
                reason = "length"
            elif "\n" in text:
                reason = "multi-line"
            elif detect_language(text) != "es":
                reason = "not Spanish"
            elif text in seen:
                reason = "duplicate"
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            seen.add(text)
            idx = counters.get(domain, 0)
            counters[domain] = idx + 1
            topic = str(row.get("topic", ""))
            f.write(json.dumps({
                "id": f"llm-{args.generator}-{domain}-{idx:04d}", "text": text, "domain": domain,
                "label": "llm", "lang": "es", "source": args.source, "title": topic,
                "generator": args.generator, "model": args.model, "topic": topic,
            }, ensure_ascii=False) + "\n")
            new += 1
    print(f"imported {new} rows -> {out_path.relative_to(ROOT)}; rejected: {rejected or 'none'}")
    print("totals by domain:", counters)


if __name__ == "__main__":
    main()
