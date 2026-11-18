"""Generate Spanish AI-written paragraphs with local instruction-tuned LLMs.

Produces the "llm" class of the dataset, with the SAME domains and record
format as the human data (scripts/collect_human_es.py,
scripts/collect_reviews_es.py), so both can be read together.

Each generator is a different Hugging Face chat model; the output goes to
``data/generated/<generator>.jsonl``. Records have the fields
``id, text, domain, label, lang, source, title, generator, model, topic``
where ``label`` is always "llm" (human text is never fabricated here).

Qwen2.5-0.5B is deliberately NOT offered: it is the observer/performer pair of
the zero-shot (Binoculars) signal, and using it as generator would make that
signal artificially easy.

Usage (from the repo root):
    py scripts/generate_data.py --generator qwen1.5b --per-domain 80
    py scripts/generate_data.py --list
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.cleaner import clean_text  # noqa: E402
from ingest.langid import detect_language  # noqa: E402

# name -> Hugging Face model id. Add more entries to add generators.
GENERATORS: dict[str, str] = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",  # 3B in fp16 (~6.2 GB) does not fit in 6 GB of VRAM
}

# domain -> (instruction template, topics). Topics are NOT taken from the human
# articles; they are generic so the generated text does not copy human content.
DOMAINS: dict[str, tuple[str, list[str]]] = {
    "ciencia": (
        "Escribe un párrafo enciclopédico en español sobre {topic}.",
        ["la fotosíntesis", "los agujeros negros", "la tectónica de placas", "el sistema inmunológico",
         "la tabla periódica", "la teoría de la evolución", "las ondas electromagnéticas", "el ciclo del agua",
         "los ácidos y las bases", "la mecánica cuántica", "el efecto invernadero", "las células madre",
         "la gravedad", "los volcanes", "la genética mendeliana", "las estrellas de neutrones",
         "la energía nuclear", "el sistema solar", "los ecosistemas marinos", "la termodinámica"],
    ),
    "historia": (
        "Escribe un párrafo de historia en español sobre {topic}, con estilo enciclopédico.",
        ["la conquista de México-Tenochtitlan", "la independencia de Argentina", "la Revolución Mexicana",
         "el Imperio inca", "la Guerra Civil Española", "la caída del Imperio romano", "la Revolución francesa",
         "la colonización de América", "las guerras de independencia hispanoamericanas", "la Primera Guerra Mundial",
         "el imperio azteca", "la Reconquista", "la Guerra Fría", "la Revolución cubana", "la dictadura militar en Chile",
         "el Virreinato del Perú", "el descubrimiento de América", "la Ilustración", "la Revolución industrial",
         "la Guerra de los Mil Días"],
    ),
    "geografia": (
        "Escribe un párrafo de geografía en español sobre {topic}, con estilo enciclopédico.",
        ["la cordillera de los Andes", "el desierto de Atacama", "el río Amazonas", "la península de Yucatán",
         "los Pirineos", "la Patagonia", "el lago Titicaca", "la sierra Madre Occidental", "las islas Canarias",
         "el valle del Cauca", "la costa del Pacífico de Colombia", "el altiplano boliviano", "el río Ebro",
         "la meseta central de España", "el golfo de California", "las cataratas del Iguazú", "el Caribe mexicano",
         "la pampa argentina", "el istmo de Tehuantepec", "los fiordos de Chile"],
    ),
    "cultura": (
        "Escribe un párrafo en español sobre {topic}, con estilo enciclopédico.",
        ["el realismo mágico", "el muralismo mexicano", "el tango", "la novela del Siglo de Oro", "el cine de autor",
         "la música barroca", "el modernismo literario", "la pintura impresionista", "el flamenco",
         "el cine mexicano de la época de oro", "la poesía del 27", "el rock en español", "la arquitectura gótica",
         "el teatro del Siglo de Oro", "la fotografía documental", "el surrealismo", "la cumbia", "el jazz",
         "la ópera", "el arte prehispánico"],
    ),
    "tecnologia": (
        "Escribe un párrafo técnico en español sobre {topic}, con estilo enciclopédico.",
        ["los sistemas operativos", "las bases de datos relacionales", "el protocolo TCP/IP", "la computación en la nube",
         "los algoritmos de ordenamiento", "las redes neuronales", "el cifrado de clave pública", "los microprocesadores",
         "la fibra óptica", "los lenguajes de programación", "la ingeniería de software", "el internet de las cosas",
         "las energías renovables", "los motores de combustión interna", "la robótica industrial", "el sistema GPS",
         "las redes inalámbricas", "la impresión 3D", "los semiconductores", "el control de versiones"],
    ),
    "noticias": (
        "Redacta un párrafo de noticia en español sobre {topic}, en tono periodístico.",
        ["la inauguración de un hospital regional", "el aumento del precio de la gasolina", "una ola de calor en el país",
         "la aprobación de un presupuesto municipal", "el cierre de una carretera por lluvias", "un nuevo récord en el maratón de la ciudad",
         "la apertura de un museo de arte", "una huelga de transporte público", "la llegada de un huracán a la costa",
         "las elecciones locales", "el lanzamiento de un satélite", "un festival de cine internacional",
         "la sequía en el campo", "una reforma educativa", "la reapertura de una estación de tren",
         "un descubrimiento arqueológico", "un partido de fútbol por el campeonato", "el cierre de una fábrica",
         "una campaña de vacunación", "la visita de un presidente extranjero"],
    ),
    "resenas": (
        "Escribe un párrafo de reseña crítica en español de la película «{topic}», como si fueras un crítico de cine.",
        ["El padrino", "Titanic", "Matrix", "Amélie", "Pulp Fiction", "El laberinto del fauno", "Cinema Paradiso",
         "Volver", "Amores perros", "El secreto de sus ojos", "Blade Runner", "La vida es bella", "Forrest Gump",
         "Parque Jurásico", "Los otros", "Y tu mamá también", "Casablanca", "El Rey León", "Gladiator", "Alien"],
    ),
}

MIN_CHARS, MAX_CHARS = 250, 1000
SYSTEM = "Eres un asistente que escribe en español. Responde únicamente con el párrafo pedido, sin títulos ni listas."


def build_jobs(per_domain: int, seed: int) -> list[tuple[str, str, str]]:
    """Return (domain, topic, prompt) jobs, cycling through each domain's topics."""
    rng = random.Random(seed)
    jobs = []
    for domain, (template, topics) in DOMAINS.items():
        order = [topics[i % len(topics)] for i in range(per_domain)]
        rng.shuffle(order)
        jobs += [(domain, t, template.format(topic=t)) for t in order]
    return jobs


def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return tok, model


def generate_batch(tok, model, prompts: list[str], device: str, temperature: float, max_new_tokens: int):
    import torch

    chats = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]  # ignored by non-Qwen3 templates
    enc = tok(chats, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                             top_p=0.92, repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


def accept(text: str) -> str | None:
    """Clean a raw generation; return None if it is not a usable paragraph."""
    text = clean_text(text).strip("\"“”«» ")
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    if text.count("\n") or text.startswith(("#", "*", "-", "1.")):
        return None  # lists / headings
    if detect_language(text) != "es":
        return None
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", choices=sorted(GENERATORS))
    ap.add_argument("--per-domain", type=int, default=80, help="target accepted paragraphs per domain")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N jobs (smoke test)")
    ap.add_argument("--list", action="store_true", help="list generators and exit")
    args = ap.parse_args()

    if args.list or not args.generator:
        for name, mid in GENERATORS.items():
            print(f"{name}: {mid}")
        return

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = GENERATORS[args.generator]
    out_path = ROOT / "data" / "generated" / f"{args.generator}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: keep what was already accepted and only generate the missing amount.
    done: dict[str, int] = {d: 0 for d in DOMAINS}
    seen: set[str] = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["domain"]] = done.get(r["domain"], 0) + 1
            seen.add(r["text"])
    total_existing = sum(done.values())

    # Over-request (acceptance is < 100%), then stop each domain at the target.
    jobs = build_jobs(int(args.per_domain * 1.6) + 4, args.seed + total_existing)
    jobs = [j for j in jobs if done[j[0]] < args.per_domain]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{args.generator} ({model_id}) on {device}: {sum(max(args.per_domain - v, 0) for v in done.values())} paragraphs missing", flush=True)
    if not jobs:
        print("nothing to do")
        return

    tok, model = load_model(model_id, device)
    start = time.time()
    n_new = 0
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for i in range(0, len(jobs), args.batch_size):
            batch = [j for j in jobs[i:i + args.batch_size] if done[j[0]] < args.per_domain]
            if not batch:
                continue
            raws = generate_batch(tok, model, [b[2] for b in batch], device, args.temperature, args.max_new_tokens)
            for (domain, topic, _), raw in zip(batch, raws):
                text = accept(raw)
                if text is None or text in seen or done[domain] >= args.per_domain:
                    continue
                seen.add(text)
                idx = done[domain]
                done[domain] += 1
                n_new += 1
                f.write(json.dumps({
                    "id": f"llm-{args.generator}-{domain}-{idx:04d}", "text": text, "domain": domain,
                    "label": "llm", "lang": "es", "source": model_id, "title": topic,
                    "generator": args.generator, "model": model_id, "topic": topic,
                }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {n_new} new | {dict(done)} | {time.time() - start:.0f}s", flush=True)
            if all(v >= args.per_domain for v in done.values()):
                break
    print(f"wrote {n_new} new rows -> {out_path.relative_to(ROOT)} (total {sum(done.values())})")


if __name__ == "__main__":
    main()
