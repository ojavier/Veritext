"""Collect Spanish movie-review paragraphs from MuchoCine for the "resenas" domain.

MuchoCine (Universidad de Sevilla, reviews from muchocine.net) is an academic
corpus published on Hugging Face in early 2022, i.e. before ChatGPT, so it
cannot contain LLM-generated text. The original download link
(lsi.us.es/~fermin/corpusCine.zip) is dead, so this script downloads the
parquet conversion from Hugging Face. Reading it needs ``pyarrow``
(``py -m pip install pyarrow``); it is only needed for this script.

Reviews have no paragraph breaks, so the segmenter splits them by sentence into
chunks of at most 1000 characters. Reviews (not chunks) are split between
``reference`` and ``eval`` so no review appears in both.

Output: data/reference/reviews_es.jsonl and data/eval/reviews_es.jsonl, with the
same fields as human_es.jsonl.

Usage (from the repo root):
    py scripts/collect_reviews_es.py --ref 120 --eval 60
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.cleaner import clean_segments  # noqa: E402
from ingest.langid import detect_language  # noqa: E402
from ingest.segmenter import segment_text  # noqa: E402

# Force IPv4 (IPv6 connects hang on some networks, see collect_human_es.py).
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **kw: _orig_getaddrinfo(
    host, port, socket.AF_INET, *a, **kw
)

PARQUET_URL = (
    "https://huggingface.co/datasets/us-lsi/muchocine/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
CACHE = ROOT / "data" / ".cache" / "muchocine.parquet"
SOURCE = "huggingface.co/datasets/us-lsi/muchocine"
MIN_CHARS, MAX_CHARS = 250, 1000
MAX_PER_REVIEW = 3


def load_reviews() -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: py -m pip install pyarrow")
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "veritext-student-project/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            CACHE.write_bytes(resp.read())
    return pq.read_table(CACHE).column("review_body").to_pylist()


def review_records(idx: int, body: str) -> list[dict]:
    out = []
    for seg in clean_segments(segment_text(body)):
        if not (MIN_CHARS <= len(seg.text) <= MAX_CHARS):
            continue
        if detect_language(seg.text) != "es":
            continue
        # Skip the movie-sheet header ("Director: ... Duración: ... Estreno: ...").
        if "Duración:" in seg.text and ("Director:" in seg.text or "Estreno:" in seg.text):
            continue
        out.append({"text": seg.text, "domain": "resenas", "label": "human", "lang": "es",
                    "source": SOURCE, "title": f"muchocine-{idx}", "revid": 0})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=int, default=120)
    ap.add_argument("--eval", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reviews = load_reviews()
    order = list(range(len(reviews)))
    random.Random(args.seed).shuffle(order)

    # Fill eval first from the shuffled reviews, then reference from the rest,
    # taking whole reviews so the two sets never share one.
    seen: set[str] = set()
    sets: dict[str, list[dict]] = {"eval": [], "reference": []}
    targets = {"eval": args.eval, "reference": args.ref}
    pos = 0
    for name in ("eval", "reference"):
        while len(sets[name]) < targets[name] and pos < len(order):
            idx = order[pos]
            pos += 1
            for rec in review_records(idx, reviews[idx])[:MAX_PER_REVIEW]:
                if rec["text"] not in seen:
                    seen.add(rec["text"])
                    sets[name].append(rec)
        sets[name] = sets[name][: targets[name]]

    for name, rows in sets.items():
        path = ROOT / "data" / name / "reviews_es.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for i, row in enumerate(rows):
                f.write(json.dumps({"id": f"{name}-resenas-{i:04d}", **row}, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
