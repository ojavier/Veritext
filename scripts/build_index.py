"""Embed data/reference/ and save the similarity index.

    py scripts/build_index.py
    py scripts/build_index.py --device cuda --out data/.cache/reference_index

The index lands in data/.cache/ (git-ignored); rebuild it after changing the
reference data or the embedding model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signals.similarity import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL,
    ReferenceIndex,
    SentenceTransformerEmbedder,
)
from src.dataset import load_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps...")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args(argv)

    records = load_dataset(["reference"], labels=["human"])
    if not records:
        print("error: no records in data/reference/", file=sys.stderr)
        return 1
    print(f"Embedding {len(records)} reference paragraphs with {args.model}...")

    embedder = SentenceTransformerEmbedder(
        args.model, args.device, args.batch_size, args.max_seq_length
    )
    start = time.time()
    try:
        index = ReferenceIndex.build(records, embedder, args.model)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    out = index.save(args.out)
    print(f"Saved {len(index)} vectors "
          f"({index.embeddings.shape[1]} dims) to {out} in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
