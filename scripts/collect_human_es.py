"""Collect human-written Spanish paragraphs for data/reference and data/eval.

Sources: Spanish Wikipedia (several topics) and Spanish Wikinews (news).
To avoid LLM contamination every article is fetched at its last revision
BEFORE the cutoff date (default 2022-11-01, before ChatGPT's release).

Articles are split between ``reference`` (corpus for the similarity signal)
and ``eval`` (human class of the validation set), so no article appears in
both. Output is JSONL, one paragraph per line.

Usage (from the repo root):
    py scripts/collect_human_es.py --ref-per-domain 120 --eval-per-domain 60
"""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.cleaner import clean_segments  # noqa: E402
from ingest.langid import detect_language  # noqa: E402
from ingest.segmenter import segment_text  # noqa: E402

# Force IPv4: on some networks IPv6 connects hang ~21 s before falling back,
# which made every API call (and the whole collection) extremely slow.
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **kw: _orig_getaddrinfo(
    host, port, socket.AF_INET, *a, **kw
)

USER_AGENT ="veritext-student-project/0.1 (academic research)"

# domain -> (host, categories). Wikinews has no stable topic categories, so it
# uses random pages instead (categories=None).
DOMAINS: dict[str, tuple[str, list[str] | None]] = {
    "ciencia": ("es.wikipedia.org", ["Física", "Biología", "Química", "Astronomía", "Geología"]),
    "historia": ("es.wikipedia.org", ["Historia de España", "Historia de México", "Historia de Argentina", "Historia de Colombia"]),
    "geografia": ("es.wikipedia.org", ["Geografía de España", "Geografía de México", "Geografía de Argentina", "Geografía de Chile"]),
    "cultura": ("es.wikipedia.org", ["Literatura", "Música", "Cine", "Pintura"]),
    "tecnologia": ("es.wikipedia.org", ["Informática", "Ingeniería", "Telecomunicaciones", "Internet"]),
    "noticias": ("es.wikinews.org", None),
}

MIN_CHARS, MAX_CHARS = 250, 1000
MAX_PER_ARTICLE = 3


def api(host: str, **params) -> dict:
    params["format"] = "json"
    url = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {}


class _ParagraphExtractor(HTMLParser):
    """Collect text inside <p> tags, skipping refs, tables and math."""

    SKIP = {"sup", "style", "script", "table", "math"}

    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self._buf: list[str] = []
        self._in_p = False
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._in_p, self._buf = True, []
        elif tag in self.SKIP and self._in_p:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag == "p" and self._in_p:
            self.paragraphs.append("".join(self._buf))
            self._in_p = False
        elif tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._in_p and not self._skip:
            self._buf.append(data)


def candidate_titles(host: str, categories: list[str] | None, want: int) -> list[str]:
    titles: set[str] = set()
    if categories is None:
        for _ in range(60):  # bounded, in case the wiki has few pages
            if len(titles) >= want:
                break
            data = api(host, action="query", list="random", rnnamespace=0, rnlimit=50)
            titles.update(p["title"] for p in data.get("query", {}).get("random", []))
        return sorted(titles)
    for cat in categories:
        subcats: list[str] = []
        for kind, ns in (("page", 0), ("subcat", 14)):
            data = api(host, action="query", list="categorymembers", cmtitle=f"Categoría:{cat}",
                       cmnamespace=ns, cmlimit=200, cmtype=kind)
            members = [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
            if kind == "page":
                titles.update(members)
            else:
                subcats = members
        for sub in subcats[:15]:  # one level of subcategories
            data = api(host, action="query", list="categorymembers", cmtitle=sub,
                       cmnamespace=0, cmlimit=60, cmtype="page")
            titles.update(m["title"] for m in data.get("query", {}).get("categorymembers", []))
    return sorted(titles)


def fetch_paragraphs(host: str, title: str, cutoff: str) -> tuple[int, list[str]]:
    """Return (revid, paragraphs) of the last revision before ``cutoff``."""
    data = api(host, action="query", prop="revisions", titles=title, rvprop="ids|timestamp",
               rvlimit=1, rvstart=cutoff, rvdir="older")
    pages = data.get("query", {}).get("pages", {})
    revs = next(iter(pages.values()), {}).get("revisions") if pages else None
    if not revs:
        return 0, []
    revid = revs[0]["revid"]
    parsed = api(host, action="parse", oldid=revid, prop="text", disablelimitreport=1)
    html = parsed.get("parse", {}).get("text", {}).get("*", "")
    extractor = _ParagraphExtractor()
    extractor.feed(html)
    return revid, extractor.paragraphs


def article_records(host: str, title: str, domain: str, cutoff: str) -> list[dict]:
    revid, raw = fetch_paragraphs(host, title, cutoff)
    out = []
    for para in raw:
        para = re.sub(r"\[\s*\d+\s*\]|\[[a-z]\]", "", para)  # leftover [1], [a]
        for seg in clean_segments(segment_text(para)):
            if not (MIN_CHARS <= len(seg.text) <= MAX_CHARS):
                continue
            if detect_language(seg.text) != "es":
                continue
            out.append({"text": seg.text, "domain": domain, "label": "human", "lang": "es",
                        "source": host, "title": title, "revid": revid})
    return out[:MAX_PER_ARTICLE]


def collect(domain: str, n_ref: int, n_eval: int, cutoff: str, rng: random.Random):
    host, cats = DOMAINS[domain]
    needed = n_ref + n_eval
    titles = candidate_titles(host, cats, want=needed)
    rng.shuffle(titles)
    # Over-fetch: many articles yield no usable paragraph.
    pool = titles[: needed * 2]
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda t: article_records(host, t, domain, cutoff), pool))
    articles = [r for r in results if r]
    # Split by ARTICLE so reference and eval never share an article.
    n_eval_art = max(1, round(len(articles) * n_eval / max(needed, 1)))
    eval_arts, ref_arts = articles[:n_eval_art], articles[n_eval_art:]
    ref = [r for a in ref_arts for r in a][:n_ref]
    ev = [r for a in eval_arts for r in a][:n_eval]
    print(f"[{domain}] articles={len(articles)} reference={len(ref)} eval={len(ev)}", flush=True)
    return ref, ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-per-domain", type=int, default=120)
    ap.add_argument("--eval-per-domain", type=int, default=60)
    ap.add_argument("--cutoff", default="2022-11-01T00:00:00Z", help="use revisions before this date")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domains", nargs="*", default=list(DOMAINS))
    args = ap.parse_args()

    # Each finished domain is cached so an interrupted run can resume.
    cache_dir = ROOT / "data" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_all, eval_all = [], []
    for domain in args.domains:
        key = f"{domain}-{args.ref_per_domain}-{args.eval_per_domain}-{args.seed}.json"
        cache = cache_dir / key
        if cache.exists():
            ref, ev = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[{domain}] cached reference={len(ref)} eval={len(ev)}", flush=True)
        else:
            # Per-domain RNG: results do not depend on which domains ran before.
            rng = random.Random(f"{args.seed}-{domain}")
            ref, ev = collect(domain, args.ref_per_domain, args.eval_per_domain, args.cutoff, rng)
            cache.write_text(json.dumps([ref, ev], ensure_ascii=False), encoding="utf-8")
        ref_all += ref
        eval_all += ev

    # Drop exact duplicate texts across both sets (eval wins), so no eval
    # paragraph can also appear in the reference corpus.
    seen = {r["text"] for r in eval_all}
    ref_all = [r for r in ref_all if r["text"] not in seen]
    uniq, eval_unique = set(), []
    for r in eval_all:
        if r["text"] not in uniq:
            uniq.add(r["text"])
            eval_unique.append(r)
    eval_all = eval_unique
    uniq = set()
    ref_all = [r for r in ref_all if not (r["text"] in uniq or uniq.add(r["text"]))]

    for name, rows in (("reference", ref_all), ("eval", eval_all)):
        path = ROOT / "data" / name / "human_es.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for i, row in enumerate(rows):
                f.write(json.dumps({"id": f"{name}-{i:04d}", **row}, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
