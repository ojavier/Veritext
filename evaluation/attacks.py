"""Text attacks used to test robustness of the detector.

All attacks are deterministic given a seed and need no model downloads:

* ``typos``: random character-level errors (swap, drop, duplicate).
* ``synonyms``: swap words using a small Spanish synonym table.
* ``paraphrase``: light rule-based rewrite (connectors and clause openers).
  A real model-based paraphraser can be plugged in via ``paraphrase_with``.

``attack_records`` turns a list of dataset records into attacked copies with a
new id and an ``attack`` field, so splits and metrics can group by attack.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

Record = dict[str, Any]

# Lower-case; matched case-insensitively and the original capitalisation is kept.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "importante": ("relevante", "significativo"),
    "grande": ("amplio", "enorme"),
    "pequeño": ("reducido", "menor"),
    "rápido": ("veloz", "ágil"),
    "comenzó": ("empezó", "inició"),
    "comienza": ("empieza", "inicia"),
    "mostrar": ("exhibir", "presentar"),
    "permite": ("posibilita", "facilita"),
    "ayuda": ("apoya", "colabora"),
    "utiliza": ("emplea", "usa"),
    "utilizan": ("emplean", "usan"),
    "también": ("asimismo", "igualmente"),
    "además": ("también", "asimismo"),
    "porque": ("debido a que", "ya que"),
    "sin embargo": ("no obstante", "pese a ello"),
    "por ejemplo": ("como muestra", "a modo de ejemplo"),
    "muchos": ("numerosos", "diversos"),
    "muchas": ("numerosas", "diversas"),
    "gran": ("notable", "enorme"),
    "desarrollo": ("evolución", "avance"),
    "proceso": ("procedimiento", "mecanismo"),
    "problema": ("dificultad", "inconveniente"),
    "objetivo": ("meta", "propósito"),
    "ciudad": ("urbe", "localidad"),
    "país": ("nación", "estado"),
    "historia": ("trayectoria", "pasado"),
}

# Phrase-level rewrites for the cheap paraphrase.
REWRITES: tuple[tuple[str, str], ...] = (
    (r"\bes decir\b", "o sea"),
    (r"\bpor lo tanto\b", "en consecuencia"),
    (r"\ben conclusión\b", "para concluir"),
    (r"\bpor otro lado\b", "en cambio"),
    (r"\bademás de\b", "junto con"),
    (r"\bdebido a\b", "a causa de"),
    (r"\bse encuentra\b", "está ubicado"),
    (r"\bse considera\b", "se tiene por"),
    (r"\bhoy en día\b", "actualmente"),
    (r"\bactualmente\b", "hoy en día"),
)


def _match_case(original: str, replacement: str) -> str:
    return replacement.capitalize() if original[:1].isupper() else replacement


def typos(text: str, rate: float = 0.03, seed: int = 0) -> str:
    """Introduce character errors in about ``rate`` of the letters."""
    rng = random.Random(seed)
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            kind = rng.choice(("swap", "drop", "dup"))
            if kind == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                out.extend((chars[i + 1], c))
                i += 2
                continue
            if kind == "drop":
                i += 1
                continue
            if kind == "dup":
                out.extend((c, c))
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def synonyms(text: str, rate: float = 0.5, seed: int = 0) -> str:
    """Swap known words for a synonym, each with probability ``rate``."""
    rng = random.Random(seed)
    keys = sorted(SYNONYMS, key=len, reverse=True)  # multi-word entries first
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.IGNORECASE)

    def swap(m: re.Match[str]) -> str:
        if rng.random() >= rate:
            return m.group(0)
        return _match_case(m.group(0), rng.choice(SYNONYMS[m.group(0).lower()]))

    return pattern.sub(swap, text)


def paraphrase(text: str, seed: int = 0) -> str:
    """Cheap rule-based paraphrase: rewrite connectors, then swap synonyms."""
    out = text
    for pattern, replacement in REWRITES:
        out = re.sub(pattern, lambda m, r=replacement: _match_case(m.group(0), r), out, flags=re.IGNORECASE)
    return synonyms(out, rate=0.7, seed=seed)


def paraphrase_with(model: Callable[[str], str]) -> Callable[..., str]:
    """Wrap a model-based paraphraser as an attack with the standard signature."""

    def attack(text: str, seed: int = 0) -> str:
        return model(text)

    return attack


ATTACKS: dict[str, Callable[..., str]] = {
    "typos": typos,
    "synonyms": synonyms,
    "paraphrase": paraphrase,
}


def apply_attack(text: str, name: str, seed: int = 0, **kwargs: Any) -> str:
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name!r}; expected one of {sorted(ATTACKS)}")
    return ATTACKS[name](text, seed=seed, **kwargs)


def attack_records(
    records: Iterable[Record],
    name: str,
    seed: int = 0,
    labels: Sequence[str] | None = None,
    **kwargs: Any,
) -> list[Record]:
    """Return attacked copies: new id ``<id>~<name>``, ``attack`` set and
    ``source_id`` pointing at the original (so indexes can hide it).

    ``labels`` restricts which records are attacked (e.g. only ``["llm"]``,
    since an evasion attack is applied to machine text).
    """
    out: list[Record] = []
    for i, record in enumerate(records):
        if labels is not None and record.get("label") not in labels:
            continue
        copy = dict(record)
        copy["id"] = f"{record['id']}~{name}"
        copy["text"] = apply_attack(str(record["text"]), name, seed=seed + i, **kwargs)
        copy["attack"] = name
        copy["source_id"] = record["id"]
        out.append(copy)
    return out
