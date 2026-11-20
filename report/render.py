"""Render a report built by :mod:`report.alert` (JSON first, HTML later)."""

from __future__ import annotations

import html
import json
from typing import Any, Iterable

from report.alert import build_report
from src.schemas import Alert


def render_json(alerts: Iterable[Alert] | dict[str, Any], indent: int = 2) -> str:
    report = alerts if isinstance(alerts, dict) else build_report(alerts)
    return json.dumps(report, ensure_ascii=False, indent=indent)


_LEVEL_LABEL = {"high": "Alta", "uncertain": "Incierta", "low": "Baja"}

_CSS = """
body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1b1b1b;background:#fff}
.note{color:#555;font-size:.9rem}
.seg{border-left:6px solid #999;padding:.6rem 1rem;margin:1rem 0;background:#f7f7f7;border-radius:4px}
.seg.high{border-color:#c62828;background:#fdecea}
.seg.uncertain{border-color:#ef6c00;background:#fff4e5}
.seg.low{border-color:#2e7d32;background:#edf7ee}
.meta{font-size:.8rem;color:#444;margin-bottom:.4rem}
.badge{font-weight:600}
details{margin-top:.5rem;font-size:.85rem}
blockquote{margin:.3rem 0 .3rem 1rem;color:#333}
@media (prefers-color-scheme:dark){
body{background:#161616;color:#eee}.note,.meta{color:#bbb}
.seg{background:#222}.seg.high{background:#3a1c1a}.seg.uncertain{background:#3a2a14}.seg.low{background:#17301a}
blockquote{color:#ddd}}
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _evidence_html(evidence: dict[str, Any]) -> str:
    parts: list[str] = []
    src = evidence.get("closest_source")
    if src:
        parts.append(
            f"<p><b>Fuente más cercana</b>: {_e(src.get('title') or src.get('id'))} "
            f"({_e(src.get('source') or '?')}), similitud {_e(src.get('similarity'))}</p>"
            f"<blockquote>{_e(src.get('excerpt'))}</blockquote>"
        )
    gen = evidence.get("closest_known_generation")
    if gen:
        shared = f", texto compartido: «{_e(gen['shared_text'])}»" if gen.get("shared_text") else ""
        parts.append(
            f"<p><b>Generación conocida más cercana</b>: {_e(gen.get('id'))} "
            f"({_e(gen.get('generator') or '?')}), puntuación {_e(gen.get('score'))}{shared}</p>"
            f"<blockquote>{_e(gen.get('excerpt'))}</blockquote>"
        )
    zs = evidence.get("zeroshot")
    if zs:
        parts.append("<p><b>Zero-shot</b>: " + ", ".join(f"{_e(k)}={_e(v)}" for k, v in zs.items()) + "</p>")
    fusion = evidence.get("fusion")
    if fusion:
        parts.append(f"<p><b>Fusión</b>: {_e(fusion)}</p>")
    return "".join(parts)


def render_html(alerts: Iterable[Alert] | dict[str, Any]) -> str:
    """Self-contained HTML page: segments colored by level, evidence in a panel."""
    report = alerts if isinstance(alerts, dict) else build_report(alerts)
    levels = report["summary"]["levels"]
    body = [
        "<h1>Alertas para revisión humana</h1>",
        f"<p class='note'>{_e(report['note'])}</p>",
        f"<p>{report['summary']['segments']} segmentos: "
        f"{levels['high']} alta, {levels['uncertain']} incierta, {levels['low']} baja.</p>",
    ]
    for item in report["alerts"]:
        seg, sig = item["segment"], item["signals"]
        level = item["level"] if item["level"] in _LEVEL_LABEL else "low"
        evidence = _evidence_html(item["evidence"])
        panel = f"<details><summary>Evidencia</summary>{evidence}</details>" if evidence else ""
        body.append(
            f"<div class='seg {level}'>"
            f"<div class='meta'><span class='badge'>#{seg['id']} · {_LEVEL_LABEL[level]}</span> · "
            f"puntuación {item['score']} · similitud {sig['similarity']} · "
            f"coincidencia {sig['matching']} · IA {sig['ai_prob']} · idioma {_e(seg['lang'])}</div>"
            f"<div>{_e(seg['text'])}</div>{panel}</div>"
        )
    return (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Alertas Veritext</title>"
        f"<style>{_CSS}</style></head><body>{''.join(body)}</body></html>"
    )
