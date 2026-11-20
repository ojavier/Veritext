"""Report building and JSON rendering (no models needed)."""

import json

from report.alert import build_report
from report.render import render_json
from src.schemas import Alert, Segment, Signals


def make_alert(i: int, level: str, evidence: dict | None = None) -> Alert:
    seg = Segment(id=i, text=f"Párrafo {i} " + "palabra " * 10, lang="es", start=0, end=10)
    return Alert(seg, 0.5, level, Signals(0.4, 0.1, 0.9, evidence or {}))


def test_summary_counts_and_review_ids():
    report = build_report([make_alert(0, "high"), make_alert(1, "low"), make_alert(2, "uncertain")])
    assert report["summary"]["segments"] == 3
    assert report["summary"]["levels"] == {"high": 1, "uncertain": 1, "low": 1}
    assert report["summary"]["needs_review"] == [0, 2]
    assert "verdict" in report["note"]


def test_evidence_is_distilled_and_excerpts_are_short():
    evidence = {
        "similarity": {"nearest": [{"id": "r1", "title": "T", "source": "s", "score": 0.9, "text": "x" * 500}]},
        "matching": {"nearest": [{"id": "g1", "generator": "m", "score": 0.2, "shared_text": "a b c", "text": "y"}]},
        "zeroshot": {"binoculars": 0.8, "threshold": 0.9, "n_tokens": 30},
    }
    ev = build_report([make_alert(0, "high", evidence)])["alerts"][0]["evidence"]
    assert ev["closest_source"]["id"] == "r1" and len(ev["closest_source"]["excerpt"]) <= 200
    assert ev["closest_known_generation"]["shared_text"] == "a b c"
    assert ev["zeroshot"]["binoculars"] == 0.8


def test_missing_evidence_and_empty_input():
    assert build_report([make_alert(0, "low")])["alerts"][0]["evidence"] == {}
    assert build_report([])["summary"]["segments"] == 0


def test_render_json_roundtrip_keeps_accents():
    out = render_json([make_alert(0, "high")])
    assert "Párrafo" in out
    assert json.loads(out)["alerts"][0]["level"] == "high"


def test_render_html_escapes_and_colors():
    from report.render import render_html

    alert = make_alert(0, "high", {"similarity": {"nearest": [{"id": "r", "score": 0.9, "text": "<script>x</script>"}]}})
    alert.segment.text = "<b>hola</b>"
    out = render_html([alert])
    assert "<b>hola</b>" not in out and "&lt;b&gt;hola&lt;/b&gt;" in out
    assert "<script>x" not in out
    assert "seg high" in out and "Evidencia" in out
    assert "<html" in render_html([])
