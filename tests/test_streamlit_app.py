import pytest

pytest.importorskip("streamlit")

from app.streamlit_app import evidence_text, paragraph_style


def test_paragraph_styles():
    high = paragraph_style("high")
    uncertain = paragraph_style("uncertain")
    low = paragraph_style("low")
    assert "c62828" in high["border"]
    assert "ef6c00" in uncertain["border"]
    assert "2e7d32" in low["border"]


def test_evidence_text():
    evidence = {
        "closest_source": {
            "title": "Historia",
            "similarity": 0.8,
            "excerpt": "Texto cercano",
        },
        "closest_known_generation": {
            "id": "llm-001",
            "generator": "modelo",
            "score": 0.9,
        },
    }
    rows = evidence_text(evidence)
    assert any("Historia" in row for row in rows)
    assert any("llm-001" in row for row in rows)
