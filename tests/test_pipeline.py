"""End-to-end checks of the pipeline with a fake scorer (no models needed)."""

import json

from src.pipeline import Pipeline, alerts_to_json, load_config


class FakeScorer:
    """Returns a fixed Binoculars score; low scores look machine-written."""

    def __init__(self, value: float):
        self.value = value

    def score(self, text: str) -> tuple[float, int]:
        return self.value, 100


TEXT = (
    "La astronomía surge desde que la humanidad dejó de ser nómada y se empezó a "
    "contemplar el cielo con atención.\n\n"
    "The second paragraph is written in English and describes how telescopes work."
)


def test_segments_language_and_offsets():
    pipe = Pipeline(load_config(), scorer=FakeScorer(1.0))
    segments = pipe.ingest(TEXT)
    assert [s.lang for s in segments] == ["es", "en"]
    for s in segments:
        assert TEXT[s.start : s.end].split() == s.text.split()


def test_scores_are_probabilities_and_levels_follow_score():
    for raw, expected in ((0.5, "high"), (1.5, "low")):
        alerts = Pipeline(load_config(), scorer=FakeScorer(raw)).run(TEXT)
        assert alerts
        for a in alerts:
            assert 0.0 <= a.score <= 1.0
            assert 0.0 <= a.signals.ai_prob <= 1.0
            assert a.level == expected


def test_json_output_is_valid_and_keeps_evidence():
    alerts = Pipeline(load_config(), scorer=FakeScorer(1.0)).run(TEXT)
    payload = json.loads(alerts_to_json(alerts))
    assert len(payload["alerts"]) == len(alerts)
    assert "zeroshot" in payload["alerts"][0]["signals"]["evidence"]


def test_empty_text_gives_no_alerts():
    assert Pipeline(load_config(), scorer=FakeScorer(1.0)).run("  \n\n ") == []
