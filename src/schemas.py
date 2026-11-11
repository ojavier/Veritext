from dataclasses import dataclass


@dataclass
class Segment:
    id: int; text: str; lang: str; start: int; end: int

@dataclass
class Signals:
    similarity: float          # 0-1
    matching: float            # 0-1
    ai_prob: float             # 0-1
    evidence: dict             # nearest source, matching text, etc.

@dataclass
class Alert:
    segment: Segment
    score: float
    level: str                 # "bajo" | "incierto" | "alto"
    signals: Signals