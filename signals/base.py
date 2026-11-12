from abc import ABC, abstractmethod
from typing import ClassVar

from src.schemas import Segment, Signals


class BaseSignal(ABC):
    """Common interface for Veritext signal detectors."""

    name: ClassVar[str]

    @abstractmethod
    def analyze(self, segment: Segment) -> Signals:
        """Analyze a segment and return its signals along with the evidence."""
        raise NotImplementedError