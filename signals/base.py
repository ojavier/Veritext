from abc import ABC, abstractmethod
from typing import ClassVar

from src.schemas import Segment, Signals


class BaseSignal(ABC):
    """Interfaz común para los detectores de señales de Veritext."""

    name: ClassVar[str]

    @abstractmethod
    def analyze(self, segment: Segment) -> Signals:
        """Analiza un segmento y devuelve sus señales junto con la evidencia."""
        raise NotImplementedError