from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class ModelTrainingError(ValueError):
    class Reason(str, Enum):
        INSUFFICIENT_DATA = "insufficient_data"
        CONSTANT_DATA = "constant_data"

    def __init__(self, reason: "ModelTrainingError.Reason", detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class AnomalyDetector(ABC):
    """Interface para detectores de anomalia em séries temporais univariadas."""

    @abstractmethod
    def fit(self, values: List[float]) -> None:
        """Treina o detector com os valores históricos."""

    @abstractmethod
    def predict(self, value: float) -> bool:
        """Retorna True se o valor for uma anomalia."""
