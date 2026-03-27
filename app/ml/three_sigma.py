import numpy as np
from typing import List

from app.core.config import settings
from app.ml.base import AnomalyDetector, ModelTrainingError


class ThreeSigmaDetector(AnomalyDetector):
    """
    Detector de anomalias baseado na regra 3-sigma.

    Um ponto x é anômalo se: x > mean + 3 * std
    Apenas o lado superior é verificado — anomalia = valor excepcionalmente alto.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.0, training_values: List[float] | None = None) -> None:
        self.mean = mean
        self.std = std
        self.training_values: List[float] = training_values or []

    def fit(self, values: List[float]) -> None:
        if len(values) < settings.min_training_points:
            raise ModelTrainingError(
                reason=ModelTrainingError.Reason.INSUFFICIENT_DATA,
                detail=f"At least {settings.min_training_points} data points are required for training.",
            )
        if len(set(values)) == 1:
            raise ModelTrainingError(
                reason=ModelTrainingError.Reason.CONSTANT_DATA,
                detail="Training data is constant — model cannot learn a meaningful distribution.",
            )
        arr = np.array(values, dtype=float)
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr))
        self.training_values = list(values)

    def predict(self, value: float) -> bool:
        if self.std == 0:
            return False
        return value > self.mean + 3 * self.std
