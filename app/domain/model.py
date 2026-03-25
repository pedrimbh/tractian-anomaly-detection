import numpy as np
from pydantic import BaseModel
from typing import List


class AnomalyDetectionModel(BaseModel):
    """
    Detector estatístico de anomalias baseado na regra 3-sigma.

    Um ponto x é anômalo se:  x > mean + 3 * std

    Apenas o lado superior é verificado pois o case original
    define anomalia como valor excepcionalmente alto (ex: sensor industrial).
    """
    mean: float = 0.0
    std: float = 0.0

    def fit(self, values: List[float]) -> None:
        """Calcula e armazena a média e desvio padrão da série de treinamento."""
        arr = np.array(values, dtype=float)
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr))

    def predict(self, value: float) -> bool:
        """Retorna True se o valor exceder mean + 3 * std (regra 3-sigma)."""
        if self.std == 0:
            return False
        return value > self.mean + 3 * self.std
