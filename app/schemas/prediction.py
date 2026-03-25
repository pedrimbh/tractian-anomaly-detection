from pydantic import BaseModel


class PredictData(BaseModel):
    """Payload de entrada para o endpoint de predição."""
    timestamp: str
    value: float


class PredictResponse(BaseModel):
    """Resposta do endpoint de predição com resultado da classificação."""
    anomaly: bool
    model_version: str
