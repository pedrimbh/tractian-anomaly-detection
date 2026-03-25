from pydantic import BaseModel
from typing import Optional


class Metrics(BaseModel):
    """Estatísticas de latência (média e percentis) em milissegundos."""
    avg: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None


class HealthCheckResponse(BaseModel):
    """Resposta do endpoint de healthcheck com estado geral da aplicação."""
    series_trained: int
    total_models: int
    inference_latency_ms: Metrics
    training_latency_ms: Metrics
