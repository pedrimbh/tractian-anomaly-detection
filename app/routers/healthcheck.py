from fastapi import APIRouter
from app.schemas.health import HealthCheckResponse, Metrics
from app.core.metrics import metrics_collector
from app.infra.storage import count_trained_series, count_total_models

router = APIRouter(tags=["Health Check"])


@router.get("/healthcheck", response_model=HealthCheckResponse)
async def healthcheck() -> HealthCheckResponse:
    """Retorna o estado da aplicação: séries treinadas e latências de inferência/treino."""
    return HealthCheckResponse(
        series_trained=await count_trained_series(),
        total_models=await count_total_models(),
        inference_latency_ms=Metrics(**await metrics_collector.get_inference_metrics()),
        training_latency_ms=Metrics(**await metrics_collector.get_training_metrics()),
    )
