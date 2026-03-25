import asyncio
import time
from fastapi import APIRouter
from app.schemas.training import TrainData, TrainResponse
from app.domain.model import AnomalyDetectionModel
from app.infra import storage
from app.core.metrics import metrics_collector
from app.core.logging import get_logger

router = APIRouter(tags=["Training"])
log = get_logger("training")


@router.post("/fit/{series_id}", response_model=TrainResponse)
async def fit(series_id: str, body: TrainData) -> TrainResponse:
    """Treina um novo modelo 3-sigma para a série informada e persiste em disco."""
    t0 = time.perf_counter()

    version = await storage.get_next_version(series_id)
    model = AnomalyDetectionModel()
    await asyncio.to_thread(model.fit, body.values)
    await storage.save_model(series_id, version, model)

    latency_ms = (time.perf_counter() - t0) * 1000
    await metrics_collector.record_training(latency_ms)

    log.info(
        "model_trained",
        series_id=series_id,
        version=version,
        points_used=len(body.values),
        duration_ms=round(latency_ms, 3),
    )

    return TrainResponse(
        series_id=series_id,
        version=version,
        points_used=len(body.values),
    )
