import asyncio
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.schemas.prediction import PredictData, PredictResponse
from app.core import model_cache
from app.infra import storage
from app.core.metrics import metrics_collector
from app.core.logging import get_logger

router = APIRouter(tags=["Prediction"])
log = get_logger("prediction")


@router.post("/predict/{series_id}", response_model=PredictResponse)
async def predict(
    series_id: str,
    body: PredictData,
    version: Optional[str] = Query(default=None),
) -> PredictResponse:
    """Classifica um valor como anomalia ou normal usando o modelo treinado da série."""
    t0 = time.perf_counter()

    model = await model_cache.get(series_id, version)
    if model is None:
        log.error("model_not_found", series_id=series_id, version=version)
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for series_id='{series_id}'"
            + (f" version='{version}'" if version else ""),
        )

    resolved_version = version or await storage.get_latest_version(series_id)
    is_anomaly = await asyncio.to_thread(model.predict, body.value)

    latency_ms = (time.perf_counter() - t0) * 1000
    await metrics_collector.record_inference(latency_ms)

    if is_anomaly:
        log.warning(
            "anomaly_detected",
            series_id=series_id,
            value=body.value,
            model_version=resolved_version,
        )
    else:
        log.info(
            "prediction_ok",
            series_id=series_id,
            value=body.value,
            model_version=resolved_version,
        )

    return PredictResponse(anomaly=is_anomaly, model_version=resolved_version)
