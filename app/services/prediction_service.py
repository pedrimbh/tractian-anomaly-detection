import asyncio
import time

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core import model_cache
from app.core.logging import get_logger
from app.core.metrics import metrics_collector
from app.infra import storage

log = get_logger("prediction")


@retry(
    retry=retry_if_exception_type(OSError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
)
async def predict(series_id: str, value: float, version: str | None) -> dict | None:
    """Orquestra a predição: cache, inferência, métricas e logging.

    Retorna dict com anomaly+model_version, ou None se a série não for encontrada.
    """
    t0 = time.perf_counter()

    model = await model_cache.get(series_id, version)
    if model is None:
        log.error("model_not_found", series_id=series_id, version=version)
        return None

    resolved_version = version or await storage.get_latest_version(series_id)
    is_anomaly = await asyncio.to_thread(model.predict, value)

    latency_ms = (time.perf_counter() - t0) * 1000
    await metrics_collector.record_inference(latency_ms)

    (log.warning if is_anomaly else log.info)(
        "anomaly_detected" if is_anomaly else "prediction_ok",
        series_id=series_id,
        value=value,
        model_version=resolved_version,
    )

    return {"anomaly": is_anomaly, "model_version": resolved_version}
