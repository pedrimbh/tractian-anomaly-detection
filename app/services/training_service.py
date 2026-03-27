import asyncio
import json
import os
import time
from pathlib import Path

from app.core import model_cache, series_lock
from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import metrics_collector
from app.ml.three_sigma import ThreeSigmaDetector

log = get_logger("training")


async def fit(series_id: str, values: list[float]) -> dict:
    """Orquestra o treinamento: versão, fit, persistência, métricas e logging.

    O bloco crítico (versão + treino + escrita) roda inteiramente em uma única
    thread com FileLock, garantindo serialização cross-process sem conflito de
    thread affinity (requisito do FileLock no Windows).
    """
    t0 = time.perf_counter()

    def _fit_locked() -> dict:
        """Seção crítica: lock → versão → fit → escrita atômica. Tudo síncrono."""
        lock = series_lock.get(series_id)
        with lock:
            series_dir = Path(settings.models_dir) / series_id
            os.makedirs(series_dir, exist_ok=True)
            meta = series_dir / "meta.json"

            # Determina próxima versão e guarda a anterior para invalidar o cache
            if not meta.exists():
                version, previous_latest = "v1", None
            else:
                data = json.loads(meta.read_text())
                previous_latest = data["latest"]
                version = f"v{data['latest_version'] + 1}"

            # Treina — ModelTrainingError sobe normalmente via asyncio.to_thread
            model = ThreeSigmaDetector()
            model.fit(values)

            # Escrita atômica do modelo
            model_path = series_dir / f"{version}.json"
            tmp = model_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"mean": model.mean, "std": model.std, "training_values": model.training_values}))
            os.replace(tmp, model_path)

            # Escrita atômica do meta.json
            version_num = int(version[1:])
            tmp_meta = meta.with_suffix(".tmp")
            tmp_meta.write_text(json.dumps({"latest_version": version_num, "latest": version}))
            os.replace(tmp_meta, meta)

            return {"version": version, "points_used": len(values), "previous_latest": previous_latest}

    result = await asyncio.to_thread(_fit_locked)

    # Invalida o latest anterior no cache (fora do lock — não precisa de exclusão)
    if result["previous_latest"]:
        await model_cache.invalidate(series_id, result["previous_latest"])

    latency_ms = (time.perf_counter() - t0) * 1000
    await metrics_collector.record_training(latency_ms)

    log.info(
        "model_trained",
        series_id=series_id,
        version=result["version"],
        points_used=result["points_used"],
        duration_ms=round(latency_ms, 3),
    )

    return {"version": result["version"], "points_used": result["points_used"]}
