import asyncio
import numpy as np
from typing import List, Dict, Optional


class MetricsCollector:
    """Async-safe in-memory latency collector."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._inference_latencies: List[float] = []
        self._training_latencies: List[float] = []

    async def record_inference(self, latency_ms: float) -> None:
        """Registra a latência de uma inferência em milissegundos."""
        async with self._lock:
            self._inference_latencies.append(latency_ms)

    async def record_training(self, latency_ms: float) -> None:
        """Registra a latência de um treino em milissegundos."""
        async with self._lock:
            self._training_latencies.append(latency_ms)

    async def get_inference_metrics(self) -> Dict[str, Optional[float]]:
        """Retorna média e percentis das latências de inferência registradas."""
        async with self._lock:
            data = list(self._inference_latencies)
        return await asyncio.to_thread(self._compute, data)

    async def get_training_metrics(self) -> Dict[str, Optional[float]]:
        """Retorna média e percentis das latências de treino registradas."""
        async with self._lock:
            data = list(self._training_latencies)
        return await asyncio.to_thread(self._compute, data)

    @staticmethod
    def _compute(data: List[float]) -> Dict[str, Optional[float]]:
        """Calcula média e percentis de uma lista de floats. Retorna None se vazia."""
        if not data:
            return {"avg": None, "p50": None, "p95": None, "p99": None}
        arr = np.array(data)
        return {
            "avg": round(float(np.mean(arr)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
        }


# Singleton compartilhado pela aplicação
metrics_collector = MetricsCollector()
