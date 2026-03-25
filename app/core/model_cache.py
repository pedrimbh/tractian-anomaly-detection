import asyncio
from typing import Optional, Dict, Tuple
from app.domain.model import AnomalyDetectionModel

# ──────────────────────────────────────────
# In-memory model cache
#
# Chave: (series_id, version) — ex: ("sensor_01", "v2")
#
# Regras:
#   - Versões antigas são imutáveis → ficam em cache para sempre
#   - Após retreino, apenas a entrada do latest anterior é invalidada
#   - get_latest_version() sempre vai ao disco (meta.json é fonte de verdade)
#   - Routers chamam model_cache, nunca storage diretamente para modelos
# ──────────────────────────────────────────

_cache: Dict[Tuple[str, str], AnomalyDetectionModel] = {}
_lock: asyncio.Lock = asyncio.Lock()


async def get(series_id: str, version: Optional[str] = None) -> Optional[AnomalyDetectionModel]:
    """
    Retorna o modelo para (series_id, version).
    Se version=None, resolve o latest via disco antes de consultar o cache.
    """
    from app.infra import storage

    resolved = version or await storage.get_latest_version(series_id)
    if resolved is None:
        return None

    async with _lock:
        cached = _cache.get((series_id, resolved))

    if cached is not None:
        return cached

    # Cache miss — carrega do disco
    model = await storage.load_from_disk(series_id, resolved)
    if model is None:
        return None

    async with _lock:
        _cache[(series_id, resolved)] = model

    return model


async def invalidate(series_id: str, version: str) -> None:
    """Remove seletivamente uma entrada do cache. Chamado após retreino."""
    async with _lock:
        _cache.pop((series_id, version), None)


async def size() -> int:
    """Retorna o número de modelos atualmente em cache."""
    async with _lock:
        return len(_cache)
