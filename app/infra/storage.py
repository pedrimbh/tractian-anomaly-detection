import asyncio
import json
import os
from typing import Optional

import aiofiles

from app.domain.model import AnomalyDetectionModel
from app.core.config import settings


# ──────────────────────────────────────────
# Paths
# ──────────────────────────────────────────

async def _series_dir(series_id: str) -> str:
    """Retorna o caminho do diretório da série, criando-o se não existir."""
    path = os.path.join(settings.models_dir, series_id)
    await asyncio.to_thread(os.makedirs, path, exist_ok=True)
    return path


async def _meta_path(series_id: str) -> str:
    """Retorna o caminho do arquivo de metadados (versão atual) da série."""
    return os.path.join(await _series_dir(series_id), "meta.json")


def _model_path(series_id: str, version: str) -> str:
    """Retorna o caminho do arquivo JSON de um modelo específico."""
    return os.path.join(settings.models_dir, series_id, f"{version}.json")


# ──────────────────────────────────────────
# Model persistence — I/O puro, sem cache
# ──────────────────────────────────────────

async def get_next_version(series_id: str) -> str:
    """Lê o meta.json e retorna a próxima versão disponível (ex: v1, v2...)."""
    meta = await _meta_path(series_id)
    if not await asyncio.to_thread(os.path.exists, meta):
        return "v1"
    async with aiofiles.open(meta) as f:
        data: dict[str, int] = json.loads(await f.read())
    return f"v{data['latest_version'] + 1}"


async def save_model(series_id: str, version: str, model: AnomalyDetectionModel) -> None:
    """Persiste o modelo em disco e invalida o latest anterior no cache."""
    from app.core import model_cache

    previous_latest = await get_latest_version(series_id)
    if previous_latest:
        await model_cache.invalidate(series_id, previous_latest)

    async with aiofiles.open(_model_path(series_id, version), "w") as f:
        await f.write(json.dumps({"mean": model.mean, "std": model.std}))

    meta = await _meta_path(series_id)
    version_number = int(version[1:])
    async with aiofiles.open(meta, "w") as f:
        await f.write(json.dumps({"latest_version": version_number, "latest": version}))


async def load_from_disk(series_id: str, version: str) -> Optional[AnomalyDetectionModel]:
    """Carrega modelo diretamente do disco. Chamado pelo model_cache em caso de miss."""
    path = _model_path(series_id, version)
    if not await asyncio.to_thread(os.path.exists, path):
        return None
    async with aiofiles.open(path) as f:
        data: dict[str, float] = json.loads(await f.read())
    return AnomalyDetectionModel(mean=data["mean"], std=data["std"])


async def get_latest_version(series_id: str) -> Optional[str]:
    """Sempre lê do disco — meta.json é a fonte de verdade após retreino."""
    meta = await _meta_path(series_id)
    if not await asyncio.to_thread(os.path.exists, meta):
        return None
    async with aiofiles.open(meta) as f:
        return json.loads(await f.read())["latest"]


async def count_trained_series() -> int:
    """Conta quantas séries possuem modelos treinados em disco."""
    if not await asyncio.to_thread(os.path.exists, settings.models_dir):
        return 0
    entries = await asyncio.to_thread(os.listdir, settings.models_dir)
    checks = await asyncio.gather(*[
        asyncio.to_thread(os.path.isdir, os.path.join(settings.models_dir, d))
        for d in entries
    ])
    return len([d for d, is_dir in zip(entries, checks) if is_dir])


async def count_total_models() -> int:
    """Conta o total de versões de modelos treinados em disco (todos os series_ids)."""
    if not await asyncio.to_thread(os.path.exists, settings.models_dir):
        return 0
    entries = await asyncio.to_thread(os.listdir, settings.models_dir)
    total = 0
    for series_dir in entries:
        series_path = os.path.join(settings.models_dir, series_dir)
        if await asyncio.to_thread(os.path.isdir, series_path):
            files = await asyncio.to_thread(os.listdir, series_path)
            total += len([f for f in files if f.startswith("v") and f.endswith(".json")])
    return total
