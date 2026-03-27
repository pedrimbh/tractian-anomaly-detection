import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import aiofiles
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.ml.base import AnomalyDetector
from app.ml.three_sigma import ThreeSigmaDetector
from app.core.config import settings


# ──────────────────────────────────────────
# Paths
# ──────────────────────────────────────────

async def _series_dir(series_id: str) -> str:
    """Retorna o caminho do diretório da série, criando-o se não existir."""
    path = os.path.join(settings.models_dir, series_id)
    await asyncio.to_thread(os.makedirs, path, exist_ok=True)
    return path


async def _meta_path(series_id: str) -> Path:
    """Retorna o caminho do arquivo de metadados (versão atual) da série."""
    return Path(await _series_dir(series_id)) / "meta.json"


def _model_path(series_id: str, version: str) -> Path:
    """Retorna o caminho do arquivo JSON de um modelo específico."""
    return Path(settings.models_dir) / series_id / f"{version}.json"


# ──────────────────────────────────────────
# Escrita atômica — helper interno
# ──────────────────────────────────────────

async def _write_atomic(path: Path, content: str) -> None:
    """Escreve content em arquivo temporário e renomeia atomicamente para path.

    os.replace é atômico no SO — nunca deixa estado parcial em disco.
    """
    tmp = path.with_suffix(".tmp")
    async with aiofiles.open(tmp, "w") as f:
        await f.write(content)
    await asyncio.to_thread(os.replace, tmp, path)


# ──────────────────────────────────────────
# Model persistence — I/O puro, sem cache
# ──────────────────────────────────────────

async def get_next_version(series_id: str) -> str:
    """Lê o meta.json e retorna a próxima versão disponível (ex: v1, v2...)."""
    meta = await _meta_path(series_id)
    if not await asyncio.to_thread(meta.exists):
        return "v1"
    async with aiofiles.open(meta) as f:
        data: dict[str, int] = json.loads(await f.read())
    return f"v{data['latest_version'] + 1}"


async def save_model(series_id: str, version: str, model: AnomalyDetector) -> None:
    """Persiste o modelo em disco com escrita atômica e invalida o latest anterior no cache."""
    from app.core import model_cache

    previous_latest = await get_latest_version(series_id)
    if previous_latest:
        await model_cache.invalidate(series_id, previous_latest)

    await _write_atomic(
        _model_path(series_id, version),
        json.dumps({"mean": model.mean, "std": model.std, "training_values": model.training_values}),
    )

    version_number = int(version[1:])
    await _write_atomic(
        await _meta_path(series_id),
        json.dumps({"latest_version": version_number, "latest": version}),
    )


@retry(
    retry=retry_if_exception_type(OSError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
)
async def load_from_disk(series_id: str, version: str) -> Optional[AnomalyDetector]:
    """Carrega modelo diretamente do disco. Retentar até 3x em erros de I/O transientes."""
    path = _model_path(series_id, version)
    if not await asyncio.to_thread(path.exists):
        return None
    async with aiofiles.open(path) as f:
        data: dict[str, float] = json.loads(await f.read())
    return ThreeSigmaDetector(
        mean=data["mean"],
        std=data["std"],
        training_values=data.get("training_values", []),
    )


@retry(
    retry=retry_if_exception_type(OSError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
)
async def get_latest_version(series_id: str) -> Optional[str]:
    """Sempre lê do disco — meta.json é a fonte de verdade após retreino."""
    meta = await _meta_path(series_id)
    if not await asyncio.to_thread(meta.exists):
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
