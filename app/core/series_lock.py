import os
import tempfile

from filelock import FileLock

_locks_dir = os.path.join(tempfile.gettempdir(), "tractian_locks")
os.makedirs(_locks_dir, exist_ok=True)


def get(series_id: str) -> FileLock:
    """Retorna um FileLock exclusivo para a série.

    FileLock usa um arquivo em disco — funciona entre múltiplos processos/workers,
    evitando race conditions no versionamento mesmo com uvicorn --workers N.
    """
    lock_path = os.path.join(_locks_dir, f"{series_id}.lock")
    return FileLock(lock_path, timeout=10)
