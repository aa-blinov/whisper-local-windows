"""Single-instance guard.

On Windows: uses a named mutex via ``win32event.CreateMutex`` so the
OS itself enforces uniqueness across user sessions and bypasses any
filesystem state. On macOS / Linux: uses ``filelock.FileLock`` over
a cache-dir lockfile, with ``acquire(timeout=0)`` to fail fast when
another instance is already running.

Both branches return an opaque handle the caller must keep alive
for the lifetime of the process — the mutex object on Windows or
the ``FileLock`` instance elsewhere — and ``None`` if another
instance already owns the lock.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


_ERROR_ALREADY_EXISTS = 183


def _windows_acquire(app_name: str) -> Optional[object]:
    import win32api
    import win32event

    mutex_name = f"{app_name}_SingleInstance"
    try:
        handle = win32event.CreateMutex(None, True, mutex_name)
    except Exception as exc:
        logger.error("Single-instance mutex creation raised: %s", exc)
        return None

    if win32api.GetLastError() == _ERROR_ALREADY_EXISTS:
        logger.info("Another instance detected (mutex %s)", mutex_name)
        return None

    logger.info("Primary instance acquired mutex %s", mutex_name)
    return handle


def _posix_acquire(app_name: str) -> Optional[object]:
    from filelock import FileLock, Timeout
    from platformdirs import user_cache_dir

    cache_dir = Path(user_cache_dir("LazyToText", appauthor=False))
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / f"{app_name}.lock"

    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        logger.info("Another instance detected (lockfile %s)", lock_path)
        return None
    except OSError as exc:
        logger.error("Single-instance lock acquisition raised: %s", exc)
        return None

    logger.info("Primary instance acquired lockfile %s", lock_path)
    return lock


def try_acquire_single_instance(app_name: str = "LazyToTextLocal") -> Optional[object]:
    """Attempt to acquire the single-instance lock.

    Returns the mutex / FileLock handle on success — the caller must
    keep it alive for the lifetime of the process so the OS treats
    this as the running instance. Returns ``None`` if another
    instance already owns the lock (or any unexpected error
    occurred). The caller is then responsible for showing the
    appropriate UI feedback and exiting.
    """
    if sys.platform == "win32":
        return _windows_acquire(app_name)
    return _posix_acquire(app_name)
