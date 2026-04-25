"""Single-instance guard via a Windows named mutex."""

from __future__ import annotations

import logging
from typing import Optional

import win32api
import win32event


logger = logging.getLogger(__name__)


_ERROR_ALREADY_EXISTS = 183


def try_acquire_single_instance(app_name: str = "LazyToTextLocal") -> Optional[object]:
    """Attempt to acquire the named single-instance mutex.

    Returns the mutex handle on success — the caller must keep it alive for
    the lifetime of the process so the OS treats this as the running
    instance. Returns ``None`` if another instance already owns the mutex
    (or any unexpected error occurred). The caller is then responsible for
    showing the appropriate UI feedback and exiting.
    """
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
