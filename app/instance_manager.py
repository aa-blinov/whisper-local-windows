import logging
import sys
import time
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


def guard_against_multiple_instances(app_name: str = "LazyToTextLocal"):
    """Legacy entry point: detects duplicate instances and exits the
    process after a short countdown. Kept for the old tk UI; new code
    should use ``try_acquire_single_instance`` instead.
    """
    handle = try_acquire_single_instance(app_name)
    if handle is None:
        _exit_to_prevent_duplicate()
    return handle


def _exit_to_prevent_duplicate():
    logger.info("Lazy to text is already running!", extra={'user_message': True})
    logger.info("This app will close in 3 seconds...", extra={'user_message': True})

    for i in range(3, 0, -1):
        time.sleep(1)

    logger.info("Goodbye!", extra={'user_message': True})
    sys.exit(0)