"""Background resource sampler for the topbar's live stats widget.

Polls CPU / RAM via ``psutil`` and GPU VRAM / utilisation via
``nvidia-ml-py`` on a Qt timer (so emissions land on the main
thread without an extra worker thread). Designed to be cheap —
``psutil.cpu_percent(interval=None)`` is non-blocking and
``nvmlDeviceGetMemoryInfo`` returns immediately from the driver.

The widget accepts a partial dict — keys are present only when the
corresponding subsystem is available (``gpu_*`` keys disappear on
machines with no NVIDIA driver).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QTimer, Signal


log = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MS = 2000


class ResourceMonitor(QObject):
    metrics_updated = Signal(dict)

    def __init__(
        self,
        interval_ms: int = _DEFAULT_INTERVAL_MS,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(max(500, int(interval_ms)))
        self._timer.timeout.connect(self._poll)

        self._gpu_handle = None
        self._nvml_initialized = False

    # ---- public API ---------------------------------------------------------

    def start(self) -> None:
        self._init_nvml()
        # Fire one sample immediately so the UI doesn't show empty
        # values for the first ``interval_ms`` after boot.
        self._poll()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._teardown_nvml()

    def sample(self) -> Dict[str, Any]:
        """Public hook used by tests to read current values without
        spinning a Qt event loop."""
        return self._collect()

    # ---- internal -----------------------------------------------------------

    def _poll(self) -> None:
        try:
            metrics = self._collect()
        except Exception as exc:  # pragma: no cover — defensive
            log.debug("resource sample failed: %s", exc)
            return
        self.metrics_updated.emit(metrics)

    def _collect(self) -> Dict[str, Any]:
        try:
            import psutil  # type: ignore[import]
        except ImportError:
            return {}

        ram = psutil.virtual_memory()
        metrics: Dict[str, Any] = {
            # ``interval=None`` returns the average since the
            # previous call — first call returns 0.0, subsequent
            # calls give meaningful per-tick deltas.
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "ram_used_mb": float(ram.used) / (1024 * 1024),
            "ram_total_mb": float(ram.total) / (1024 * 1024),
            "ram_percent": float(ram.percent),
        }

        if self._gpu_handle is not None:
            try:
                from pynvml import (  # type: ignore[import]
                    nvmlDeviceGetMemoryInfo,
                    nvmlDeviceGetUtilizationRates,
                )
                mem = nvmlDeviceGetMemoryInfo(self._gpu_handle)
                util = nvmlDeviceGetUtilizationRates(self._gpu_handle)
                metrics["gpu_vram_used_mb"] = float(mem.used) / (1024 * 1024)
                metrics["gpu_vram_total_mb"] = float(mem.total) / (1024 * 1024)
                metrics["gpu_util_percent"] = float(util.gpu)
            except Exception as exc:  # pragma: no cover — defensive
                log.debug("nvml sample failed, dropping GPU keys: %s", exc)

        return metrics

    def _init_nvml(self) -> None:
        if self._nvml_initialized:
            return
        try:
            from pynvml import (  # type: ignore[import]
                nvmlDeviceGetCount,
                nvmlDeviceGetHandleByIndex,
                nvmlInit,
            )
            nvmlInit()
            self._nvml_initialized = True
            if nvmlDeviceGetCount() > 0:
                self._gpu_handle = nvmlDeviceGetHandleByIndex(0)
                log.info("Resource monitor: NVML initialised, watching GPU 0")
            else:
                log.info("Resource monitor: NVML up but no GPUs reported")
        except Exception as exc:
            log.info(
                "Resource monitor: NVML unavailable, GPU metrics disabled: %s",
                exc,
            )

    def _teardown_nvml(self) -> None:
        if not self._nvml_initialized:
            return
        try:
            from pynvml import nvmlShutdown  # type: ignore[import]
            nvmlShutdown()
        except Exception:  # pragma: no cover — defensive
            pass
        self._nvml_initialized = False
        self._gpu_handle = None
