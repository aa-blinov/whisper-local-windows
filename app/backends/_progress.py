"""HuggingFace download progress hook (backend-agnostic).

Patches ``tqdm.tqdm`` so download bars produced inside ``huggingface_hub``
forward their ``(current_bytes, total_bytes, desc)`` triple to our UI
callback.  Used by every backend that downloads weights through HF (which
is all of them now that the app is ONNX-only).

Module-level state — both for the patch (tqdm class swap is process-wide,
no point per-instance) and for the callback (one progress bar in the UI
at a time, so a single global callback is plenty)::

    >>> set_progress_callback(lambda cur, tot, desc: ...)
    >>> install_tqdm_progress()  # idempotent

The custom ``tqdm`` subclass also has to survive the indignities of
running under ``pythonw.exe`` (``sys.stdout is None``) and the random
``OverflowError`` that ``tqdm`` raises when it tries to render
``ETA = ∞``.  Both are caught and silently ignored — we don't render
to a console at all, the UI gets its updates via ``_fire()``.
"""

from __future__ import annotations

from typing import Callable, Optional

_tqdm_patched = False
# Module-level callback called by the custom tqdm. Set via
# ``set_progress_callback``. Signature:
#   callback(current_bytes: int, total_bytes: int, desc: str) -> None
_progress_callback: Optional[Callable[[int, int, str], None]] = None

# Minimum gap between two callback invocations from the same tqdm
# instance, in seconds.  HuggingFace download tqdm fires ``update()``
# for every chunk it pulls off the socket — easily 200+ times per
# second for a multi-GB file.  Each of those becomes a Qt signal
# emitted into the main thread's event queue, and at that rate the
# main thread can't drain the queue fast enough — the smooth-scroll
# QTimer (running at 144 Hz to match the display) gets starved out,
# the wheel events queue up, and the user feels the scroll "stick"
# while a download is in flight.  200 ms (5 Hz) is still smooth
# visually for a percentage display and leaves three full smooth-
# scroll frames between every signal emit.
_FIRE_THROTTLE_S: float = 0.2


def set_progress_callback(
    callback: Optional[Callable[[int, int, str], None]],
) -> None:
    """Register the UI callback to receive HF download progress.

    Pass ``None`` to clear.  Safe to call multiple times — only the most
    recent callback is used.
    """
    global _progress_callback
    _progress_callback = callback


def install_tqdm_progress() -> None:
    """Patch ``tqdm`` / ``tqdm.auto`` so HF downloads report into the
    registered progress callback.  Idempotent.
    """
    global _tqdm_patched
    if _tqdm_patched:
        return
    try:
        import tqdm as _tqdm
        import tqdm.auto as _tqdm_auto
    except ImportError:
        return

    base_cls = _tqdm.tqdm

    import time as _time

    class _ProgressTqdm(base_cls):  # type: ignore[misc, valid-type]
        # Last-fire timestamp (monotonic) per instance — throttles
        # callback emission so the Qt main thread isn't drowned in
        # signals during a fast download.
        _last_fire_ts: float = 0.0

        def update(self, n=1):
            ret = super().update(n)
            # PyQt apps usually run without a TTY, so huggingface_hub's
            # tqdm gets ``disable=None`` which auto-resolves to ``True``.
            # When disabled, vanilla tqdm short-circuits ``update()`` and
            # leaves ``self.n`` at zero — meaning our callback would see
            # 0 bytes forever. Mirror the count ourselves in that case so
            # the progress callback reports accurate bytes.
            if getattr(self, "disable", False) and n:
                try:
                    self.n = (self.n or 0) + n
                except (TypeError, ValueError):
                    pass
            self._fire()
            return ret

        def display(self, *args, **kwargs):
            # Guard against any error in tqdm's console-rendering path.
            # Common culprits when running without a real terminal:
            #   AttributeError — sys.stdout is None (pythonw.exe)
            #   OverflowError  — int(float('inf')) in rate/ETA formatting
            # We don't need console output; progress goes via _fire().
            try:
                return super().display(*args, **kwargs)
            except Exception:
                pass

        def refresh(self, *args, **kwargs):
            try:
                ret = super().refresh(*args, **kwargs)
            except Exception:
                ret = None
            self._fire()
            return ret

        def close(self):
            # ``close()`` is the final state — bypass the throttle so
            # the UI gets a guaranteed final update at the real total
            # (otherwise a download that finishes in <100 ms after the
            # last throttled fire would be stuck at e.g. 87 % forever).
            self._fire(force=True)
            try:
                return super().close()
            except Exception:
                pass

        def _fire(self, force: bool = False):
            cb = _progress_callback
            if cb is None:
                return
            if not force:
                now = _time.monotonic()
                if now - self._last_fire_ts < _FIRE_THROTTLE_S:
                    return
                self._last_fire_ts = now
            # ``disable=True`` makes tqdm.__init__ return early before
            # ``self.desc`` is assigned — and PyQt apps run without a TTY,
            # which auto-disables every bar huggingface_hub creates. Use
            # getattr so we still report progress instead of swallowing
            # an AttributeError silently.
            total = int(getattr(self, "total", 0) or 0)
            # Hugging Face spawns one tqdm bar per file (config.json,
            # tokenizer.json, vocabulary.txt, model.bin, …). The small
            # ones complete in milliseconds, so the user sees the bar
            # bounce 0%→99%→0%→99%→0%→% as each file is touched. Skip
            # bars whose total weight is trivial — only the actual
            # weights file is worth surfacing in the UI.
            if 0 < total < 1_000_000:
                return
            try:
                cb(
                    int(getattr(self, "n", 0) or 0),
                    total,
                    str(getattr(self, "desc", "") or ""),
                )
            except Exception:
                pass

    _tqdm.tqdm = _ProgressTqdm
    _tqdm_auto.tqdm = _ProgressTqdm
    _tqdm_patched = True
