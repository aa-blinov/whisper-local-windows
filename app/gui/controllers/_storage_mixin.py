"""Storage card behaviour as a mixin.

Pulls the models-folder management out of the (already large)
``AppController`` into a focused unit that handles:

- showing the resolved path,
- recomputing the on-disk size on a worker thread,
- the Open-folder shortcut,
- the Change… picker with optional weight migration,
- Reset to default.

The mixin assumes the host class provides:

- ``self._config``      — config manager
- ``self._window``      — main window
- ``self._storage_size_ready``  — Qt signal emitting the rendered
  human-readable size string from a worker thread back onto the UI
  thread.

Public entry points (called from ``AppController.__init__`` /
``_wire_shortcuts``):

- ``_wire_storage()`` — connects the view's three signals and seeds
  the path + size on first paint.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QRunnable, QThreadPool
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

if TYPE_CHECKING:  # pragma: no cover — typing only
    from PySide6.QtCore import Signal


log = logging.getLogger(__name__)


def _ctrl_module():
    """Resolve the AppController module lazily.

    Storage utilities (``get_models_root``, ``cached_models_size``,
    ``move_cached_dir``) are re-exported from ``app_controller`` so
    existing tests can ``monkeypatch.setattr(controller_module, X, …)``
    without knowing about this mixin file.  We look them up through
    that module on every call so those patches actually land.
    """
    from app.gui.controllers import app_controller as _ctrl
    return _ctrl


def _apply_env_for_models_root(configured: str) -> str:
    """Mirror the user's chosen models root into the live process
    environment so the next ``onnx_asr.load_model(...)`` call's
    ``huggingface_hub`` download picks it up without a restart.

    Mirrors the startup logic in ``app.gui.app._apply_storage_path``
    — kept in lockstep so 'change live' and 'apply on next launch'
    end up at the same env state.

    Returns the resolved root for logging.
    """
    root = _ctrl_module().get_models_root(configured)
    os.environ["HF_HOME"] = root
    return root


def _human_size(num_bytes: int) -> str:
    """Compact human size for status / dialog text. KB/MB/GB to one
    decimal — close enough for 'will this fit?' reasoning."""
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


class StorageMixin:
    """Storage-card slice of ``AppController``."""

    # ---- wiring -------------------------------------------------------------

    def _wire_storage(self) -> None:
        view = self._window.shortcuts_view
        view.storage_path_change_requested.connect(self._on_storage_path_change)
        view.storage_reset_requested.connect(self._on_storage_reset)
        view.storage_open_requested.connect(self._on_storage_open)
        self._refresh_storage_path()
        self._refresh_storage_size()

    # ---- read-only refreshers ----------------------------------------------

    def _refresh_storage_path(self) -> None:
        """Push the resolved storage path into the Settings card. The
        view shows ``(default)`` after the path when nothing's been
        overridden — same source-of-truth (``get_models_root``) the
        rest of the app uses at startup."""
        configured = self._config.get_setting("storage", "models_dir")
        is_default = not (configured and str(configured).strip())
        resolved = _ctrl_module().get_models_root(configured)
        self._window.shortcuts_view.set_storage_path(
            resolved, is_default=is_default,
        )

    def _refresh_storage_size(self) -> None:
        """Compute the on-disk size of the cache and push it into the
        Settings card.

        Walks ``<root>/hub`` recursively which can take 100-300 ms for
        a multi-GB cache; runs in a ``QThreadPool`` worker so the UI
        thread stays responsive.  The result lands back via a Qt
        signal that this method connects to ``set_storage_size``.
        """
        view = self._window.shortcuts_view
        view.set_storage_size("computing…")

        configured = self._config.get_setting("storage", "models_dir")
        resolved = _ctrl_module().get_models_root(configured)

        signal = self._storage_size_ready

        class _Worker(QRunnable):
            def run(self_inner) -> None:  # noqa: N805 (Qt-style)
                try:
                    nbytes = _ctrl_module().cached_models_size(resolved)
                except Exception:  # pragma: no cover — defensive
                    nbytes = 0
                try:
                    signal.emit(_human_size(nbytes))
                except RuntimeError:
                    # The controller was destroyed before the worker
                    # finished — common at app shutdown / test teardown.
                    # Drop the result silently; the next AppController
                    # will recompute on init.
                    pass

        QThreadPool.globalInstance().start(_Worker())

    def _on_storage_size_ready(self, text: str) -> None:
        self._window.shortcuts_view.set_storage_size(text)

    # ---- user actions ------------------------------------------------------

    def _on_storage_open(self) -> None:
        """Open the resolved models directory in the OS file manager.

        Windows uses ``os.startfile`` which honours the user's default
        Explorer; macOS uses ``open`` (Finder); Linux uses
        ``xdg-open`` (whatever the desktop environment registers as
        the file-manager handler). Creates the directory first if it
        doesn't exist (might happen on a brand-new install before any
        model has been downloaded) so the user doesn't get a 'path
        not found' error popup.
        """
        configured = self._config.get_setting("storage", "models_dir")
        resolved = _ctrl_module().get_models_root(configured)
        try:
            Path(resolved).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("Failed to create storage dir for open: %s", exc)
        try:
            if sys.platform == "win32":
                os.startfile(resolved)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", resolved])
            else:
                subprocess.Popen(["xdg-open", resolved])
        except Exception as exc:
            log.warning("Failed to open storage dir %s: %s", resolved, exc)
            QMessageBox.warning(
                self._window,
                "Open folder",
                f"Could not open the folder:\n{resolved}\n\n{exc}",
            )

    def _on_storage_path_change(self) -> None:
        """User clicked Change….

        Flow:
          1. Open the folder picker; bail on Cancel.
          2. If the pick equals the current root → no-op.
          3. Sum cached weights at the old root. If non-zero, ask
             Yes/No/Cancel about migrating them. Cancel here aborts
             the whole change so the user can re-pick without leaving
             config in a half-applied state.
          4. On Yes — block UI with a wait cursor and call
             ``move_cached_dir`` for the ``hub/`` subtree (single-
             engine app — only one subdirectory exists today).
          5. Write the new path to config and pop a single info
             dialog summarising what moved + the live-vs-restart
             nuance.
        """
        configured = self._config.get_setting("storage", "models_dir") or ""
        old_root = _ctrl_module().get_models_root(configured)
        start_dir = configured or str(Path(old_root).parent)
        chosen = QFileDialog.getExistingDirectory(
            self._window,
            "Choose models folder",
            start_dir,
        )
        if not chosen:
            return  # Cancelled at the folder picker.

        try:
            same = Path(chosen).resolve() == Path(old_root).resolve()
        except OSError:
            same = chosen == old_root
        if same:
            return  # Picked the same folder — nothing to do.

        old_size = _ctrl_module().cached_models_size(old_root)
        move_outcomes: list[tuple[str, dict]] = []

        if old_size > 0:
            answer = QMessageBox.question(
                self._window,
                "Move existing weights?",
                (
                    f"You have {_human_size(old_size)} of cached models at:\n"
                    f"{old_root}\n\n"
                    f"Move them to the new location?\n{chosen}\n\n"
                    "Yes — relocate now (intra-drive is instant; "
                    "across drives can take several minutes for large "
                    "caches).\n"
                    "No  — leave them in place; new downloads go to "
                    "the new folder.\n"
                    "Cancel — go back without changing anything."
                ),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer == QMessageBox.Cancel:
                return  # Bail without writing config.
            if answer == QMessageBox.Yes:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    # Single-engine app — only the HF hub subtree
                    # exists.  The legacy ``gigaam`` subtree (used by
                    # the old gigaam-Python backend) was retired with
                    # the ONNX-only refactor.
                    result = _ctrl_module().move_cached_dir(
                        str(Path(old_root) / "hub"),
                        str(Path(chosen) / "hub"),
                    )
                    move_outcomes.append(("hub", result))
                finally:
                    QApplication.restoreOverrideCursor()

        self._config.update_user_setting("storage", "models_dir", chosen)
        # Mirror into the live process env — both backends read these
        # at every load, so the change takes effect on the very next
        # ``Download`` click without a restart.
        _apply_env_for_models_root(chosen)
        self._refresh_storage_path()
        self._refresh_storage_size()
        # Refresh every model card's cache state so the Download ↔ Select
        # button reflects the new directory immediately — without this the
        # cards keep showing "Download" even when the chosen folder already
        # contains the model weights.
        self._window.models_view.refresh_cache_state()

        # Build a user-friendly summary so they know what landed
        # where and what didn't.
        summary_lines = [f"Models folder set to:\n{chosen}\n"]
        if move_outcomes:
            for name, result in move_outcomes:
                if result.get("moved"):
                    summary_lines.append(
                        f"  • {name}: moved {_human_size(int(result['bytes']))}"
                    )
                else:
                    reason = result.get("reason", "no source")
                    if "missing" in reason or "same" in reason:
                        # Don't bother surfacing 'gigaam: source missing'
                        # — that's the normal case for Whisper-only
                        # users and would clutter the dialog.
                        continue
                    summary_lines.append(f"  • {name}: skipped ({reason})")
            summary_lines.append("")
        elif old_size > 0:
            summary_lines.append(
                f"Existing {_human_size(old_size)} of weights left at:\n"
                f"{old_root}\n"
            )
        summary_lines.append(
            "New downloads will land at the new location immediately."
        )

        QMessageBox.information(
            self._window,
            "Models folder updated",
            "\n".join(summary_lines),
        )

    def _on_storage_reset(self) -> None:
        """Reset Storage to default and mirror that into the live
        env so subsequent loads/downloads use the default path."""
        self._config.update_user_setting("storage", "models_dir", "")
        resolved = _apply_env_for_models_root("")
        self._refresh_storage_path()
        self._refresh_storage_size()
        self._window.models_view.refresh_cache_state()
        QMessageBox.information(
            self._window,
            "Models folder reset",
            (
                f"Models folder set to default:\n{resolved}\n\n"
                "New downloads will land there immediately."
            ),
        )
