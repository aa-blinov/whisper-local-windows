"""History view + signal wiring as a mixin.

Owns the History tab: load entries on first paint, clear / export
buttons, copy-to-clipboard signal from row, and the
``history_updated`` signal from the recording pipeline that pops a
toast + prepends the newest entry to the table.

Expects the host class to provide:

- ``self._window``   — main window (uses ``self._window.history_view``
                        and ``self._window.toast``)
- ``self._history``  — history manager (or ``None``)

Public entry point: ``_wire_history()``.

The ``_on_history_updated_signal`` handler is connected from the
recording-mixin (where the signal source lives), so it stays public
to that mixin too.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from app.gui.widgets.dialogs import confirm, notify


log = logging.getLogger(__name__)


class HistoryMixin:
    """History-tab slice of ``AppController``."""

    def _wire_history(self) -> None:
        view = self._window.history_view
        if self._history is not None:
            view.set_entries(self._history.get_entries())
            view.clear_requested.connect(self._on_history_clear)
            view.export_requested.connect(self._on_history_export)
        view.copy_requested.connect(self._on_history_copy)

    def _on_history_clear(self) -> None:
        if self._history is None:
            return
        # Wipes the on-disk history file too — confirm before doing
        # anything irreversible.
        entries = self._history.get_entries()
        if not entries:
            return
        if not confirm(
            self._window,
            "Clear history?",
            f"Delete all {len(entries)} transcriptions? This cannot be undone.",
        ):
            return
        self._history.clear_history()
        self._window.history_view.set_entries(self._history.get_entries())

    def _on_history_export(self) -> None:
        if self._history is None:
            return
        if getattr(self, "_history_export_in_progress", False):
            return
        entries = self._history.get_entries()
        if not entries:
            notify(
                self._window,
                "Nothing to export",
                "Your history is empty — record a transcription first.",
            )
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self._window,
            "Export history",
            "transcription_history.txt",
            "Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return
        self._history_export_in_progress = True
        self._window.history_view.set_export_busy(True)

        def worker() -> None:
            try:
                ok = bool(self._history.export_to_text(path))
            except Exception as exc:
                log.warning("History export raised: %s", exc)
                payload = {"ok": False, "path": path, "count": len(entries)}
                try:
                    self._history_export_finished.emit(payload)
                except RuntimeError:
                    pass
                return
            payload = {"ok": ok, "path": path, "count": len(entries)}
            try:
                self._history_export_finished.emit(payload)
            except RuntimeError:
                pass

        threading.Thread(
            target=worker,
            daemon=True,
            name="export-history",
        ).start()

    def _on_history_export_finished(self, payload: dict) -> None:
        self._history_export_in_progress = False
        self._window.history_view.set_export_busy(False)
        ok = bool(payload.get("ok"))
        path = str(payload.get("path", ""))
        count = int(payload.get("count", 0))
        if ok:
            notify(
                self._window,
                "History exported",
                f"Saved {count} transcriptions to:\n{path}",
            )
            return
        notify(
            self._window,
            "Export failed",
            "Could not write the history file. Check the destination "
            "path and permissions.",
            kind="warning",
        )

    def _on_history_copy(self, text: str) -> None:
        QApplication.clipboard().setText(text)

    def _on_history_updated_signal(self) -> None:
        """Called from the recording-controller signal when a new
        transcription lands.  Prepend the newest entry without
        resetting the table model (so scroll + selection survive),
        then pop the confirmation toast."""
        if self._history is None:
            return
        entries = self._history.get_entries()
        if not entries:
            return
        max_entries = getattr(self._history, "max_entries", 0)
        self._window.history_view.prepend_entry(entries[0], max_entries)
        try:
            latest_text = getattr(entries[0], "text", "") or ""
        except Exception:  # pragma: no cover — defensive
            latest_text = ""
        if latest_text:
            try:
                self._window.toast.show_message(latest_text)
            except Exception:  # pragma: no cover — defensive
                pass
