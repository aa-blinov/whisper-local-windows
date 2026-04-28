"""File-transcribe view behaviour as a mixin.

Pulls the Transcribe-tab dispatch out of ``AppController``: file
picks go to ``RecordingController.transcribe_file_async`` on a
worker thread, results land back via Qt signals which this mixin
filters by current-file path (so a stale result for the previous
drop doesn't overwrite a fresh one).

Expects the host class to provide:

- ``self._window`` — main window (uses ``self._window.transcribe_view``)
- ``self._recording`` — recording controller (or ``None``)

Public entry point: ``_wire_transcribe()``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt


class TranscribeMixin:
    """Transcribe-tab slice of ``AppController``."""

    def _wire_transcribe(self) -> None:
        """Hook the file-transcribe view into the recording controller.

        File picks dispatch through
        ``RecordingController.transcribe_file_async`` which runs
        ``backend.transcribe_file`` on a worker and emits a result
        signal.  The view stays responsive — busy state is shown
        until the result lands.
        """
        view = self._window.transcribe_view
        view.file_dropped.connect(self._on_transcribe_file_picked)

    def _on_transcribe_file_picked(self, path: str) -> None:
        view = self._window.transcribe_view
        view.set_busy(path)
        if self._recording is None:
            view.set_error(
                "Recording stack isn't initialised — try restarting the app."
            )
            return
        target = getattr(self._recording, "transcribe_file_async", None)
        if target is None:
            view.set_error(
                "This build doesn't support file transcription."
            )
            return
        # Connect once, lazily — multiple connects from repeated picks
        # are guarded by Qt.UniqueConnection.
        try:
            self._recording.file_transcribed.connect(
                self._on_transcribe_done, Qt.UniqueConnection,
            )
        except (TypeError, RuntimeError):
            pass
        try:
            self._recording.file_transcription_failed.connect(
                self._on_transcribe_failed, Qt.UniqueConnection,
            )
        except (TypeError, RuntimeError):
            pass
        target(path)

    def _on_transcribe_done(self, path: str, text: str) -> None:
        view = self._window.transcribe_view
        # Ignore stale results: if the user picked a second file the
        # view's current_file() is the latter; only render the latest.
        if view.current_file() != path:
            return
        view.set_result(text)

    def _on_transcribe_failed(self, path: str, message: str) -> None:
        view = self._window.transcribe_view
        if view.current_file() != path:
            return
        view.set_error(message)
