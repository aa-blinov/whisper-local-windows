"""File-transcription view — drag-and-drop or browse for audio files,
get text back.

Single-file workflow for v1: pick one audio file, get a transcript,
copy or save it.  Diarization, batch processing and per-segment
timestamps are out of scope until the underlying backend grows
support for them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# Audio / video extensions our two-tier decoder can handle.
# ``soundfile`` covers the lossless / OGG family; ``imageio-ffmpeg``
# (the bundled static ffmpeg) covers everything else.  We accept
# common video container extensions too — the audio track is what
# the model cares about, not the codec.
_AUDIO_EXTS: tuple[str, ...] = (
    # libsndfile native
    ".wav", ".flac", ".ogg", ".oga", ".opus", ".aiff", ".aif",
    # ffmpeg fallback — audio
    ".mp3", ".m4a", ".aac", ".wma", ".amr", ".ac3", ".alac",
    # ffmpeg fallback — video containers (we extract the audio track)
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".flv", ".3gp", ".m4v",
    ".wmv", ".ts",
)


def _looks_like_audio(path: str) -> bool:
    return path.lower().endswith(_AUDIO_EXTS)


class TranscribeView(QWidget):
    """Drag-and-drop / browse zone wired to a single transcription
    pipeline.

    Emits one signal up to the controller:

    - ``file_dropped(path)`` — user picked a file (browse button or
      drag-drop).  The controller is expected to dispatch this to a
      worker thread and call back via ``set_result`` / ``set_error``.

    Convenience signals for telemetry / Logs view:

    - ``copy_requested()`` — user clicked Copy.
    - ``save_requested(path)`` — user clicked Save and confirmed a path.
    """

    file_dropped = Signal(str)
    copy_requested = Signal()
    save_requested = Signal(str)

    # ---- States ----
    _STATE_IDLE = "idle"
    _STATE_BUSY = "busy"
    _STATE_DONE = "done"
    _STATE_ERROR = "error"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("TranscribeView")
        # Top-level layout: drop zone, action row, transcript box.
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # The drop zone is a QFrame (so the dashed border applies to
        # the whole region) with two labels stacked inside: the prompt
        # and a smaller line listing the supported formats.  Without
        # the format hint the user has to guess which extensions work.
        from PySide6.QtWidgets import QFrame

        self._drop_zone = QFrame(self)
        self._drop_zone.setObjectName("TranscribeDropZone")
        self._drop_zone.setMinimumHeight(120)
        self._drop_zone.setProperty("dropState", "idle")
        drop_layout = QVBoxLayout(self._drop_zone)
        drop_layout.setContentsMargins(24, 18, 24, 18)
        drop_layout.setSpacing(8)
        drop_layout.addStretch(1)
        prompt = QLabel(
            "Drop an audio or video file here — or click Browse",
            self._drop_zone,
        )
        prompt.setObjectName("TranscribeDropPrompt")
        prompt.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(prompt)
        formats_hint = QLabel(
            "Supports: WAV, MP3, FLAC, OGG, OPUS, M4A, AAC, WMA, AIFF · "
            "MP4, MOV, MKV, WebM, AVI, FLV, 3GP",
            self._drop_zone,
        )
        formats_hint.setObjectName("TranscribeDropFormats")
        formats_hint.setAlignment(Qt.AlignCenter)
        formats_hint.setWordWrap(True)
        drop_layout.addWidget(formats_hint)
        drop_layout.addStretch(1)
        # AcceptDrops on the parent widget; the QFrame is just a
        # visual cue (it doesn't have its own dragEnterEvent).
        self.setAcceptDrops(True)

        self._status_label = QLabel("", self)
        self._status_label.setObjectName("TranscribeStatus")
        self._status_label.setVisible(False)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._browse_btn = QPushButton("Browse…", self)
        self._browse_btn.setObjectName("TranscribeBrowseButton")
        self._browse_btn.clicked.connect(self._on_browse_clicked)

        actions.addWidget(self._browse_btn)
        actions.addStretch(1)

        self._copy_btn = QPushButton("Copy", self)
        self._copy_btn.setObjectName("TranscribeCopyButton")
        self._copy_btn.setEnabled(False)
        self._copy_btn.clicked.connect(self._on_copy_clicked)

        self._save_btn = QPushButton("Save .txt", self)
        self._save_btn.setObjectName("TranscribeSaveButton")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_clicked)

        actions.addWidget(self._copy_btn)
        actions.addWidget(self._save_btn)

        self._transcript = QPlainTextEdit(self)
        self._transcript.setObjectName("TranscribeOutput")
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText(
            "The transcript will appear here…"
        )

        root.addWidget(self._drop_zone)
        root.addLayout(actions)
        root.addWidget(self._status_label)
        root.addWidget(self._transcript, 1)

        self._current_path: Optional[str] = None
        self._set_state(self._STATE_IDLE)

    # ---- public API ----------------------------------------------------------

    def current_file(self) -> Optional[str]:
        return self._current_path

    def transcript(self) -> str:
        return self._transcript.toPlainText()

    def set_busy(self, file_path: str) -> None:
        """Controller calls this when transcription starts."""
        self._current_path = file_path
        self._transcript.clear()
        self._copy_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._browse_btn.setEnabled(False)
        self._status_label.setVisible(True)
        self._status_label.setText(
            f"Transcribing {Path(file_path).name}…"
        )
        self._status_label.setProperty("status", "busy")
        self._set_state(self._STATE_BUSY)

    def set_result(self, text: str) -> None:
        """Controller calls this when transcription succeeds."""
        self._transcript.setPlainText(text or "")
        has_text = bool(text and text.strip())
        self._copy_btn.setEnabled(has_text)
        self._save_btn.setEnabled(has_text)
        self._browse_btn.setEnabled(True)
        self._status_label.setText(
            "Done."
            if has_text
            else "Empty transcript — the model didn't hear any speech."
        )
        self._status_label.setProperty(
            "status", "done" if has_text else "warning"
        )
        self._set_state(self._STATE_DONE)

    def set_error(self, message: str) -> None:
        """Controller calls this when transcription fails."""
        self._transcript.clear()
        self._copy_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._browse_btn.setEnabled(True)
        self._status_label.setVisible(True)
        self._status_label.setText(f"Error: {message}")
        self._status_label.setProperty("status", "error")
        self._set_state(self._STATE_ERROR)

    # ---- Qt drag-drop --------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._state == self._STATE_BUSY:
            event.ignore()
            return
        mime: QMimeData = event.mimeData()
        if not mime.hasUrls():
            event.ignore()
            return
        # Accept if at least one local URL is an audio file by extension.
        for url in mime.urls():
            if url.isLocalFile() and _looks_like_audio(url.toLocalFile()):
                event.acceptProposedAction()
                self._drop_zone.setProperty("dropState", "active")
                self._reapply_drop_style()
                return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._drop_zone.setProperty("dropState", "idle")
        self._reapply_drop_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._drop_zone.setProperty("dropState", "idle")
        self._reapply_drop_style()
        if self._state == self._STATE_BUSY:
            event.ignore()
            return
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if _looks_like_audio(path):
                event.acceptProposedAction()
                self.file_dropped.emit(path)
                return
        event.ignore()

    # ---- internal ------------------------------------------------------------

    def _reapply_drop_style(self) -> None:
        """Force a stylesheet recalculation for the drop zone after a
        property change.  Without this Qt won't notice the
        ``dropState`` flip during drag-enter/leave events."""
        self._drop_zone.style().unpolish(self._drop_zone)
        self._drop_zone.style().polish(self._drop_zone)
        self._drop_zone.update()

    def _on_browse_clicked(self) -> None:
        if self._state == self._STATE_BUSY:
            return
        # Build a filter string from the recognised extensions.
        glob = " ".join(f"*{ext}" for ext in _AUDIO_EXTS)
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose an audio or video file",
            "",
            f"Audio / video ({glob});;All files (*.*)",
        )
        if path:
            self.file_dropped.emit(path)

    def _on_copy_clicked(self) -> None:
        from PySide6.QtWidgets import QApplication

        text = self.transcript()
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.copy_requested.emit()
        self._status_label.setText("Copied to clipboard.")
        self._status_label.setProperty("status", "done")

    def _on_save_clicked(self) -> None:
        text = self.transcript()
        if not text:
            return
        suggested_dir = ""
        suggested_name = "transcript.txt"
        if self._current_path:
            stem = Path(self._current_path).stem
            suggested_name = f"{stem}.txt"
            suggested_dir = str(Path(self._current_path).parent)
        suggested = (
            os.path.join(suggested_dir, suggested_name)
            if suggested_dir
            else suggested_name
        )
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save transcript",
            suggested,
            "Text (*.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as exc:
            self._status_label.setText(f"Save failed: {exc}")
            self._status_label.setProperty("status", "error")
            return
        self.save_requested.emit(path)
        self._status_label.setText(f"Saved: {path}")
        self._status_label.setProperty("status", "done")

    def _set_state(self, state: str) -> None:
        self._state = state
