"""Common interface for speech-to-text backends.

Every concrete backend (faster-whisper, GigaAM, future cloud APIs) implements
this Protocol so the rest of the app — StateManager, RecordingFactory, the
status poller — can swap between them with no special-casing.

The contract is intentionally narrow: load the model in the background, tell
callers what state we're in, and turn audio into text. Backends are expected
to be thread-safe — ``transcribe`` is invoked from the recording pipeline
thread and ``status`` from the UI poller thread.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


# Possible values returned by ``status()``. ``loading`` covers both initial
# load and a model swap; the recording pipeline treats anything other than
# ``ready`` as "do not transcribe right now".
BACKEND_STATES = ("stopped", "loading", "ready", "error")


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Speech-to-text backend interface."""

    def status(self) -> str:
        """Return one of ``stopped`` | ``loading`` | ``ready`` | ``error``."""
        ...

    def health_check(self) -> bool:
        """``True`` if the backend is loaded and ready to transcribe."""
        ...

    def load(self) -> None:
        """Begin loading the configured model (idempotent, non-blocking)."""
        ...

    def change_model(self, model: str) -> None:
        """Switch to a different model. Triggers a background reload."""
        ...

    def current_model(self) -> str:
        """Name / identifier of the model currently configured (not necessarily loaded)."""
        ...

    def current_language(self) -> Optional[str]:
        """Language code the backend will request from the model (or ``None``
        for auto-detect). Used by the recording pipeline when stamping
        history entries."""
        ...

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Optional[str]:
        """Run inference on a mono ``float32`` numpy array.

        Returns the transcribed text, or ``None`` when the backend is not
        ready or the model produced an empty result.
        """
        ...

    def transcribe_file(self, path: str) -> Optional[str]:
        """Transcribe an audio file from disk.

        The backend handles its own audio decoding (WAV / FLAC / OGG /
        MP3 depending on what its loader supports).  Returns the
        transcribed text, or ``None`` when the backend is not ready,
        the file is unreadable, or the model produced an empty result.
        """
        ...

    def shutdown(self) -> None:
        """Release resources, stop background threads. Idempotent."""
        ...
