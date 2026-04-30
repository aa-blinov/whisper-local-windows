import logging
import os
import struct
import sys
import threading

from app.utils import resolve_asset_path

if sys.platform == "win32":
    import winsound
else:
    winsound = None  # type: ignore[assignment]


def _build_silent_wav() -> bytes:
    """Return a minimal valid silent WAV buffer.

    Used to prewarm the Windows multimedia stack: the first
    ``winsound.PlaySound`` after process start spends 100-300 ms opening the
    audio device, and any SND_ASYNC sound issued during that window is
    silently dropped. Playing this buffer at startup forces the device open
    without any audible output (it is a fraction of a second of zero
    samples at low rate).
    """
    sample_rate = 8000
    bits_per_sample = 8
    channels = 1
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    sample_count = 100  # ~12.5 ms of silence — inaudible
    data_size = sample_count * block_align
    file_size = 36 + data_size
    header = b"".join(
        [
            b"RIFF",
            struct.pack("<I", file_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),
            struct.pack("<H", 1),  # PCM
            struct.pack("<H", channels),
            struct.pack("<I", sample_rate),
            struct.pack("<I", byte_rate),
            struct.pack("<H", block_align),
            struct.pack("<H", bits_per_sample),
            b"data",
            struct.pack("<I", data_size),
        ]
    )
    # 8-bit unsigned PCM: silence is 0x80 (128).
    silence = bytes([0x80] * data_size)
    return header + silence


_SILENT_WAV = _build_silent_wav() if sys.platform == "win32" else b""


class AudioFeedback:
    def __init__(self, enabled=True, start_sound='', stop_sound='', cancel_sound=''):
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)

        self.start_sound_path = resolve_asset_path(start_sound)
        self.stop_sound_path = resolve_asset_path(stop_sound)
        self.cancel_sound_path = resolve_asset_path(cancel_sound)

        if not self.enabled:
            self.logger.info("Audio feedback disabled by configuration", extra={'user_message': True})
            self.logger.info("Audio feedback disabled", extra={'user_message': True})
        else:
            self._validate_sound_files()
            self.prewarm()
            self.logger.info("Audio feedback enabled...", extra={'user_message': True})

    def prewarm(self) -> None:
        """Wake the Windows audio subsystem so the next play_*_sound call is
        heard rather than silently dropped during device-open latency.

        Windows releases an idle audio device after a few seconds.  The
        initial call happens at construction time; call this again whenever a
        long pause (e.g. model loading) may have caused the device to close.
        No-ops immediately if audio feedback is disabled, and on
        non-Windows platforms (``playsound3`` opens the device on
        every call without the silent-drop pathology).
        """
        if not self.enabled or sys.platform != "win32":
            return

        def warm():
            try:
                winsound.PlaySound(
                    _SILENT_WAV,
                    winsound.SND_MEMORY | winsound.SND_ASYNC,
                )
            except Exception as exc:
                self.logger.debug(f"Audio prewarm failed: {exc}")

        threading.Thread(target=warm, daemon=True, name="audio-prewarm").start()
    
    def _validate_sound_files(self):
        if self.start_sound_path and not os.path.isfile(self.start_sound_path):
            self.logger.warning(f"Start sound file not found: {self.start_sound_path}")
        
        if self.stop_sound_path and not os.path.isfile(self.stop_sound_path):
            self.logger.warning(f"Stop sound file not found: {self.stop_sound_path}")
        
        if self.cancel_sound_path and not os.path.isfile(self.cancel_sound_path):
            self.logger.warning(f"Cancel sound file not found: {self.cancel_sound_path}")
    
    def _play_sound_file_async(self, file_path: str):
        """Play ``file_path`` on a daemon thread.

        Windows: we deliberately do NOT pass ``winsound.SND_ASYNC``.
        With ``SND_ASYNC``, ``PlaySound`` returns immediately and
        Windows plays the buffer from a system worker — but if the
        calling thread exits before Windows finishes setting up that
        play, the sound is dropped silently.  We've seen this happen
        intermittently when the start sound fires right before the
        microphone stream opens (the recorder grabbing the audio
        device disrupts the still-pending async play).

        Synchronous ``PlaySound`` blocks the thread for the duration
        of the clip (~80-150 ms for our cues) — but the thread is
        daemon and isolated from the main loop, so the user-facing
        latency is identical: the start sound still kicks off in
        parallel with everything else, just without the racy
        Windows-side queue.

        macOS / Linux: delegates to ``playsound3.playsound`` with
        ``block=True`` on the daemon thread (same rationale — keep
        the play synchronous to avoid any interpreter-shutdown race
        on the underlying backend).
        """
        if not file_path:
            return

        if sys.platform == "win32":
            def play_sound():
                try:
                    winsound.PlaySound(file_path, winsound.SND_FILENAME)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to play sound file {file_path}: {e}"
                    )
        else:
            def play_sound():
                try:
                    from playsound3 import playsound

                    playsound(file_path, block=True)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to play sound file {file_path}: {e}"
                    )

        sound_thread = threading.Thread(target=play_sound, daemon=True)
        sound_thread.start()
    
    def play_start_sound(self):
        if self.enabled:
            self._play_sound_file_async(self.start_sound_path)
    
    def play_stop_sound(self):
        if self.enabled:        
            self._play_sound_file_async(self.stop_sound_path)
    
    def play_cancel_sound(self):
        if self.enabled:
            self._play_sound_file_async(self.cancel_sound_path)