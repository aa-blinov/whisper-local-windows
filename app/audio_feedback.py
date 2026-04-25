import logging
import os
import struct
import threading
import winsound

from app.utils import resolve_asset_path


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


_SILENT_WAV = _build_silent_wav()


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
            self._prewarm()
            self.logger.info("Audio feedback enabled...", extra={'user_message': True})

    def _prewarm(self):
        """Wake the Windows audio subsystem so the first user-triggered
        play_*_sound is heard rather than dropped during device-open latency.
        """
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
        def play_sound():
            try:
                # SND_FILENAME = play from file, SND_ASYNC = don't block
                winsound.PlaySound(file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                
            except Exception as e:
                self.logger.warning(f"Failed to play sound file {file_path}: {e}")
        
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