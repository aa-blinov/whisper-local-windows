import logging
import threading
import time
from typing import Optional, Union

import numpy as np
import sounddevice as sd

class AudioRecorder:
    WHISPER_SAMPLE_RATE = 16000
    THREAD_JOIN_TIMEOUT = 2.0
    RECORDING_SLEEP_INTERVAL = 100
    STREAM_DTYPE = np.float32

    def __init__(self,
                 channels: int = 1,
                 dtype: str = "float32",
                 max_duration: int = 30,
                 on_max_duration_reached: callable = None,
                 device: Optional[Union[int, str]] = None):

        self.sample_rate = self.WHISPER_SAMPLE_RATE
        self.channels = channels
        self.dtype = dtype
        self.max_duration = max_duration
        self.on_max_duration_reached = on_max_duration_reached
        # ``device`` may be ``None`` (use system default), an int index, or a
        # case-insensitive substring of the device name. Resolved to an int
        # at __init__ time so the rest of the code only deals with integers.
        self.device: Optional[int] = self._resolve_device(device)
        self.is_recording = False
        self.audio_data = []
        self.recording_thread = None
        self.recording_start_time = None
        self.logger = logging.getLogger(__name__)

        self._test_microphone()

    def _wait_for_thread_finish(self):
        if self.recording_thread:
            self.recording_thread.join(timeout=self.THREAD_JOIN_TIMEOUT)

    @staticmethod
    def list_input_devices() -> list:
        """Return ``[(index, name), ...]`` for every input-capable device.

        On Windows the MME host API truncates device names to 32 characters
        ("Микрофон (Razer BlackShark V2 P" instead of "Pro"). Prefer WASAPI
        if it's available — same hardware, full names, modern API. Falls
        back to listing every input across every host API on platforms
        where WASAPI is not present (Linux, macOS).
        """
        out = []
        try:
            preferred_hostapi = AudioRecorder._find_hostapi(("WASAPI",))
            for idx, info in enumerate(sd.query_devices()):
                if info.get("max_input_channels", 0) <= 0:
                    continue
                if preferred_hostapi is not None and info.get("hostapi") != preferred_hostapi:
                    continue
                out.append((idx, info.get("name", "<unnamed>")))
        except Exception:
            pass
        return out

    @staticmethod
    def _find_hostapi(needles) -> Optional[int]:
        """Return the index of the first matching host API, or None."""
        try:
            for i, ha in enumerate(sd.query_hostapis()):
                name = (ha.get("name") or "").upper()
                if any(n.upper() in name for n in needles):
                    return i
        except Exception:
            pass
        return None

    def set_device(self, raw: Optional[Union[int, str]]) -> Optional[int]:
        """Switch the input device used for the next recording. Returns the
        resolved index (or ``None`` for system default)."""
        self.device = self._resolve_device(raw)
        return self.device

    def _resolve_device(self, raw: Optional[Union[int, str]]) -> Optional[int]:
        if raw is None or raw == "":
            return None
        try:
            if isinstance(raw, int) or (isinstance(raw, str) and raw.lstrip("-").isdigit()):
                return int(raw)
            needle = str(raw).lower()
            for idx, info in enumerate(sd.query_devices()):
                if info.get("max_input_channels", 0) <= 0:
                    continue
                if needle in str(info.get("name", "")).lower():
                    return idx
        except Exception:
            pass
        return None

    def _test_microphone(self):
        try:
            # List every input device so the user can pick one if the
            # default is wrong.
            for idx, info in enumerate(sd.query_devices()):
                if info.get("max_input_channels", 0) <= 0:
                    continue
                self.logger.info(
                    "Input device [%d]: %s (%s)",
                    idx,
                    info.get("name", "<unnamed>"),
                    info.get("hostapi_name") or info.get("hostapi", ""),
                )

            if self.device is not None:
                info = sd.query_devices(self.device)
                self.logger.info(
                    "Recording from device [%d]: %s",
                    self.device, info.get("name", "<unnamed>"),
                )
            else:
                default_input = sd.query_devices(kind="input")
                self.logger.info(
                    "Recording from system default: %s",
                    default_input.get("name", "<unnamed>"),
                )
        except Exception as e:
            self.logger.error(f"Microphone test failed: {e}")
            raise
    
    def start_recording(self):
        if self.is_recording:
            return False
        
        try:
            self.logger.info("Starting audio recording...")
            self.is_recording = True
            self.audio_data = []
            self.recording_start_time = time.time()
            
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True  # Thread will close when main program closes
            self.recording_thread.start()
            
            self.logger.info("Recording started! Speak now...", extra={'user_message': True})
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start audio recording: {e}")
            self.logger.error("Failed to start recording!", extra={'user_message': True})
            self.is_recording = False
            return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        if not self.is_recording:
            return None
        
        self.is_recording = False
        self._wait_for_thread_finish()
        
        return self._process_audio_data()
    
    def _process_audio_data(self) -> Optional[np.ndarray]:
        if len(self.audio_data) == 0:
            self.logger.warning("No audio data recorded!", extra={'user_message': True})
            return None

        # Convert list of audio chunks into a single numpy array
        audio_array = np.concatenate(self.audio_data, axis=0)
        duration = self.get_audio_duration(audio_array)
        # Peak / RMS amplitude is the cheapest "did the mic actually pick
        # anything up?" check. Whisper returning empty on a quiet recording
        # is impossible to debug without this.
        try:
            peak = float(np.max(np.abs(audio_array)))
            rms = float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)))
        except Exception:
            peak = 0.0
            rms = 0.0
        self.logger.info(
            "Recorded %.2f seconds of audio (peak=%.3f, rms=%.4f)",
            duration, peak, rms,
        )
        if peak < 0.01:
            self.logger.warning(
                "Audio looks silent (peak<0.01). Wrong microphone selected, "
                "muted in Windows Sound settings, or input gain too low?",
                extra={'user_message': True},
            )
        return audio_array
    
    def cancel_recording(self):
        if not self.is_recording:
            return
        
        self.is_recording = False
        self._wait_for_thread_finish()
        
        self.audio_data = []
        self.recording_start_time = None
    
    def _record_audio(self):
        try:
            def audio_callback(audio_data, frames, time, status):                
                if self.is_recording:
                    self.audio_data.append(audio_data.copy())

                if status:
                    self.logger.debug(f"Audio callback status: {status}")
            
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=self.channels,
                                callback=audio_callback,
                                dtype=self.STREAM_DTYPE,
                                device=self.device):
                
                while self.is_recording:
                    if self._check_max_duration_exceeded():
                        break
                    
                    sd.sleep(self.RECORDING_SLEEP_INTERVAL)
                
        except Exception as e:
            self.logger.error(f"Error during audio recording: {e}")
            self.is_recording = False
    
    def _check_max_duration_exceeded(self) -> bool:
        if self.max_duration > 0 and self.recording_start_time:
            elapsed_time = time.time() - self.recording_start_time
            if elapsed_time >= self.max_duration:
                self.logger.info(f"Maximum recording duration of {self.max_duration}s reached")
                self.logger.info(f"Maximum recording duration of {self.max_duration}s reached - stopping recording", extra={'user_message': True})
                
                self.is_recording = False
                audio_data = self._process_audio_data()
                
                if self.on_max_duration_reached:
                    self.on_max_duration_reached(audio_data)
                return True
        return False
    
    def get_recording_status(self) -> bool:
        return self.is_recording
    
    def get_audio_duration(self, audio_data: np.ndarray) -> float:
        if audio_data is None or len(audio_data) == 0:
            return 0.0
        return len(audio_data) / self.sample_rate