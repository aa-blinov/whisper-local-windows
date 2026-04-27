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
        self._capture_sample_rate: Optional[int] = None
        # Updated from the audio callback on the recording thread; read
        # by the UI poller. Plain float assignment is atomic under the
        # GIL — no lock needed for the worst-case stale-by-one-tick read.
        self._current_input_level: float = 0.0
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

    def test_input_level(self, duration_s: float = 3.0) -> dict:
        """Synchronous mic check — capture from the selected device for
        ``duration_s`` seconds and report peak / RMS amplitude.

        Uses ``sd.InputStream`` instead of the simpler blocking
        ``sd.rec`` so the per-block callback can update
        ``self._current_input_level`` in real time — the UI's VU
        meter polls that field and follows the level live during
        the test, same as during a real recording. Without this the
        meter sat at zero for the whole 3 s window.

        Captures at the device's native sample rate (so WASAPI
        doesn't complain about 16 kHz like it does for the real
        recording path); we only want amplitude statistics so no
        resampling is needed.

        Returns ``{"peak": float, "rms": float, "duration_s": float}``.
        Both amplitudes are normalised to the [0, 1] range. Raises
        any underlying ``sounddevice`` error so the caller can
        surface a readable message.
        """
        device_idx = self.device
        if device_idx is not None:
            try:
                info = sd.query_devices(device_idx)
                rate = int(info.get("default_samplerate", self.sample_rate))
            except Exception:
                rate = self.sample_rate
        else:
            rate = self.sample_rate

        # ~50 ms blocks — fast enough to feel responsive on the
        # VU meter (~20 Hz updates) without piling up Python
        # callbacks. Floor at 256 frames so very low rates don't
        # trip sounddevice's minimum block size.
        blocksize = max(256, int(rate * 0.05))
        chunks: list = []

        def _on_block(indata, frames, time_info, status):  # noqa: ARG001
            chunks.append(np.asarray(indata).copy())
            flat = np.asarray(indata).flatten()
            if flat.size:
                # Per-block PEAK amplitude rather than RMS so the
                # value matches the meter's 0–1 scale and the
                # post-test frozen reading. RMS for speech is 3–5×
                # smaller than peak — feeding RMS made the meter
                # barely twitch even on yelling, then jump on the
                # frozen-peak overlay after the test ended. The UI
                # widget applies its own peak-and-decay envelope so
                # we don't need any smoothing here.
                self._current_input_level = float(
                    np.abs(flat).max()
                )

        try:
            with sd.InputStream(
                samplerate=rate,
                channels=self.channels,
                device=device_idx,
                dtype=self.STREAM_DTYPE,
                blocksize=blocksize,
                callback=_on_block,
            ):
                # Block the calling (worker) thread for the test
                # window — sounddevice runs the callback on its own
                # thread, so this sleep doesn't starve the audio
                # pipeline.
                time.sleep(float(duration_s))
        finally:
            # Reset the live level so the meter doesn't keep
            # showing the last reading after the test ends.
            self._current_input_level = 0.0

        if not chunks:
            return {"peak": 0.0, "rms": 0.0, "duration_s": float(duration_s)}
        audio = np.concatenate(chunks, axis=0)
        flat = audio.flatten()
        if flat.size == 0:
            return {"peak": 0.0, "rms": 0.0, "duration_s": float(duration_s)}
        peak = float(np.abs(flat).max())
        rms = float(np.sqrt(np.mean(flat ** 2)))
        return {
            "peak": peak,
            "rms": rms,
            "duration_s": float(duration_s),
        }

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
        # Reset the live level so the UI's VU meter doesn't keep
        # showing the last chunk's reading after the user lets go.
        self._current_input_level = 0.0

        return self._process_audio_data()

    def current_input_level(self) -> float:
        """Most-recent RMS amplitude (0..1) of the live audio stream.

        Updated from the recording callback while ``is_recording`` is
        True; reset to 0 on stop. Polled by the UI's VU meter at a
        few dozen Hz.
        """
        return self._current_input_level
    
    def _process_audio_data(self) -> Optional[np.ndarray]:
        if len(self.audio_data) == 0:
            self.logger.warning("No audio data recorded!", extra={'user_message': True})
            return None

        # Convert list of audio chunks into a single numpy array (still at
        # the device's native sample rate at this point).
        audio_array = np.concatenate(self.audio_data, axis=0)

        # Whisper expects 16 kHz mono float32. Resample if the device gave
        # us anything else (typically 44.1 / 48 kHz on WASAPI).
        capture_sr = self._capture_sample_rate or self.WHISPER_SAMPLE_RATE
        if capture_sr != self.WHISPER_SAMPLE_RATE:
            audio_array = self._resample_to(audio_array, capture_sr, self.WHISPER_SAMPLE_RATE)

        # If the recorder ran in stereo, average to mono so Whisper sees a
        # 1-D buffer.
        if audio_array.ndim == 2 and audio_array.shape[1] > 1:
            audio_array = audio_array.mean(axis=1)
        elif audio_array.ndim == 2:
            audio_array = audio_array[:, 0]

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
            "Recorded %.2f seconds of audio (peak=%.3f, rms=%.4f, captured @ %d Hz)",
            duration, peak, rms, capture_sr,
        )
        if peak < 0.01:
            self.logger.warning(
                "Audio looks silent (peak<0.01). Wrong microphone selected, "
                "muted in Windows Sound settings, or input gain too low?",
                extra={'user_message': True},
            )
        return audio_array

    @staticmethod
    def _resample_to(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """Resample a numpy audio buffer to ``target_sr``.

        Uses ``scipy.signal.resample_poly`` (polyphase FIR) instead of
        linear interpolation for two reasons:

        1. **Memory** — linear interpolation required two float64 arrays of
           ``len(audio)`` samples as coordinate vectors (x_old, x_new),
           inflating peak RAM by ~4× the audio size during downsampling.
           ``resample_poly`` uses a fixed-size FIR filter kernel regardless
           of audio length — O(filter_len), not O(N).

        2. **Quality** — polyphase filtering applies a proper anti-aliasing
           low-pass before decimation; linear interpolation has no such
           filter, allowing content above the target Nyquist to alias back
           into the speech band.

        For multi-channel (stereo) input the resampling is done along the
        time axis (axis=0), preserving channel count.  The caller
        (``_process_audio_data``) is responsible for the subsequent
        stereo-to-mono mix-down.
        """
        if source_sr == target_sr or len(audio) == 0:
            return audio.astype(np.float32)

        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(source_sr), int(target_sr))
        up = int(target_sr) // g    # e.g. 48 kHz → 16 kHz: up=1, down=3
        down = int(source_sr) // g

        axis = 0 if audio.ndim == 2 else -1
        return resample_poly(audio, up, down, axis=axis).astype(np.float32)
    
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

                # Compute RMS amplitude of this chunk for the live VU
                # meter. Cheap (numpy on a few hundred samples) and
                # decoupled from the UI thread that polls the value.
                try:
                    flat = np.asarray(audio_data).flatten()
                    if flat.size:
                        # Peak rather than RMS — see comment in
                        # ``test_input_level`` for the rationale.
                        # Same scale as the post-recording UI
                        # treatments (color thresholds calibrated
                        # for peak amplitude).
                        self._current_input_level = float(
                            np.abs(flat).max()
                        )
                except Exception:
                    pass

                if status:
                    self.logger.debug(f"Audio callback status: {status}")

            # Many devices (especially WASAPI) only support their native rate
            # — opening at 16 kHz raises ``Invalid sample rate`` (PaErrorCode
            # -9997). Probe the device's default rate, capture there, and
            # resample to Whisper's 16 kHz in ``_process_audio_data``.
            try:
                if self.device is not None:
                    info = sd.query_devices(self.device)
                else:
                    info = sd.query_devices(kind="input")
                native_sr = int(info.get("default_samplerate", self.WHISPER_SAMPLE_RATE))
            except Exception:
                native_sr = self.WHISPER_SAMPLE_RATE
            if native_sr <= 0:
                native_sr = self.WHISPER_SAMPLE_RATE
            self._capture_sample_rate = native_sr
            self.logger.info(
                "Capturing at %d Hz (target: %d Hz)",
                native_sr, self.WHISPER_SAMPLE_RATE,
            )

            with sd.InputStream(samplerate=native_sr,
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