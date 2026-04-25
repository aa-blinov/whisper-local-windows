import logging
import time
import threading
from typing import Optional

from app.audio_recorder import AudioRecorder
from app.whisper_engine import WhisperEngine
from app.clipboard_manager import ClipboardManager
from app.system_tray import SystemTray
from app.config_manager import ConfigManager
from app.audio_feedback import AudioFeedback
from app.utils import OptionalComponent
from app.history_manager import HistoryManager

class StateManager:
    def __init__(self, 
                 audio_recorder: AudioRecorder,
                 whisper_engine: WhisperEngine,
                 clipboard_manager: ClipboardManager,
                 config_manager: ConfigManager,
                 system_tray: Optional[SystemTray] = None,
                 audio_feedback: Optional[AudioFeedback] = None,
                 docker_backend_manager = None):

        self.audio_recorder = audio_recorder
        self.whisper_engine = whisper_engine
        self.clipboard_manager = clipboard_manager
        self.system_tray = OptionalComponent(system_tray)
        self.config_manager = config_manager
        self.audio_feedback = OptionalComponent(audio_feedback)
        self.docker_backend_manager = docker_backend_manager
        
        # Initialize history manager
        from app.utils import get_project_logs_path
        import os
        history_config = self.config_manager.get_history_config()
        if history_config.get('enabled', True):
            history_file = os.path.join(get_project_logs_path(), "transcription_history.json")
            max_entries = history_config.get('max_entries', 1000)
            self.history_manager = HistoryManager(max_entries=max_entries, history_file=history_file)
        else:
            self.history_manager = None
        
        # History update callback (to be set by UI)
        self.history_update_callback = None
        
        self.is_processing = False
        self.is_model_loading = False
        self.last_transcription = None
        self._pending_model_change = None  # Store pending model change request
        self._state_lock = threading.Lock()  # Thread safety for state operations

        self.logger = logging.getLogger(__name__)
    
    def handle_max_recording_duration_reached(self, audio_data):
        """Called when audio recorder reaches max duration with audio data"""
        self.logger.info("Max recording duration reached - starting transcription")
        self._transcription_pipeline(audio_data, use_auto_enter=False)
    
    def stop_recording(self, use_auto_enter: bool = False) -> bool:
        currently_recording = self.audio_recorder.get_recording_status()
        
        if currently_recording:
            audio_data = self.audio_recorder.stop_recording()
            self._transcription_pipeline(audio_data, use_auto_enter)
            return True
        else:
            return False
    
    def cancel_active_recording(self):
        self.audio_recorder.cancel_recording()
        self.audio_feedback.play_cancel_sound()
        self.system_tray.update_state("idle")
    
    def cancel_recording_hotkey_pressed(self) -> bool:
        current_state = self.get_current_state()
        
        if current_state == "recording":
            self.logger.info("Recording cancelled!", extra={'user_message': True})            
            self.cancel_active_recording()
            return True
        else:
            return False
    
    def toggle_recording(self):
        was_recording = self.stop_recording(use_auto_enter=False)

        if not was_recording:
            current_state = self.get_current_state()
            can_start = self.can_start_recording()
            self.logger.debug(f"toggle_recording: current_state={current_state}, can_start={can_start}")

            if can_start:
                self._start_recording()
            else:
                # Acoustic feedback that the hotkey was actually caught even
                # when recording cannot start (backend not ready, busy, etc.).
                # Without this the user has no signal that the keypress arrived.
                self.audio_feedback.play_start_sound()
                if self.is_processing:
                    self.logger.info("Still processing previous recording...", extra={'user_message': True})
                elif self.is_model_loading:
                    self.logger.info("Still loading model...", extra={'user_message': True})
                elif current_state == "idle" and self.docker_backend_manager:
                    # If idle but can't record, likely model not ready
                    self.logger.info("Model is not ready yet. Please wait...", extra={'user_message': True})
                else:
                    self.logger.info(f"Cannot record while {current_state}...", extra={'user_message': True})

    def _start_recording(self):
        self.logger.debug("_start_recording: attempting to start audio recorder")
        success = self.audio_recorder.start_recording()
        
        if success:
            self.logger.debug("_start_recording: audio recorder started successfully, playing start sound")
            self.config_manager.print_stop_instructions_based_on_config()
            self.audio_feedback.play_start_sound()
            self.system_tray.update_state("recording")
        else:
            self.logger.debug("_start_recording: audio recorder failed to start")
    
    def _transcription_pipeline(self, audio_data, use_auto_enter: bool = False):
        try:
            self.logger.debug("[Pipeline] Enter _transcription_pipeline (auto_enter=%s)" % use_auto_enter)
            # Prevent multiple threads from starting simultaneous transcription
            with self._state_lock:
                self.is_processing = True
                self.logger.debug(f"[Pipeline] is_processing set True; model_loading={self.is_model_loading}")

            self.audio_feedback.play_stop_sound()
            
            if audio_data is None:
                self.logger.debug("[Pipeline] audio_data is None -> early return")
                return
            
            duration = self.audio_recorder.get_audio_duration(audio_data)
            self.logger.info(f"Recorded {duration:.1f} seconds! Transcribing...", extra={'user_message': True})
            self.logger.debug(f"[Pipeline] Recorded duration={duration:.3f}s; starting transcription")
            
            transcribed_text = self.whisper_engine.transcribe_audio(audio_data)
            self.logger.debug(f"[Pipeline] transcribe_audio returned length={0 if not transcribed_text else len(transcribed_text)}")
            
            if not transcribed_text:
                self.logger.debug("[Pipeline] No transcription text -> return path")
                return
            
            self.system_tray.update_state("processing")
            self.logger.debug("[Pipeline] System tray set to processing; delivering transcription (auto_enter=%s)" % use_auto_enter)

            success = self.clipboard_manager.deliver_transcription(
                transcribed_text, use_auto_enter
            )
            self.logger.debug(f"[Pipeline] deliver_transcription success={success}")
            
            if success:
                self.last_transcription = transcribed_text
                self.logger.debug("[Pipeline] last_transcription updated")
                
                # Add to history
                if self.history_manager:
                    try:
                        self.history_manager.add_entry(
                            text=transcribed_text,
                            duration=duration,
                            model=self.whisper_engine.model_size,
                            language=self.whisper_engine.language or "auto"
                        )
                        self.logger.debug("[Pipeline] Added entry to history")
                        
                        # Notify UI to update history display
                        if self.history_update_callback:
                            self.history_update_callback()
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to add entry to history: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in processing workflow: {e}", exc_info=True)
            self.logger.error(f"Error processing recording: {e}", extra={'user_message': True})
        
        finally:
            self.logger.debug("[Pipeline] Enter finally block")
            
            # Explicitly free audio data memory
            try:
                if audio_data is not None:
                    del audio_data
                    import gc
                    gc.collect()
                    self.logger.debug("[Pipeline] Audio data memory freed")
            except Exception as e:
                self.logger.debug(f"[Pipeline] Failed to free audio memory: {e}")
            
            with self._state_lock:
                self.is_processing = False
                self.logger.debug(f"[Pipeline] is_processing set False; model_loading={self.is_model_loading}")
                
                pending_model = self._pending_model_change                    
            
            # Execute pending model change outside of lock to avoid deadlock
            if 'pending_model' in locals() and pending_model:
                self.logger.info(f"Executing pending model change to: {pending_model}")
                self.logger.info(f"Processing complete, now switching to {pending_model} model...", extra={'user_message': True})
                self._execute_model_change(pending_model)
                self._pending_model_change = None
            else:
                self.system_tray.update_state("idle")
                self.logger.debug("[Pipeline] System tray set to idle; pipeline end")
    
    def get_application_state(self) -> dict:
        status = {
            "recording": self.audio_recorder.get_recording_status(),
            "processing": self.is_processing,
            "model_loading": self.is_model_loading,
        }
        
        return status
    
    def manual_transcribe_test(self, duration_seconds: int = 5):
        try:
            self.logger.info(f"Recording for {duration_seconds} seconds...", extra={'user_message': True})
            self.logger.info("Speak now!", extra={'user_message': True})
            
            self.audio_recorder.start_recording()
            
            time.sleep(duration_seconds)
            
            audio_data = self.audio_recorder.stop_recording()
            self._transcription_pipeline(audio_data)
            
        except Exception as e:
            self.logger.error(f"Manual test failed: {e}")
            self.logger.error(f"Test failed: {e}", extra={'user_message': True})
    
    def shutdown(self):        
        self.logger.info("Lazy to text is shutting down... goodbye!", extra={'user_message': True})

        if self.audio_recorder.get_recording_status():
            self.audio_recorder.stop_recording()
        
        self.system_tray.stop()
    
    def set_model_loading(self, loading: bool):
        with self._state_lock:
            old_state = self.is_model_loading
            self.is_model_loading = loading
            
            if old_state != loading:
                if loading:
                    self.system_tray.update_state("processing")
                else:
                    self.system_tray.update_state("idle")
    
    def can_start_recording(self) -> bool:
        with self._state_lock:
            basic_check = not (self.is_processing or self.is_model_loading or self.audio_recorder.get_recording_status())
            
            # If basic checks fail, don't proceed
            if not basic_check:
                self.logger.debug(f"can_start_recording: basic_check failed (processing={self.is_processing}, model_loading={self.is_model_loading}, recording={self.audio_recorder.get_recording_status()})")
                return False
            
            # Check if Docker backend is ready (model loaded)
            if self.docker_backend_manager:
                try:
                    # Check if container is running and model is ready
                    is_ready = self.docker_backend_manager.has_ready_log_line(r"Connection to .*10300 port .* succeeded!")
                    self.logger.debug(f"can_start_recording: docker_backend check -> is_ready={is_ready}")
                    return is_ready
                except Exception as e:
                    self.logger.debug(f"Failed to check Docker readiness: {e}")
                    # If we can't check Docker status, fallback to basic checks
                    return basic_check
            
            self.logger.debug("can_start_recording: no docker_backend_manager, returning basic_check=True")
            return basic_check
    
    def get_current_state(self) -> str:
        with self._state_lock:
            if self.is_model_loading:
                return "model_loading"
            elif self.is_processing:
                return "processing"
            elif self.audio_recorder.get_recording_status():
                return "recording"
            else:
                return "idle"
    
    def request_model_change(self, new_model_size: str) -> bool:
        current_state = self.get_current_state()
        
        if new_model_size == self.whisper_engine.model_size:
            return True
        
        if current_state == "model_loading":
            self.logger.info("Model already loading, please wait...", extra={'user_message': True})
            return False
        
        if current_state == "recording":
            self.logger.info(f"Cancelling recording to switch to {new_model_size} model...", extra={'user_message': True})
            self.cancel_active_recording()
            self._execute_model_change(new_model_size)
            return True
        
        if current_state == "processing":
            self.logger.info(f"Queueing model change to {new_model_size} until transcription completes...", extra={'user_message': True})
            self._pending_model_change = new_model_size
            return True
        
        if current_state == "idle":
            self._execute_model_change(new_model_size)
            return True
        
        self.logger.warning(f"Unexpected state for model change: {current_state}")
        return False
    
    def update_transcription_mode(self, value):            
        self.config_manager.update_user_setting('clipboard', 'auto_paste', value)
        self.clipboard_manager.update_auto_paste(value)

    def _execute_model_change(self, new_model_size: str):
        def progress_callback(message: str):
            if "ready" in message.lower() or "already loaded" in message.lower():
                self.logger.info(f"Successfully switched to {new_model_size} model", extra={'user_message': True})
                self.set_model_loading(False)
            elif "failed" in message.lower():
                self.logger.error(f"Failed to change model: {message}", extra={'user_message': True})
                self.set_model_loading(False)
            else:
                self.logger.info(f"{message}", extra={'user_message': True})
                self.set_model_loading(True)
        
        try:
            self.set_model_loading(True)
            self.logger.info(f"Switching to {new_model_size} model...", extra={'user_message': True})
            
            self.whisper_engine.change_model(new_model_size, progress_callback)
            
        except Exception as e:
            self.logger.error(f"Failed to initiate model change: {e}")
            self.logger.error(f"Failed to change model: {e}", extra={'user_message': True})
            self.set_model_loading(False)