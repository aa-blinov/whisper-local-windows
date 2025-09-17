import logging

from global_hotkeys import register_hotkeys, start_checking_hotkeys, stop_checking_hotkeys
# Extra imports to encourage PyInstaller to collect dependencies used internally
try:
    import keyboard  # type: ignore
except Exception:  # pragma: no cover
    keyboard = None
try:
    import ctypes  # noqa: F401
except Exception:
    pass

from .state_manager import StateManager

class HotkeyListener:
    def __init__(self, state_manager: StateManager, start_recording_hotkey: str,
                 stop_recording_hotkey: str | None = None,
                 cancel_combination: str = None):
        """Simplified listener: only start, optional stop, optional cancel.
        Flow:
          1. Press start_recording_hotkey -> start (only if idle)
          2. Speak
          3. Press stop_recording_hotkey (or cancel) -> stop & transcribe
        Clipboard auto-paste handled downstream; no auto-enter / modifier stop logic now.
        """
        self.state_manager = state_manager
        self.start_recording_hotkey = start_recording_hotkey
        self.stop_recording_hotkey = stop_recording_hotkey
        self.cancel_combination = cancel_combination
        self.is_listening = False
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"[hotkeys] Initializing (frozen={getattr(__import__('sys'),'frozen',False)}) start='{start_recording_hotkey}' stop='{stop_recording_hotkey}' cancel='{cancel_combination}'")

        self._setup_hotkeys()
        self.start_listening()
    
    def _setup_hotkeys(self):
        hotkey_configs = []
        
        same_toggle = False
        if self.stop_recording_hotkey and self.stop_recording_hotkey.strip().lower() == self.start_recording_hotkey.strip().lower():
            same_toggle = True

        if same_toggle:
            # Single toggle binding
            hotkey_configs.append({
                'combination': self.start_recording_hotkey,
                'callback': self._toggle_hotkey_pressed,
                'name': 'toggle'
            })
        else:
            hotkey_configs.append({
                'combination': self.start_recording_hotkey,
                'callback': self._start_hotkey_pressed,
                'name': 'start'
            })
            if self.stop_recording_hotkey:
                hotkey_configs.append({
                    'combination': self.stop_recording_hotkey,
                    'callback': self._stop_hotkey_pressed,
                    'name': 'stop'
                })
        if self.cancel_combination and not same_toggle:
            hotkey_configs.append({
                'combination': self.cancel_combination,
                'callback': self._cancel_hotkey_pressed,
                'name': 'cancel'
            })
        hotkey_configs.sort(key=self._get_hotkey_combination_specificity, reverse=True)
        self.hotkey_bindings = []
        for config in hotkey_configs:
            formatted_hotkey = self._convert_hotkey_to_global_hotkeys_format(config['combination'])
            self.hotkey_bindings.append([
                formatted_hotkey,
                config['callback'],
                config.get('release_callback') or None,
                False
            ])
            self.logger.info(f"Configured {config['name']} hotkey: {config['combination']} -> {formatted_hotkey}")
        self.logger.info(f"Total hotkeys configured: {len(self.hotkey_bindings)}")
    
    def _get_hotkey_combination_specificity(self, hotkey_config: dict) -> int:
        """
        Returns specificity score to ensure combos with more keys take priority
        """
        combination = hotkey_config['combination'].lower()
        return len(combination.split('+'))
    
    def _start_hotkey_pressed(self):
        self.logger.info(f"Start hotkey pressed: {self.start_recording_hotkey}")
        if self.state_manager.get_current_state() == "idle":
            self.state_manager.toggle_recording()
        else:
            self.logger.debug("Start hotkey ignored - not idle")

    def _stop_hotkey_pressed(self):
        self.logger.info(f"Stop hotkey pressed: {self.stop_recording_hotkey}")
        if self.state_manager.get_current_state() == "recording":
            self.state_manager.stop_recording(use_auto_enter=False)
        else:
            self.logger.debug("Stop hotkey ignored - not recording")

    def _toggle_hotkey_pressed(self):
        # Unified toggle when start == stop
        current = self.state_manager.get_current_state()
        if current == "idle":
            self.logger.info(f"Toggle hotkey pressed (start): {self.start_recording_hotkey}")
            self.state_manager.toggle_recording()
        elif current == "recording":
            self.logger.info(f"Toggle hotkey pressed (stop): {self.start_recording_hotkey}")
            self.state_manager.stop_recording(use_auto_enter=False)
        else:
            self.logger.debug("Toggle hotkey ignored - busy state")
    
    def _cancel_hotkey_pressed(self):
        self.logger.info(f"Cancel hotkey pressed: {self.cancel_combination}")
        self.state_manager.cancel_recording_hotkey_pressed()
    
    def start_listening(self):
        if self.is_listening:
            return
        try:
            self.logger.debug(f"[hotkeys] Registering {len(self.hotkey_bindings)} bindings: {self.hotkey_bindings}")
            register_hotkeys(self.hotkey_bindings)
            start_checking_hotkeys()
            self.is_listening = True
            self.logger.info("Global hotkey listener active")
        except Exception as e:
            self.logger.error(f"Failed to start hotkey listener: {e}")
            raise
    
    def stop_listening(self):
        if not self.is_listening:
            return
        try:
            stop_checking_hotkeys()
            self.is_listening = False
            self.logger.info("Hotkey listener stopped")
        except Exception as e:
            self.logger.error(f"Error stopping hotkey listener: {e}")
    
    def _convert_hotkey_to_global_hotkeys_format(self, hotkey_str: str) -> str:
        key_mapping = {
            'ctrl': 'control',
            'shift': 'shift',
            'alt': 'alt',
            'win': 'window',
            'windows': 'window',
            'cmd': 'window',
            'super': 'window',
            'space': 'space',
            'enter': 'enter',
            'esc': 'escape'
        }
        keys = hotkey_str.lower().split('+')
        converted_keys = []
        for key in keys:
            key = key.strip()
            converted_keys.append(key_mapping.get(key, key))
        return ' + '.join(converted_keys)    
    
    def change_hotkey_config(self, setting: str, value):
        valid_settings = ['start_recording_hotkey', 'stop_recording_hotkey', 'cancel_combination']
        if setting not in valid_settings:
            raise ValueError(f"Invalid setting '{setting}'. Valid options: {valid_settings}")
        old_value = getattr(self, setting)
        if old_value == value:
            return
        setattr(self, setting, value)
        self.logger.info(f"Changed {setting}: {old_value} -> {value}")
        self.stop_listening()
        self._setup_hotkeys()
        self.start_listening()
    
    def is_active(self) -> bool:
        return self.is_listening