import logging
import sys
import time

if sys.platform == "win32":
    from global_hotkeys import (
        register_hotkeys,
        start_checking_hotkeys,
        stop_checking_hotkeys,
    )
    # Extra imports to encourage PyInstaller to collect dependencies
    # used internally by ``global_hotkeys``.
    try:
        import keyboard  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover
        pass
    try:
        import ctypes  # noqa: F401
    except Exception:
        pass
else:
    register_hotkeys = None  # type: ignore[assignment]
    start_checking_hotkeys = None  # type: ignore[assignment]
    stop_checking_hotkeys = None  # type: ignore[assignment]

from app.state_manager import StateManager


# Push-to-talk key vocabulary → pynput Key attribute name.  We
# resolve the actual ``pynput.keyboard.Key`` enum lazily inside
# ``_resolve_ptt_key`` so the module import path doesn't pull in
# pynput on Windows where the legacy ``global-hotkeys`` listener
# is used instead.
_PTT_KEY_TO_PYNPUT: dict[str, str] = {
    "right_cmd": "cmd_r",
    "cmd_r": "cmd_r",
    "right_command": "cmd_r",
    "command_r": "cmd_r",
    "right_option": "alt_r",
    "option_r": "alt_r",
    "right_alt": "alt_r",
    "alt_r": "alt_r",
    "right_shift": "shift_r",
    "shift_r": "shift_r",
    "right_ctrl": "ctrl_r",
    "ctrl_r": "ctrl_r",
    "right_control": "ctrl_r",
    "control_r": "ctrl_r",
    "fn": "fn",
}


class HotkeyListener:
    def __init__(
        self,
        state_manager: StateManager,
        start_recording_hotkey: str,
        stop_recording_hotkey: str | None = None,
        cancel_combination: str | None = None,
        *,
        mode: str = "two_keys",
        push_to_talk_key: str | None = None,
        push_to_talk_min_hold_seconds: float = 0.2,
    ):
        """Global keyboard listener with three recording modes.

        ``mode`` selects the binding shape:

        - ``"two_keys"`` (default) — one binding fires
          ``state_manager.toggle_recording`` when ``idle``, another
          fires ``state_manager.stop_recording`` when ``recording``.
          Classic fire-and-forget UX, same as the project shipped
          since day one.
        - ``"toggle"`` — single binding that flips between idle ↔
          recording on each press (use when ``start`` and ``stop``
          would otherwise be the same combo).
        - ``"push_to_talk"`` — hold ``push_to_talk_key`` to record,
          release to stop and transcribe.  The PTT path uses a raw
          ``pynput.keyboard.Listener`` (or the
          ``global_hotkeys`` low-level hook on Windows) instead of
          ``GlobalHotKeys`` because the latter only fires on press.
          Bindings shorter than ``push_to_talk_min_hold_seconds``
          are treated as accidental taps and cancel the recording
          before transcription, so a glancing right-Cmd press
          doesn't litter the history with empty entries.

        ``cancel_combination`` is honoured in all three modes —
        registered via ``GlobalHotKeys`` even in PTT mode (the
        listener and the GlobalHotKeys instance share the underlying
        macOS event tap and don't conflict).
        """
        self.state_manager = state_manager
        self.start_recording_hotkey = start_recording_hotkey
        self.stop_recording_hotkey = stop_recording_hotkey
        self.cancel_combination = cancel_combination
        self.mode = mode if mode in {"two_keys", "toggle", "push_to_talk"} else "two_keys"
        self.push_to_talk_key = (push_to_talk_key or "").strip().lower() or None
        self.push_to_talk_min_hold_seconds = float(push_to_talk_min_hold_seconds)
        self.is_listening = False
        self.logger = logging.getLogger(__name__)
        # Non-Windows path: pynput's GlobalHotKeys instance (created
        # by ``_setup_hotkeys`` and started by ``start_listening``).
        self._pynput_listener = None
        self._pynput_callbacks: dict[str, callable] = {}
        # Push-to-talk: separate ``pynput.keyboard.Listener`` that
        # tracks raw on_press / on_release for the configured key.
        # Lives alongside the GlobalHotKeys instance (which
        # registers the cancel combo, if any).
        self._ptt_listener = None
        self._ptt_target_key = None  # pynput.keyboard.Key enum
        self._ptt_held = False
        self._ptt_press_ts: float = 0.0
        self.logger.debug(
            f"[hotkeys] Initializing mode='{self.mode}' "
            f"start='{start_recording_hotkey}' stop='{stop_recording_hotkey}' "
            f"cancel='{cancel_combination}' "
            f"ptt_key='{self.push_to_talk_key}'"
        )

        self._setup_hotkeys()
        self.start_listening()
    
    def _setup_hotkeys(self):
        hotkey_configs = []

        if self.mode == "push_to_talk":
            # PTT path:  the press/release listener is set up
            # separately in ``start_listening``.  Register only the
            # cancel binding through GlobalHotKeys so a global
            # "abort recording" combo still works while the user
            # holds the PTT key.
            self._ptt_target_key = self._resolve_ptt_key(self.push_to_talk_key)
            if self._ptt_target_key is None:
                self.logger.warning(
                    "Push-to-talk key %r is not in the recognised "
                    "vocabulary (%s) — PTT will be inactive.",
                    self.push_to_talk_key,
                    sorted(_PTT_KEY_TO_PYNPUT.keys()),
                )
            else:
                self.logger.info(
                    "Configured push-to-talk: hold %r (%.0f ms minimum)",
                    self.push_to_talk_key,
                    self.push_to_talk_min_hold_seconds * 1000,
                )
            if self.cancel_combination:
                hotkey_configs.append({
                    'combination': self.cancel_combination,
                    'callback': self._cancel_hotkey_pressed,
                    'name': 'cancel',
                })
        elif self.mode == "toggle":
            hotkey_configs.append({
                'combination': self.start_recording_hotkey,
                'callback': self._toggle_hotkey_pressed,
                'name': 'toggle',
            })
            if self.cancel_combination:
                hotkey_configs.append({
                    'combination': self.cancel_combination,
                    'callback': self._cancel_hotkey_pressed,
                    'name': 'cancel',
                })
        else:
            # ``two_keys`` (default).  Legacy escape hatch: if the
            # caller passes the same combo for start and stop, fold
            # into a single toggle binding instead of trying to
            # register the same key twice — the old default-config
            # migration path still ends up here for some users.
            same_toggle = (
                self.stop_recording_hotkey
                and self.stop_recording_hotkey.strip().lower()
                == self.start_recording_hotkey.strip().lower()
            )
            if same_toggle:
                hotkey_configs.append({
                    'combination': self.start_recording_hotkey,
                    'callback': self._toggle_hotkey_pressed,
                    'name': 'toggle',
                })
            else:
                hotkey_configs.append({
                    'combination': self.start_recording_hotkey,
                    'callback': self._start_hotkey_pressed,
                    'name': 'start',
                })
                if self.stop_recording_hotkey:
                    hotkey_configs.append({
                        'combination': self.stop_recording_hotkey,
                        'callback': self._stop_hotkey_pressed,
                        'name': 'stop',
                    })
            if self.cancel_combination and not same_toggle:
                hotkey_configs.append({
                    'combination': self.cancel_combination,
                    'callback': self._cancel_hotkey_pressed,
                    'name': 'cancel',
                })

        hotkey_configs.sort(key=self._get_hotkey_combination_specificity, reverse=True)
        self.hotkey_bindings = []
        self._pynput_callbacks = {}
        for config in hotkey_configs:
            if sys.platform == "win32":
                formatted_hotkey = self._convert_hotkey_to_global_hotkeys_format(config['combination'])
                self.hotkey_bindings.append([
                    formatted_hotkey,
                    config['callback'],
                    config.get('release_callback') or None,
                    False
                ])
            else:
                formatted_hotkey = self._convert_hotkey_to_pynput_format(config['combination'])
                self._pynput_callbacks[formatted_hotkey] = config['callback']
            self.logger.info(f"Configured {config['name']} hotkey: {config['combination']} -> {formatted_hotkey}")
        if sys.platform == "win32":
            self.logger.info(f"Total hotkeys configured: {len(self.hotkey_bindings)}")
        else:
            self.logger.info(f"Total hotkeys configured: {len(self._pynput_callbacks)}")

    def _resolve_ptt_key(self, name: str | None):
        """Translate a user-facing PTT key name (``"right_cmd"``,
        ``"cmd_r"``, …) to a ``pynput.keyboard.Key`` enum.  Returns
        ``None`` when the name is unrecognised or pynput isn't
        importable in this environment.
        """
        if not name:
            return None
        attr = _PTT_KEY_TO_PYNPUT.get(name.strip().lower())
        if attr is None:
            return None
        try:
            from pynput import keyboard as _pk
        except ImportError:
            return None
        return getattr(_pk.Key, attr, None)

    # ---- push-to-talk callbacks (called from listener thread) ---------------

    def _ptt_on_press(self, key) -> None:
        """Listener callback — start recording on the first press of
        the PTT key.  Auto-repeat presses while the key is held are
        ignored (``_ptt_held`` guard).
        """
        if self._ptt_target_key is None or key != self._ptt_target_key:
            return
        if self._ptt_held:
            return  # auto-repeat
        self._ptt_held = True
        self._ptt_press_ts = time.monotonic()
        if self.state_manager.get_current_state() != "idle":
            self.logger.debug("PTT press ignored — state is not idle")
            return
        if not self.state_manager.can_start_recording():
            self.logger.info(
                "Model is not ready yet. Please wait...",
                extra={'user_message': True},
            )
            return
        self.logger.info(
            "PTT press: %r — starting recording",
            self.push_to_talk_key,
        )
        self.state_manager.toggle_recording()

    def _ptt_on_release(self, key) -> None:
        """Listener callback — stop recording when the PTT key is
        released.  Holds shorter than ``push_to_talk_min_hold_seconds``
        are treated as accidental taps:  cancel the recording so the
        history doesn't fill up with empty entries.
        """
        if self._ptt_target_key is None or key != self._ptt_target_key:
            return
        if not self._ptt_held:
            return
        held_for = time.monotonic() - self._ptt_press_ts
        self._ptt_held = False
        if self.state_manager.get_current_state() != "recording":
            return
        if held_for < self.push_to_talk_min_hold_seconds:
            self.logger.info(
                "PTT release after %.0f ms (< %.0f ms minimum) — "
                "cancelling recording.",
                held_for * 1000,
                self.push_to_talk_min_hold_seconds * 1000,
                extra={'user_message': True},
            )
            self.state_manager.cancel_recording_hotkey_pressed()
            return
        self.logger.info(
            "PTT release after %.0f ms — stopping + transcribing",
            held_for * 1000,
        )
        self.state_manager.stop_recording(use_auto_enter=False)
    
    def _get_hotkey_combination_specificity(self, hotkey_config: dict) -> int:
        """
        Returns specificity score to ensure combos with more keys take priority
        """
        combination = hotkey_config['combination'].lower()
        return len(combination.split('+'))
    
    def _start_hotkey_pressed(self):
        self.logger.info(f"Start hotkey pressed: {self.start_recording_hotkey}")
        if self.state_manager.get_current_state() == "idle":
            if self.state_manager.can_start_recording():
                self.state_manager.toggle_recording()
            else:
                self.logger.info("Model is not ready yet. Please wait...", extra={'user_message': True})
                self.logger.debug("Start hotkey ignored - model not ready")
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
            if self.state_manager.can_start_recording():
                self.state_manager.toggle_recording()
            else:
                self.logger.info("Model is not ready yet. Please wait...", extra={'user_message': True})
                self.logger.debug("Toggle hotkey ignored - model not ready")
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
            if sys.platform == "win32":
                if self.hotkey_bindings:
                    self.logger.debug(
                        f"[hotkeys] Registering {len(self.hotkey_bindings)} bindings: {self.hotkey_bindings}"
                    )
                    register_hotkeys(self.hotkey_bindings)
                    start_checking_hotkeys()
                if self.mode == "push_to_talk" and self._ptt_target_key is not None:
                    # Windows PTT also needs the raw press/release
                    # listener — ``global-hotkeys`` only fires on
                    # press.  pynput is shipped on Windows too
                    # (transitively via several deps), so the same
                    # code path works.
                    from pynput import keyboard as _pk

                    self._ptt_listener = _pk.Listener(
                        on_press=self._ptt_on_press,
                        on_release=self._ptt_on_release,
                    )
                    self._ptt_listener.start()
            else:
                from pynput import keyboard as _pk

                if self._pynput_callbacks:
                    self._pynput_listener = _pk.GlobalHotKeys(
                        self._pynput_callbacks,
                    )
                    self._pynput_listener.start()
                if self.mode == "push_to_talk" and self._ptt_target_key is not None:
                    self._ptt_listener = _pk.Listener(
                        on_press=self._ptt_on_press,
                        on_release=self._ptt_on_release,
                    )
                    self._ptt_listener.start()
            self.is_listening = True
            self.logger.info("Global hotkey listener active")
        except Exception as e:
            self.logger.error(f"Failed to start hotkey listener: {e}")
            raise

    def stop_listening(self):
        if not self.is_listening:
            return
        try:
            if sys.platform == "win32":
                stop_checking_hotkeys()
                # Принудительно очищаем все регистрации горячих клавиш
                try:
                    from global_hotkeys import clear_hotkeys
                    clear_hotkeys()
                    self.logger.debug("Cleared all hotkey registrations")
                except (ImportError, AttributeError):
                    # Если функция clear_hotkeys недоступна, используем альтернативный подход
                    self.logger.debug("clear_hotkeys not available, using alternative cleanup")
            else:
                if self._pynput_listener is not None:
                    self._pynput_listener.stop()
                    self._pynput_listener = None
            if self._ptt_listener is not None:
                try:
                    self._ptt_listener.stop()
                except Exception:
                    pass
                self._ptt_listener = None
            # Reset PTT state so a stop-during-hold doesn't leave the
            # next start_listening() in a stuck "_ptt_held=True" state.
            self._ptt_held = False
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

    def _convert_hotkey_to_pynput_format(self, hotkey_str: str) -> str:
        """Convert ``ctrl+f2`` → ``<ctrl>+<f2>`` for pynput's GlobalHotKeys.

        pynput expects modifiers and named keys wrapped in angle
        brackets and single character keys bare. We map the same
        aliases the Settings tab accepts (``ctrl``, ``win``, ``cmd``,
        etc.) to pynput's canonical names so the same config.yaml
        works on both platforms.
        """
        modifier_aliases = {
            'ctrl': 'ctrl',
            'control': 'ctrl',
            'shift': 'shift',
            'alt': 'alt',
            'option': 'alt',
            'win': 'cmd',
            'windows': 'cmd',
            'cmd': 'cmd',
            'command': 'cmd',
            'super': 'cmd',
        }
        named_keys = {
            'space', 'enter', 'tab', 'backspace', 'delete',
            'esc', 'escape', 'home', 'end', 'page_up', 'page_down',
            'up', 'down', 'left', 'right', 'insert',
            *(f'f{i}' for i in range(1, 25)),
        }
        parts = []
        for raw in hotkey_str.lower().split('+'):
            key = raw.strip()
            if key in modifier_aliases:
                parts.append(f"<{modifier_aliases[key]}>")
            elif key == 'esc':
                parts.append("<esc>")
            elif key in named_keys:
                parts.append(f"<{key}>")
            elif len(key) == 1:
                parts.append(key)
            else:
                # Unknown long token — let pynput surface the error
                # at registration time instead of silently passing it
                # through.
                parts.append(f"<{key}>")
        return '+'.join(parts)
    
    def change_hotkey_config(self, setting: str, value):
        valid_settings = [
            'start_recording_hotkey',
            'stop_recording_hotkey',
            'cancel_combination',
            'mode',
            'push_to_talk_key',
            'push_to_talk_min_hold_seconds',
        ]
        if setting not in valid_settings:
            raise ValueError(f"Invalid setting '{setting}'. Valid options: {valid_settings}")
        old_value = getattr(self, setting)
        if setting == "push_to_talk_key" and isinstance(value, str):
            value = value.strip().lower() or None
        if setting == "push_to_talk_min_hold_seconds":
            value = float(value)
        if setting == "mode" and value not in {"two_keys", "toggle", "push_to_talk"}:
            raise ValueError(
                f"mode must be one of two_keys / toggle / push_to_talk, got {value!r}"
            )
        if old_value == value:
            return
        setattr(self, setting, value)
        self.logger.info(f"Changed {setting}: {old_value} -> {value}")
        self.stop_listening()
        self._setup_hotkeys()
        self.start_listening()
    
    def is_active(self) -> bool:
        return self.is_listening