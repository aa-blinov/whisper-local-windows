import importlib
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
# ``_resolve_pynput_ptt_key`` so the module import path doesn't pull in
# pynput on Windows where the legacy ``global-hotkeys`` listener
# is used instead.
_PTT_KEY_TO_PYNPUT: dict[str, str] = {
    "right_cmd": "cmd_r",
    "cmd_r": "cmd_r",
    "right_command": "cmd_r",
    "command_r": "cmd_r",
    "right_win": "cmd_r",
    "win_r": "cmd_r",
    "right_windows": "cmd_r",
    "windows_r": "cmd_r",
    "right_super": "cmd_r",
    "super_r": "cmd_r",
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

_PTT_KEY_TO_WINDOWS: dict[str, str] = {
    "right_cmd": "right_window",
    "cmd_r": "right_window",
    "right_command": "right_window",
    "command_r": "right_window",
    "right_win": "right_window",
    "win_r": "right_window",
    "right_windows": "right_window",
    "windows_r": "right_window",
    "right_super": "right_window",
    "super_r": "right_window",
    "right_window": "right_window",
    "right_option": "right_menu",
    "option_r": "right_menu",
    "right_alt": "right_menu",
    "alt_r": "right_menu",
    "right_menu": "right_menu",
    "right_shift": "right_shift",
    "shift_r": "right_shift",
    "right_ctrl": "right_control",
    "ctrl_r": "right_control",
    "right_control": "right_control",
    "control_r": "right_control",
}


def _patch_pynput_darwin_listener() -> None:
    """Skip pynput's unused ``keycode_context`` on macOS.

    Recent macOS builds can abort the whole process with
    ``dispatch_assert_queue_fail`` when ``pynput`` calls
    ``TISGetInputSourceProperty`` from the listener thread during
    startup. ``pynput`` 1.8.x stores the resulting context on
    ``self._context`` but the Darwin keyboard listener never reads it
    afterwards, so we can safely bypass that preflight and jump
    straight to the real event-tap loop.
    """
    if sys.platform != "darwin":
        return
    try:
        _pk_darwin = importlib.import_module("pynput.keyboard._darwin")
    except ImportError:
        return

    listener_cls = _pk_darwin.Listener
    if getattr(listener_cls, "_lazy_to_text_skip_keycode_context", False):
        return

    def _run_without_keycode_context(self):
        self._context = None
        return _pk_darwin.ListenerMixin._run(self)

    listener_cls._lazy_to_text_original_run = listener_cls._run
    listener_cls._lazy_to_text_skip_keycode_context = True
    listener_cls._run = _run_without_keycode_context
    logging.getLogger(__name__).debug(
        "Patched pynput Darwin listener to skip keycode_context()"
    )


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
          ``pynput.keyboard.Listener`` on macOS/Linux and the
          documented ``global_hotkeys`` press/release callbacks on
          Windows, because that platform already routes the normal
          hotkey path through ``global-hotkeys``.
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
        # Non-Windows push-to-talk: separate ``pynput.keyboard.Listener``
        # that tracks raw on_press / on_release for the configured key.
        # Lives alongside the GlobalHotKeys instance (which registers
        # the cancel combo, if any).
        self._ptt_listener = None
        self._ptt_windows_binding = None
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
            # separately on macOS/Linux via ``pynput``;  on Windows
            # we keep everything inside ``global-hotkeys`` and bind a
            # press + release callback pair for the PTT key itself.
            self._ptt_windows_binding = None
            self._ptt_target_key = None
            if sys.platform == "win32":
                self._ptt_windows_binding = self._resolve_windows_ptt_binding(
                    self.push_to_talk_key
                )
                ptt_available = self._ptt_windows_binding is not None
            else:
                self._ptt_target_key = self._resolve_pynput_ptt_key(
                    self.push_to_talk_key
                )
                ptt_available = self._ptt_target_key is not None
            if not ptt_available:
                self.logger.warning(
                    "Push-to-talk key %r is not in the recognised "
                    "vocabulary (%s) for %s — PTT will be inactive.",
                    self.push_to_talk_key,
                    sorted(
                        _PTT_KEY_TO_WINDOWS.keys()
                        if sys.platform == "win32"
                        else _PTT_KEY_TO_PYNPUT.keys()
                    ),
                    sys.platform,
                )
            else:
                self.logger.info(
                    "Configured push-to-talk: hold %r (%.0f ms minimum)",
                    self.push_to_talk_key,
                    self.push_to_talk_min_hold_seconds * 1000,
                )
                if sys.platform == "win32":
                    hotkey_configs.append({
                        'combination': self.push_to_talk_key,
                        'callback': self._ptt_press,
                        'release_callback': self._ptt_release,
                        'name': 'push_to_talk',
                    })
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

    def _resolve_pynput_ptt_key(self, name: str | None):
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

    def _resolve_windows_ptt_binding(self, name: str | None) -> str | None:
        """Translate a user-facing PTT key name to a documented
        ``global-hotkeys`` Windows key token such as
        ``"right_menu"`` or ``"right_control"``.
        """
        if not name:
            return None
        return _PTT_KEY_TO_WINDOWS.get(name.strip().lower())

    # ---- push-to-talk callbacks (called from listener thread) ---------------

    def _ptt_press(self) -> None:
        """Start recording on the first press of the configured PTT
        key. Called either by ``pynput`` (after key filtering) or by
        the Windows ``global-hotkeys`` press callback.
        """
        if self._ptt_held:
            return  # auto-repeat / duplicate callback
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

    def _ptt_release(self) -> None:
        """Stop recording when the held PTT key is released.

        Called either by ``pynput`` (after key filtering) or by the
        Windows ``global-hotkeys`` release callback.
        """
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

    def _ptt_on_press(self, key) -> None:
        """Listener callback — start recording on the first press of
        the PTT key.  Auto-repeat presses while the key is held are
        ignored (``_ptt_held`` guard).
        """
        if self._ptt_target_key is None or key != self._ptt_target_key:
            return
        self._ptt_press()

    def _ptt_on_release(self, key) -> None:
        """Listener callback — stop recording when the PTT key is
        released.  Holds shorter than ``push_to_talk_min_hold_seconds``
        are treated as accidental taps:  cancel the recording so the
        history doesn't fill up with empty entries.
        """
        if self._ptt_target_key is None or key != self._ptt_target_key:
            return
        self._ptt_release()
    
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

    def _stop_pynput_thread(self, listener, label: str) -> None:
        """Best-effort stop + join for pynput threads before rebinding."""
        if listener is None:
            return
        try:
            listener.stop()
        except Exception as exc:
            self.logger.debug("Stopping %s listener raised: %s", label, exc)
        join = getattr(listener, "join", None)
        if callable(join):
            try:
                join(timeout=1.0)
            except Exception as exc:
                self.logger.debug("Joining %s listener raised: %s", label, exc)
    
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
            else:
                _patch_pynput_darwin_listener()
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
                    self._stop_pynput_thread(
                        self._pynput_listener, "global hotkey"
                    )
                    self._pynput_listener = None
            if self._ptt_listener is not None:
                self._stop_pynput_thread(
                    self._ptt_listener, "push-to-talk"
                )
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
            'control': 'control',
            'shift': 'shift',
            'alt': 'alt',
            'option': 'alt',
            'win': 'window',
            'windows': 'window',
            'cmd': 'window',
            'command': 'window',
            'super': 'window',
            'space': 'space',
            'enter': 'enter',
            'return': 'enter',
            'esc': 'escape',
            'escape': 'escape',
            'pageup': 'page_up',
            'pagedown': 'page_down',
            'del': 'delete',
            'left_cmd': 'left_window',
            'cmd_l': 'left_window',
            'left_command': 'left_window',
            'command_l': 'left_window',
            'left_win': 'left_window',
            'win_l': 'left_window',
            'left_windows': 'left_window',
            'windows_l': 'left_window',
            'left_super': 'left_window',
            'super_l': 'left_window',
            'right_cmd': 'right_window',
            'cmd_r': 'right_window',
            'right_command': 'right_window',
            'command_r': 'right_window',
            'right_win': 'right_window',
            'win_r': 'right_window',
            'right_windows': 'right_window',
            'windows_r': 'right_window',
            'right_super': 'right_window',
            'super_r': 'right_window',
            'left_option': 'left_menu',
            'option_l': 'left_menu',
            'left_alt': 'left_menu',
            'alt_l': 'left_menu',
            'right_option': 'right_menu',
            'option_r': 'right_menu',
            'right_alt': 'right_menu',
            'alt_r': 'right_menu',
            'left_ctrl': 'left_control',
            'ctrl_l': 'left_control',
            'left_control': 'left_control',
            'control_l': 'left_control',
            'right_ctrl': 'right_control',
            'ctrl_r': 'right_control',
            'right_control': 'right_control',
            'control_r': 'right_control',
            'left_shift': 'left_shift',
            'shift_l': 'left_shift',
            'right_shift': 'right_shift',
            'shift_r': 'right_shift',
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
        key_aliases = {
            'return': 'enter',
            'esc': 'esc',
            'escape': 'esc',
            'pageup': 'page_up',
            'pagedown': 'page_down',
            'del': 'delete',
        }
        named_keys = {
            'space', 'enter', 'tab', 'backspace', 'delete',
            'esc', 'home', 'end', 'page_up', 'page_down',
            'up', 'down', 'left', 'right', 'insert',
            *(f'f{i}' for i in range(1, 25)),
        }
        parts = []
        for raw in hotkey_str.lower().split('+'):
            key = key_aliases.get(raw.strip(), raw.strip())
            if key in modifier_aliases:
                parts.append(f"<{modifier_aliases[key]}>")
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
