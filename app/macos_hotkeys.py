"""Native macOS hotkey monitor built on top of AppKit NSEvent.

The app previously relied on ``pynput`` on macOS, which made the
permission story brittle and forced full-process restarts after the
user granted access. Here we use native Cocoa event monitors instead:

- global monitor for events going to other apps
- local monitor for events while Lazy to Text itself is focused

The callbacks are forwarded onto a dedicated worker thread so start /
stop / cancel logic stays off the GUI thread, matching the old
listener-thread behaviour.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from AppKit import (  # type: ignore[attr-defined]
    NSEvent,
    NSEventMaskFlagsChanged,
    NSEventMaskKeyDown,
    NSEventMaskKeyUp,
    NSEventModifierFlagCommand,
    NSEventModifierFlagControl,
    NSEventModifierFlagFunction,
    NSEventModifierFlagOption,
    NSEventModifierFlagShift,
    NSEventTypeFlagsChanged,
    NSEventTypeKeyDown,
)


_COMBO_MODIFIER_FLAGS = {
    "ctrl": int(NSEventModifierFlagControl),
    "shift": int(NSEventModifierFlagShift),
    "alt": int(NSEventModifierFlagOption),
    "cmd": int(NSEventModifierFlagCommand),
}
_COMBO_RELEVANT_FLAGS = 0
for _flag in _COMBO_MODIFIER_FLAGS.values():
    _COMBO_RELEVANT_FLAGS |= int(_flag)

_MODIFIER_ALIASES = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "shift": "shift",
    "alt": "alt",
    "option": "alt",
    "win": "cmd",
    "windows": "cmd",
    "cmd": "cmd",
    "command": "cmd",
    "super": "cmd",
}
_KEY_ALIASES = {
    "return": "enter",
    "esc": "esc",
    "escape": "esc",
    "pageup": "page_up",
    "pagedown": "page_down",
    "del": "delete",
}

_MAC_KEY_CODES = {
    "a": 0x00,
    "b": 0x0B,
    "c": 0x08,
    "d": 0x02,
    "e": 0x0E,
    "f": 0x03,
    "g": 0x05,
    "h": 0x04,
    "i": 0x22,
    "j": 0x26,
    "k": 0x28,
    "l": 0x25,
    "m": 0x2E,
    "n": 0x2D,
    "o": 0x1F,
    "p": 0x23,
    "q": 0x0C,
    "r": 0x0F,
    "s": 0x01,
    "t": 0x11,
    "u": 0x20,
    "v": 0x09,
    "w": 0x0D,
    "x": 0x07,
    "y": 0x10,
    "z": 0x06,
    "0": 0x1D,
    "1": 0x12,
    "2": 0x13,
    "3": 0x14,
    "4": 0x15,
    "5": 0x17,
    "6": 0x16,
    "7": 0x1A,
    "8": 0x1C,
    "9": 0x19,
    "space": 0x31,
    "enter": 0x24,
    "tab": 0x30,
    "backspace": 0x33,
    "delete": 0x75,
    "insert": 0x72,
    "home": 0x73,
    "end": 0x77,
    "page_up": 0x74,
    "page_down": 0x79,
    "up": 0x7E,
    "down": 0x7D,
    "left": 0x7B,
    "right": 0x7C,
    "esc": 0x35,
    "f1": 0x7A,
    "f2": 0x78,
    "f3": 0x63,
    "f4": 0x76,
    "f5": 0x60,
    "f6": 0x61,
    "f7": 0x62,
    "f8": 0x64,
    "f9": 0x65,
    "f10": 0x6D,
    "f11": 0x67,
    "f12": 0x6F,
    "f13": 0x69,
    "f14": 0x6B,
    "f15": 0x71,
    "f16": 0x6A,
    "f17": 0x40,
    "f18": 0x4F,
    "f19": 0x50,
    "f20": 0x5A,
}

_PTT_KEY_SPECS = {
    "right_cmd": (0x36, int(NSEventModifierFlagCommand)),
    "cmd_r": (0x36, int(NSEventModifierFlagCommand)),
    "right_command": (0x36, int(NSEventModifierFlagCommand)),
    "command_r": (0x36, int(NSEventModifierFlagCommand)),
    "right_win": (0x36, int(NSEventModifierFlagCommand)),
    "win_r": (0x36, int(NSEventModifierFlagCommand)),
    "right_windows": (0x36, int(NSEventModifierFlagCommand)),
    "windows_r": (0x36, int(NSEventModifierFlagCommand)),
    "right_super": (0x36, int(NSEventModifierFlagCommand)),
    "super_r": (0x36, int(NSEventModifierFlagCommand)),
    "right_option": (0x3D, int(NSEventModifierFlagOption)),
    "option_r": (0x3D, int(NSEventModifierFlagOption)),
    "right_alt": (0x3D, int(NSEventModifierFlagOption)),
    "alt_r": (0x3D, int(NSEventModifierFlagOption)),
    "right_shift": (0x3C, int(NSEventModifierFlagShift)),
    "shift_r": (0x3C, int(NSEventModifierFlagShift)),
    "right_ctrl": (0x3E, int(NSEventModifierFlagControl)),
    "ctrl_r": (0x3E, int(NSEventModifierFlagControl)),
    "right_control": (0x3E, int(NSEventModifierFlagControl)),
    "control_r": (0x3E, int(NSEventModifierFlagControl)),
    "fn": (0x3F, int(NSEventModifierFlagFunction)),
}


@dataclass(frozen=True)
class MacHotkeySpec:
    raw: str
    key_code: int
    required_flags: int
    callback: Callable[[], None]
    name: str


@dataclass(frozen=True)
class MacPushToTalkSpec:
    raw: str
    key_code: int
    pressed_flag: int


def parse_hotkey_spec(
    hotkey: str,
    callback: Callable[[], None],
    *,
    name: str,
) -> MacHotkeySpec:
    raw_parts = [part.strip().lower() for part in hotkey.split("+")]
    parts = [
        _MODIFIER_ALIASES.get(part, _KEY_ALIASES.get(part, part))
        for part in raw_parts
    ]
    *modifiers, key = parts
    required_flags = 0
    for modifier in modifiers:
        required_flags |= _COMBO_MODIFIER_FLAGS[modifier]
    key_code = _MAC_KEY_CODES[key]
    return MacHotkeySpec(
        raw=hotkey,
        key_code=int(key_code),
        required_flags=int(required_flags),
        callback=callback,
        name=name,
    )


def resolve_push_to_talk_spec(name: str | None) -> Optional[MacPushToTalkSpec]:
    if not name:
        return None
    resolved = _PTT_KEY_SPECS.get(name.strip().lower())
    if resolved is None:
        return None
    key_code, pressed_flag = resolved
    return MacPushToTalkSpec(
        raw=name,
        key_code=int(key_code),
        pressed_flag=int(pressed_flag),
    )


class MacHotkeyMonitor:
    def __init__(
        self,
        hotkeys: list[MacHotkeySpec],
        *,
        push_to_talk: Optional[MacPushToTalkSpec] = None,
        on_push_to_talk_press: Optional[Callable[[], None]] = None,
        on_push_to_talk_release: Optional[Callable[[], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._hotkeys = list(hotkeys)
        self._push_to_talk = push_to_talk
        self._on_push_to_talk_press = on_push_to_talk_press
        self._on_push_to_talk_release = on_push_to_talk_release
        self._logger = logger or logging.getLogger(__name__)
        self._global_monitor = None
        self._local_monitor = None
        self._callback_queue: Optional[queue.SimpleQueue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._ptt_held = False
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._callback_queue = queue.SimpleQueue()
        self._worker_thread = threading.Thread(
            target=self._drain_callback_queue,
            daemon=True,
            name="macos-hotkeys",
        )
        self._worker_thread.start()

        mask = (
            int(NSEventMaskKeyDown)
            | int(NSEventMaskKeyUp)
            | int(NSEventMaskFlagsChanged)
        )
        self._global_monitor = (
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                mask,
                self._handle_global_event,
            )
        )
        self._local_monitor = (
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                mask,
                self._handle_local_event,
            )
        )
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        if self._global_monitor is not None:
            NSEvent.removeMonitor_(self._global_monitor)
            self._global_monitor = None
        if self._local_monitor is not None:
            NSEvent.removeMonitor_(self._local_monitor)
            self._local_monitor = None
        self._ptt_held = False
        if self._callback_queue is not None:
            self._callback_queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
        self._callback_queue = None
        self._started = False

    def _drain_callback_queue(self) -> None:
        assert self._callback_queue is not None
        while True:
            callback = self._callback_queue.get()
            if callback is None:
                return
            try:
                callback()
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.warning(
                    "macOS hotkey callback raised: %s", exc,
                )

    def _enqueue(self, callback: Optional[Callable[[], None]]) -> None:
        if callback is None or self._callback_queue is None:
            return
        self._callback_queue.put(callback)

    def _handle_global_event(self, event) -> None:
        self._handle_event(event)

    def _handle_local_event(self, event):
        self._handle_event(event)
        return event

    def _handle_event(self, event) -> None:
        try:
            event_type = int(event.type())
            key_code = int(event.keyCode())
            flags = int(event.modifierFlags())
        except Exception:
            return

        if event_type == int(NSEventTypeFlagsChanged):
            self._handle_flags_changed(key_code, flags)
            return

        if event_type != int(NSEventTypeKeyDown):
            return
        if self._is_repeat(event):
            return

        normalized_flags = self._normalize_combo_flags(flags)
        for spec in self._hotkeys:
            if (
                key_code == spec.key_code
                and normalized_flags == spec.required_flags
            ):
                self._enqueue(spec.callback)
                break

    def _handle_flags_changed(self, key_code: int, flags: int) -> None:
        if self._push_to_talk is None:
            return
        if key_code != self._push_to_talk.key_code:
            return
        pressed = bool(flags & self._push_to_talk.pressed_flag)
        if pressed and not self._ptt_held:
            self._ptt_held = True
            self._enqueue(self._on_push_to_talk_press)
        elif not pressed and self._ptt_held:
            self._ptt_held = False
            self._enqueue(self._on_push_to_talk_release)

    @staticmethod
    def _normalize_combo_flags(flags: int) -> int:
        return int(flags) & int(_COMBO_RELEVANT_FLAGS)

    @staticmethod
    def _is_repeat(event) -> bool:
        repeat = getattr(event, "isARepeat", None)
        if callable(repeat):
            try:
                return bool(repeat())
            except Exception:
                return False
        return False
