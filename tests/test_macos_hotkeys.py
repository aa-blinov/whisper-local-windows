import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="macOS-native hotkey monitor is only available on Darwin",
)


class _FakeEvent:
    def __init__(
        self,
        event_type: int,
        key_code: int,
        flags: int,
        *,
        is_repeat: bool = False,
    ) -> None:
        self._event_type = event_type
        self._key_code = key_code
        self._flags = flags
        self._is_repeat = is_repeat

    def type(self):
        return self._event_type

    def keyCode(self):
        return self._key_code

    def modifierFlags(self):
        return self._flags

    def isARepeat(self):
        return self._is_repeat


def test_parse_hotkey_spec_maps_ctrl_f8():
    from app.macos_hotkeys import (
        _COMBO_MODIFIER_FLAGS,
        parse_hotkey_spec,
    )

    spec = parse_hotkey_spec("ctrl+f8", lambda: None, name="start")

    assert spec.key_code == 0x64
    assert spec.required_flags == _COMBO_MODIFIER_FLAGS["ctrl"]
    assert spec.name == "start"


def test_macos_hotkey_monitor_handles_combo_and_ptt():
    from app.macos_hotkeys import (
        MacHotkeyMonitor,
        _COMBO_MODIFIER_FLAGS,
        parse_hotkey_spec,
        resolve_push_to_talk_spec,
    )

    calls: list[str] = []
    ptt_spec = resolve_push_to_talk_spec("right_cmd")
    assert ptt_spec is not None
    monitor = MacHotkeyMonitor(
        [
            parse_hotkey_spec(
                "ctrl+f8",
                lambda: calls.append("start"),
                name="start",
            )
        ],
        push_to_talk=ptt_spec,
        on_push_to_talk_press=lambda: calls.append("ptt_press"),
        on_push_to_talk_release=lambda: calls.append("ptt_release"),
    )
    monitor._enqueue = lambda callback: callback() if callback else None

    monitor._handle_event(
        _FakeEvent(
            event_type=10,
            key_code=0x64,
            flags=_COMBO_MODIFIER_FLAGS["ctrl"],
        )
    )
    monitor._handle_event(
        _FakeEvent(
            event_type=10,
            key_code=0x64,
            flags=_COMBO_MODIFIER_FLAGS["ctrl"],
            is_repeat=True,
        )
    )
    monitor._handle_event(
        _FakeEvent(
            event_type=12,
            key_code=0x36,
            flags=ptt_spec.pressed_flag,
        )
    )
    monitor._handle_event(
        _FakeEvent(
            event_type=12,
            key_code=0x36,
            flags=ptt_spec.pressed_flag,
        )
    )
    monitor._handle_event(
        _FakeEvent(
            event_type=12,
            key_code=0x36,
            flags=0,
        )
    )

    assert calls == ["start", "ptt_press", "ptt_release"]
