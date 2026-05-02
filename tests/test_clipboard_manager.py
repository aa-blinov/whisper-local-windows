import logging


def _manager():
    from app.clipboard_manager import ClipboardManager

    manager = ClipboardManager.__new__(ClipboardManager)
    manager.logger = logging.getLogger("test.clipboard")
    manager.key_simulation_delay = 0.0
    manager.auto_paste = True
    manager.preserve_clipboard = False
    return manager


def test_send_paste_combo_prefers_accessibility_path_on_macos(monkeypatch):
    import app.clipboard_manager as clipboard_module

    manager = _manager()
    calls = []
    monkeypatch.setattr(clipboard_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        manager,
        "_send_mac_accessibility_key_sequence",
        lambda steps, label: calls.append((label, steps)) or True,
    )
    monkeypatch.setattr(
        manager,
        "_send_mac_keystroke_with_cmd",
        lambda key_code, label: calls.append(("quartz", key_code, label)),
    )

    manager._send_paste_combo()

    assert calls == [
        (
            "Cmd+V",
            [
                (0, clipboard_module._MAC_KEYCODE_LEFT_COMMAND, True),
                (ord("v"), clipboard_module._MAC_KEYCODE_V, True),
                (ord("v"), clipboard_module._MAC_KEYCODE_V, False),
                (0, clipboard_module._MAC_KEYCODE_LEFT_COMMAND, False),
            ],
        )
    ]


def test_send_paste_combo_falls_back_to_quartz_on_macos(monkeypatch):
    import app.clipboard_manager as clipboard_module

    manager = _manager()
    calls = []
    monkeypatch.setattr(clipboard_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        manager,
        "_send_mac_accessibility_key_sequence",
        lambda steps, label: False,
    )
    monkeypatch.setattr(
        manager,
        "_send_mac_keystroke_with_cmd",
        lambda key_code, label: calls.append((key_code, label)),
    )

    manager._send_paste_combo()

    assert calls == [(clipboard_module._MAC_KEYCODE_V, "Cmd+V")]


def test_send_enter_prefers_accessibility_path_on_macos(monkeypatch):
    import app.clipboard_manager as clipboard_module

    manager = _manager()
    calls = []
    monkeypatch.setattr(clipboard_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        manager,
        "_send_mac_accessibility_key_sequence",
        lambda steps, label: calls.append((label, steps)) or True,
    )
    monkeypatch.setattr(
        manager,
        "_send_mac_keystroke",
        lambda key_code, label: calls.append(("quartz", key_code, label)),
    )

    manager._send_enter()

    assert calls == [
        (
            "Enter",
            [
                (0x0D, clipboard_module._MAC_KEYCODE_RETURN, True),
                (0x0D, clipboard_module._MAC_KEYCODE_RETURN, False),
            ],
        )
    ]
