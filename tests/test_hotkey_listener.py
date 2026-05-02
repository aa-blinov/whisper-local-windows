import importlib
import logging
import sys
import types


def test_patch_pynput_darwin_listener_skips_keycode_context(monkeypatch):
    hotkey_listener = importlib.import_module("app.hotkey_listener")

    class FakeListenerMixin:
        def _run(self):
            self.mixin_runs = getattr(self, "mixin_runs", 0) + 1

    class FakeListener(FakeListenerMixin):
        def __init__(self):
            self._context = "live"
            self.original_runs = 0
            self.mixin_runs = 0

        def _run(self):
            self.original_runs += 1

    class FakeGlobalHotKeys:
        def __init__(self):
            self.press_calls = []
            self.release_calls = []

        def _on_press(self, key, injected):
            self.press_calls.append((key, injected))

        def _on_release(self, key, injected):
            self.release_calls.append((key, injected))

    fake_pynput = types.ModuleType("pynput")
    fake_keyboard = types.ModuleType("pynput.keyboard")
    fake_darwin = types.ModuleType("pynput.keyboard._darwin")
    fake_pynput.keyboard = fake_keyboard
    fake_keyboard._darwin = fake_darwin
    fake_keyboard.GlobalHotKeys = FakeGlobalHotKeys
    fake_darwin.Listener = FakeListener
    fake_darwin.ListenerMixin = FakeListenerMixin

    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", fake_keyboard)
    monkeypatch.setitem(sys.modules, "pynput.keyboard._darwin", fake_darwin)
    monkeypatch.setattr(hotkey_listener.sys, "platform", "darwin")

    hotkey_listener._patch_pynput_darwin_listener()
    patched_run = FakeListener._run
    hotkey_listener._patch_pynput_darwin_listener()

    listener = FakeListener()
    listener._run()
    global_hotkeys = FakeGlobalHotKeys()
    global_hotkeys._on_press("volume_up")
    global_hotkeys._on_release("volume_up")

    assert FakeListener._run is patched_run
    assert listener._context is None
    assert listener.original_runs == 0
    assert listener.mixin_runs == 1
    assert getattr(FakeListener, "_lazy_to_text_skip_keycode_context", False) is True
    assert (
        getattr(
            FakeGlobalHotKeys,
            "_lazy_to_text_optional_injected_callbacks",
            False,
        )
        is True
    )
    assert global_hotkeys.press_calls == [("volume_up", False)]
    assert global_hotkeys.release_calls == [("volume_up", False)]


def test_stop_listening_joins_pynput_threads(monkeypatch):
    hotkey_listener = importlib.import_module("app.hotkey_listener")

    class FakeThread:
        def __init__(self):
            self.stop_calls = 0
            self.join_calls = []

        def stop(self):
            self.stop_calls += 1

        def join(self, timeout=None):
            self.join_calls.append(timeout)

    global_thread = FakeThread()
    ptt_thread = FakeThread()

    listener = hotkey_listener.HotkeyListener.__new__(
        hotkey_listener.HotkeyListener
    )
    listener.logger = logging.getLogger("tests.hotkey_listener")
    listener.is_listening = True
    listener._pynput_listener = global_thread
    listener._ptt_listener = ptt_thread
    listener._ptt_held = True

    monkeypatch.setattr(hotkey_listener.sys, "platform", "darwin")

    listener.stop_listening()

    assert global_thread.stop_calls == 1
    assert global_thread.join_calls == [1.0]
    assert ptt_thread.stop_calls == 1
    assert ptt_thread.join_calls == [1.0]
    assert listener._pynput_listener is None
    assert listener._ptt_listener is None
    assert listener._ptt_held is False
    assert listener.is_listening is False


def test_windows_push_to_talk_uses_global_hotkeys_callbacks(monkeypatch):
    hotkey_listener = importlib.import_module("app.hotkey_listener")

    registered = []
    started = []

    class FakeStateManager:
        def get_current_state(self):
            return "idle"

        def can_start_recording(self):
            return True

        def toggle_recording(self):
            pass

        def stop_recording(self, use_auto_enter=False):
            pass

        def cancel_recording_hotkey_pressed(self):
            pass

    def fake_register_hotkeys(bindings):
        registered.append(bindings)

    def fake_start_checking_hotkeys():
        started.append(True)

    monkeypatch.setattr(hotkey_listener.sys, "platform", "win32")
    monkeypatch.setattr(
        hotkey_listener, "register_hotkeys", fake_register_hotkeys
    )
    monkeypatch.setattr(
        hotkey_listener, "start_checking_hotkeys", fake_start_checking_hotkeys
    )

    listener = hotkey_listener.HotkeyListener.__new__(
        hotkey_listener.HotkeyListener
    )
    listener.state_manager = FakeStateManager()
    listener.start_recording_hotkey = "ctrl+f2"
    listener.stop_recording_hotkey = "ctrl+f3"
    listener.cancel_combination = "esc"
    listener.mode = "push_to_talk"
    listener.push_to_talk_key = "right_alt"
    listener.push_to_talk_min_hold_seconds = 0.2
    listener.is_listening = False
    listener.logger = logging.getLogger("tests.hotkey_listener.win32")
    listener._pynput_listener = None
    listener._pynput_callbacks = {}
    listener._ptt_listener = None
    listener._ptt_windows_binding = None
    listener._ptt_target_key = None
    listener._ptt_held = False
    listener._ptt_press_ts = 0.0

    listener._setup_hotkeys()
    listener.start_listening()

    assert started == [True]
    assert len(registered) == 1
    assert [binding[0] for binding in registered[0]] == [
        "right_menu",
        "escape",
    ]
    assert registered[0][0][1] == listener._ptt_press
    assert registered[0][0][2] == listener._ptt_release
    assert listener._ptt_listener is None
    assert listener.is_listening is True


def test_windows_hotkey_converter_normalises_documented_aliases():
    hotkey_listener = importlib.import_module("app.hotkey_listener")
    listener = hotkey_listener.HotkeyListener.__new__(
        hotkey_listener.HotkeyListener
    )

    assert (
        listener._convert_hotkey_to_global_hotkeys_format("ctrl+return")
        == "control + enter"
    )
    assert (
        listener._convert_hotkey_to_global_hotkeys_format("ctrl+pageup")
        == "control + page_up"
    )
    assert (
        listener._convert_hotkey_to_global_hotkeys_format("right_alt")
        == "right_menu"
    )
    assert listener._resolve_windows_ptt_binding("right_menu") == "right_menu"
