"""Tests for the single-instance guard wrapper."""

import pytest


@pytest.fixture
def fake_win32(monkeypatch):
    """Patch win32event.CreateMutex / win32api.GetLastError for deterministic tests."""
    import win32api
    import win32event

    state = {"last_error": 0, "create_calls": []}

    class _Handle:
        def __init__(self, name: str) -> None:
            self.name = name

    def fake_create(_sec, _initial, name):
        state["create_calls"].append(name)
        return _Handle(name)

    def fake_last_error():
        return state["last_error"]

    monkeypatch.setattr(win32event, "CreateMutex", fake_create)
    monkeypatch.setattr(win32api, "GetLastError", fake_last_error)
    return state


def test_returns_handle_when_mutex_is_fresh(fake_win32):
    fake_win32["last_error"] = 0
    from app.instance_manager import try_acquire_single_instance

    handle = try_acquire_single_instance("TestAppFresh")

    assert handle is not None
    assert "TestAppFresh" in fake_win32["create_calls"][0]


def test_returns_none_when_another_instance_holds_mutex(fake_win32):
    fake_win32["last_error"] = 183  # ERROR_ALREADY_EXISTS
    from app.instance_manager import try_acquire_single_instance

    handle = try_acquire_single_instance("TestAppDuplicate")

    assert handle is None


def test_returns_none_on_unexpected_exception(monkeypatch):
    import win32event

    def boom(*_args, **_kwargs):
        raise OSError("CreateMutex failed")

    monkeypatch.setattr(win32event, "CreateMutex", boom)

    from app.instance_manager import try_acquire_single_instance

    handle = try_acquire_single_instance("TestAppCrash")
    assert handle is None
