"""Tests for the single-instance guard wrapper.

Windows uses ``win32event.CreateMutex`` (a kernel object); macOS /
Linux use ``filelock.FileLock`` over a cache-dir lockfile. The two
branches share the same return-shape contract — opaque handle on
success, ``None`` if another instance owns the lock or the
acquisition crashed — so each platform has its own three tests
mirroring the same behaviour matrix.
"""

import sys

import pytest


# ---- Windows path: pywin32 named mutex ------------------------------------


@pytest.fixture
def fake_win32(monkeypatch):
    """Patch win32event.CreateMutex / win32api.GetLastError for
    deterministic tests."""
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


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="pywin32 is only installed on Windows",
)
def test_win_returns_handle_when_mutex_is_fresh(fake_win32):
    fake_win32["last_error"] = 0
    from app.instance_manager import try_acquire_single_instance

    handle = try_acquire_single_instance("TestAppFresh")

    assert handle is not None
    assert "TestAppFresh" in fake_win32["create_calls"][0]


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="pywin32 is only installed on Windows",
)
def test_win_returns_none_when_another_instance_holds_mutex(fake_win32):
    fake_win32["last_error"] = 183  # ERROR_ALREADY_EXISTS
    from app.instance_manager import try_acquire_single_instance

    handle = try_acquire_single_instance("TestAppDuplicate")

    assert handle is None


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="pywin32 is only installed on Windows",
)
def test_win_returns_none_on_unexpected_exception(monkeypatch):
    import win32event

    def boom(*_args, **_kwargs):
        raise OSError("CreateMutex failed")

    monkeypatch.setattr(win32event, "CreateMutex", boom)

    from app.instance_manager import try_acquire_single_instance

    handle = try_acquire_single_instance("TestAppCrash")
    assert handle is None


# ---- macOS / Linux path: filelock lockfile --------------------------------


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Redirect ``platformdirs.user_cache_dir`` so the test never
    touches the real ``~/Library/Caches/LazyToText`` directory."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(
        "platformdirs.user_cache_dir",
        lambda *_a, **_k: str(cache),
    )
    return cache


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="filelock branch is only used on non-Windows",
)
def test_posix_returns_lock_when_lockfile_is_fresh(isolated_cache):
    from app.instance_manager import try_acquire_single_instance

    handle = try_acquire_single_instance("TestAppFreshPosix")
    try:
        assert handle is not None
        # Lock file should exist next to the resolved cache dir.
        assert (isolated_cache / "TestAppFreshPosix.lock").exists()
    finally:
        if handle is not None:
            handle.release()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="filelock branch is only used on non-Windows",
)
def test_posix_returns_none_when_another_instance_holds_lock(
    isolated_cache,
):
    from app.instance_manager import try_acquire_single_instance

    first = try_acquire_single_instance("TestAppDupPosix")
    try:
        assert first is not None
        second = try_acquire_single_instance("TestAppDupPosix")
        assert second is None
    finally:
        if first is not None:
            first.release()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="filelock branch is only used on non-Windows",
)
def test_posix_returns_none_on_unexpected_exception(
    isolated_cache, monkeypatch,
):
    """Surface OSErrors from the lock filesystem (read-only volume,
    quota exhausted, …) as ``None`` so the caller falls into the
    'another instance is running' UI branch instead of crashing."""
    import filelock

    class _BrokenLock:
        def __init__(self, *_a, **_k):
            pass

        def acquire(self, timeout=0):
            raise OSError("simulated lock filesystem failure")

    monkeypatch.setattr(filelock, "FileLock", _BrokenLock)

    from app.instance_manager import try_acquire_single_instance

    handle = try_acquire_single_instance("TestAppCrashPosix")
    assert handle is None
