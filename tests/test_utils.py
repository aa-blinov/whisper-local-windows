"""Tests for cache helpers in app.utils.

Both ``is_*_cached`` (already used in production) and the new
``delete_*`` family share the same on-disk layout, so they're
covered side by side here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---- Helpers ----------------------------------------------------------------


def _make_hf_snapshot(hub_root: Path, canonical: str) -> Path:
    """Lay out a minimal HF hub directory tree mirroring what
    ``huggingface_hub`` writes after a successful snapshot download."""
    repo_dir = hub_root / f"models--{canonical.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots" / "deadbeef"
    snap_dir.mkdir(parents=True)
    (snap_dir / "model.bin").write_bytes(b"fake weights")
    (snap_dir / "config.json").write_text("{}")
    blobs = repo_dir / "blobs"
    blobs.mkdir()
    (blobs / "abc123").write_bytes(b"fake blob")
    refs = repo_dir / "refs"
    refs.mkdir()
    (refs / "main").write_text("deadbeef")
    return repo_dir


def _make_gigaam_ckpt(home_dir: Path, model_name: str) -> Path:
    cache_dir = home_dir / ".cache" / "gigaam"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cache_dir / f"{model_name}.ckpt"
    ckpt.write_bytes(b"x" * 1024)
    return ckpt


# ---- delete_cached_model (faster-whisper / HF hub) --------------------------


def test_delete_cached_model_removes_existing_repo_dir(tmp_path, monkeypatch):
    """A cached model — entire ``models--…`` subtree (snapshots, blobs,
    refs) — must be removed in one shot."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    canonical = "Systran/faster-whisper-large-v3"
    repo_dir = _make_hf_snapshot(tmp_path / "hub", canonical)
    assert repo_dir.exists()

    from app.utils import delete_cached_model

    assert delete_cached_model(canonical) is True
    assert not repo_dir.exists()


def test_delete_cached_model_returns_false_when_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    from app.utils import delete_cached_model

    assert delete_cached_model("ghost/never-downloaded") is False


def test_delete_cached_model_returns_false_for_empty_canonical(tmp_path, monkeypatch):
    """Empty inputs are a no-op rather than raising — caller may pass
    ``info.canonical`` from a partially-populated info object."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    from app.utils import delete_cached_model

    assert delete_cached_model("") is False


def test_delete_cached_model_falls_back_to_default_hf_cache(tmp_path, monkeypatch):
    """When ``HF_HOME`` is unset the helper should look at
    ``~/.cache/huggingface/hub`` — same fallback used by the
    ``is_model_cached`` reader."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    canonical = "openai/whisper-tiny"
    repo_dir = _make_hf_snapshot(tmp_path / ".cache" / "huggingface" / "hub", canonical)

    from app.utils import delete_cached_model

    assert delete_cached_model(canonical) is True
    assert not repo_dir.exists()


def test_delete_cached_model_does_not_touch_sibling_repos(tmp_path, monkeypatch):
    """Deleting one model must leave every other ``models--…`` dir
    in the hub cache untouched."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    target = _make_hf_snapshot(tmp_path / "hub", "owner/target")
    sibling = _make_hf_snapshot(tmp_path / "hub", "owner/sibling")

    from app.utils import delete_cached_model

    assert delete_cached_model("owner/target") is True
    assert not target.exists()
    assert sibling.exists()
    assert (sibling / "snapshots" / "deadbeef" / "model.bin").exists()


# ---- delete_gigaam_cached ---------------------------------------------------


def test_delete_gigaam_cached_removes_ckpt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ckpt = _make_gigaam_ckpt(tmp_path, "v3_e2e_ctc")
    assert ckpt.exists()

    from app.utils import delete_gigaam_cached

    assert delete_gigaam_cached("v3_e2e_ctc") is True
    assert not ckpt.exists()


def test_delete_gigaam_cached_returns_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from app.utils import delete_gigaam_cached

    assert delete_gigaam_cached("never-downloaded") is False


def test_delete_gigaam_cached_returns_false_for_empty_name(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from app.utils import delete_gigaam_cached

    assert delete_gigaam_cached("") is False


def test_delete_gigaam_cached_does_not_touch_other_ckpts(tmp_path, monkeypatch):
    """Each GigaAM model is a single ``.ckpt`` next to its siblings —
    deleting one must keep the others intact."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    target = _make_gigaam_ckpt(tmp_path, "v3_e2e_ctc")
    sibling = _make_gigaam_ckpt(tmp_path, "v3_e2e_rnnt")

    from app.utils import delete_gigaam_cached

    assert delete_gigaam_cached("v3_e2e_ctc") is True
    assert not target.exists()
    assert sibling.exists()


# ---- delete_cached_for_info dispatcher -------------------------------------


class _FakeInfo:
    def __init__(self, canonical: str, backend_kind: str = "faster_whisper") -> None:
        self.canonical = canonical
        self.backend_kind = backend_kind


def test_delete_cached_for_info_routes_to_hf_for_faster_whisper(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    canonical = "Systran/faster-whisper-large-v3"
    repo_dir = _make_hf_snapshot(tmp_path / "hub", canonical)

    from app.utils import delete_cached_for_info

    info = _FakeInfo(canonical=canonical, backend_kind="faster_whisper")
    assert delete_cached_for_info(info) is True
    assert not repo_dir.exists()


def test_delete_cached_for_info_routes_to_gigaam(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ckpt = _make_gigaam_ckpt(tmp_path, "v3_e2e_ctc")

    from app.utils import delete_cached_for_info

    info = _FakeInfo(canonical="v3_e2e_ctc", backend_kind="gigaam")
    assert delete_cached_for_info(info) is True
    assert not ckpt.exists()


def test_delete_cached_for_info_returns_false_when_nothing_to_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    from app.utils import delete_cached_for_info

    info = _FakeInfo(canonical="ghost/x", backend_kind="faster_whisper")
    assert delete_cached_for_info(info) is False


# ---- Sanity: existing readers still see what we just removed ---------------


def test_is_model_cached_then_delete_then_is_not_cached(tmp_path, monkeypatch):
    """End-to-end round-trip: the existing ``is_model_cached`` reader
    must agree with the deletion outcome, otherwise the UI would flash
    'Download' then re-flip back to 'Select' on the next refresh."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    canonical = "Systran/faster-whisper-large-v3"
    _make_hf_snapshot(tmp_path / "hub", canonical)

    from app.utils import delete_cached_model, is_model_cached

    assert is_model_cached(canonical) is True
    assert delete_cached_model(canonical) is True
    assert is_model_cached(canonical) is False


def test_is_gigaam_cached_then_delete_then_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _make_gigaam_ckpt(tmp_path, "v3_e2e_ctc")

    from app.utils import delete_gigaam_cached, is_gigaam_cached

    assert is_gigaam_cached("v3_e2e_ctc") is True
    assert delete_gigaam_cached("v3_e2e_ctc") is True
    assert is_gigaam_cached("v3_e2e_ctc") is False


# ---- GIGAAM_MODELS_DIR override --------------------------------------------


def test_is_gigaam_cached_honours_env_var(tmp_path, monkeypatch):
    """When ``GIGAAM_MODELS_DIR`` is set, the reader must look there
    instead of the default ``~/.cache/gigaam`` — same contract as
    ``HF_HOME`` for faster-whisper."""
    custom_root = tmp_path / "custom-models" / "gigaam"
    custom_root.mkdir(parents=True)
    (custom_root / "v3_e2e_ctc.ckpt").write_bytes(b"x" * 1024)
    monkeypatch.setenv("GIGAAM_MODELS_DIR", str(custom_root))
    # Make sure the default path is empty so we know we're reading the
    # env-pointed one, not the default.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake-home")

    from app.utils import is_gigaam_cached

    assert is_gigaam_cached("v3_e2e_ctc") is True


def test_is_gigaam_cached_env_var_misses_when_dir_empty(tmp_path, monkeypatch):
    """Env var set but the file isn't there → False, no fallback to
    the default path (otherwise the user would be confused why the
    card says 'Select' and then tries to download to the configured
    location and the existing copy in ~/.cache is ignored)."""
    custom_root = tmp_path / "custom-models" / "gigaam"
    custom_root.mkdir(parents=True)
    monkeypatch.setenv("GIGAAM_MODELS_DIR", str(custom_root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake-home")
    # Place a ckpt in the *default* location — must not be picked up
    # because the env var redirects.
    (tmp_path / "fake-home" / ".cache" / "gigaam").mkdir(parents=True)
    (tmp_path / "fake-home" / ".cache" / "gigaam" / "v3_e2e_ctc.ckpt").write_bytes(b"x")

    from app.utils import is_gigaam_cached

    assert is_gigaam_cached("v3_e2e_ctc") is False


def test_delete_gigaam_cached_honours_env_var(tmp_path, monkeypatch):
    """The deleter mirrors the reader — when ``GIGAAM_MODELS_DIR`` is
    set, deletion targets that directory."""
    custom_root = tmp_path / "custom-models" / "gigaam"
    custom_root.mkdir(parents=True)
    ckpt = custom_root / "v3_e2e_ctc.ckpt"
    ckpt.write_bytes(b"x" * 1024)
    monkeypatch.setenv("GIGAAM_MODELS_DIR", str(custom_root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake-home")

    from app.utils import delete_gigaam_cached

    assert delete_gigaam_cached("v3_e2e_ctc") is True
    assert not ckpt.exists()


def test_gigaam_helpers_unset_env_var_uses_default(tmp_path, monkeypatch):
    """Env var unset → fall back to ``~/.cache/gigaam``. Existing
    deployments must keep finding their old downloads."""
    monkeypatch.delenv("GIGAAM_MODELS_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _make_gigaam_ckpt(tmp_path, "v3_e2e_ctc")

    from app.utils import is_gigaam_cached

    assert is_gigaam_cached("v3_e2e_ctc") is True


# ---- get_models_root ------------------------------------------------------


def test_get_models_root_returns_configured_value(tmp_path):
    """The helper used by ``app.py`` to set HF_HOME / GIGAAM_MODELS_DIR
    must echo back whatever the user picked, with no surprise
    rewriting (e.g. appending ``/hub`` or normalising case)."""
    from app.utils import get_models_root

    custom = str(tmp_path / "drive-d-models")
    assert get_models_root(custom) == custom


def test_get_models_root_returns_default_when_empty(tmp_path, monkeypatch):
    """Empty / None / whitespace input → caller's default — same path
    the app has used since day one (``<project>/models``). Means the
    config schema can ship a blank value to mean 'unchanged'."""
    from app.utils import get_models_root, get_project_models_path

    default = get_project_models_path()
    assert get_models_root("") == default
    assert get_models_root(None) == default
    assert get_models_root("   ") == default
