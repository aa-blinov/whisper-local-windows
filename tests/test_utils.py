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


# ---- cached_models_size ----------------------------------------------------


def test_cached_models_size_returns_zero_for_missing_root(tmp_path):
    """Brand-new path with nothing in it → 0. Lets the controller
    skip the migration prompt entirely when there's nothing to
    move."""
    from app.utils import cached_models_size

    assert cached_models_size(str(tmp_path / "does-not-exist")) == 0


def test_cached_models_size_sums_hub_and_gigaam_subtrees(tmp_path):
    """Total bytes under ``hub/`` (HF) plus ``gigaam/`` (.ckpt files)
    — the two subdirs the app manages."""
    from app.utils import cached_models_size

    repo_dir = _make_hf_snapshot(tmp_path / "hub", "owner/repo")
    (tmp_path / "gigaam").mkdir()
    (tmp_path / "gigaam" / "v3_e2e_ctc.ckpt").write_bytes(b"x" * 2048)

    total = cached_models_size(str(tmp_path))
    # We don't assert the exact value (it depends on the test
    # fixture's bytes) — just that it sums both subtrees.
    assert total >= 2048 + len(b"fake weights") + len(b"{}")
    # Sanity: removing one subtree must reduce the total.
    import shutil
    shutil.rmtree(repo_dir)
    assert cached_models_size(str(tmp_path)) < total


def test_cached_models_size_handles_only_hub(tmp_path):
    """Hub only, no gigaam dir → returns just the hub bytes (no
    crash from missing gigaam)."""
    from app.utils import cached_models_size

    _make_hf_snapshot(tmp_path / "hub", "owner/repo")
    assert cached_models_size(str(tmp_path)) > 0


def test_cached_models_size_handles_only_gigaam(tmp_path):
    """And vice versa — gigaam only, no hub dir."""
    from app.utils import cached_models_size

    (tmp_path / "gigaam").mkdir()
    (tmp_path / "gigaam" / "v3_e2e_ctc.ckpt").write_bytes(b"x" * 4096)
    assert cached_models_size(str(tmp_path)) >= 4096


# ---- move_cached_dir -------------------------------------------------------


def test_move_cached_dir_renames_intra_volume(tmp_path):
    """Inside the same drive ``os.rename`` is atomic and instant —
    the helper should use it. We can't directly observe rename vs
    copy, but the result must put files at dst and remove them
    from src."""
    from app.utils import move_cached_dir

    src = tmp_path / "src" / "hub"
    src.mkdir(parents=True)
    (src / "model.bin").write_bytes(b"x" * 1024)

    dst = tmp_path / "dst" / "hub"
    result = move_cached_dir(str(src), str(dst))

    assert result["moved"] is True
    assert result["bytes"] >= 1024
    assert not src.exists()
    assert (dst / "model.bin").exists()


def test_move_cached_dir_skips_when_source_missing(tmp_path):
    """No source → no-op. Used to handle 'user has Whisper but never
    downloaded GigaAM' cleanly without raising."""
    from app.utils import move_cached_dir

    src = tmp_path / "ghost"
    dst = tmp_path / "dst"
    result = move_cached_dir(str(src), str(dst))

    assert result["moved"] is False
    assert "missing" in result["reason"].lower() or "exist" in result["reason"].lower()


def test_move_cached_dir_skips_when_dest_already_exists(tmp_path):
    """Destination already populated → refuse rather than merge or
    overwrite. Surface in the reason so the controller can show the
    user what to do."""
    from app.utils import move_cached_dir

    src = tmp_path / "src" / "hub"
    src.mkdir(parents=True)
    (src / "model.bin").write_bytes(b"x")

    dst = tmp_path / "dst" / "hub"
    dst.mkdir(parents=True)
    (dst / "existing.bin").write_bytes(b"already here")

    result = move_cached_dir(str(src), str(dst))

    assert result["moved"] is False
    assert "exist" in result["reason"].lower()
    # Source untouched — caller can still fall back to manual copy.
    assert (src / "model.bin").exists()
    # Destination untouched too — no overwrite.
    assert (dst / "existing.bin").exists()


def test_move_cached_dir_creates_dst_parent_dir(tmp_path):
    """Destination's parent might not exist yet (fresh path the
    user picked) — helper creates it on the way."""
    from app.utils import move_cached_dir

    src = tmp_path / "src" / "hub"
    src.mkdir(parents=True)
    (src / "model.bin").write_bytes(b"x" * 512)

    dst = tmp_path / "fresh" / "destination" / "hub"
    result = move_cached_dir(str(src), str(dst))

    assert result["moved"] is True
    assert (dst / "model.bin").exists()


def test_move_cached_dir_short_circuits_when_src_equals_dst(tmp_path):
    """User picked the same folder → no move, no failure."""
    from app.utils import move_cached_dir

    src = tmp_path / "hub"
    src.mkdir()
    (src / "model.bin").write_bytes(b"x")

    result = move_cached_dir(str(src), str(src))

    assert result["moved"] is False
    assert "same" in result["reason"].lower()
    assert (src / "model.bin").exists()
