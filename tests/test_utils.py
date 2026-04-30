"""Tests for cache helpers in app.utils.

Both ``is_*_cached`` (already used in production) and the new
``delete_*`` family share the same on-disk layout, so they're
covered side by side here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_storage_env(monkeypatch):
    """Each test starts with a clean ``HF_HOME`` so a leaked value
    from a controller test (which writes it directly via
    ``os.environ[...] = ...``) doesn't send the helpers under test
    at the wrong directory."""
    monkeypatch.delenv("HF_HOME", raising=False)


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


# ---- delete_cached_for_info dispatcher -------------------------------------


class _FakeInfo:
    def __init__(
        self,
        canonical: str,
        backend_kind: str = "onnx_asr",
        onnx_family: str = "whisper",
    ) -> None:
        self.canonical = canonical
        self.backend_kind = backend_kind
        self.onnx_family = onnx_family


def test_delete_cached_for_info_deletes_hf_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    canonical = "onnx-community/whisper-large-v3-turbo"
    repo_dir = _make_hf_snapshot(tmp_path / "hub", canonical)

    from app.utils import delete_cached_for_info

    info = _FakeInfo(canonical=canonical)
    assert delete_cached_for_info(info) is True
    assert not repo_dir.exists()


def test_delete_cached_for_info_returns_false_when_nothing_to_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    from app.utils import delete_cached_for_info

    info = _FakeInfo(canonical="ghost/x")
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


# ---- get_models_root ------------------------------------------------------


def test_get_models_root_returns_configured_value(tmp_path):
    """The helper used by ``app.py`` to set ``HF_HOME`` must echo
    back whatever the user picked, with no surprise rewriting
    (e.g. appending ``/hub`` or normalising case)."""
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


def test_get_models_root_does_not_create_filesystem_entries(tmp_path, monkeypatch):
    """Documented as 'doesn't touch the filesystem'. Caller passes a
    fresh path that doesn't exist yet — the function must just echo
    it back without ``mkdir``-ing anything. Otherwise the dev/test
    workflow accidentally seeds empty ``models/`` directories all
    over the place when probing config values."""
    from app.utils import get_models_root

    custom = tmp_path / "fresh-cache-dir"
    assert not custom.exists()

    result = get_models_root(str(custom))

    assert result == str(custom)
    assert not custom.exists(), (
        "get_models_root must not create the configured path"
    )


def test_get_models_root_default_branch_does_not_create_dir(tmp_path, monkeypatch):
    """Even on the default-fallback branch (no configured path) the
    function should just resolve a string — directory creation is
    the caller's responsibility, and HF Hub creates the cache dir
    on first download anyway."""
    from app import utils as utils_module

    fake_root = tmp_path / "fake-default-models"
    monkeypatch.setattr(
        utils_module, "get_project_models_path", lambda: str(fake_root)
    )
    assert not fake_root.exists()

    result = utils_module.get_models_root("")

    assert result == str(fake_root)
    assert not fake_root.exists(), (
        "default-fallback branch must not pre-create the models dir"
    )


# ---- cached_models_size ----------------------------------------------------


def test_cached_models_size_returns_zero_for_missing_root(tmp_path):
    """Brand-new path with nothing in it → 0. Lets the controller
    skip the migration prompt entirely when there's nothing to
    move."""
    from app.utils import cached_models_size

    assert cached_models_size(str(tmp_path / "does-not-exist")) == 0


def test_cached_models_size_sums_hub_subtree(tmp_path):
    """Total bytes under ``<root>/hub`` (HF cache layout) — the only
    subtree the app manages now that everything is ONNX-only."""
    from app.utils import cached_models_size

    repo_dir = _make_hf_snapshot(tmp_path / "hub", "owner/repo")
    # Stray files outside the hub subtree must not leak into the sum.
    (tmp_path / "scratch.bin").write_bytes(b"y" * 4096)

    total = cached_models_size(str(tmp_path))
    assert total >= len(b"fake weights") + len(b"{}") + len(b"fake blob")
    assert total < 4096, (
        "size must only walk <root>/hub, not stray files in <root>"
    )

    import shutil
    shutil.rmtree(repo_dir)
    assert cached_models_size(str(tmp_path)) == 0


def test_cached_models_size_handles_only_hub(tmp_path):
    """Hub only, no gigaam dir → returns just the hub bytes (no
    crash from missing gigaam)."""
    from app.utils import cached_models_size

    _make_hf_snapshot(tmp_path / "hub", "owner/repo")
    assert cached_models_size(str(tmp_path)) > 0


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


# ---- Dev-mode user data dirs -----------------------------------------------


def test_get_project_logs_path_dev_unchanged():
    """Dev mode walks up from the module file to find
    ``pyproject.toml`` and returns ``<project>/logs``."""
    from app.utils import get_project_logs_path

    result = get_project_logs_path()
    # Project's own logs dir; never a hidden user-data path in dev.
    assert "AppData" not in result
    assert "Application Support" not in result
    assert "logs" in result.lower()


def test_get_project_models_path_dev_unchanged():
    from app.utils import get_project_models_path

    result = get_project_models_path()
    assert "AppData" not in result
    assert "Application Support" not in result
    assert "models" in result.lower()


# ---- is_onnx_model_cached ---------------------------------------------------


def _make_onnx_snapshot(hub_root: Path, canonical: str, with_weights: bool = True) -> Path:
    """Lay out a minimal HF snapshot for an ONNX model repo.

    If ``with_weights=True``, an ``.onnx`` file is included — simulates
    a completed download. Otherwise only ``config.json`` is present —
    simulates a failed/partial download where huggingface_hub wrote the
    config before the weights transfer finished.
    """
    repo_dir = hub_root / f"models--{canonical.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots" / "deadbeef"
    snap_dir.mkdir(parents=True)
    (snap_dir / "config.json").write_text("{}")
    if with_weights:
        (snap_dir / "model.onnx").write_bytes(b"fake onnx weights")
    return repo_dir


def test_is_onnx_model_cached_true_when_onnx_file_present(tmp_path, monkeypatch):
    """A completed ONNX download has a ``.onnx`` file in the snapshot —
    must be reported as cached."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    canonical = "istupakov/parakeet-tdt-0.6b-v3-onnx"
    _make_onnx_snapshot(tmp_path / "hub", canonical, with_weights=True)

    from app.utils import is_onnx_model_cached

    assert is_onnx_model_cached(canonical) is True


def test_is_onnx_model_cached_false_when_only_config_present(tmp_path, monkeypatch):
    """A partial/failed download has only ``config.json`` — must NOT be
    reported as cached so the app doesn't attempt to auto-load and hang."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    canonical = "istupakov/parakeet-tdt-0.6b-v3-onnx"
    _make_onnx_snapshot(tmp_path / "hub", canonical, with_weights=False)

    from app.utils import is_onnx_model_cached

    assert is_onnx_model_cached(canonical) is False


def test_is_onnx_model_cached_false_when_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    from app.utils import is_onnx_model_cached

    assert is_onnx_model_cached("istupakov/parakeet-tdt-0.6b-v3-onnx") is False


def test_is_cached_for_info_uses_onnx_check_for_onnx_models(tmp_path, monkeypatch):
    """``is_cached_for_info`` must route onnx_asr models through
    ``is_onnx_model_cached`` (requires .onnx file) — not the lenient
    ``is_model_cached`` which accepts any file and gives false positives
    for partial downloads."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    canonical = "istupakov/parakeet-tdt-0.6b-v3-onnx"

    from app.model_mapping import ModelInfo
    from app.utils import is_cached_for_info

    info = ModelInfo(
        alias="parakeet-tdt-v3",
        canonical=canonical,
        display_name="Parakeet ONNX",
        size_mb=1200,
        vram_gb=2.0,
        speed="fast",
        quality="excellent",
        languages="multilingual",
        description="x",
        compute_type="float32",
        backend_kind="onnx_asr",
        family="Parakeet",
        onnx_family="parakeet",
    )

    # Partial download — only config.json.
    _make_onnx_snapshot(tmp_path / "hub", canonical, with_weights=False)
    assert is_cached_for_info(info) is False, (
        "partial ONNX download (no .onnx file) must not report as cached"
    )

    # Full download — .onnx file present.
    snap = (tmp_path / "hub" / f"models--{canonical.replace('/', '--')}"
            / "snapshots" / "deadbeef")
    (snap / "model.onnx").write_bytes(b"weights")
    assert is_cached_for_info(info) is True
