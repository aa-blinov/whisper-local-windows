import logging
import os
import shutil
import sys
import importlib.resources
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

class OptionalComponent:
    def __init__(self, component):
        self._component = component
    
    def __getattr__(self, name):
        if self._component and hasattr(self._component, name):
            attr = getattr(self._component, name)
            return attr
        else:
            return lambda *args, **kwargs: None


def is_installed_package():
    # Check if running from an installed package
    return 'site-packages' in __file__


def _project_root_or_cwd() -> Path:
    """Walk up from this module to the nearest ``pyproject.toml``.

    On a normal source / wheel install the marker is two levels up
    (``app/utils.py`` → ``<project>/pyproject.toml``) and that's
    what gets returned. The fallback when no marker is found walks
    three parents up from this module's directory — i.e. the
    grandparent of the source root — which is a hangover from an
    earlier layout. It's exotic enough to be effectively dead code
    in normal runtimes; the spelling is preserved here so existing
    builds aren't disturbed, but new callers should not rely on it.
    """
    current = Path(__file__).parent
    for p in [current, *current.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return current.parent.parent.parent


def get_project_logs_path():
    """Return the directory log files are written to.

    - **Installed wheel**: CWD ``/logs`` (assumes the user launched
      the tool from a writable cwd).
    - **Dev**: project root ``/logs``.

    Created if missing.
    """
    if is_installed_package():
        logs_dir = Path.cwd() / 'logs'
    else:
        logs_dir = _project_root_or_cwd() / 'logs'

    os.makedirs(logs_dir, exist_ok=True)
    return str(logs_dir)


def get_project_models_path() -> str:
    """Return the directory used to cache downloaded model weights.

    - **Installed wheel**: CWD ``/models``.
    - **Dev**: project root ``/models``.

    Used as the default ``HF_HOME`` so ``onnx-asr`` / ``huggingface_hub``
    keep their downloads where ``is_model_cached`` can find them.
    Settings → Storage card lets the user override this path; this
    is just the default when they haven't.

    Pure path resolution — no filesystem side effects.
    ``huggingface_hub.snapshot_download`` creates the directory
    itself on first download, so probing this function from cache-
    status code shouldn't seed empty ``models/`` dirs as a side effect.
    """
    if is_installed_package():
        base = Path.cwd()
    else:
        base = _project_root_or_cwd()
    return str(base / "models")


def cached_models_size(root: str) -> int:
    """Sum bytes used by downloaded weights under ``root``.

    Walks the ``<root>/hub`` subtree (the HuggingFace cache layout
    used by every model now that the app is ONNX-only).  Returns 0
    if the hub directory doesn't exist.  Used by the Storage card's
    migration prompt to show the user how much would move.

    Doesn't follow symlinks (``Path.stat`` would resolve them and
    inflate the count) and silently skips files that disappear
    mid-walk.
    """
    if not root:
        return 0
    target = Path(root) / "hub"
    if not target.is_dir():
        return 0
    total = 0
    for path in target.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            # Race or permission error — ignore the file rather
            # than failing the whole sum.
            continue
    return total


def move_cached_dir(src: str, dst: str) -> dict:
    """Move the directory at ``src`` to ``dst``.

    Tries ``os.rename`` first (atomic + free for intra-volume); on
    cross-volume (``OSError``) falls back to ``shutil.move`` which
    copies + deletes. Refuses to overwrite — if ``dst`` already
    exists, returns ``moved=False`` with a reason. The controller
    surfaces those reasons in the post-migration info dialog.

    Returns a dict with:
      - ``moved`` (bool) — whether anything was relocated
      - ``bytes`` (int) — size of source tree (only when ``moved``)
      - ``reason`` (str) — human-readable, only when ``moved`` is False
    """
    if not src or not dst:
        return {"moved": False, "reason": "empty source or destination path"}

    src_path = Path(src)
    dst_path = Path(dst)

    # Same path → nothing to do, but don't surface as a failure.
    try:
        same = src_path.resolve() == dst_path.resolve()
    except OSError:
        same = src_path == dst_path
    if same:
        return {"moved": False, "reason": "source and destination are the same"}

    if not src_path.exists():
        return {"moved": False, "reason": "source missing"}

    if dst_path.exists():
        return {"moved": False, "reason": "destination already exists"}

    # Measure source size BEFORE the move so we can report it
    # truthfully even after a successful rename leaves the original
    # path empty.
    bytes_moved = 0
    try:
        for p in src_path.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    bytes_moved += p.stat().st_size
            except OSError:
                continue
    except OSError as exc:
        log.warning("Failed to measure %s before move: %s", src, exc)

    # Make sure the destination's parent exists; ``os.rename`` won't
    # create intermediate directories. ``shutil.move`` will, but we
    # call ``os.rename`` first so we have to set this up either way.
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.rename(str(src_path), str(dst_path))
        log.info("Moved %s → %s (intra-volume rename)", src, dst)
        return {"moved": True, "bytes": bytes_moved}
    except OSError as rename_exc:
        # Cross-volume rename, or some other rename-time error;
        # fall back to copy + delete via shutil.move.
        log.info(
            "os.rename failed (%s) — falling back to shutil.move for %s → %s",
            rename_exc, src, dst,
        )
        try:
            shutil.move(str(src_path), str(dst_path))
            log.info("Moved %s → %s (copy + delete)", src, dst)
            return {"moved": True, "bytes": bytes_moved}
        except (OSError, shutil.Error) as move_exc:
            log.error(
                "Failed to move %s → %s: %s", src, dst, move_exc,
            )
            return {
                "moved": False,
                "reason": f"move failed: {move_exc}",
            }


def get_models_root(configured: Optional[str]) -> str:
    """Resolve the root directory for downloaded model weights.

    Returns ``configured`` when it's a non-empty, non-whitespace
    string; otherwise falls back to the same default ``app.py`` has
    used since day one (``<project>/models`` in dev,
    ``<exe-dir>/models`` when frozen).

    Used by ``app.py`` at startup to decide what to put into
    ``HF_HOME``.  No filesystem side effects — neither branch calls
    ``mkdir``; HF Hub creates the directory itself on first download.
    """
    if configured and configured.strip():
        return configured
    return get_project_models_path()


def is_model_cached(canonical: str) -> bool:
    """Return True if the given Hugging Face model id has at least one
    snapshot present in the local hub cache.

    Honours ``HF_HOME`` (which we set to ``<project>/models``); falls back
    to the user's ``~/.cache/huggingface`` if HF_HOME is not configured.
    Doesn't validate the snapshot's contents — just checks for the
    presence of a non-empty directory under ``snapshots/``.
    """
    if not canonical:
        return False
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hub_root = Path(hf_home) / "hub"
    else:
        hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_root / f"models--{canonical.replace('/', '--')}"
    if not repo_dir.is_dir():
        return False
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir() and any(snap.iterdir()):
            return True
    return False


def is_onnx_model_cached(canonical: str) -> bool:
    """Return True only when actual ONNX weight files are present in the
    HF hub snapshot for ``canonical``.

    ``is_model_cached`` is too lenient for ONNX repos: huggingface_hub
    writes ``config.json`` first, long before the large ``.onnx`` weights
    arrive, so a failed / partial download already satisfies the
    ``any(snap.iterdir())`` check.  We require at least one ``.onnx`` file
    to avoid triggering the startup auto-load on an incomplete download.
    """
    if not canonical:
        return False
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hub_root = Path(hf_home) / "hub"
    else:
        hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_root / f"models--{canonical.replace('/', '--')}"
    if not repo_dir.is_dir():
        return False
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir() and any(snap.glob("*.onnx")):
            return True
    return False


def is_cached_for_info(info) -> bool:
    """Check whether the weights for ``info`` are downloaded.

    Every model is in the HF hub cache.  For ONNX-asr models we use
    the stricter ``is_onnx_model_cached`` (requires an actual ``.onnx``
    file present, not just the ``config.json`` huggingface_hub writes
    first); for anything else we fall back to the lenient
    ``is_model_cached`` reader.
    """
    canonical = getattr(info, "canonical", "")
    onnx_family = getattr(info, "onnx_family", None)
    if onnx_family is not None:
        return is_onnx_model_cached(canonical)
    return is_model_cached(canonical)


def _hf_hub_root() -> Path:
    """Resolve the HF hub cache root the same way ``is_model_cached``
    does — keeps the reader and the deleter pointing at the same dir."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def delete_cached_model(canonical: str) -> bool:
    """Remove the entire HF hub repo directory for ``canonical``.

    Faster-whisper / huggingface_hub stores each model as
    ``models--<owner>--<repo>/`` containing ``snapshots/``, ``blobs/``
    and ``refs/``. We blow away the whole subtree in one call so no
    half-deleted state can survive — the next ``is_model_cached`` read
    must agree with the deletion outcome (no flicker between
    'Download' and 'Select' on the next refresh).

    Returns True iff the directory existed and was successfully
    removed; False on missing input, missing dir, or filesystem error
    (logged at WARNING — the UI surfaces failure as 'still cached').
    """
    if not canonical:
        return False
    repo_dir = _hf_hub_root() / f"models--{canonical.replace('/', '--')}"
    if not repo_dir.is_dir():
        return False
    try:
        shutil.rmtree(repo_dir)
    except OSError as exc:
        log.warning(
            "Failed to delete cached model %s at %s: %s",
            canonical, repo_dir, exc,
        )
        return False
    return True


def delete_cached_for_info(info) -> bool:
    """Delete the cached weights for a registry model.

    Every model in the ONNX-only registry lives in the HF hub cache,
    so this is a thin pass-through to ``delete_cached_model``.  Kept
    as a function so callers stay agnostic in case a future backend
    needs a different cache layout.
    """
    return delete_cached_model(getattr(info, "canonical", ""))


def resolve_asset_path(relative_path: str) -> str:

    if not relative_path or os.path.isabs(relative_path):
        return relative_path

    if is_installed_package():  # pip / pipx
        files = importlib.resources.files("app")
        return str(files / relative_path)

    return str(Path(__file__).parent / relative_path)  # Development