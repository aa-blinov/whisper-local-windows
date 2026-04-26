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


def _user_local_data_dir() -> Path:
    """Per-user, non-roaming app data directory on Windows.

    Honours ``%LOCALAPPDATA%`` (the canonical Local path); falls
    back to ``~/AppData/Local/LazyToText`` when the env var isn't
    exposed (sandboxed shells / unusual envs). Used for things
    that are large or machine-specific — logs and downloaded
    model weights — and shouldn't sync via Windows roaming
    profiles. Config / small settings live under ``%APPDATA%``
    instead (see ``ConfigManager._user_config_dir``).
    """
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "LazyToText"
    return Path.home() / "AppData" / "Local" / "LazyToText"


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

    - **Frozen (PyInstaller)**: ``%LOCALAPPDATA%/LazyToText/logs``.
      The .exe might be installed in ``Program Files`` — that's
      read-only for non-admin users, so writing logs next to the
      binary fails on the very first ``RotatingFileHandler.emit``.
      LOCAL appdata is always per-user-writable.
    - **Installed wheel**: CWD ``/logs`` (legacy behaviour;
      assumes the user launched from a writable cwd).
    - **Dev**: project root ``/logs``.

    Created if missing.
    """
    if getattr(sys, 'frozen', False):  # PyInstaller bundle
        logs_dir = _user_local_data_dir() / 'logs'
    elif is_installed_package():
        # For installed packages, place logs in the working directory (where user launched the tool)
        logs_dir = Path.cwd() / 'logs'
    else:
        logs_dir = _project_root_or_cwd() / 'logs'

    os.makedirs(logs_dir, exist_ok=True)
    return str(logs_dir)

def get_project_models_path() -> str:
    """Return the directory used to cache downloaded model weights.

    - **Frozen (PyInstaller)**: ``%LOCALAPPDATA%/LazyToText/models``.
      Downloaded weights are gigabytes and would either fail to
      write (Program Files install, read-only without admin) or
      bloat the install dir if they did. LOCAL appdata is the
      right bucket — per-user, writable, NOT synced across
      machines via roaming profiles.
    - **Installed wheel**: CWD ``/models``.
    - **Dev**: project root ``/models``.

    Used as the default ``HF_HOME`` so faster-whisper /
    huggingface_hub keep their downloads where ``is_model_cached``
    can find them. Settings → Storage card lets the user override
    this path; this is just the default when they haven't.

    Pure path resolution — no filesystem side effects. ``HF_HOME``
    consumers (huggingface_hub.snapshot_download, gigaam.load_model)
    create the directory themselves on first download via their own
    ``os.makedirs(..., exist_ok=True)``, so probing this function
    from cache-status code shouldn't seed empty ``models/`` dirs as
    a side effect.
    """
    if getattr(sys, "frozen", False):
        base = _user_local_data_dir()
    elif is_installed_package():
        base = Path.cwd()
    else:
        base = _project_root_or_cwd()
    return str(base / "models")


def cached_models_size(root: str) -> int:
    """Sum bytes used by downloaded weights under ``root``.

    Looks at the two subtrees the app manages — ``<root>/hub`` for
    HF-hosted Whisper models and ``<root>/gigaam`` for GigaAM ckpt
    files. Returns 0 if neither exists. Used by the Storage card's
    migration prompt to show the user how much would move.

    Doesn't follow symlinks (``Path.stat`` would resolve them and
    inflate the count) and silently skips files that disappear
    mid-walk.
    """
    if not root:
        return 0
    base = Path(root)
    if not base.is_dir():
        return 0
    total = 0
    for sub in ("hub", "gigaam"):
        target = base / sub
        if not target.is_dir():
            continue
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
    ``HF_HOME`` and ``GIGAAM_MODELS_DIR``. No filesystem side
    effects — neither branch calls ``mkdir``; HF Hub /
    GigaAM create the directory themselves on first download.
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


def _gigaam_cache_dir() -> Path:
    """Resolve the GigaAM checkpoint directory.

    Honours ``GIGAAM_MODELS_DIR`` (set by ``app.py`` at startup from
    the configured ``storage.models_dir``); falls back to the
    library's own default ``~/.cache/gigaam`` so existing installs
    keep finding their downloads after upgrading to a build that
    supports the override.
    """
    env_dir = os.environ.get("GIGAAM_MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".cache" / "gigaam"


def is_gigaam_cached(model_name: str) -> bool:
    """Return True if GigaAM has the given model checkpoint on disk.

    GigaAM downloads to ``<cache_dir>/<model_name>.ckpt`` (NOT the HF
    hub layout) — every weights file lives next to the others as a
    single ``.ckpt``. Checks the file is present and non-empty.
    """
    if not model_name:
        return False
    candidate = _gigaam_cache_dir() / f"{model_name}.ckpt"
    try:
        return candidate.is_file() and candidate.stat().st_size > 0
    except OSError:
        return False


def is_cached_for_info(info) -> bool:
    """Dispatch the cache check by ``info.backend_kind`` so the UI can
    ask one question regardless of which engine backs a model."""
    kind = getattr(info, "backend_kind", "faster_whisper")
    if kind == "gigaam":
        return is_gigaam_cached(getattr(info, "canonical", ""))
    return is_model_cached(getattr(info, "canonical", ""))


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


def delete_gigaam_cached(model_name: str) -> bool:
    """Remove the GigaAM checkpoint file for ``model_name``.

    GigaAM keeps every weights file as a single ``.ckpt`` inside
    ``<cache_dir>`` (configurable via ``GIGAAM_MODELS_DIR``, defaults
    to ``~/.cache/gigaam``) — no shared blobs, no metadata sidecars
    to worry about. Returns True iff the file existed and was
    unlinked.
    """
    if not model_name:
        return False
    candidate = _gigaam_cache_dir() / f"{model_name}.ckpt"
    if not candidate.is_file():
        return False
    try:
        candidate.unlink()
    except OSError as exc:
        log.warning(
            "Failed to delete GigaAM checkpoint %s at %s: %s",
            model_name, candidate, exc,
        )
        return False
    return True


def delete_cached_for_info(info) -> bool:
    """Dispatch the deletion by ``info.backend_kind`` — mirror of
    ``is_cached_for_info`` so the UI can ask one question regardless
    of which engine backs a model."""
    kind = getattr(info, "backend_kind", "faster_whisper")
    if kind == "gigaam":
        return delete_gigaam_cached(getattr(info, "canonical", ""))
    return delete_cached_model(getattr(info, "canonical", ""))


def resolve_asset_path(relative_path: str) -> str:
    
    if not relative_path or os.path.isabs(relative_path):
        return relative_path
    
    if getattr(sys, 'frozen', False): # PyInstaller
        return str(Path(sys._MEIPASS) / relative_path)
    
    if is_installed_package(): # pip / pipx
        files = importlib.resources.files("app")
        return str(files / relative_path)
    
    return str(Path(__file__).parent / relative_path) # Development