import logging
import os
import shutil
import sys
import importlib.resources
from pathlib import Path

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

def get_project_logs_path():
    """Return unified logs directory inside project (or next to exe when frozen).

        Decision: always write logs to the project `logs` directory per requirement.
        Behavior now:
            * PyInstaller: next to the executable /logs
            * Any other case (dev, installed package) — project root /logs
        Logs are always local in the logs directory.
    """
    if getattr(sys, 'frozen', False):  # PyInstaller bundle
        exe_dir = Path(sys.executable).parent
        logs_dir = exe_dir / 'logs'
    elif is_installed_package():
        # For installed packages, place logs in the working directory (where user launched the tool)
        logs_dir = Path.cwd() / 'logs'
    else:
        # Walk upward until we find pyproject.toml to determine project root.
        current = Path(__file__).parent
        probe = current
        project_root = None
        for p in [probe, *probe.parents]:
            if (p / 'pyproject.toml').exists():
                project_root = p
                break
        if project_root is None:
            project_root = current.parent.parent.parent
        logs_dir = project_root / 'logs'

    os.makedirs(logs_dir, exist_ok=True)
    return str(logs_dir)

def get_project_models_path() -> str:
    """Return the directory used to cache downloaded model weights.

    Lives next to the executable when frozen (PyInstaller), otherwise in the
    project root identified by ``pyproject.toml``. Created if missing.

    Used as ``HF_HOME`` so faster-whisper / huggingface_hub keep their
    downloads inside the project tree rather than the per-user
    ``~/.cache/huggingface`` location, which is invisible to most users and
    eats the system drive.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    elif is_installed_package():
        base = Path.cwd()
    else:
        current = Path(__file__).parent
        base = None
        for p in [current, *current.parents]:
            if (p / "pyproject.toml").exists():
                base = p
                break
        if base is None:
            base = current.parent.parent.parent
    models_dir = base / "models"
    os.makedirs(models_dir, exist_ok=True)
    return str(models_dir)


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


def is_gigaam_cached(model_name: str) -> bool:
    """Return True if GigaAM has the given model checkpoint on disk.

    GigaAM downloads to ``~/.cache/gigaam/<model_name>.ckpt`` (NOT the
    HF hub layout) — every weights file lives next to the others as a
    single ``.ckpt``. Checks the file is present and non-empty.
    """
    if not model_name:
        return False
    cache_dir = Path.home() / ".cache" / "gigaam"
    candidate = cache_dir / f"{model_name}.ckpt"
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
    ``~/.cache/gigaam/`` — no shared blobs, no metadata sidecars to
    worry about. Returns True iff the file existed and was unlinked.
    """
    if not model_name:
        return False
    candidate = Path.home() / ".cache" / "gigaam" / f"{model_name}.ckpt"
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