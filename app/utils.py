import os
import sys
import importlib.resources
from pathlib import Path

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


def resolve_asset_path(relative_path: str) -> str:
    
    if not relative_path or os.path.isabs(relative_path):
        return relative_path
    
    if getattr(sys, 'frozen', False): # PyInstaller
        return str(Path(sys._MEIPASS) / relative_path)
    
    if is_installed_package(): # pip / pipx
        files = importlib.resources.files("app")
        return str(files / relative_path)
    
    return str(Path(__file__).parent / relative_path) # Development