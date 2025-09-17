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


def beautify_hotkey(hotkey_string: str) -> str:
    if not hotkey_string:
        return ""
    
    return hotkey_string.replace('+', '+').upper()

def is_installed_package():
    # Check if running from an installed package
    return 'site-packages' in __file__

def get_config_path() -> str:
    """Return path to single config.yaml (root next to exe or project root).

    Rules:
      * frozen: next to executable
      * installed (site-packages): current working directory
      * dev: walk up to pyproject.toml
    """
    if getattr(sys, 'frozen', False):
        base = Path(sys.executable).parent
    elif is_installed_package():
        base = Path.cwd()
    else:
        current = Path(__file__).parent
        base = None
        for p in [current, *current.parents]:
            if (p / 'pyproject.toml').exists():
                base = p
                break
        if base is None:
            base = current.parent.parent.parent
    return str(base / 'config.yaml')

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

def resolve_asset_path(relative_path: str) -> str:
    
    if not relative_path or os.path.isabs(relative_path):
        return relative_path
    
    if getattr(sys, 'frozen', False): # PyInstaller
        return str(Path(sys._MEIPASS) / relative_path)
    
    if is_installed_package(): # pip / pipx
        files = importlib.resources.files("whisper_key")
        return str(files / relative_path)
    
    return str(Path(__file__).parent / relative_path) # Development