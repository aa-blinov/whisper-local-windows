# PyInstaller spec file for Lazy to text
# Build: pyinstaller lazy_to_text.spec
# Requires: pip install pyinstaller

import os
from pathlib import Path

block_cipher = None

"""PyInstaller spec for Lazy to text.

Note: Under some invocation methods (e.g. `uv run pyinstaller lazy_to_text.spec`),
`__file__` is not injected into the spec execution globals, causing a NameError.
We defensively resolve the project root via `__file__` when available, otherwise
fall back to the current working directory (assuming the build is launched from
the project root)."""

try:  # Preferred: actual spec file location
    spec_file_path = Path(__file__).resolve()  # type: ignore[name-defined]
except NameError:
    # Fallback: current working directory (user should run build from repo root)
    cwd = Path.cwd()
    candidate = cwd / 'lazy_to_text.spec'
    spec_file_path = candidate if candidate.exists() else cwd

project_root = spec_file_path.parent
app_dir = project_root / 'app'
assets_src = app_dir / 'assets'

# Collect data files (assets)
datas = []
for p in assets_src.rglob('*'):
    if p.is_file():
        rel = p.relative_to(app_dir)
        # Place assets at their natural relative path (so code can refer to 'assets/..')
        # Example: app/assets/tray_idle.png -> dist/.../assets/tray_idle.png
        datas.append((str(p), str(rel.parent)))

# Include top-level config.yaml next to exe so user can edit it
config_path = project_root / 'config.yaml'
if config_path.exists():
    datas.append((str(config_path), '.'))

import importlib.util

requested_hiddenimports = [
    'customtkinter',
    'PIL._tkinter_finder',  # may not exist in newer Pillow; filtered below
    'pystray._win32',       # pystray platform helper (optional)
    'win32timezone',        # pywin32 timezone helper
    'global_hotkeys',       # ensure hotkey library + submodules bundled
]

hiddenimports = [m for m in requested_hiddenimports if importlib.util.find_spec(m) is not None]

if len(hiddenimports) < len(requested_hiddenimports):
    missing = set(requested_hiddenimports) - set(hiddenimports)
    print(f"[spec] Skipping missing optional hidden imports: {', '.join(sorted(missing))}")

# Extra: sounddevice sometimes needs explicit PortAudio dynamic lib inclusion (PyInstaller usually detects)
# If не подхватит, можно явно добавить binaries сюда позже.

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

analysis = Analysis(
    ['lazy-to-text-ui.py'],
    pathex=[str(app_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

import os as _os
_console_flag = (_os.environ.get('LAZYTOTEXT_DEBUG_CONSOLE','0') == '1')

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name='LazyToText',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=_console_flag,  # set LAZYTOTEXT_DEBUG_CONSOLE=1 to debug with console
    icon=str(assets_src / 'tray_idle.ico') if (assets_src / 'tray_idle.ico').exists() else None,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=False,
    name='LazyToText'  # Output folder name
)
