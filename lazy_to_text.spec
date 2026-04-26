# PyInstaller spec file for Lazy to text
# Build: pyinstaller lazy_to_text.spec
# Requires: pip install pyinstaller

import os
from pathlib import Path

# Build-time compatibility shims — must run BEFORE the
# ``importlib.util.find_spec(...)`` probes below, otherwise PyInstaller's
# hidden-imports analysis crashes during ``import nemo.collections.asr``
# (NeMo's exp_manager touches the POSIX-only ``signal.SIGKILL`` at
# class-definition time on Windows). Same shim is applied at runtime via
# ``runtime_hooks`` so the frozen exe boots cleanly too.
import signal as _spec_signal

if not hasattr(_spec_signal, "SIGKILL"):
    _spec_signal.SIGKILL = _spec_signal.SIGTERM  # type: ignore[attr-defined]

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

# Collect data files (assets + Qt stylesheets).
#
# theme.py / resolve_asset_path use ``Path(__file__).parent`` to locate
# resources, so the destination paths inside the bundle have to mirror the
# layout under ``app/`` (PyInstaller drops them into ``_internal/`` itself).
datas = []
for p in assets_src.rglob('*'):
    if p.is_file():
        rel = p.relative_to(app_dir)
        datas.append((str(p), str(rel.parent)))

styles_src = app_dir / 'gui' / 'styles'
if styles_src.exists():
    for p in styles_src.rglob('*'):
        if p.is_file():
            rel = p.relative_to(app_dir)
            datas.append((str(p), str(rel.parent)))

# Note: ``config.yaml`` is intentionally NOT bundled in the release.
# Whatever sits in the repo's working tree at build time is the
# developer's personal config (selected model, model_overrides,
# audio device index, …) — shipping it would seed every fresh
# install with someone else's preferences.
#
# ConfigManager._load_or_create() falls back to DEFAULT_CONFIG when
# no user / bundled config is found, so the first launch writes a
# clean default config.yaml into ``%APPDATA%/LazyToText/``.
#
# If a build needs a curated 'factory defaults' file (e.g. corporate
# spin with non-default hotkeys), put it next to the .exe in the
# COLLECT output (``dist/LazyToText/config.yaml``) — the seeding
# path in ConfigManager will pick it up automatically.

import importlib.util

requested_hiddenimports = [
    'win32timezone',        # pywin32 timezone helper
    'global_hotkeys',       # ensure hotkey library + submodules bundled
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    # NeMo's submodules are loaded via hydra config + dynamic
    # ``importlib`` calls — PyInstaller's static analysis misses
    # most of them, so list the ones the ASR path actually needs.
    'nemo',
    'nemo.collections',
    'nemo.collections.asr',
    'nemo.collections.asr.models',
    'nemo.collections.asr.modules',
    'nemo.collections.asr.parts',
    'nemo.utils',
    'lhotse',
]

hiddenimports = [m for m in requested_hiddenimports if importlib.util.find_spec(m) is not None]

if len(hiddenimports) < len(requested_hiddenimports):
    missing = set(requested_hiddenimports) - set(hiddenimports)
    print(f"[spec] Skipping missing optional hidden imports: {', '.join(sorted(missing))}")

# Extra: sounddevice sometimes needs explicit PortAudio dynamic lib inclusion (PyInstaller usually detects)
# If не подхватит, можно явно добавить binaries сюда позже.

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

_runtime_hook = project_root / 'scripts' / 'pyi_runtime_hook.py'
runtime_hooks = [str(_runtime_hook)] if _runtime_hook.is_file() else []

analysis = Analysis(
    ['lazy-to-text-ui.py'],
    pathex=[str(app_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[],
    # ``noarchive=True`` keeps the .py source files alongside the
    # .pyc in ``_internal/`` instead of bundling them into the
    # base_library.zip / .pyz archive. TorchScript's
    # ``inspect.getsource`` needs to read the actual .py source for
    # any ``@torch.jit.script`` decorated function (NeMo's RNNT
    # decoder uses several — without source access the boot path
    # dies with ``Can't get source for <function snake at ...>.
    # TorchScript requires source access in order to carry out
    # compilation``). The size cost is ~50-100 MB extra in
    # ``_internal/``, mostly torch / nemo source.
    noarchive=True,
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
