# PyInstaller spec — Windows portable build for Lazy to Text.
#
# Build via the wrapper:
#     scripts\build-windows.ps1
# or directly:
#     uv run pyinstaller lazy_to_text.spec
#
# Output: dist/LazyToText/LazyToText.exe (folder bundle, ~700 MB).
# Drop the whole folder onto another Windows box — no install,
# no admin, no PATH munging.

from pathlib import Path

block_cipher = None

# Resolve the project root robustly — ``__file__`` is sometimes
# missing from the spec's exec scope depending on how PyInstaller
# is launched (``uv run pyinstaller …`` vs direct).  Falls back to
# CWD with the assumption that the build is launched from the repo
# root (which the wrapper script enforces with ``cd``).
try:
    spec_file_path = Path(__file__).resolve()  # type: ignore[name-defined]
except NameError:
    cwd = Path.cwd()
    candidate = cwd / "lazy_to_text.spec"
    spec_file_path = candidate if candidate.exists() else cwd

project_root = spec_file_path.parent
app_dir = project_root / "app"
assets_src = app_dir / "assets"

# Mirror the ``app/`` layout inside the bundle so
# ``resolve_asset_path`` (which walks from ``app/__file__``) finds
# every shipped resource at runtime.  PyInstaller drops these into
# ``_internal/`` itself, with the relative path we hand it as the
# destination directory.
datas = []
for sub in (assets_src, app_dir / "gui" / "styles"):
    if not sub.exists():
        continue
    for p in sub.rglob("*"):
        if p.is_file():
            rel = p.relative_to(app_dir)
            datas.append((str(p), str(rel.parent)))

# ``config.yaml`` is intentionally NOT bundled — whatever sits in
# the working tree at build time is the developer's personal
# config (selected model, audio device index, …) and shipping it
# would seed every fresh install with someone else's preferences.
# ``ConfigManager._load_or_create()`` writes a clean
# ``DEFAULT_CONFIG`` to ``%APPDATA%/LazyToText/config.yaml`` on
# first launch — that's the source of truth for portable builds.

import importlib.util
from PyInstaller.utils.hooks import collect_submodules

# Imports PyInstaller's static analyser misses.  Each of these is
# loaded via late-bound ``importlib`` / hydra-style string targets,
# so the dependency graph never sees them at analysis time.
requested_hiddenimports = [
    "win32timezone",          # pywin32 timezone helper
    "global_hotkeys",         # Win32 global hotkey listener
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",          # Heroicons SVG render path
]

hiddenimports = [
    m for m in requested_hiddenimports
    if importlib.util.find_spec(m) is not None
]

if len(hiddenimports) < len(requested_hiddenimports):
    missing = set(requested_hiddenimports) - set(hiddenimports)
    print(
        f"[spec] Skipping missing optional hidden imports: "
        f"{', '.join(sorted(missing))}"
    )

# ``onnx_asr`` enumerates engine adapters via entry-point lookup at
# import time, but a couple of its helper modules are also imported
# by string from configs.  Pull the whole tree so any future model
# adapter (parakeet / whisper / gigaam / canary / vosk / t-one)
# works straight from the bundle without re-building.
for _pkg in ("onnx_asr", "onnxruntime"):
    if importlib.util.find_spec(_pkg) is not None:
        try:
            hiddenimports.extend(collect_submodules(_pkg))
        except Exception as _exc:
            print(f"[spec] collect_submodules({_pkg!r}) failed: {_exc!r}")

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

_runtime_hook = project_root / "scripts" / "pyi_runtime_hook.py"
runtime_hooks = [str(_runtime_hook)] if _runtime_hook.is_file() else []

analysis = Analysis(
    ["lazy-to-text-ui.py"],
    pathex=[str(app_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[],
    # Keep .py source alongside .pyc in ``_internal/`` rather than
    # zipping into ``base_library.zip`` — onnx_asr / huggingface_hub
    # occasionally use ``inspect.getsource`` on user-facing helper
    # functions and the zipped path returns empty.  Costs ~30 MB
    # extra; well worth the boot reliability.
    noarchive=True,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

import os as _os

# Set ``LAZYTOTEXT_DEBUG_CONSOLE=1`` before building to get a
# console window for stdout / stderr (useful for debugging frozen
# crashes that don't reach the file logger).  Default is windowed.
_console_flag = (_os.environ.get("LAZYTOTEXT_DEBUG_CONSOLE", "0") == "1")

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="LazyToText",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=_console_flag,
    icon=str(assets_src / "tray_idle.ico")
    if (assets_src / "tray_idle.ico").exists()
    else None,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=False,
    name="LazyToText",  # → dist/LazyToText/
)
