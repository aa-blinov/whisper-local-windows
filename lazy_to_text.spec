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
            # Preserve the full ``app/…`` prefix so runtime paths like
            # ``Path(__file__).parent / "styles"`` resolve correctly in
            # the frozen bundle.
            rel = p.relative_to(project_root)
            datas.append((str(p), str(rel.parent)))

# ``config.yaml`` is intentionally NOT bundled — whatever sits in
# the working tree at build time is the developer's personal
# config (selected model, audio device index, …) and shipping it
# would seed every fresh install with someone else's preferences.
# ``ConfigManager._load_or_create()`` writes a clean
# ``DEFAULT_CONFIG`` to ``%APPDATA%/LazyToText/config.yaml`` on
# first launch — that's the source of truth for portable builds.

import importlib.util
import sys
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

# When the ``[cuda]`` extra is installed, the nvidia-* wheels drop
# their DLLs into ``site-packages/nvidia/<pkg>/bin/``.  PyInstaller
# sometimes auto-collects these when it resolves dependencies of
# ``onnxruntime_providers_cuda.dll``, but only if they happen to be
# on the host PATH at build time.  To guarantee they ship regardless
# of the build environment, we explicitly bundle them as data files
# (preserving the ``nvidia/<pkg>/bin`` directory structure).  The
# runtime hook then prepends every ``bin/`` path to PATH so the
# Windows loader finds them at launch.
_datas = list(datas)
for _sp in sys.path:
    _nvidia_base = Path(_sp) / "nvidia"
    if not _nvidia_base.is_dir():
        continue
    for _pkg_dir in _nvidia_base.iterdir():
        _bin_dir = _pkg_dir / "bin"
        if not _bin_dir.is_dir():
            continue
        for _dll in _bin_dir.glob("*.dll"):
            # Strip the leading ``site-packages/`` (or ``Lib/``) so
            # the DLLs land at ``_internal/nvidia/<pkg>/bin/`` rather
            # than ``_internal/site-packages/nvidia/<pkg>/bin/``.
            _rel = _dll.relative_to(_nvidia_base.parent)
            _dest = str(_rel.parent).replace("site-packages\\", "").replace("Lib\\", "")
            _datas.append((str(_dll), _dest))
            print(f"[spec] Bundling CUDA DLL: {_dll.name} -> {_dest}")

# ``onnx_asr`` uses ``importlib.metadata`` at import time to read its
# own version from ``onnx_asr-*.dist-info``.  PyInstaller does not
# auto-collect ``.dist-info`` directories, so the import crashes in
# the frozen build with "No package metadata was found for onnx-asr".
# We explicitly ship the dist-info folder so metadata queries work.
for _sp in sys.path:
    _sp_path = Path(_sp)
    if not _sp_path.is_dir():
        continue
    for _di in _sp_path.glob("onnx_asr-*.dist-info"):
        if _di.is_dir():
            _rel = _di.relative_to(_sp_path)
            _datas.append((str(_di), str(_rel)))
            print(f"[spec] Bundling dist-info: {_rel}")

    # ``onnx_asr`` also bundles small ONNX preprocessor graphs inside
    # ``preprocessors/data/`` (resample kernels, feature extractors,
    # …).  PyInstaller does not auto-collect them because they're
    # loaded at runtime via ``Path(__file__).parent / "data"`` rather
    # than imported as Python modules.  Ship the whole subtree so
    # every model family works out of the box.
    _onnx_asr_pkg = _sp_path / "onnx_asr"
    if _onnx_asr_pkg.is_dir():
        for _data_file in _onnx_asr_pkg.rglob("*"):
            if _data_file.is_file() and _data_file.suffix not in (".py", ".pyc") and "__pycache__" not in _data_file.parts:
                _rel = _data_file.relative_to(_sp_path)
                _datas.append((str(_data_file), str(_rel.parent)))
                print(f"[spec] Bundling onnx_asr data: {_rel}")

analysis = Analysis(
    ["lazy-to-text-ui.py"],
    pathex=[str(app_dir)],
    binaries=[],
    datas=_datas,
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

# Allow wrapper scripts to produce differently-named bundles so CPU
# and CUDA variants can coexist in ``dist/`` without overwriting.
_build_name = _os.environ.get("LAZYTOTEXT_BUILD_NAME", "LazyToText")

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=_build_name,
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
    name=_build_name,  # → dist/<_build_name>/
)
