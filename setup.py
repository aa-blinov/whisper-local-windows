"""py2app build script for the macOS .app bundle.

Run via the helper:

    ./scripts/build-macos.sh

or directly with::

    uv run python setup.py py2app -A   # alias mode (dev — symlinks back to source)
    uv run python setup.py py2app      # release mode (full self-contained .app)

Why py2app instead of PyInstaller
---------------------------------
- **Alias mode** rebuilds in ~5–10 s and symlinks the bundle into
  the source tree, so editing a ``.py`` file is reflected on the
  next launch with no rebuild — exactly the iteration speed we
  need while solving the macOS Accessibility flow.
- The host process inside the bundle reports as ``Lazy to Text``
  (not ``python3.12``), so System Settings → Privacy & Security →
  Accessibility shows our app with our icon — drag-and-drop / +
  Add works the way every other Mac app teaches.

The release path stays available (``./scripts/build-macos.sh
--release``) for distribution work later — same setup.py, just
without the ``-A`` flag.

Information stays in setup.py rather than pyproject.toml because
py2app reads ``setup.py``/setuptools options directly; there is
no PEP 621 equivalent for the ``OPTIONS`` dict it expects.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup


# Bail loudly if someone runs the build off-platform — the
# generated bundle wouldn't work anyway and the error from
# ``py2app`` itself is much less informative than this hint.
if sys.platform != "darwin":
    sys.stderr.write(
        "setup.py is only useful for the macOS .app bundle; on "
        "Windows / Linux run the app directly via "
        "``uv run lazy-to-text-ui``.\n"
    )
    sys.exit(1)


_PROJECT_ROOT = Path(__file__).resolve().parent
_ICON_PATH = _PROJECT_ROOT / "app" / "assets" / "lazy_to_text.icns"


def _patch_builtin_zlib_for_py2app() -> None:
    """Work around py2app 0.28 assuming ``zlib.__file__`` exists.

    Our uv-managed Python 3.12 build exposes ``zlib`` as a built-in
    module (``__spec__.origin == "built-in"``), so py2app's release
    path crashes when it blindly tries to copy ``zlib.__file__`` into
    the bundle. The stdlib zlib implementation is already linked into
    libpython in this configuration, so runtime does not need a
    separate extension file. We hand py2app a tiny placeholder path so
    the copy step succeeds instead of aborting the whole build.
    """
    import zlib

    if getattr(zlib, "__file__", None):
        return

    placeholder = _PROJECT_ROOT / ".py2app-zlib-built-in-placeholder"
    if not placeholder.exists():
        placeholder.write_bytes(b"")
    zlib.__file__ = str(placeholder)


_patch_builtin_zlib_for_py2app()


def _seed_tcl_tk_env_for_py2app() -> None:
    """Point py2app's tkinter recipe at the live Tcl/Tk runtime.

    py2app 0.28 unconditionally imports ``_tkinter`` during its
    recipe pass and calls ``_tkinter.create()`` to inspect the Tk
    version. Under the uv-managed Python build that powers this repo,
    the runtime libraries live under ``~/.local/share/uv/python/...``
    and are not discovered by py2app's fallback search path, which
    makes release builds fail before our app code is even considered.

    We do not ship or use tkinter at runtime, but exporting the live
    library paths here keeps py2app's probe from aborting the build.
    """
    if os.environ.get("TCL_LIBRARY") and os.environ.get("TK_LIBRARY"):
        return
    try:
        import _tkinter
    except Exception:
        return
    try:
        tcl = _tkinter.create()
        tcl_library = tcl.call("info", "library")
    except Exception:
        return
    if tcl_library:
        os.environ.setdefault("TCL_LIBRARY", str(tcl_library))
        tk_library = str(Path(str(tcl_library)).with_name("tk8.6"))
        if Path(tk_library).exists():
            os.environ.setdefault("TK_LIBRARY", tk_library)


_seed_tcl_tk_env_for_py2app()

# py2app expects the entry script as a positional argument under
# ``app=[...]``.  We point it at our existing thin shim
# ``lazy-to-text-ui.py`` which already exists for ``uv run`` —
# avoids duplicating the import + ``main()`` call.
_ENTRY_SCRIPT = str(_PROJECT_ROOT / "lazy-to-text-ui.py")


# All Python packages py2app should treat as "first-party" and
# bundle in full — anything imported lazily inside these still
# gets traced.  ``app`` is enough; py2app picks up its sub-
# packages (``app.gui``, ``app.backends``, …) automatically.
_PACKAGES = [
    "app",
    # Heavy native deps that py2app's modulegraph sometimes
    # misses — listing them here forces full bundling so the
    # release ``.app`` still works.
    "PySide6",
    "shiboken6",
    "onnxruntime",
    "onnx_asr",
    "huggingface_hub",
    "soundfile",
    "sounddevice",
    "_sounddevice_data",
    "pynput",
    "pyperclip",
    "pyautogui",
    "playsound3",
    "platformdirs",
    "filelock",
    "psutil",
]


# ``Info.plist`` overrides — py2app merges these into the
# template plist it generates.  Each entry exists for a reason:
#
# - ``CFBundleIdentifier`` is the unique key macOS uses for
#   per-app permissions (Accessibility, microphone) — pinning it
#   means future builds keep the same Privacy & Security entries
#   instead of asking for permission again.
# - ``LSMinimumSystemVersion`` matches what PySide 6.8+ wheels
#   ship with; setting it any lower would crash on launch
#   because Qt's frameworks won't load.
# - ``NSMicrophoneUsageDescription`` is required by macOS for
#   the microphone TCC prompt — without it the app crashes on
#   first ``sounddevice.InputStream`` open.
# - ``NSAppleEventsUsageDescription`` covers our use of the
#   ``open`` URL scheme to launch System Settings from the
#   Accessibility-permission banner.
# - ``LSUIElement = False`` keeps the Dock icon visible (we want
#   Cmd-Tab + Dock-click flow); set it to ``True`` to hide.
_PLIST = {
    "CFBundleName": "Lazy to Text",
    "CFBundleDisplayName": "Lazy to Text",
    "CFBundleIdentifier": "ai.eora.lazytotext",
    "CFBundleVersion": "0.0.1",
    "CFBundleShortVersionString": "0.0.1",
    "CFBundleExecutable": "Lazy to Text",
    "LSMinimumSystemVersion": "12.0",
    "LSUIElement": False,
    "NSHighResolutionCapable": True,
    "NSMicrophoneUsageDescription": (
        "Lazy to Text records audio from the microphone to "
        "transcribe it locally on your Mac. Audio never leaves "
        "the device."
    ),
    "NSAppleEventsUsageDescription": (
        "Lazy to Text opens System Settings to help you grant "
        "the Accessibility permission required for global hotkeys."
    ),
    # Make sure the bundle is treated as a regular app, not a
    # helper / agent.
    "LSApplicationCategoryType": "public.app-category.productivity",
}


_OPTIONS = {
    "argv_emulation": False,  # Qt has its own event loop; Apple's
                              # AppleEvent emulation interferes.
    "iconfile": str(_ICON_PATH) if _ICON_PATH.exists() else None,
    "plist": _PLIST,
    "packages": _PACKAGES,
    "includes": [
        # Stdlib modules onnxruntime / huggingface_hub import lazily
        # that py2app's tracer occasionally misses.
        "json", "logging", "threading", "queue", "subprocess",
        "ctypes", "ctypes.util",
    ],
    # Drop the test scaffolding from the release bundle — saves
    # ~30 MB and tests reference Qt's offscreen platform that
    # the Finder-launched .app shouldn't carry.
    "excludes": [
        "tests", "pytest", "pytest_qt",
        # PyInstaller is a build-only tool — never wanted at
        # runtime in a competing bundler's output.
        "pyinstaller",
        # Qt-приложение не использует Tk. py2app 0.28 всё равно
        # заходит в tkinter-recipe и на uv Python 3.12 падает,
        # если в окружении нет полноценного Tcl/Tk runtime.
        "tkinter", "Tkinter", "_tkinter",
        # ``rubicon-objc`` installs ``rubicon`` as a namespace
        # package (no top-level ``__init__.py``). py2app 0.28's
        # package collector trips over that during release builds
        # and aborts with ``ImportError: No module named 'rubicon'``
        # even though our app never imports it. Excluding the
        # namespace keeps the standalone bundle buildable.
        "rubicon",
    ],
    # Bundle ``onnxruntime``'s shared libraries instead of leaving
    # them as broken symlinks pointing back into the source venv.
    # Only matters in non-alias (release) mode.
    "frameworks": [],
    "resources": [],
    # ``alias`` is set on the command line via ``-A`` for dev
    # builds — keep it out of the static config so release builds
    # are full bundles by default.
    "optimize": 0,
    "strip": False,
}


# py2app 0.28 raises ``DistutilsOptionError: install_requires is
# no longer supported`` the moment it sees a populated
# ``Distribution.install_requires`` attribute.  Modern setuptools
# (80+) auto-derives that attribute from PEP 621
# ``[project] dependencies`` in ``pyproject.toml``, which our
# project uses.  Passing ``install_requires=[]`` explicitly here
# wins over the auto-derived value (setuptools' ``setup()``
# kwargs always take precedence over ``pyproject.toml``); the
# empty list is falsy so py2app's ``if self.distribution.install_requires``
# check passes.  The runtime still resolves dependencies through
# pyproject.toml directly via ``uv sync`` — this only quiets the
# legacy bridge between the two metadata systems for the duration
# of a py2app invocation.
setup(
    app=[_ENTRY_SCRIPT],
    name="Lazy to Text",
    version="0.0.1",
    install_requires=[],
    options={"py2app": _OPTIONS},
)
