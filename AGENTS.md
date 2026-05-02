# AGENTS.md — agent context for `lazy-to-text`

Local-first speech-to-text with a Qt UI. Cross-platform (Windows /
macOS), single ONNX inference path via [`onnx-asr`](https://github.com/istupakov/onnx-asr).
This file is your project quickstart — keep it skim-able.

## Run / build / test

```bash
uv sync                          # bootstrap venv from uv.lock
uv run lazy-to-text-ui           # launch the app in dev mode

# Portable bundle — wrapper picks the right tool per OS:
./scripts/build-macos.sh         # py2app alias .app, 5–10 s (Mac)
./scripts/build-macos.sh --release   # py2app full .app, 5–10 min (Mac)
.\scripts\build-windows.ps1      # PyInstaller folder bundle (Win)
.\scripts\build-windows.ps1 -OneFile  # PyInstaller single .exe (Win)

# Tests — always with offscreen Qt platform plugin so Cocoa /
# WinAPI never opens a real window in CI
QT_QPA_PLATFORM=offscreen uv run python -m pytest tests/
QT_QPA_PLATFORM=offscreen uv run python -m pytest tests/gui/    # GUI subset (~25s)
QT_QPA_PLATFORM=offscreen uv run python -m pytest tests/backends/test_subprocess_backend.py    # ~90s, real spawn
```

The full suite is **624 passed, 10 skipped** at last commit. Skipped
tests are mostly engine-specific paths that need a real model.

## Where things live

| Data | Dev (`uv run`) | macOS `.app` (frozen) | Windows portable (frozen) |
| --- | --- | --- | --- |
| `config.yaml` | `<project>/config.yaml` | `~/Library/Application Support/LazyToText/` | `%APPDATA%\LazyToText\` |
| `app.log` + history | `<project>/logs/` | `~/Library/Logs/LazyToText/` | `%LOCALAPPDATA%\LazyToText\Log\` |
| Model weights (HF hub) | `<project>/models/hub/` | `~/Library/Caches/LazyToText/models/hub/` | `%LOCALAPPDATA%\LazyToText\Cache\models\hub\` |
| Single-instance lock | filelock under cache | `~/Library/Caches/LazyToText/LazyToTextQt.lock` | named mutex |

`platformdirs.user_{config,log,cache}_dir("LazyToText", appauthor=False)`
is the single source of truth — `app/utils.py:_is_frozen()` and
`app/config_manager.py:_resolve_base_dir()` gate the frozen paths.
Two frozen targets right now: py2app's `.app` on macOS and
PyInstaller's folder bundle on Windows.

## Platform conditionals — what to grep for

- **`sys.platform == "win32"`** — `clipboard_manager`, `audio_feedback`,
  `instance_manager`, `hotkey_listener` (Win32-API replacements:
  `win32api`, `winsound`, named mutex). Each Win-specific import is
  guarded under the same conditional, so the modules import cleanly
  on macOS / Linux.
- **`sys.platform == "darwin"`** — Accessibility / Microphone permission
  banners (`app/gui/views/_accessibility_check.py`,
  `_microphone_check.py`), Dock / tray icon rendering
  (`app/gui/widgets/tray_icon.py`, `app/gui/app.py:_render_dock_icon_at`),
  bundle-mode backend swap (`app/gui/app.py:677`), CoreML provider
  selection (`app/backends/onnx_backend.py:412+`).
- **`getattr(sys, "frozen", False)`** — flips data paths to `~/Library/*`
  and selects the in-process `RegistryBackend` over the spawn-based
  `SubprocessBackend`.

## Backend topology

```
                    SubprocessBackend                RegistryBackend
                    ──────────────────                ─────────────────
                    spawn worker process              in-process
   used on:         Windows / dev macOS               macOS .app (frozen)
   IPC:             multiprocessing.Pipe              direct method calls
   why:             dodge Win32 DLL-loader-lock       py2app + spawn fight
                                                      (launcher binary
                                                      can't be re-execed)
```

Decision lives in `app/gui/app.py:677`:

```python
if sys.platform == "darwin" and getattr(sys, "frozen", False):
    _early_backend = RegistryBackend(**_backend_kwargs)
else:
    _early_backend = SubprocessBackend(**_backend_kwargs)
```

When editing `subprocess_backend.py`, remember it now uses
`multiprocessing.get_context("spawn")` instead of the bare
`multiprocessing.Pipe()` / `Process()`. The fixture in
`tests/backends/test_subprocess_backend.py` patches `get_context`
to return a fake ctx — patching the bare module attributes alone
won't intercept the calls.

## CoreML / Apple Silicon

Provider tuple for ORT on Mac:

```python
("CoreMLExecutionProvider", {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "ALL",         # NE + GPU + CPU
    "RequireStaticInputShapes": "0",
    "EnableOnSubgraphs": "0",
}),
"CPUExecutionProvider",
```

`onnx_backend.py:_detect_active_provider` probes the live session
and falls back to CPU on accelerator failure. It unwraps model
adapters (like GigaAM's `TextResultsAsrAdapter`) to find the
inner session. Some models (GigaAM / T-One) are pinned to CPU
via `prefer_cpu_provider=True` in `model_mapping.py` to avoid
known CoreML compilation crashes.

## macOS HUD / Fullscreen Overlay

The recording overlay (`app/gui/widgets/recording_overlay.py`) is
designed to stay visible over native full-screen applications.
Key technical choices:

- **Agent Mode:** `setup.py` sets `LSUIElement = True`. This
  background-app mode is required for windows to reliably pierce
  other applications' full-screen Spaces.
- **Window Level:** Uses `NSScreenSaverWindowLevel` to stay above
  full-screen shields, the Notch, and the Menu Bar.
- **Collection Behavior:** Uses `CanJoinAllSpaces | FullScreenAuxiliary | IgnoresCycle | Stationary`. This ensures it's on every desktop, follows Cmd+Tab ignores, and doesn't slide during swipes.
- **Focus Policy:** We strictly avoid `raise_()` on background events
  (like finishing transcription) to prevent focus stealing. Toasts
  use `WA_ShowWithoutActivating`.

## UI Performance & Layout

- **FlowLayout:** Optimized to use uniform `spacing()` instead of
  calling `style().layoutSpacing()` thousands of times per frame
  during list resize events.
- **Batching:** `setUpdatesEnabled(False/True)` is used during
  multi-widget updates (like model selection) to prevent layout
  thrashing and scroll freezes.
- **Deferred Tasks:** Heavy UI/IPC handshakes (like model change)
  are deferred via `QTimer.singleShot(0)` to keep animations fluid.

## Hotkey defaults — and why they differ per OS

```python
# config_manager.py
if sys.platform == "darwin":
    _DEFAULT_START_HOTKEY = "ctrl+f2"
    _DEFAULT_STOP_HOTKEY  = "ctrl+f3"
    _DEFAULT_CANCEL_HOTKEY = "ctrl+f6"
else:
    _DEFAULT_START_HOTKEY = "ctrl+f2"
    _DEFAULT_STOP_HOTKEY  = "ctrl+f3"
    _DEFAULT_CANCEL_HOTKEY = "ctrl+f6"
```

macOS reserves `Ctrl+F1`..`Ctrl+F7` for system keyboard navigation
(focus → menu bar / Dock / window / toolbar / floating window /
next window / status menu). We use higher F-keys or safe ones like
F2/F3/F6 where possible to avoid conflicts.


## macOS App menu (Cmd+, / Cmd+Q / About)

`app/gui/main_window.py:_install_app_menu` builds a single
`QMenuBar` with three `QAction`s:

| MenuRole | Text | Shortcut | Handler |
| --- | --- | --- | --- |
| `AboutRole` | About Lazy to Text | — | `_show_about_dialog` |
| `PreferencesRole` | Settings… | `Ctrl+,` (→ `Cmd+,`) | `_open_settings_view` |
| `QuitRole` | Quit Lazy to Text | `Ctrl+Q` (→ `Cmd+Q`) | `_quit_application` |

Qt's Cocoa platform plugin promotes these into the global App menu
(under the Apple logo) regardless of which submenu they're attached
to — the host menu's title is irrelevant on Mac. On Windows / Linux
the same actions appear in a regular `Lazy to Text` top menu.
`Ctrl+,` / `Ctrl+Q` → `Cmd+,` / `Cmd+Q` translation comes from Qt's
portable `QKeySequence` layer.

Quit goes through `request_quit()` (flips `_quitting=True` so
close-to-tray override doesn't kick in) **and** `QApplication.quit()`
(drops the event loop — `setQuitOnLastWindowClosed(False)` is set
when the tray is alive).

## Permissions UX (macOS)

Two TCC-gated capabilities are checked at startup:

- **Accessibility** (`AXIsProcessTrusted`) — for `pynput` global
  hotkeys + `pyautogui` autopaste keystrokes. macOS won't prompt;
  Settings shows a two-state banner that opens *Privacy & Security
  → Accessibility* and a separate "granted but needs restart" banner
  with an in-app relaunch button.
- **Microphone** (`AVCaptureDevice.authorizationStatus`) — system
  prompt fires automatically via `requestAccessForMediaType_` the
  first time we call it. The Settings banner shows authorized /
  denied / not_determined states, with *Open Microphone Settings*
  for the denied case.

Both checks are no-ops on non-macOS (early-return on `sys.platform
!= "darwin"`).

## Dialogs — `app/gui/widgets/dialogs.py`

Three helpers, all stamped with the live `QApplication.windowIcon()`
to replace QMessageBox's macOS template glyph (system "?"):

- `confirm(parent, title, text, *, default_yes, yes_label, cancel_label)`
- `confirm_three_way(parent, title, text, *, yes_label, no_label, cancel_label)`
- `notify(parent, title, text, *, kind, informative, rich_text)` —
  `informative` is the QMessageBox secondary body (lighter weight);
  `rich_text=True` enables HTML and clickable `<a href="…">`.

`tests/gui/conftest.py:stub_modal_dialogs` autouse-mocks all three
at every callsite so headless tests don't hang on `exec()`. When
adding a new dialog, route through these helpers — direct
`QMessageBox.question/.information` will (a) show the system glyph
on Mac instead of the app icon and (b) bypass the test mock.

## Tests — naming conventions

- `tests/gui/` — Qt widget logic, run with `QT_QPA_PLATFORM=offscreen`
- `tests/backends/` — backend dispatch + worker process tests
- `tests/services/` — pure-Python services (audio recorder, clipboard)
- `tests/utils/` — `app/utils.py` helpers

Always pass `-x` during dev so the first failure surfaces fast;
GUI failures often cascade through the autouse fixtures.

## Style / safety rails

- **Don't reach for emojis** — they don't render right in the dark
  QSS theme and the Logs view is monospaced. Same for source files.
- **Don't add new dependencies casually** — the project ships a
  curated set in `pyproject.toml`; any addition should justify itself
  against the package size and Apple Silicon wheel availability
  (some ML libs only ship x86_64 wheels).
- **Preserve `Co-Authored-By: Codex Opus 4.7 <noreply@anthropic.com>`
  in commit footers** — matches existing branch style.
