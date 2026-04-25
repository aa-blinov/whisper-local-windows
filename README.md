# Lazy to Text

Press a global hotkey, speak, paste. Local Whisper-based speech-to-text for Windows, with a Qt UI.

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![Qt](https://img.shields.io/badge/UI-PySide6-41cd52)](https://doc.qt.io/qtforpython-6/)
[![faster-whisper](https://img.shields.io/badge/backend-faster--whisper-orange)](https://github.com/SYSTRAN/faster-whisper)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![Hero](docs/screenshots/hero.png)

---

## What it is

A Windows desktop application that records microphone audio on a global hotkey,
transcribes it locally via the [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
Python library (CTranslate2 under the hood), and pastes the resulting text
into the focused window.

The audio never leaves the machine. No Docker, no Wyoming protocol, no cloud
APIs — the model runs in-process and downloads from Hugging Face on first use.

## Features

- Global hotkey activation (`Ctrl+F2` / `Ctrl+F3` by default).
- Auto-paste into the foreground window after transcription.
- 6 model presets (`tiny` → `large-v3`) with size / VRAM / speed / quality
  metadata; switching swaps the in-process model on the fly.
- Accepts any CTranslate2-converted Whisper model from Hugging Face — point
  ``whisper.model`` at the HF id and it works (e.g. ``bzikst/faster-whisper-large-v3-russian``).
- Searchable transcription history persisted to JSON.
- Settings auto-save on edit; "Reset to defaults" button.
- Live application log stream inside the UI.
- Native Windows system tray (close button hides to tray; right-click → Show / Quit).
- Single-instance guard via a named mutex with a branded warning dialog.
- Audio feedback for start / stop / cancel events with prewarmed playback so
  the first keypress is never silent.

## Quick start

Requires Windows 10/11, Python 3.12, and a microphone. A CUDA-capable GPU is
optional but strongly recommended for the larger models.

```powershell
git clone https://github.com/aa-blinov/lazy-to-text.git
cd lazy-to-text

uv sync                      # creates the venv from uv.lock
uv run lazy-to-text-ui       # launch the app
```

The first launch downloads the configured model from Hugging Face (≈3 GB for
``large-v3``) into ``~/.cache/huggingface/hub`` — subsequent launches use the
local cache. Press `Ctrl+F2`, speak, press `Ctrl+F3` — the transcript is
pasted into whatever has focus when you stop recording.

## Screenshots

| Models | Shortcuts |
| :---: | :---: |
| ![Models](docs/screenshots/models.png) | ![Shortcuts](docs/screenshots/shortcuts.png) |

| History | Logs |
| :---: | :---: |
| ![History](docs/screenshots/history.png) | ![Logs](docs/screenshots/logs.png) |

## Hotkeys

| Action | Default | Configurable |
| --- | --- | --- |
| Start recording | `Ctrl+F2` | yes — Shortcuts tab |
| Stop recording + transcribe | `Ctrl+F3` | yes — Shortcuts tab |
| Hide / show window | close button / tray click | no |

Hotkey edits in the Shortcuts tab persist immediately on focus loss; no Save
button. The auto-paste toggle behaves the same way.

## Configuration

`config.yaml` is created on first launch in the project root (or next to the
executable in a built distribution). Most fields are exposed in the UI; the
file is the source of truth.

```yaml
whisper:
  model: large-v3                # alias from app/model_mapping.py, or any
                                 # CTranslate2-converted Whisper model on
                                 # Hugging Face (e.g.
                                 # bzikst/faster-whisper-large-v3-russian)
  device: auto                   # auto | cpu | cuda
  compute_type: float16          # float16 | int8_float16 | int8 | float32
  language: ru                   # ISO code, or omit/null for auto-detect
  beam_size: 5

hotkey:
  start_recording_hotkey: ctrl+f2
  stop_recording_hotkey: ctrl+f3

clipboard:
  auto_paste: true
  preserve_clipboard: false
  key_simulation_delay: 0.05

audio:
  channels: 1
  dtype: float32
  max_duration: 300

audio_feedback:
  enabled: true
  start_sound: assets/sounds/record_start.wav
  stop_sound: assets/sounds/record_stop.wav
  cancel_sound: assets/sounds/record_cancel.wav

history:
  enabled: true
  max_entries: 1000
  auto_cleanup_days: 30
```

To reset settings, delete `config.yaml` and relaunch — defaults are written
back. The Shortcuts tab "Reset to defaults" button does the same for hotkey
and auto-paste fields without touching the rest of the file.

## Architecture

```
                                ┌─────────────────────────────┐
                                │  Qt UI (app/gui)            │
                                │  views, widgets, controllers│
                                └─────────────┬───────────────┘
                                              │
                                Qt signals    │
                                              ▼
┌─────────────┐    callbacks   ┌──────────────────────────────┐
│ Hotkey      ├───────────────▶│  StateManager  (app/)        │
│ Listener    │                │  recording / processing /    │
└─────────────┘                │  model_loading state machine │
                                └─────┬───────────────┬─────────┘
                                      │               │
                           audio_data │               │ transcribe(audio)
                                      ▼               ▼
                         ┌──────────────────┐  ┌──────────────────────┐
                         │ AudioRecorder    │  │ TranscriptionBackend │
                         │ sounddevice +    │  │   (Protocol)         │
                         │ daemon thread    │  └─────────┬────────────┘
                         └──────────────────┘            │
                                                         ▼
                                            ┌────────────────────────┐
                                            │ FasterWhisperBackend   │
                                            │ in-process via         │
                                            │ faster_whisper +       │
                                            │ ctranslate2 (CPU/GPU)  │
                                            └────────────────────────┘
```

The Qt layer (``app/gui/``) holds every UI concern. The domain layer
(``app/state_manager.py``, ``app/audio_recorder.py``,
``app/clipboard_manager.py``) is what ``RecordingController`` wires Qt
signals into. The speech-to-text engine sits behind the
``TranscriptionBackend`` protocol in ``app/backends/`` so swapping in
GigaAM, distilled Whisper variants, or a cloud API is a drop-in change.
``BackendStatusPoller`` polls ``backend.status()`` off the UI thread and
feeds the result into the TopBar pill.

## Building a standalone executable

```powershell
.\build-exe.ps1            # folder build via lazy_to_text.spec
.\build-exe.ps1 -OneFile   # single-file build (assets unpacked from _MEIPASS)
.\build-exe.ps1 -Clean     # remove dist/ and build/ first
```

Output: `dist\LazyToText\LazyToText.exe` (folder mode, ~144 MB, 264 files,
fastest startup) or `dist\LazyToText.exe` (`-OneFile`, ~120 MB, slower
startup because assets unpack on each run). The build script delegates to
`pyinstaller` via `uv run`; a fresh `uv sync` runs first unless
`-SkipSync` is passed.

## Installing locally

After a folder build, install per-user (no admin required) into
`%LOCALAPPDATA%\Programs\LazyToText` and add a Start Menu shortcut. Run
from PowerShell, in the repository root:

```powershell
$dest = "$env:LOCALAPPDATA\Programs\LazyToText"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
robocopy "dist\LazyToText" $dest /MIR /NFL /NDL /NJH /NJS /NP

$start = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Lazy to Text.lnk"
$wsh   = New-Object -ComObject WScript.Shell
$lnk   = $wsh.CreateShortcut($start)
$lnk.TargetPath       = "$dest\LazyToText.exe"
$lnk.WorkingDirectory = $dest
$lnk.IconLocation     = "$dest\LazyToText.exe"
$lnk.Description      = "Local Whisper speech-to-text for Windows"
$lnk.Save()
```

After this, `Win` + typing "Lazy to Text" launches it from the Start
menu. You can then right-click the taskbar icon → **Pin to taskbar** for
single-click access.

To launch automatically at login, copy the shortcut into the user's
Startup folder:

```powershell
Copy-Item $start "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Lazy to Text.lnk"
```

To uninstall:

```powershell
Get-Process LazyToText -ErrorAction SilentlyContinue | Stop-Process -Force
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Programs\LazyToText"
Remove-Item -Force "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Lazy to Text.lnk" -ErrorAction SilentlyContinue
Remove-Item -Force "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Lazy to Text.lnk" -ErrorAction SilentlyContinue
```

The `config.yaml` and `logs\` folder live next to the executable; they
are deleted along with the install folder. Your transcription history
(`logs\transcription_history.json`) goes with them — back it up first if
you want to keep it.

## GPU support

By default the backend uses ``device: auto`` and ``compute_type: float16``
so faster-whisper picks the GPU when CTranslate2 finds a CUDA-capable card
and falls back to CPU otherwise. For a CUDA setup you need:

- An NVIDIA GPU with up-to-date drivers (CUDA 12 era).
- ``cuBLAS`` and ``cuDNN`` libraries on ``PATH`` — the simplest way is
  to install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
  and [cuDNN 9.x](https://developer.nvidia.com/cudnn).

CPU-only is fine for ``tiny`` / ``base`` / ``small``; ``medium`` and
``large-v*`` will be slow without a GPU.

## Development

```powershell
uv sync --group dev
uv run pytest                         # full test suite (~210 tests)
uv run python lazy-to-text-ui.py      # run from source
uv run python scripts/generate_screenshots.py  # regenerate docs/screenshots
```

Tests live in `tests/` and use `pytest-qt` with the real Windows Qt platform.
GUI tests can run headless via `QT_QPA_PLATFORM=offscreen`.

## Roadmap

- [ ] Live audio level meter while recording
- [ ] System tray notification when transcription completes
- [ ] Light theme + custom QSS
- [ ] GigaAM-v3 backend (Sber's Russian-specialised Conformer)
- [ ] Optional cloud backends (OpenAI, Groq) behind the same `TranscriptionBackend`
- [ ] Distil / turbo model presets in the registry
- [ ] Auto-detect language toggle in the Models tab

## Tech stack

- Python 3.12
- PySide6 (Qt 6.11) for the UI
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + CTranslate2 for in-process inference
- `sounddevice` for audio capture, `pyautogui` + `pyperclip` for paste,
  `global-hotkeys` + `pywin32` for Windows-native hotkey registration

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (SYSTRAN) and
  [CTranslate2](https://github.com/OpenNMT/CTranslate2) under the hood.
- [OpenAI Whisper](https://github.com/openai/whisper) — the underlying model.
- UI direction borrowed from [Spokenly](https://spokenly.app/) (macOS).

## License

[MIT](LICENSE)
