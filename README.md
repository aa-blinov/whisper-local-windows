# Lazy to Text

Press a global hotkey, speak, paste. Local Whisper / GigaAM / Parakeet speech-to-text for Windows, with a Qt UI.

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![Qt](https://img.shields.io/badge/UI-PySide6-41cd52)](https://doc.qt.io/qtforpython-6/)
[![faster-whisper](https://img.shields.io/badge/backend-faster--whisper-orange)](https://github.com/SYSTRAN/faster-whisper)
[![GigaAM](https://img.shields.io/badge/backend-gigaam-7a3fff)](https://github.com/salute-developers/GigaAM)
[![NeMo](https://img.shields.io/badge/backend-NVIDIA%20NeMo-76b900)](https://github.com/NVIDIA/NeMo)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![Hero](docs/screenshots/hero.png)

---

## What it is

A Windows desktop application that records microphone audio on a global hotkey,
transcribes it locally via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2), Sber's [GigaAM](https://github.com/salute-developers/GigaAM), or
NVIDIA's [NeMo](https://github.com/NVIDIA/NeMo) running Parakeet TDT v3, and
pastes the resulting text into the focused window.

The audio never leaves the machine. No Docker, no cloud APIs — every model
runs in-process and downloads from Hugging Face / Sber's CDN on first use.

## Features

- **Three engines, one Protocol.** A `RoutedBackend` facade picks between
  faster-whisper, GigaAM, and NVIDIA NeMo based on the selected model and
  rebuilds the inner backend transparently when the user switches across
  engines.
- **Per-model inference settings.** Each card surfaces its own inline
  panel — language, VAD filter, beam size, temperature, initial prompt —
  persisted under `model_overrides.<alias>` and pushed live into the
  running backend without a restart.
- **11 model presets.** Whisper Distil, Turbo (fp16/int8), Large v3
  (fp16/int8), Russian fine-tunes (fp16/int8), GigaAM v3 e2e CTC + RNN-T,
  NVIDIA Parakeet TDT v3 (multilingual, 25 EU languages incl. Russian).
  Family chips on each card colour-code the lineage.
- **Long-form audio.** Captures over 25 s on GigaAM route through
  `transcribe_longform` with pyannote VAD; faster-whisper goes through
  Silero VAD when the toggle is on; Parakeet TDT v3 handles multi-minute
  audio natively via local relative-position attention.
- **Cancel-load button.** Mis-clicked a heavy model card? A red Cancel
  pill appears next to the loading indicator in the topbar. Clicking it
  abandons the in-flight load, rolls back the active card / config /
  topbar pill to whatever was active before, and clears the loading
  state immediately — no waiting for a stuck import to finish.
- **Splash screen on cold start.** When the persisted model is cached,
  the app pre-loads it in the main thread on a movable / minimisable
  splash widget BEFORE the main window appears — so the main window
  never freezes for the 11–30 s it takes NeMo or torch to import.
- **Live resource monitor.** A 4-block widget in the topbar tracks
  CPU / RAM / GPU utilisation / VRAM via `psutil` + `nvidia-ml-py`,
  refreshing every two seconds.
- **Recording status chip with live VU.** A `STATUS · Idle / Recording /
  Processing` chip pinned to the sidebar's bottom-left corner mirrors the
  topbar's resource cards — the VU meter under it animates while
  recording, so silent / muted mics show up before you finish speaking.
- **Toast confirmation.** A bottom-right banner with a preview of the
  latest transcription pops up after every successful run — silent
  paste flow used to be invisible.
- **Searchable model browser.** A search box plus family-filter chips
  (All / Whisper / Whisper Turbo / Whisper Distil / Whisper RU /
  GigaAM / Parakeet) narrow the grid; an empty-state placeholder appears
  if nothing matches.
- **Coloured logs view.** Records colour-coded by level + logger
  source, with a "Show network logs" toggle that hides httpx /
  huggingface_hub noise and a search field that filters the buffer
  on the fly.
- **History detail dialog.** Double-click any history row to read the
  full transcription in a modal, copy it with one button. Hover for a
  full-text tooltip on the truncated cell. Export to a plain-text file
  via the toolbar button.
- **Auto-paste into the focused window.** Triggered by simulated
  keystrokes; the live `ClipboardManager` picks up toggle changes in
  Settings without a restart.
- **Modern visual treatment.** Soft drop-shadow cards, Heroicons in
  the sidebar, bundled Inter font, color-coded family badges, smooth
  pixel-level scrolling everywhere.
- **Native Windows polish.** System tray with state-aware icon,
  single-instance guard via named mutex, confirmation dialog before
  destructive history wipes, Ctrl+1..4 keyboard shortcuts to switch
  tabs.

## Quick start

Requires Windows 10/11, Python 3.12, and a microphone. A CUDA-capable GPU is
optional but strongly recommended — every preset in the registry now is
either a Large-class Whisper variant or GigaAM v3.

```powershell
git clone https://github.com/aa-blinov/lazy-to-text.git
cd lazy-to-text

uv sync                      # creates the venv from uv.lock
uv run lazy-to-text-ui       # launch the app
```

The first launch leaves no model loaded — pick one from the Models tab and
click **Download**. Weights for Whisper-derivatives and NeMo / Parakeet
land in `<project>/models/hub/` (HF cache); GigaAM weights go to
`~/.cache/gigaam/<name>.ckpt`. Press `Ctrl+F2`, speak, `Ctrl+F3` — the
transcript pastes into whatever has focus, and a banner confirms in the
bottom-right. Subsequent launches pre-load the persisted active model on a
splash screen so the main window appears already responsive.

## Screenshots

| Models | Settings |
| :---: | :---: |
| ![Models](docs/screenshots/models.png) | ![Settings](docs/screenshots/shortcuts.png) |

| History | Logs |
| :---: | :---: |
| ![History](docs/screenshots/history.png) | ![Logs](docs/screenshots/logs.png) |

## Hotkeys

| Action | Default | Configurable |
| --- | --- | --- |
| Start recording | `Ctrl+F2` | yes — Settings tab |
| Stop recording + transcribe | `Ctrl+F3` | yes — Settings tab |
| Switch tab (Models / Settings / History / Logs) | `Ctrl+1..4` | no |
| Hide / show window | close button / tray click | no |

Hotkey edits in the Settings tab persist immediately on focus loss; no Save
button. The auto-paste toggle behaves the same way and is also pushed live
into the running `ClipboardManager`.

## Configuration

`config.yaml` lives in the project root (or next to the executable in a
built distribution). Most fields are exposed in the UI; the file is the
source of truth.

```yaml
whisper:
  model: gigaam-v3-e2e-rnnt    # any alias from app/model_mapping.py, or
                               # a Hugging Face id for an unregistered
                               # CTranslate2 Whisper model
  device: auto                 # auto | cpu | cuda
  compute_type: float16        # float16 | int8_float16 | int8 | float32
  language: ru                 # ISO code, or omit for auto-detect
  beam_size: 5

# Per-model overrides — written by the inline Inference settings panel
# on the active card. Each entry is keyed by alias.
model_overrides:
  large-v3:
    language: ru
    vad_filter: true
    beam_size: 7
    temperature: 0.0
    initial_prompt: "Anthropic, Claude, faster-whisper, ctranslate2."
  gigaam-v3-e2e-rnnt: {}       # GigaAM ignores all of these (end-to-end)

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
  device: 31                   # input device index from sounddevice; the
                               # Settings tab dropdown writes this for you

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
back. The Settings tab "Reset to defaults" button does the same for hotkey
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
┌─────────────┐    callbacks    ┌──────────────────────────────┐
│ Hotkey      ├────────────────▶│  StateManager  (app/)        │
│ Listener    │                 │  recording / processing /    │
└─────────────┘                 │  model_loading state machine │
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
                                          ┌────────────────────────────┐
                                          │ RoutedBackend  (facade)    │
                                          │ picks engine by model.     │
                                          │ backend_kind, swaps inner  │
                                          │ on cross-engine change.    │
                                          └────┬─────────────┬─────────┴───┐
                                               │             │             │
                                faster_whisper ▼             ▼ gigaam      ▼ nemo
                          ┌───────────────────────┐ ┌───────────────────┐ ┌────────────────────┐
                          │ FasterWhisperBackend  │ │ GigaamBackend     │ │ NemoBackend        │
                          │ ctranslate2 + cuBLAS  │ │ pyannote VAD for  │ │ Parakeet TDT v3    │
                          │ + cuDNN via cu12      │ │ longform >25 s    │ │ rel_pos_local_attn │
                          │ wheels; tqdm progress │ │ via               │ │ for multi-minute   │
                          │ piped through to UI   │ │ gigaam[longform]  │ │ audio; nemo_toolkit│
                          └───────────────────────┘ └───────────────────┘ │ + signal/numpy     │
                                                                          │ shims for Windows  │
                                                                          └────────────────────┘
```

The Qt layer (`app/gui/`) holds every UI concern. The domain layer
(`app/state_manager.py`, `app/audio_recorder.py`,
`app/clipboard_manager.py`) is what `RecordingController` wires Qt
signals into. The speech-to-text engine sits behind the
`TranscriptionBackend` Protocol in `app/backends/` and is dispatched by
`RoutedBackend` based on the selected model's `backend_kind` —
`faster_whisper`, `gigaam`, and `nemo` ship today; cloud APIs would be a
drop-in addition. `ResourceMonitor` polls CPU / RAM / GPU on a Qt timer
to drive the topbar widget. `app/gui/splash.py` pre-loads the persisted
model in the main thread before the main window appears, so the GIL-
heavy `import nemo` / `import torch` step never freezes a half-built UI.

## Building a standalone executable

```powershell
.\build-exe.ps1            # folder build via lazy_to_text.spec
.\build-exe.ps1 -OneFile   # single-file build (assets unpacked from _MEIPASS)
.\build-exe.ps1 -Clean     # remove dist/ and build/ first
```

Output: `dist\LazyToText\LazyToText.exe` (folder mode, fastest startup) or
`dist\LazyToText.exe` (`-OneFile`, slower because assets unpack on each
run). The build script delegates to `pyinstaller` via `uv run`; a fresh
`uv sync` runs first unless `-SkipSync` is passed.

> **Bundle size note** — with the `gigaam[longform]` and
> `nemo_toolkit[asr]` extras pulling in torch + pyannote + transformers
> + lhotse + hydra, the folder build now lands in the 6–8 GB range (vs
> ~144 MB before GigaAM). One-file is similarly large. Model weights
> are NOT bundled and download on first use of each card.

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
$lnk.Description      = "Local Whisper / GigaAM speech-to-text for Windows"
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

By default the backend uses `device: auto` and `compute_type: float16`,
so faster-whisper picks the GPU when CTranslate2 finds a CUDA-capable
card and falls back to CPU otherwise.

For the bundled `dist\LazyToText` you don't need to install anything
extra — `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` ride along in the
build. From source, `uv sync` installs the same wheels. **Driver-side**,
you still need an NVIDIA GPU with up-to-date drivers (CUDA 12 era).

CPU-only just about works for `gigaam-v3-e2e-ctc`; everything else
(`turbo*`, `large-v*`, `gigaam-v3-e2e-rnnt`, `parakeet-tdt-v3`) will
be too slow to be useful without a GPU.

## GigaAM longform — pyannote VAD requirements

`GigaAMASR.transcribe` is documented as good up to 25 seconds. For
longer captures `GigaamBackend` routes through `transcribe_longform`,
which depends on `pyannote/segmentation-3.0` — a **gated** Hugging Face
model. To enable longform end-to-end:

1. Generate an HF token with read access at <https://huggingface.co/settings/tokens>.
2. Visit <https://huggingface.co/pyannote/segmentation-3.0> and accept
   the user-conditions form.
3. Set `HF_TOKEN` in the environment before launching the app.

If the gated model isn't accepted, `transcribe_longform` falls back to
plain `transcribe` (which truncates at 25 s) and the app keeps working
on short captures.

## Development

```powershell
uv sync                                         # production deps + dev tools
uv run pytest                                   # full test suite (485 tests)
uv run python lazy-to-text-ui.py                # run from source
uv run python scripts/generate_screenshots.py  # regenerate docs/screenshots
```

Tests live in `tests/` and use `pytest-qt`. Cross-thread paths (resource
monitor, mic test, model load) are covered with fakes that don't touch
the actual hardware or libraries.

## Roadmap

- [ ] Onboarding overlay for first-launch users (which model to pick, how the hotkey works)
- [ ] Per-model VRAM forecasting that warns before downloading something the GPU can't fit
- [ ] Cloud-API backend behind the same `TranscriptionBackend` Protocol (OpenAI / Groq)
- [ ] Light theme + theme switcher
- [ ] Replace bundled Inter with system-installed UI font detection (saves ~880 KB)

## Tech stack

- Python 3.12
- PySide6 (Qt 6.11) for the UI, bundled Inter Variable + Heroicons
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + CTranslate2 for in-process Whisper inference
- [GigaAM](https://github.com/salute-developers/GigaAM) (PyTorch) for the Russian-only end-to-end engine, with `gigaam[longform]` for >25 s captures
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (`nemo_toolkit[asr]`) for Parakeet TDT v3 (25 EU languages), with `signal.SIGKILL` and `np.sctypes` shims so it boots cleanly on Windows + NumPy 2.x
- `sounddevice` for audio capture, `pyautogui` + `pyperclip` for paste,
  `global-hotkeys` + `pywin32` for Windows-native hotkey registration
- `psutil` + `nvidia-ml-py` for the live resource monitor

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (SYSTRAN) and
  [CTranslate2](https://github.com/OpenNMT/CTranslate2) under the Whisper hood.
- [OpenAI Whisper](https://github.com/openai/whisper) — the underlying model.
- [GigaAM](https://github.com/salute-developers/GigaAM) (Sber) — the Russian-specialised acoustic model.
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) and
  [Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — the multilingual European-language engine.
- [Heroicons](https://heroicons.com) (Tailwind Labs) — sidebar icons.
- [Inter](https://rsms.me/inter/) (Rasmus Andersson) — bundled UI font.
- UI direction borrowed from [Spokenly](https://spokenly.app/) (macOS).

## License

[MIT](LICENSE)
