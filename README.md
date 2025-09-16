# Lazy to Text – Hotkey Speech-to-Text with Remote Whisper Backend

Lazy to Text is a lightweight Windows desktop application that enables speech-to-text transcription using customizable hotkeys. Simply press a hotkey to start recording, press another to stop, and the transcribed text is automatically pasted into your active application.

The application uses a remote [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) backend for speech recognition, providing high-quality transcription without requiring powerful local hardware.

## Features

- **Global Hotkeys**: Start and stop recording with customizable keyboard shortcuts
- **Remote Transcription**: Uses a remote Faster-Whisper backend for high-quality speech recognition
- **Auto-Paste Functionality**: Automatically paste transcribed text into the active application
- **Simple Configuration**: Single configuration file (`config.yaml`) for all settings
- **Modern UI**: Intuitive CustomTkinter-based interface for model selection and settings
- **System Tray Integration**: Minimize to system tray for unobtrusive operation
- **Audio Feedback**: Optional sound cues for recording start/stop events
- **Docker Integration**: Built-in Docker container management for the backend service
- **Multiple Language Support**: Supports transcription in multiple languages
- **Model Flexibility**: Switch between different Whisper model sizes (turbo, base, small, medium, large-v3)

## Quick Start

### Prerequisites

- Windows 10/11
- Microphone
- Docker Desktop (for running the backend service)
- Python 3.12 (if running from source)

### Installation Options

#### Option 1: Using uv (Recommended)

```bash
# Install uv if not already installed
pip install --upgrade uv

# Create a Python 3.12 virtual environment
uv venv --python 3.12

# Install dependencies
uv sync

# Run the application
uv run lazy-to-text-ui
```

#### Option 2: Install as a Global Tool

```bash
uv tool install . --force
lazy-to-text-ui
```

#### Option 3: Development Setup

```bash
# Clone the repository
git clone https://github.com/aa-blinov/whisper-local-windows.git
cd whisper-local-windows

# Install uv if not already installed
pip install --upgrade uv

# Create a Python 3.12 virtual environment
uv venv --python 3.12

# Install dependencies
uv sync

# Run the application
uv run lazy-to-text-ui
```

#### Option 4: Portable Executable

Download a packaged build, extract it, and run `LazyToText.exe`. Note that the model container must be running for the application to work.

## Building a Windows Executable (PyInstaller)

You can create a standalone `LazyToText.exe` so end users do not need to install Python and dependencies.

### Prerequisites

- Python 3.12 (x64) installed and on PATH
- (Recommended) Virtual environment with project dependencies installed: `pip install -e .`
- `pip install pyinstaller`

### Fast One‑Shot Build (no spec)

```powershell
pyinstaller -y --clean --name LazyToText `
  --icon app\assets\tray_idle.ico `
  --add-data "app\\assets;app\\assets" `
  --add-data "config.yaml;." `
  --hidden-import customtkinter `
  --hidden-import PIL._tkinter_finder `
  --hidden-import pystray._win32 `
  --hidden-import win32timezone `
  lazy-to-text-ui.py
```

### Recommended (Spec File)

A curated spec file `lazy_to_text.spec` is included. It bundles assets and the root `config.yaml`.

If you're using uv, you can build with:

```bash
uv run pyinstaller lazy_to_text.spec
```

Or if you're not using uv:

```powershell
python -m pip install pyinstaller
pyinstaller lazy_to_text.spec
```

Result: `dist\LazyToText\LazyToText.exe`

### PowerShell Helper Script

Script `build-exe.ps1` automates the build:

```powershell
./build-exe.ps1            # folder build using spec
./build-exe.ps1 -Clean     # clean + build
./build-exe.ps1 -OneFile   # experimental one-file build
```

### One-File Mode Notes

`--onefile` unpacks to a temp folder at runtime (slower first start). Assets are still accessible via `_MEIPASS` (already handled in `resolve_asset_path`). Use only if you need a single binary; otherwise prefer the folder build for faster startup and easier inspection.

### Docker Integration Safety

The app communicates with Docker via the standard Python SDK (`docker` package) and the host's Docker daemon (named pipe on Windows). Packaging with PyInstaller does not break this, because:

- No dynamic code generation required for core calls
- We are not vendoring the daemon itself, only the client library
- Network / pipe access functions the same from a frozen exe

If Docker appears "unavailable" in the packaged build:

1. Ensure Docker Desktop is running
2. Run the exe as the same user that can run `docker ps`
3. Test from PowerShell: `docker version`
4. If corporate security blocks named pipes, run the app elevated or configure Docker to expose TCP (not recommended on unsecured networks)

### Updating After Changes

Re-run the build (add `-Clean` to purge old bundles). If you change only config defaults, users may still have an existing `config.yaml` beside the exe—document that they may need to delete it to pick up new defaults.

### Reducing Size (Optional)

- Remove unused locales / assets
- Use `--exclude-module` for modules you are certain are not needed
- UPX packing (enable after verifying antivirus does not flag): `--upx-dir <path>` and set `upx=True` in spec
- (Advanced) Switch to Nuitka for further optimization

### Troubleshooting Build

| Issue | Cause | Fix |
|-------|-------|-----|
| Missing icon | Path wrong | Check `tray_idle.ico` exists |
| Blank window / theme issues | customtkinter styles not bundled | Confirm `customtkinter` in hiddenimports |
| Audio fails | PortAudio DLL not found | Let me know; can add binary inclusion rule |
| Docker unavailable | Daemon not running / permissions | Start Docker Desktop / check user rights |

### Verification Checklist

After building run:

1. Start `LazyToText.exe`
2. Confirm tray icon appears (if enabled)
3. Open UI → Start Server (or detect running container)
4. Press hotkeys and verify transcription
5. Inspect `logs/app.log` created next to the exe

---

If you need a signed build or an installer (MSIX / Inno Setup), that can be layered on top of the `dist/LazyToText` output.

## Core Runtime Dependencies

The client application depends on the following Python packages:

- `sounddevice`: Audio recording
- `global-hotkeys`: Global hotkey detection
- `pyperclip`: Clipboard management
- `ruamel.yaml`: YAML configuration handling
- `pyautogui`: Automated key simulation for pasting
- `pystray`: System tray integration
- `Pillow`: Image handling for tray icons
- `customtkinter`: Modern UI components
- `requests`: HTTP requests
- `docker`: Docker container management
- `wyoming`: Wyoming protocol communication

Note: Local model inference libraries (e.g., `faster-whisper`) have been removed. All inference happens on the remote service.

## Usage

1. **Start the Backend Service**: Launch the Docker container that provides the speech recognition service (see Docker Backend section below).
2. **Launch the UI**: Run `uv run lazy-to-text-ui` or execute the portable executable.
3. **Record Audio**: Press the start hotkey (default `ctrl+f2`) to begin recording, then press the stop hotkey (`ctrl+f3`) to end.
4. **View Results**: The transcribed text appears in the log panel and is automatically copied (and optionally pasted) into your active application.
5. **Change Models**: Select a different model from the dropdown and click the `Switch` button to change the transcription model.

### Hotkey Controls

- **Start Recording**: Default `Ctrl+F2` (customizable in config)
- **Stop Recording**: Default `Ctrl+F3` (customizable in config)

### User Interface

The application features a modern UI built with CustomTkinter:

1. **Model Section**: Select and switch between different Whisper models
2. **Status Panel**: View server and container status
3. **Hotkeys Section**: Configure recording hotkeys and auto-paste settings
4. **Logs Section**: View application logs and status messages

## Configuration

The `config.yaml` file contains all application settings and is located at the repository root (or beside the executable). To reset to defaults, simply delete this file and restart the application.

### Configuration Sections

#### whisper
- `backend_mode`: Either "local" or "external" (default: "local")
- `model`: Model alias to use (default: "turbo")
- `language`: Language for transcription, "auto" for automatic detection (default: "ru")
- `beam_size`: Beam search size for better accuracy (default: 5)
- `local_url`: URL for the local Docker backend (default: "http://localhost:10300")
- `external_url`: URL for an external backend service

#### hotkey
- `start_recording_hotkey`: Hotkey to start recording (default: "ctrl+f2")
- `stop_recording_hotkey`: Hotkey to stop recording (default: "ctrl+f3")

#### audio
- `channels`: Number of audio channels (default: 1)
- `dtype`: Audio data type (default: "float32")
- `max_duration`: Maximum recording duration in seconds (default: 300)

#### clipboard
- `auto_paste`: Automatically paste transcribed text (default: true)
- `preserve_clipboard`: Restore original clipboard content after pasting (default: false)
- `key_simulation_delay`: Delay between key presses in seconds (default: 0.05)

#### logging
- `level`: Log level (default: "INFO")
- `file`: File logging settings
  - `enabled`: Enable file logging (default: true)
  - `filename`: Log file name (default: "app.log")
  - `rotation`: Log rotation settings
    - `enabled`: Enable log rotation (default: true)
    - `max_bytes`: Maximum file size before rotation (default: 1048576)
    - `backup_count`: Number of backup files to keep (default: 5)
    - `encoding`: File encoding (default: "utf-8")
- `console`: Console logging settings
  - `enabled`: Enable console logging (default: true)
  - `level`: Console log level (default: "WARNING")

#### audio_feedback
- `enabled`: Enable audio feedback sounds (default: true)
- `start_sound`: Sound file for recording start (default: "assets/sounds/record_start.wav")
- `stop_sound`: Sound file for recording stop (default: "assets/sounds/record_stop.wav")
- `cancel_sound`: Sound file for recording cancel (default: "assets/sounds/record_cancel.wav")

#### system_tray
- `enabled`: Enable system tray integration (default: true)
- `tooltip`: Tooltip text for system tray icon (default: "Lazy to text")

## Docker Backend (Model Service)

Lazy to text uses a Docker container running the LinuxServer Faster-Whisper image as its backend service. The reference `docker-compose.yml` file is configured with sensible defaults:

```yaml
services:
  faster-whisper:
    image: linuxserver/faster-whisper:gpu
    container_name: faster-whisper
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=Etc/UTC
      - WHISPER_MODEL=turbo
      - WHISPER_BEAM=5
      - WHISPER_LANG=ru
    ports:
      - "10300:10300"
    volumes:
      - models_cache:/config
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [ gpu ]
    restart: unless-stopped

volumes:
  models_cache:
```

### Starting the Backend

```bash
docker compose up -d
```

The UI connects to `http://localhost:10300` by default. This can be configured in the `whisper.local_url` setting in `config.yaml`.

### Environment Variables

You can override settings using environment variables when launching the UI:

Bash:
```bash
WHISPER_URL=http://localhost:10300 WHISPER_MODEL=base uv run lazy-to-text-ui
```

PowerShell:
```powershell
$env:WHISPER_URL="http://localhost:10300"; $env:WHISPER_MODEL="base"; uv run lazy-to-text-ui
```

### Model Switching

The application supports switching between different Whisper models:

- Model aliases map to full identifiers (e.g., `turbo` → `mobiuslabsgmbh/faster-whisper-large-v3-turbo`)
- Use the UI dropdown to select a model and click `Switch` to change models
- The switch is instantaneous for the client but may require the server to load the new model
- Supported models include: turbo, base, small, medium, large-v3

### Backend Management

The application includes built-in Docker container management:

- **Start**: Launch the backend container
- **Stop**: Stop the backend container
- **Automatic Restart**: When switching models, the container is automatically restarted with the new model configuration

## Health Check

The client periodically checks the backend service health by calling `GET /health` and displays the status in the UI as `Server status: running` or `Server status: not running`.

## Troubleshooting

- **Hotkey not working**: Try changing `hotkey.start_recording_hotkey` as another application may be intercepting it
- **Empty transcription**: Check your microphone device and input levels in Windows settings
- **Text not pasted**: Toggle auto-paste off and on, or check Windows permissions for simulated key presses
- **Reset settings**: Delete `config.yaml` to regenerate with default values
- **View logs**: Check `logs/app.log` for detailed application logs
- **Docker issues**: Ensure Docker Desktop is running and you have sufficient permissions
- **GPU not detected**: The container uses GPU by default; ensure NVIDIA Container Toolkit is installed

## Technical Architecture

Lazy to text follows a modular architecture with clearly separated components:

### Core Components

1. **UI Layer**: CustomTkinter-based interface for user interaction
2. **State Manager**: Central coordinator for application state
3. **Audio Recorder**: Handles microphone input and audio processing
4. **Whisper Engine**: Communicates with the backend service using the Wyoming protocol
5. **Clipboard Manager**: Manages text copying and pasting operations
6. **Hotkey Listener**: Detects and responds to global keyboard shortcuts
7. **Docker Backend Manager**: Controls the Docker container lifecycle
8. **Configuration Manager**: Handles application settings
9. **System Tray**: Provides system tray integration
10. **Audio Feedback**: Plays sound cues for user actions

### Data Flow

1. User presses a hotkey → Hotkey Listener detects it
2. State Manager coordinates the recording process
3. Audio Recorder captures microphone input
4. Audio data is sent to Whisper Engine
5. Whisper Engine communicates with backend via Wyoming protocol
6. Transcribed text is processed by Clipboard Manager
7. Text is either copied to clipboard or auto-pasted to active application
8. System Tray updates with status information

## Contribution Guidelines

We welcome contributions to Lazy to text! Here's how you can help:

### Reporting Issues

- Check existing issues before creating a new one
- Provide detailed information about your environment
- Include steps to reproduce the issue
- Attach relevant log files if applicable

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

1. Follow the development installation instructions above
2. Make your changes
3. Test thoroughly
4. Ensure code follows existing style conventions
5. Update documentation as needed

### Code Style

- Follow PEP 8 Python style guide
- Use descriptive variable and function names
- Include docstrings for public functions and classes
- Write clear, concise comments for complex logic

## Notes

All configuration is stored in `config.yaml`. Local model inference was removed to reduce complexity and resource usage. For offline operation, you would need to reintroduce a local backend (outside the current scope).

### Using uv / Python 3.12

```bash
# Optional: deactivate old virtualenv
pip install --upgrade uv
uv venv --python 3.12
uv sync
uv run lazy-to-text-ui
```

The `uv sync` command installs dependencies strictly from `pyproject.toml`.

