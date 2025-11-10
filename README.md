# Lazy to Text

Speech-to-text application for Windows using hotkeys and the Whisper model. Records audio via global hotkey, transcribes it using a Docker-based backend, and automatically pastes the result.

## Features

- Global hotkey activation (default: Ctrl+F2 to start, Ctrl+F3 to stop)
- Automatic text insertion into active application
- System tray integration for background operation
- Built-in Docker container management
- Multiple Whisper model support (base, small, medium, turbo, large-v3)
- Audio feedback with configurable sound alerts
- Transcription history with search capability
- Automatic punctuation and capitalization improvement
- Configurable beam search parameters for quality control

## Requirements

- Windows 10 or 11
- Microphone (built-in or USB)
- Docker Desktop installed and running
- Python 3.12 (for source installation)

## Installation

### Option 1: Portable Executable

1. Download `LazyToText.exe` from releases
2. Start Docker Desktop
3. Open PowerShell in the application directory:

   ```powershell
   docker compose up -d
   ```

4. Run `LazyToText.exe`

### Option 2: From Source

```bash
git clone https://github.com/aa-blinov/whisper-local-windows.git
cd whisper-local-windows

# Install uv package manager if not present
# Create virtual environment and install dependencies
uv venv --python 3.12
uv sync

# Start Docker backend
docker compose up -d

# Run application
uv run lazy-to-text-ui
```

## Usage

1. Ensure Docker Desktop is running (system tray icon visible)
2. Launch the application (window and system tray icon appear)
3. Click "Start Server" to launch the Docker container (first run downloads ~5GB model)
4. Wait for status to show "ready"
5. Press Ctrl+F2 to start recording (audio feedback plays)
6. Speak clearly with natural pauses between sentences
7. Press Ctrl+F3 to stop recording
8. Transcribed text is automatically inserted into the active window

### Default Hotkeys

- Ctrl+F2: Start recording
- Ctrl+F3: Stop recording

Hotkeys can be customized in the application UI.

## Transcription Quality

The application automatically applies post-processing to improve text quality:

- Capitalizes first letter of text
- Capitalizes letters after sentence-ending punctuation (. ! ?)
- Adds spaces after commas where missing
- Removes excessive whitespace
- Normalizes punctuation spacing

### Model Comparison

| Model | Quality | Speed | Punctuation |
|-------|---------|-------|-------------|
| base | Low | Very Fast | Poor |
| small | Medium | Fast | Fair |
| medium | Good | Moderate | Good |
| turbo | Excellent | Fast | Excellent |
| large-v3 | Excellent | Moderate | Excellent |

Recommended: turbo (best balance of quality and speed)

### Quality Settings

Use the UI to configure transcription parameters:

- **Model**: Select from dropdown (base, small, medium, turbo, large-v3)
- **Beam Size**: Range 3-10 (higher values improve quality but reduce speed)
- **Language**: Specify language code or use "auto" for detection

### Recording Best Practices

- Speak clearly with natural pauses between sentences
- Minimize background noise
- Use 3-20 second recordings for optimal results
- Maintain consistent microphone distance and volume

## Building Executable

To build a standalone executable:

```powershell
# Clean build
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
uv run pyinstaller lazy_to_text.spec
```

Output: `dist\LazyToText\LazyToText.exe`

Alternatively, use the provided script:

```powershell
./build-exe.ps1           # Standard build
./build-exe.ps1 -Clean    # Clean build
```

## User Interface

### Main Sections

1. **Model Configuration**: Select model, language, and beam size parameters
2. **System Status**: Docker daemon and container status with readiness indicators
3. **Hotkeys**: Configure recording hotkeys and auto-paste behavior
4. **History**: Browse and search previous transcriptions
5. **Logs**: Application events and status messages

### Control Buttons

- **Apply Changes**: Apply new model configuration (restarts container)
- **Start Server**: Launch Docker container
- **Stop Server**: Stop Docker container
- **View Logs**: Display container logs in separate window

## Configuration

The `config.yaml` file is created automatically on first run and stores all application settings. All parameters can be configured through the UI.

### Configuration Structure

```yaml
whisper:
  backend_mode: local        # local or external
  model: large-v3            # base, small, medium, turbo, large-v3
  beam_size: 7               # 3-10 range
  language: ru               # Language code or "auto"
  local_url: http://localhost:10300

hotkey:
  start_recording_hotkey: ctrl+f2
  stop_recording_hotkey: ctrl+f3

clipboard:
  auto_paste: 1              # 0=disabled, 1=enabled
  preserve_clipboard: false
  key_simulation_delay: 0.05

audio_feedback:
  enabled: true
  start_sound: assets/sounds/record_start.wav
  stop_sound: assets/sounds/record_stop.wav
  cancel_sound: assets/sounds/record_cancel.wav

system_tray:
  enabled: true
  tooltip: Lazy to text

history:
  enabled: true
  max_entries: 1000
  auto_cleanup_days: 30
```

To reset configuration: delete `config.yaml` and restart the application.

## Docker Backend

The application uses a Docker container running the faster-whisper model. Container lifecycle is managed through the UI.

### Initial Setup

```powershell
docker compose up -d
```

First run downloads the model (~5GB). Subsequent starts are faster.

### Container Management

Use UI buttons "Start Server" and "Stop Server" to control the container. Manual Docker commands are not required during normal operation.

### Container Status

- **ready**: Container running, model loaded, service operational
- **waiting**: Container running, model loading or service initializing
- **stopped**: Container not running
- **error**: Container or service failure

Status is displayed in the System Status panel with color indicators.

### Model Switching

Model changes are applied through the UI:

1. Select new model from dropdown
2. Click "Apply Changes"
3. Container automatically restarts with new model configuration

The application verifies readiness by monitoring container logs for the service initialization message.

## Troubleshooting

### Hotkeys Not Working

- Change hotkey combination in UI (another application may be intercepting)
- Close other applications with global hotkeys
- Verify hotkey configuration in Hotkeys section

### Empty Transcription

- Verify microphone input in Windows sound settings
- Check microphone levels and test recording
- Ensure container status shows "ready"
- Speak clearly at normal volume

### Text Not Pasting

- Toggle auto-paste setting in UI
- Verify Windows permissions for simulated keystrokes
- Run application as administrator if necessary
- Check that target application accepts pasted input

### Poor Punctuation Quality

- Increase beam_size parameter (7-10 range)
- Speak with natural pauses between sentences
- Automatic post-processing is always applied

### Docker Issues

- Verify Docker Desktop is running (system tray icon)
- Restart Docker Desktop if necessary
- Test Docker functionality: `docker ps` in PowerShell
- Check Docker logs for error messages

### Performance Issues

- Reduce beam_size parameter (5 or lower)
- Switch to "base" or "small" model
- Close resource-intensive applications
- Monitor CPU/GPU usage during transcription

### Diagnostics

- Application logs: `logs/app.log` (adjacent to executable)
- Container logs: "View Logs" button in UI
- Configuration reset: delete `config.yaml` and restart

## Technical Architecture

### Data Flow

```
Microphone → Audio Recording → Processing → Whisper Model → Text → Insertion
                                              ↓
                                        Docker Container
```

### Component Overview

1. **Audio Recorder**: Captures microphone input using sounddevice library
2. **Hotkey Listener**: Detects global keyboard shortcuts
3. **State Manager**: Coordinates recording lifecycle and application state
4. **Whisper Engine**: Communicates with Docker backend via Wyoming protocol
5. **Clipboard Manager**: Handles text copying and simulated paste operations
6. **Docker Backend Manager**: Controls container lifecycle and health monitoring
7. **UI Layer**: CustomTkinter-based interface for configuration and status
8. **System Tray**: Background operation with status indicators

### Technology Stack

- Python 3.12
- CustomTkinter (modern UI framework)
- Docker SDK for Python (container management)
- Wyoming Protocol (model communication)
- sounddevice (audio capture)
- PyAutoGUI (automated text insertion)
- pystray (system tray integration)

### Container Details

- Image: linuxserver/faster-whisper:gpu
- Protocol: Wyoming (TCP-based)
- Port: 10300 (configurable)
- GPU Support: NVIDIA with Container Toolkit
- Models: Stored in persistent volume

## Contributing

Contributions are welcome. To report bugs or request features, open an issue in the repository with detailed information about the problem or proposed enhancement.

For code contributions:

1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate tests
4. Submit a pull request with clear description
