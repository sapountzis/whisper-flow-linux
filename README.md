# WhisperFlow

AI-powered voice-to-text flow application with context-aware processing and system tray integration.

## Features

- 🎤 **Voice Transcription**: Real-time speech-to-text with OpenAI Whisper
- 🤖 **AI Completion**: Context-aware text completion and commands
- 🔧 **System Tray**: Background daemon with tray icon and global hotkeys
- ⌨️ **Global Hotkeys**: Push-to-talk and single-press voice activation
- 📝 **Multiple Modes**: Transcribe, Auto-Transcribe, and Command modes
- ⚙️ **Configurable**: Customizable prompts, models, and settings

## Quick Setup (One-Click Install)

### Prerequisites

- **Linux** (Ubuntu/Debian/Mint) with desktop environment
- **Python 3.12** (required for system tray support)
- **OpenAI API Key** (for transcription and completion)

### 1. Install System Dependencies

```bash
# Install required system packages for tray icon support
sudo apt update && sudo apt install -y \
    python3-gi \
    gir1.2-gtk-3.0 \
    gir1.2-appindicator3-0.1 \
    libappindicator3-1 \
    python3-venv \
    python3-pip
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/sapountzis/whisper-flow.git
cd whisper-flow

# Create virtual environment with system site packages
python3.12 -m venv .venv --system-site-packages

# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 3. Configure API Key

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
```

### 4. Start the Daemon

```bash
# Start the daemon with tray icon (GTK backend configured by default)
whisper-flow daemon --foreground

# Or start in background
whisper-flow daemon
```

You should see a microphone icon in your system tray. Right-click it to access the menu!

## Usage

### System Tray

- **Right-click** the tray icon to access:
  - Settings
  - Test Configuration  
  - Exit

### Global Hotkeys

- **🎤 Transcribe**: `Ctrl+Cmd` (push-to-talk)
- **🔴 Auto-Transcribe**: `Ctrl+Cmd+Space` (single press)
- **🤖 Command**: `Ctrl+Cmd+Alt` (single press)
- **🛑 Cancel**: `Escape`
- **📋 Menu**: `F1`

### CLI Commands

```bash
# Initialize configuration
whisper-flow init-config

# Start daemon
whisper-flow daemon

# Stop daemon
whisper-flow stop

# Check status
whisper-flow status

# Test configuration
whisper-flow validate
```

## Troubleshooting

### Tray Icon Not Working?

If you see the tray icon but the menu doesn't appear:

1. **Check backend**: The daemon should show "Pystray backend: gtk"
2. **Verify gi module**: `python -c "import gi; print('OK')"`
3. **Restart daemon**: `whisper-flow stop && whisper-flow daemon`

### Common Issues

- **"No module named 'gi'"**: Install system packages from step 1
- **"XOrg backend"**: Ensure Python 3.12 and system packages are installed
- **No tray icon**: Check if your desktop environment supports system tray

### Manual Backend Selection

The GTK backend is configured by default. To use a different backend:

```bash
# Set environment variable
export PYSTRAY_BACKEND=appindicator
whisper-flow daemon

# Or edit config file
# ~/.config/whisper-flow/config.yaml: pystray_backend: "appindicator"
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/
```

### Project Structure

```
whisper-flow/
├── src/whisper_flow/
│   ├── app.py          # Main application logic
│   ├── daemon.py       # System tray daemon
│   ├── cli.py          # Command-line interface
│   ├── audio.py        # Audio recording
│   ├── transcription.py # OpenAI Whisper integration
│   ├── completion.py   # AI completion
│   └── config.py       # Configuration management
├── pyproject.toml      # Project configuration
└── README.md          # This file
```

## Configuration

Configuration files are stored in `~/.config/whisper-flow/`:

- `config.yaml` - Main settings
- `prompts.yaml` - Default prompts
- `transcribe.yaml` - Transcription mode prompts
- `auto_transcribe.yaml` - Auto-transcribe mode prompts
- `command.yaml` - Command mode prompts

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/sapountzis/open-whisper-flow/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sapountzis/open-whisper-flow/discussions)

### Manual Testing

Once setup is complete, you can run individual tests:

```bash
# Test tray functionality
python tests/test_tray.py

# Test daemon components
python tests/test_daemon_tray.py
```