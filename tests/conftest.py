"""Pytest configuration and fixtures for whisper-flow tests."""

import os

# Add src to path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_config_dir():
    """Create a temporary configuration directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir)
        yield config_path


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.openai_api_key = "test-api-key"
    config.transcription_model = "gpt-4o-mini"
    config.completion_model = "gpt-4o-mini"
    config.sample_rate = 16000
    config.frame_ms = 30
    config.vad_mode = 3
    config.silence_timeout = 2.0
    config.mic_device_index = None
    config.daemon_enabled = True
    config.hotkey_transcribe = "ctrl+shift+t"
    config.hotkey_auto_transcribe = "ctrl+shift+a"
    config.hotkey_command = "ctrl+shift+c"
    config.notification_timeout = 5000
    config.pystray_backend = "gtk"
    config.auto_stop_silence_duration = 2.0
    config.config_dir = Path("/tmp/test-config")
    return config


@pytest.fixture
def mock_system_manager():
    """Create a mock system manager."""
    manager = Mock()
    manager.notify.return_value = None
    manager.get_active_window_title.return_value = "Test Window"
    manager.get_highlighted_text.return_value = None
    manager.paste_text.return_value = True
    manager._copy_to_clipboard.return_value = None
    return manager


@pytest.fixture
def mock_audio_recorder():
    """Create a mock audio recorder."""
    recorder = Mock()
    recorder.record_push_to_talk.return_value = "/tmp/test_audio.wav"
    recorder.record_until_silence.return_value = "/tmp/test_audio.wav"
    return recorder


@pytest.fixture
def mock_transcription_service():
    """Create a mock transcription service."""
    service = Mock()
    service.transcribe_audio.return_value = "This is a test transcript"
    return service


@pytest.fixture
def mock_completion_service():
    """Create a mock completion service."""
    service = Mock()
    service.complete_text.return_value = "This is a test completion"
    return service


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager."""
    manager = Mock()
    manager.get_system_message.return_value = "You are a helpful assistant."
    manager.get_user_message.return_value = (
        "Date: 2024-01-01\nTime: 12:00:00\nUser input: test transcript"
    )
    manager.get_messages.return_value = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Date: 2024-01-01\nTime: 12:00:00\nUser input: test transcript",
        },
    ]
    manager.should_use_completion.return_value = True
    manager.get_prompt_info.return_value = {
        "system": "single_template",
        "system_message": "You are a helpful assistant.",
        "variables": ["date", "time", "highlighted_text", "user_input"],
        "description": "Single prompt template with context variables",
    }
    return manager


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        # Create minimal WAV file for testing
        import struct
        import wave

        with wave.open(tmp_file.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # Write 1 second of silence
            silence = struct.pack("<h", 0) * 16000
            wav_file.writeframes(silence)

        yield tmp_file.name

        # Cleanup
        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress warnings during tests."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*pkg_resources.*")


@pytest.fixture(autouse=True)
def mock_environment():
    """Mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "ALSA_SUPPRESS_WARNINGS": "1",
            "ALSA_PCM_CARD": "0",
            "ALSA_PCM_DEVICE": "0",
            "PYSTRAY_BACKEND": "gtk",
        },
    ):
        yield
