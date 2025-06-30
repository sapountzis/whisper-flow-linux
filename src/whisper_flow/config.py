"""Configuration management for whisper-flow using Pydantic Settings."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration manager for whisper-flow using Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_prefix="WHISPER_FLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # File paths
    config_dir: Path = Field(
        default_factory=lambda: Path.home() / ".config" / "whisper-flow",
        description="Configuration directory path",
    )

    # API configuration
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    openai_url: str = Field(
        default="https://api.openai.com/v1/chat/completions",
        description="OpenAI API URL for completions",
        env="OPENAI_URL",
    )
    audio_url: str = Field(
        default="https://api.openai.com/v1/audio/transcriptions",
        description="OpenAI API URL for audio transcription",
        env="AUDIO_URL",
    )

    # Audio configuration
    vad_mode: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Voice Activity Detection mode (0-3)",
        env="VAD_MODE",
    )
    mic_device_index: int | None = Field(
        default=None,
        description="Microphone device index",
        env="MIC_DEVICE_INDEX",
    )
    frame_ms: int = Field(
        default=30,
        gt=0,
        description="Audio frame duration in milliseconds",
    )
    silence_timeout: float = Field(
        default=1.5,
        gt=0.0,
        description="Silence timeout in seconds",
        env="SILENCE_TIMEOUT",
    )
    sample_rate: int = Field(
        default=16000,
        gt=0,
        description="Audio sample rate in Hz",
        env="SAMPLE_RATE",
    )
    speedup_audio: float = Field(
        default=1,
        description="Audio speed multiplier (1.0 = normal speed, 1.5 = 1.5x speed, etc.)",
        env="WHISPER_FLOW_SPEEDUP_AUDIO",
    )

    # AI model configuration
    transcription_model: str = Field(
        default="gpt-4o-mini-transcribe",
        description="OpenAI model for transcription",
        env="WHISPER_FLOW_TRANSCRIPTION_MODEL",
    )
    completion_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for completion",
        env="WHISPER_FLOW_COMPLETION_MODEL",
    )
    temperature: float = Field(
        default=0.4,
        description="Temperature for completion",
        ge=0.0,
        le=2.0,
        env="WHISPER_FLOW_TEMPERATURE",
    )

    # UI configuration
    notification_timeout: int = Field(
        default=3000,
        description="Notification timeout in milliseconds",
        ge=1000,
        le=10000,
        env="WHISPER_FLOW_NOTIFICATION_TIMEOUT",
    )

    # Daemon Configuration
    daemon_enabled: bool = Field(
        default=True,
        description="Enable daemon mode with system tray",
        env="WHISPER_FLOW_DAEMON_ENABLED",
    )
    auto_stop_silence_duration: float = Field(
        default=2.0,
        description="Seconds of silence before auto-stopping recording",
        ge=0.5,
        le=10.0,
        env="WHISPER_FLOW_AUTO_STOP_SILENCE",
    )
    pystray_backend: str = Field(
        default="gtk",
        description="Pystray backend for system tray (gtk, appindicator, xorg)",
        env="PYSTRAY_BACKEND",
    )

    # Hotkey Configuration
    hotkey_transcribe: str = Field(
        default="ctrl+cmd",
        description="Hotkey for push-to-talk transcription",
        env="WHISPER_FLOW_HOTKEY_TRANSCRIBE",
    )
    hotkey_auto_transcribe: str = Field(
        default="ctrl+cmd+space",
        description="Hotkey for auto-stop transcription",
        env="WHISPER_FLOW_HOTKEY_AUTO_TRANSCRIBE",
    )
    hotkey_command: str = Field(
        default="ctrl+cmd+alt",
        description="Hotkey for push-to-talk command mode",
        env="WHISPER_FLOW_HOTKEY_COMMAND",
    )

    # Logging configuration
    logging_enabled: bool = Field(
        default=False,
        description="Enable debug logging and print statements",
        env="WHISPER_FLOW_LOGGING_ENABLED",
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def get_openai_api_key(cls, v):
        """Get OpenAI API key from multiple possible environment variables."""
        if v is not None:
            return v
        # Check standard OPENAI_API_KEY first, then our prefixed version
        return os.getenv("OPENAI_API_KEY") or os.getenv("WHISPER_FLOW_OPENAI_API_KEY")

    @field_validator("config_dir", mode="before")
    @classmethod
    def expand_config_dir(cls, v):
        """Expand config directory path and ensure it exists."""
        if isinstance(v, str):
            path = Path(os.path.expanduser(v))
        else:
            path = v
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def openai_headers(self) -> dict[str, str]:
        """Get OpenAI API headers."""
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return {"Authorization": f"Bearer {self.openai_api_key}"}

    def get_prompts_config(self, mode: str = "default") -> list[dict[str, Any]]:
        """Load prompts configuration (legacy method for compatibility).

        Args:
            mode: Configuration mode (no longer used in simplified system)

        Returns:
            Empty list since we now use a single template approach

        """
        # Return empty list since we now use a single template approach
        return []

    def _get_default_prompts(self, mode: str) -> list[dict[str, Any]]:
        """Get default prompt configurations (legacy method for compatibility).

        Args:
            mode: Configuration mode (no longer used in simplified system)

        Returns:
            Empty list since we now use a single template approach

        """
        # Return empty list since we now use a single template approach
        return []

    def ensure_config_files(self):
        """Ensure configuration files exist with default content."""
        # No longer need to create prompt configuration files
        # The system now uses a single template approach

    def model_dump_config(self) -> dict[str, Any]:
        """Export current configuration as a dictionary."""
        return {
            "config_dir": str(self.config_dir),
            "whisper_bin": str(self.whisper_bin),
            "model_path": str(self.model_path),
            "openai_api_key_set": self.openai_api_key is not None,
            "openai_url": self.openai_url,
            "audio_url": self.audio_url,
            "vad_mode": self.vad_mode,
            "mic_device_index": self.mic_device_index,
            "frame_ms": self.frame_ms,
            "silence_timeout": self.silence_timeout,
            "sample_rate": self.sample_rate,
            "transcription_model": self.transcription_model,
            "completion_model": self.completion_model,
            "temperature": self.temperature,
            "notification_timeout": self.notification_timeout,
        }

    @classmethod
    def load_from_file(cls, config_file: Path) -> "Config":
        """Load configuration from a YAML file.

        Args:
            config_file: Path to configuration file

        Returns:
            Config instance with loaded settings

        """
        if config_file.exists():
            with open(config_file) as f:
                config_data = yaml.safe_load(f) or {}
            return cls(**config_data)
        return cls()

    def save_to_file(self, config_file: Path):
        """Save current configuration to a YAML file.

        Args:
            config_file: Path to save configuration

        """
        config_data = self.model_dump_config()
        # Remove sensitive data
        config_data.pop("openai_api_key_set", None)

        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
