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
        """Load prompts configuration.

        Args:
            mode: Configuration mode ('default', 'dictation', 'transcribe', 'auto_transcribe', 'command')

        Returns:
            List of prompt rules

        """
        # Map modes to filenames
        mode_files = {
            "dictation": "dictation.yaml",
            "transcribe": "transcribe.yaml",
            "auto_transcribe": "auto_transcribe.yaml",
            "command": "command.yaml",
            "default": "prompts.yaml",
        }

        filename = mode_files.get(mode, "prompts.yaml")
        prompts_file = self.config_dir / filename

        if prompts_file.exists():
            with open(prompts_file) as f:
                return yaml.safe_load(f) or []

        # Return default prompts if file doesn't exist
        return self._get_default_prompts(mode)

    def _get_default_prompts(self, mode: str) -> list[dict[str, Any]]:
        """Get default prompt configurations."""
        if mode == "transcribe" or mode == "auto_transcribe":
            # Plain transcription modes - no LLM processing
            return [
                {
                    "match": "",
                    "prompt": "{{text}}",  # Just return the transcript as-is
                },
            ]
        if mode == "command":
            # Command mode - full LLM processing with context awareness
            return [
                {
                    "match": r"\.py .* Cursor",
                    "prompt": "Turn transcript into idiomatic Python code that passes basic linters.\nNo markdown. Formatting: 4 spaces and \\n for new line.\n<transcript>{{text}}</transcript>",
                },
                {
                    "match": r"\.js .* Cursor|\.ts .* Cursor",
                    "prompt": "Turn transcript into clean JavaScript/TypeScript code.\nNo markdown. Use modern ES6+ syntax.\n<transcript>{{text}}</transcript>",
                },
                {
                    "match": "Gmail",
                    "prompt": "Rewrite transcript as a concise, professional email. Keep paragraphs short.\n<transcript>{{text}}</transcript>",
                },
                {
                    "match": r"terminal|Terminal|bash|zsh|fish",
                    "prompt": "Convert transcript to appropriate shell commands. Output only the commands, no explanations.\n<transcript>{{text}}</transcript>",
                },
                {
                    "match": "",
                    "prompt": "Process the transcript and provide a helpful response. Fix grammar and spelling if needed, or answer the question if one was asked.\n<transcript>{{text}}</transcript>",
                },
            ]
        if mode == "dictation":
            # Legacy dictation mode
            return [
                {
                    "match": "",
                    "prompt": "Output the transcript exactly as spoken. Do not rephrase, summarize, or fix grammar.\n<transcript>{{text}}</transcript>",
                },
            ]
        # Default/legacy mode
        return [
            {
                "match": r"\.py .* Cursor",
                "prompt": "Turn transcript into idiomatic Python code that passes basic linters.\nNo markdown. Formatting: 4 spaces and \\n for new line.\n<transcript>{{text}}</transcript>",
            },
            {
                "match": "Gmail",
                "prompt": "Rewrite transcript as a concise, friendly email. Keep paragraphs short.\n<transcript>{{text}}</transcript>",
            },
            {
                "match": "",
                "prompt": "Output the transcript and fix any grammatical, spelling, syntax problems and or transcription errors.\n<transcript>{{text}}</transcript>",
            },
        ]

    def ensure_config_files(self):
        """Ensure configuration files exist with default content."""
        # Create default prompts.yaml (legacy/default mode)
        prompts_file = self.config_dir / "prompts.yaml"
        if not prompts_file.exists():
            with open(prompts_file, "w") as f:
                yaml.dump(
                    self._get_default_prompts("default"),
                    f,
                    default_flow_style=False,
                )

        # Create default dictation.yaml (legacy dictation mode)
        dictation_file = self.config_dir / "dictation.yaml"
        if not dictation_file.exists():
            with open(dictation_file, "w") as f:
                yaml.dump(
                    self._get_default_prompts("dictation"),
                    f,
                    default_flow_style=False,
                )

        # Create transcribe.yaml (plain transcription mode)
        transcribe_file = self.config_dir / "transcribe.yaml"
        if not transcribe_file.exists():
            with open(transcribe_file, "w") as f:
                yaml.dump(
                    self._get_default_prompts("transcribe"),
                    f,
                    default_flow_style=False,
                )

        # Create auto_transcribe.yaml (auto-stop transcription mode)
        auto_transcribe_file = self.config_dir / "auto_transcribe.yaml"
        if not auto_transcribe_file.exists():
            with open(auto_transcribe_file, "w") as f:
                yaml.dump(
                    self._get_default_prompts("auto_transcribe"),
                    f,
                    default_flow_style=False,
                )

        # Create command.yaml (LLM command mode)
        command_file = self.config_dir / "command.yaml"
        if not command_file.exists():
            with open(command_file, "w") as f:
                yaml.dump(
                    self._get_default_prompts("command"),
                    f,
                    default_flow_style=False,
                )

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
