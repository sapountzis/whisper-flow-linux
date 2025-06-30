"""Audio transcription functionality for whisper-flow."""

import time
from pathlib import Path

from openai import OpenAI

from .config import Config
from .logging import log


class TranscriptionService:
    """Audio transcription service using OpenAI API."""

    def __init__(self, config: Config):
        """Initialize transcription service.

        Args:
            config: Configuration object

        """
        self.config = config
        self.client = (
            OpenAI(api_key=config.openai_api_key) if config.openai_api_key else None
        )

    def transcribe_audio(self, audio_path: str, max_retries: int = 3) -> str | None:
        """Transcribe audio file using OpenAI API.

        Args:
            audio_path: Path to the audio file
            max_retries: Maximum number of retry attempts

        Returns:
            Transcribed text or None if failed

        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        for attempt in range(max_retries):
            try:
                return self._transcribe_with_openai(audio_path)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Transcription failed after {max_retries} attempts: {e}",
                    )

                # Wait before retry (exponential backoff)
                wait_time = 2**attempt
                log(
                    f"Transcription attempt {attempt + 1} failed, retrying in {wait_time}s...",
                )
                time.sleep(wait_time)

        return None

    def _transcribe_with_openai(self, audio_path: str) -> str:
        """Transcribe audio using OpenAI API.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text

        Raises:
            RuntimeError: If API key is not configured or client is not initialized

        """
        if not self.client:
            raise RuntimeError("OpenAI API key not configured")

        with open(audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=self.config.transcription_model,
                file=audio_file,
                response_format="text",
            )
            return transcription.strip()

    def get_available_models(self) -> list:
        """Get list of available transcription models.

        Returns:
            List of model names

        """
        # OpenAI models (as of current knowledge)
        openai_models = [
            "gpt-4o-mini-transcribe",
            "whisper-1",
        ]

        available_models = []

        # Check if OpenAI is available
        if self.config.openai_api_key:
            available_models.extend(openai_models)

        return available_models

    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate audio file format and size.

        Args:
            audio_path: Path to the audio file

        Returns:
            True if valid, False otherwise

        """
        try:
            audio_file = Path(audio_path)

            # Check if file exists
            if not audio_file.exists():
                return False

            # Check file size (OpenAI has 25MB limit)
            max_size = 25 * 1024 * 1024  # 25MB in bytes
            if audio_file.stat().st_size > max_size:
                log(
                    f"Audio file too large: {audio_file.stat().st_size} bytes (max: {max_size})",
                )
                return False

            # Check file extension
            valid_extensions = {
                ".wav",
                ".mp3",
                ".m4a",
                ".mp4",
                ".mpeg",
                ".mpga",
                ".webm",
            }
            if audio_file.suffix.lower() not in valid_extensions:
                log(f"Unsupported audio format: {audio_file.suffix}")
                return False

            return True

        except Exception as e:
            log(f"Error validating audio file: {e}")
            return False

    def get_transcription_info(self) -> dict:
        """Get information about transcription capabilities.

        Returns:
            Dictionary containing transcription service information

        """
        return {
            "openai_available": self.config.openai_api_key is not None,
            "current_model": self.config.transcription_model,
            "available_models": self.get_available_models(),
        }
