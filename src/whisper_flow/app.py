"""Main application class for whisper-flow."""

from pathlib import Path

from .audio import AudioRecorder
from .completion import CompletionService
from .config import Config
from .logging import log, set_logging_enabled
from .prompts import PromptManager
from .system import SystemManager
from .transcription import TranscriptionService


class WhisperFlow:
    """Main application class for whisper-flow."""

    def __init__(self, config_dir: Path | None = None, mode: str = "default"):
        """Initialize WhisperFlow application.

        Args:
            config_dir: Custom configuration directory
            mode: Processing mode (default, dictation, transcribe, auto_transcribe, command)

        """
        self.config = Config(config_dir=config_dir) if config_dir else Config()
        self.mode = mode

        # Initialize logging based on configuration
        set_logging_enabled(self.config.logging_enabled)

        # Initialize components
        self.system_manager = SystemManager(self.config)
        self.audio_recorder = AudioRecorder(self.config, self.system_manager)
        self.transcription_service = TranscriptionService(self.config)
        self.completion_service = CompletionService(self.config)
        self.prompt_manager = PromptManager(self.config, self.system_manager)

    def run_voice_flow_push_to_talk_daemon(self, stop_key: str, stop_event) -> bool:
        """Run voice flow with daemon-controlled push-to-talk recording.

        Args:
            stop_key: Hotkey combination that stops recording (for display)
            stop_event: Threading event to control recording stop

        Returns:
            True if successful, False otherwise

        """
        try:
            # Record audio with daemon-controlled stop event
            audio_file = self.audio_recorder.record_push_to_talk(stop_key, stop_event)

            if not audio_file:
                log("No audio recorded")
                return False

            return self._process_recorded_audio(audio_file)

        except Exception as e:
            log(f"Error in daemon push-to-talk flow: {e}")
            self.system_manager.notify(f"Push-to-talk failed: {e}")
            return False

    def run_voice_flow_auto_stop(self, silence_duration: float = 2.0) -> bool:
        """Run voice flow with auto-stop on silence.

        Args:
            silence_duration: Seconds of silence before stopping

        Returns:
            True if successful, False otherwise

        """
        try:
            # Record audio until silence detected
            log(f"Recording... Will auto-stop after {silence_duration}s of silence")
            audio_file = self.audio_recorder.record_until_silence(silence_duration)

            if not audio_file:
                log("No audio recorded")
                return False

            return self._process_recorded_audio(audio_file)

        except Exception as e:
            log(f"Error in auto-stop flow: {e}")
            self.system_manager.notify(f"Auto-stop recording failed: {e}")
            return False

    def _process_recorded_audio(self, audio_file: str) -> bool:
        """Process recorded audio file through the full pipeline.

        Args:
            audio_file: Path to the recorded audio file

        Returns:
            True if successful, False otherwise

        """
        try:
            # Transcribe audio
            log("Transcribing...")
            transcript = self.transcription_service.transcribe_audio(audio_file)

            if not transcript:
                log("Transcription failed")
                return False

            log("Transcript completed")

            # For transcribe and auto_transcribe modes, return transcript as-is
            if self.mode in ["transcribe", "auto_transcribe"]:
                final_result = transcript
            # Use the new simplified prompt system
            elif self.prompt_manager.should_use_completion():
                log("Processing with AI...")
                messages = self.prompt_manager.get_messages(transcript)

                final_result = self.completion_service.complete_text(messages)

                if not final_result:
                    log("AI processing failed, using raw transcript")
                    final_result = transcript
            else:
                final_result = transcript

            log("Final result completed")

            # Paste the result
            if not self.system_manager.paste_text(final_result):
                log("Failed to paste text, copying to clipboard...")
                self.system_manager._copy_to_clipboard(final_result)

            return True

        except Exception as e:
            log(f"Error processing audio: {e}")
            return False
        finally:
            # Cleanup temporary file
            try:
                Path(audio_file).unlink()
            except Exception:
                pass

    def run_comprehensive_validation(self) -> dict:
        """Run comprehensive system validation.

        Returns:
            Dictionary with validation results by category

        """
        results = {}

        # API Configuration validation
        results["api_config"] = self._validate_api_config()

        # System Dependencies validation
        results["system_deps"] = self._validate_system_dependencies()

        # Audio System validation
        results["audio_system"] = self._validate_audio_system()

        # Services validation
        results["services"] = self._validate_services()

        # Configuration Files validation
        results["config_files"] = self._validate_config_files()

        # Environment validation
        results["environment"] = self._validate_environment()

        return results

    def _validate_api_config(self) -> list[dict]:
        """Validate API configuration."""
        tests = []

        # API Key validation
        if self.config.openai_api_key:
            tests.append(
                {
                    "name": "OpenAI API Key",
                    "status": "pass",
                    "message": "API key is configured",
                },
            )
        else:
            tests.append(
                {
                    "name": "OpenAI API Key",
                    "status": "fail",
                    "message": "OpenAI API key not set",
                },
            )

        # Model validation
        valid_models = ["gpt-4o-mini", "gpt-4o-mini-transcribe", "gpt-4o", "gpt-4"]
        if self.config.transcription_model in valid_models:
            tests.append(
                {
                    "name": "Transcription Model",
                    "status": "pass",
                    "message": f"Valid model: {self.config.transcription_model}",
                },
            )
        else:
            tests.append(
                {
                    "name": "Transcription Model",
                    "status": "warn",
                    "message": f"Unknown model: {self.config.transcription_model}",
                },
            )

        return tests

    def _validate_system_dependencies(self) -> list[dict]:
        """Validate system dependencies."""
        tests = []
        deps = ["xdotool", "xclip", "xsel", "notify-send", "wmctrl"]

        for dep in deps:
            try:
                import subprocess

                result = subprocess.run(
                    [dep, "--version"],
                    capture_output=True,
                    check=True,
                )
                tests.append(
                    {
                        "name": f"System Tool: {dep}",
                        "status": "pass",
                        "message": "Available",
                    },
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                tests.append(
                    {
                        "name": f"System Tool: {dep}",
                        "status": "warn",
                        "message": "Not available",
                    },
                )

        return tests

    def _validate_audio_system(self) -> list[dict]:
        """Validate audio system."""
        tests = []

        # PyAudio availability
        try:
            import pyaudio

            tests.append(
                {
                    "name": "PyAudio Library",
                    "status": "pass",
                    "message": "Available",
                },
            )
        except ImportError:
            tests.append(
                {
                    "name": "PyAudio Library",
                    "status": "fail",
                    "message": "Not installed",
                },
            )
            return tests

        # Audio devices
        try:
            pa = pyaudio.PyAudio()
            device_count = pa.get_device_count()
            tests.append(
                {
                    "name": "Audio Devices",
                    "status": "pass",
                    "message": f"{device_count} devices found",
                },
            )
            pa.terminate()
        except Exception as e:
            tests.append(
                {
                    "name": "Audio Devices",
                    "status": "fail",
                    "message": f"Error: {e}",
                },
            )

        return tests

    def _validate_services(self) -> list[dict]:
        """Validate external services."""
        tests = []

        # Transcription service
        if self.config.openai_api_key:
            tests.append(
                {
                    "name": "Transcription Service",
                    "status": "pass",
                    "message": "OpenAI API configured",
                },
            )
        else:
            tests.append(
                {
                    "name": "Transcription Service",
                    "status": "fail",
                    "message": "No API key configured",
                },
            )

        # Completion service
        if self.config.openai_api_key:
            tests.append(
                {
                    "name": "Completion Service",
                    "status": "pass",
                    "message": "OpenAI API configured",
                },
            )
        else:
            tests.append(
                {
                    "name": "Completion Service",
                    "status": "fail",
                    "message": "No API key configured",
                },
            )

        return tests

    def _validate_config_files(self) -> list[dict]:
        """Validate configuration files."""
        tests = []

        # No longer need to validate prompt configuration files
        # The system now uses a single template approach
        tests.append(
            {
                "name": "Prompt System",
                "status": "pass",
                "message": "Using simplified single template approach",
            },
        )

        return tests

    def _validate_environment(self) -> list[dict]:
        """Validate environment setup."""
        tests = []

        # Python version
        import sys

        if sys.version_info >= (3, 11):
            tests.append(
                {
                    "name": "Python Version",
                    "status": "pass",
                    "message": f"Python {sys.version_info.major}.{sys.version_info.minor}",
                },
            )
        else:
            tests.append(
                {
                    "name": "Python Version",
                    "status": "warn",
                    "message": f"Python {sys.version_info.major}.{sys.version_info.minor} (3.11+ recommended)",
                },
            )

        return tests
