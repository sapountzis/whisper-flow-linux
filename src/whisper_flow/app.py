"""Main application class for whisper-flow."""

import tempfile
from pathlib import Path

from .audio import AudioRecorder
from .completion import CompletionService
from .config import Config
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
                print("No audio recorded")
                return False

            return self._process_recorded_audio(audio_file)

        except Exception as e:
            print(f"Error in daemon push-to-talk flow: {e}")
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
            print(f"Recording... Will auto-stop after {silence_duration}s of silence")
            audio_file = self.audio_recorder.record_until_silence(silence_duration)

            if not audio_file:
                print("No audio recorded")
                return False

            return self._process_recorded_audio(audio_file)

        except Exception as e:
            print(f"Error in auto-stop flow: {e}")
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
            print("Transcribing...")
            transcript = self.transcription_service.transcribe_audio(audio_file)

            if not transcript:
                print("Transcription failed")
                return False

            print(f"Transcript: {transcript}")

            # For transcribe and auto_transcribe modes, return transcript as-is
            if self.mode in ["transcribe", "auto_transcribe"]:
                final_result = transcript
            else:
                # Get window context and choose prompt for command mode
                window_title = self.system_manager.get_active_window_title()
                prompt_template = self.prompt_manager.choose_prompt(
                    window_title,
                    self.mode,
                )

                # Process with completion if needed
                if self.prompt_manager.should_use_completion(prompt_template):
                    print("Processing with AI...")
                    processed_prompt = self.prompt_manager.process_prompt(
                        prompt_template,
                        transcript,
                    )
                    final_result = self.completion_service.complete_text(
                        processed_prompt,
                    )

                    if not final_result:
                        print("AI processing failed, using raw transcript")
                        final_result = transcript
                else:
                    final_result = transcript

            print(f"Final result: {final_result}")

            # Paste the result
            if not self.system_manager.paste_text(final_result):
                print("Failed to paste text, copying to clipboard...")
                self.system_manager._copy_to_clipboard(final_result)

            # Show notification with mode-specific message
            mode_messages = {
                "transcribe": "Transcription completed",
                "auto_transcribe": "Auto-transcription completed",
                "command": "Command processed",
            }
            message = mode_messages.get(self.mode, "Voice input processed")
            self.system_manager.notify(message)

            return True

        except Exception as e:
            print(f"Error processing audio: {e}")
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

    def run_comprehensive_tests(self, verbose: bool = False) -> dict:
        """Run comprehensive system tests.

        Args:
            verbose: Whether to show detailed output

        Returns:
            Dictionary with test results by category

        """
        results = {}

        # API Configuration tests
        results["api_config"] = self._test_api_config(verbose)

        # System Dependencies tests
        results["system_deps"] = self._test_system_dependencies(verbose)

        # Audio System tests
        results["audio_system"] = self._test_audio_system(verbose)

        # Services tests
        results["services"] = self._test_services(verbose)

        # Configuration Files tests
        results["config_files"] = self._test_config_files(verbose)

        # Environment tests
        results["environment"] = self._test_environment(verbose)

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
        deps = {
            "xdotool": ["xdotool", "--version"],
            "xclip": ["xclip", "-version"],
            "xsel": ["xsel", "--version"],
            "notify-send": ["notify-send", "--version"],
            "wmctrl": ["wmctrl", "-m"],
        }

        for name, cmd in deps.items():
            try:
                import subprocess

                result = subprocess.run(cmd, capture_output=True, check=True)
                tests.append(
                    {
                        "name": f"System Tool: {name}",
                        "status": "pass",
                        "message": "Available",
                    },
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                tests.append(
                    {
                        "name": f"System Tool: {name}",
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

        config_files = [
            "prompts.yaml",
            "dictation.yaml",
            "transcribe.yaml",
            "auto_transcribe.yaml",
            "command.yaml",
        ]

        for filename in config_files:
            file_path = self.config.config_dir / filename
            if file_path.exists():
                try:
                    import yaml

                    with open(file_path) as f:
                        yaml.safe_load(f)
                    tests.append(
                        {
                            "name": f"Config File: {filename}",
                            "status": "pass",
                            "message": "Valid YAML",
                        },
                    )
                except Exception as e:
                    tests.append(
                        {
                            "name": f"Config File: {filename}",
                            "status": "fail",
                            "message": f"Invalid YAML: {e}",
                        },
                    )
            else:
                tests.append(
                    {
                        "name": f"Config File: {filename}",
                        "status": "warn",
                        "message": "Missing (will use defaults)",
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

        # Disk space
        try:
            import shutil

            free_space = shutil.disk_usage(self.config.config_dir).free
            if free_space > 100 * 1024 * 1024:  # 100MB
                tests.append(
                    {
                        "name": "Disk Space",
                        "status": "pass",
                        "message": f"{free_space // (1024 * 1024)}MB available",
                    },
                )
            else:
                tests.append(
                    {
                        "name": "Disk Space",
                        "status": "warn",
                        "message": "Low disk space",
                    },
                )
        except Exception:
            tests.append(
                {
                    "name": "Disk Space",
                    "status": "warn",
                    "message": "Could not check",
                },
            )

        return tests

    def _test_api_config(self, verbose: bool) -> list[dict]:
        """Test API configuration with actual calls."""
        tests = []

        if not self.config.openai_api_key:
            tests.append(
                {
                    "name": "API Connectivity",
                    "status": "skip",
                    "message": "No API key configured",
                },
            )
            return tests

        # Test API connectivity
        try:
            import requests

            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=self.config.openai_headers,
                timeout=10,
            )
            if response.status_code == 200:
                tests.append(
                    {
                        "name": "API Connectivity",
                        "status": "pass",
                        "message": "OpenAI API reachable",
                    },
                )
            else:
                tests.append(
                    {
                        "name": "API Connectivity",
                        "status": "fail",
                        "message": f"API returned {response.status_code}",
                    },
                )
        except Exception as e:
            tests.append(
                {
                    "name": "API Connectivity",
                    "status": "fail",
                    "message": f"Connection failed: {e}",
                },
            )

        return tests

    def _test_system_dependencies(self, verbose: bool) -> list[dict]:
        """Test system dependencies functionality."""
        tests = []

        # Test clipboard functionality
        try:
            test_text = "test_whisper_flow"
            self.system_manager._copy_to_clipboard(test_text)
            tests.append(
                {
                    "name": "Clipboard Functionality",
                    "status": "pass",
                    "message": "Clipboard operations working",
                },
            )
        except Exception as e:
            tests.append(
                {
                    "name": "Clipboard Functionality",
                    "status": "fail",
                    "message": f"Clipboard failed: {e}",
                },
            )

        # Test window detection
        try:
            window_title = self.system_manager.get_active_window_title()
            tests.append(
                {
                    "name": "Window Detection",
                    "status": "pass",
                    "message": f"Active window: {window_title[:50]}..."
                    if window_title
                    else "No window detected",
                },
            )
        except Exception as e:
            tests.append(
                {
                    "name": "Window Detection",
                    "status": "fail",
                    "message": f"Window detection failed: {e}",
                },
            )

        return tests

    def _test_audio_system(self, verbose: bool) -> list[dict]:
        """Test audio system functionality."""
        tests = []

        # Test microphone access
        try:
            import pyaudio

            pa = pyaudio.PyAudio()

            # Try to open a stream
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.mic_device_index,
                frames_per_buffer=1024,
            )

            # Read a small amount of data
            data = stream.read(1024)
            stream.close()
            pa.terminate()

            tests.append(
                {
                    "name": "Microphone Access",
                    "status": "pass",
                    "message": "Microphone accessible",
                },
            )
        except Exception as e:
            tests.append(
                {
                    "name": "Microphone Access",
                    "status": "fail",
                    "message": f"Microphone failed: {e}",
                },
            )

        return tests

    def _test_services(self, verbose: bool) -> list[dict]:
        """Test external services with real requests."""
        tests = []

        if not self.config.openai_api_key:
            tests.append(
                {
                    "name": "Service Tests",
                    "status": "skip",
                    "message": "No API key for testing",
                },
            )
            return tests

        # Test with a minimal audio file (create a short silent audio)
        try:
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

                # Test transcription
                try:
                    result = self.transcription_service.transcribe_audio(tmp_file.name)
                    tests.append(
                        {
                            "name": "Transcription Service",
                            "status": "pass",
                            "message": "Service responding",
                        },
                    )
                except Exception as e:
                    tests.append(
                        {
                            "name": "Transcription Service",
                            "status": "fail",
                            "message": f"Service failed: {e}",
                        },
                    )

                # Cleanup
                Path(tmp_file.name).unlink()

        except Exception as e:
            tests.append(
                {
                    "name": "Service Tests",
                    "status": "fail",
                    "message": f"Test setup failed: {e}",
                },
            )

        return tests

    def _test_config_files(self, verbose: bool) -> list[dict]:
        """Test configuration file functionality."""
        tests = []

        # Test prompt loading
        try:
            prompts = self.config.get_prompts_config("command")
            tests.append(
                {
                    "name": "Prompt Loading",
                    "status": "pass",
                    "message": f"Loaded {len(prompts)} prompts",
                },
            )
        except Exception as e:
            tests.append(
                {
                    "name": "Prompt Loading",
                    "status": "fail",
                    "message": f"Failed to load prompts: {e}",
                },
            )

        # Test prompt matching
        try:
            test_window = "test.py - Cursor"
            prompt = self.prompt_manager.choose_prompt(test_window, "command")
            tests.append(
                {
                    "name": "Prompt Matching",
                    "status": "pass",
                    "message": "Prompt matching functional",
                },
            )
        except Exception as e:
            tests.append(
                {
                    "name": "Prompt Matching",
                    "status": "fail",
                    "message": f"Prompt matching failed: {e}",
                },
            )

        return tests

    def _test_environment(self, verbose: bool) -> list[dict]:
        """Test environment functionality."""
        tests = []

        # Test config directory access
        try:
            test_file = self.config.config_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            tests.append(
                {
                    "name": "Config Directory Access",
                    "status": "pass",
                    "message": "Read/write access confirmed",
                },
            )
        except Exception as e:
            tests.append(
                {
                    "name": "Config Directory Access",
                    "status": "fail",
                    "message": f"Access failed: {e}",
                },
            )

        # Test network connectivity
        try:
            import socket

            socket.create_connection(("8.8.8.8", 53), timeout=3)
            tests.append(
                {
                    "name": "Network Connectivity",
                    "status": "pass",
                    "message": "Internet connection available",
                },
            )
        except Exception:
            tests.append(
                {
                    "name": "Network Connectivity",
                    "status": "warn",
                    "message": "Limited network access",
                },
            )

        return tests
