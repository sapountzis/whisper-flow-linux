"""Unit tests for the WhisperFlow application class."""

import tempfile
from unittest.mock import Mock, patch

from whisper_flow.app import WhisperFlow


class TestWhisperFlow:
    """Test the WhisperFlow application class."""

    def test_init_default(self, temp_config_dir):
        """Test WhisperFlow initialization with default parameters."""
        with (
            patch("whisper_flow.app.Config") as mock_config_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_recorder_class,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_audio_recorder_class.return_value = Mock()
            app = WhisperFlow()
            assert app.mode == "default"
            mock_config_class.assert_called_once_with()

    def test_init_with_config_dir(self, temp_config_dir):
        """Test WhisperFlow initialization with custom config directory."""
        with (
            patch("whisper_flow.app.Config") as mock_config_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_recorder_class,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_audio_recorder_class.return_value = Mock()
            app = WhisperFlow(config_dir=temp_config_dir, mode="transcribe")
            assert app.mode == "transcribe"
            mock_config_class.assert_called_once_with(config_dir=temp_config_dir)

    def test_run_voice_flow_push_to_talk_daemon_success(self, mock_config):
        """Test successful push-to-talk voice flow."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_class,
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
            patch("whisper_flow.app.CompletionService") as mock_completion_class,
            patch("whisper_flow.app.PromptManager") as mock_prompt_class,
        ):
            # Setup mocks
            mock_system = Mock()
            mock_system_class.return_value = mock_system

            mock_audio = Mock()
            mock_audio.record_push_to_talk.return_value = "/tmp/test.wav"
            mock_audio_class.return_value = mock_audio

            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = "Test transcript"
            mock_transcription_class.return_value = mock_transcription

            mock_completion = Mock()
            mock_completion_class.return_value = mock_completion

            mock_prompt = Mock()
            mock_prompt.choose_prompt.return_value = "Test prompt"
            mock_prompt.should_use_completion.return_value = False
            mock_prompt_class.return_value = mock_prompt

            # Create app and test
            app = WhisperFlow(mode="command")
            stop_event = Mock()

            result = app.run_voice_flow_push_to_talk_daemon("ctrl+shift+t", stop_event)

            assert result is True
            mock_audio.record_push_to_talk.assert_called_once_with(
                "ctrl+shift+t",
                stop_event,
            )
            mock_transcription.transcribe_audio.assert_called_once_with("/tmp/test.wav")

    def test_run_voice_flow_push_to_talk_daemon_no_audio(self, mock_config):
        """Test push-to-talk voice flow when no audio is recorded."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_class,
            patch("whisper_flow.app.TranscriptionService"),
            patch("whisper_flow.app.CompletionService"),
            patch("whisper_flow.app.PromptManager"),
        ):
            mock_system = Mock()
            mock_system_class.return_value = mock_system

            mock_audio = Mock()
            mock_audio.record_push_to_talk.return_value = None
            mock_audio_class.return_value = mock_audio

            app = WhisperFlow()
            stop_event = Mock()

            result = app.run_voice_flow_push_to_talk_daemon("ctrl+shift+t", stop_event)

            assert result is False

    def test_run_voice_flow_auto_stop_success(self, mock_config):
        """Test successful auto-stop voice flow."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_class,
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
            patch("whisper_flow.app.CompletionService"),
            patch("whisper_flow.app.PromptManager"),
        ):
            mock_system = Mock()
            mock_system_class.return_value = mock_system

            mock_audio = Mock()
            mock_audio.record_until_silence.return_value = "/tmp/test.wav"
            mock_audio_class.return_value = mock_audio

            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = "Test transcript"
            mock_transcription_class.return_value = mock_transcription

            app = WhisperFlow(mode="transcribe")

            result = app.run_voice_flow_auto_stop(silence_duration=3.0)

            assert result is True
            mock_audio.record_until_silence.assert_called_once_with(3.0)

    def test_run_comprehensive_validation(self, mock_config):
        """Test comprehensive validation method."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService"),
            patch("whisper_flow.app.CompletionService"),
            patch("whisper_flow.app.PromptManager"),
        ):
            app = WhisperFlow()

            results = app.run_comprehensive_validation()

            assert isinstance(results, dict)
            assert "api_config" in results
            assert "system_deps" in results
            assert "audio_system" in results
            assert "services" in results
            assert "config_files" in results
            assert "environment" in results

    def test_run_comprehensive_tests(self, mock_config):
        """Test comprehensive tests method."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService"),
            patch("whisper_flow.app.CompletionService"),
            patch("whisper_flow.app.PromptManager"),
        ):
            app = WhisperFlow()

            results = app.run_comprehensive_tests(verbose=True)

            assert isinstance(results, dict)
            assert "api_config" in results
            assert "system_deps" in results
            assert "audio_system" in results
            assert "services" in results
            assert "config_files" in results
            assert "environment" in results

    def test_process_recorded_audio_transcribe_mode(self, mock_config):
        """Test processing recorded audio in transcribe mode."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
            patch("whisper_flow.app.CompletionService"),
            patch("whisper_flow.app.PromptManager"),
        ):
            mock_system = Mock()
            mock_system.paste_text.return_value = True
            mock_system_class.return_value = mock_system
            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = "Test transcript"
            mock_transcription_class.return_value = mock_transcription
            app = WhisperFlow(mode="transcribe")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(b"fake audio data")
                tmp_file.flush()
                result = app._process_recorded_audio(tmp_file.name)
                assert result is True
                mock_transcription.transcribe_audio.assert_called_once_with(
                    tmp_file.name,
                )
                mock_system.paste_text.assert_called_once_with("Test transcript")

    def test_process_recorded_audio_command_mode_with_completion(self, mock_config):
        """Test processing recorded audio in command mode with completion."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
            patch("whisper_flow.app.CompletionService") as mock_completion_class,
            patch("whisper_flow.app.PromptManager") as mock_prompt_class,
        ):
            mock_system = Mock()
            mock_system.get_active_window_title.return_value = "Test Window"
            mock_system.paste_text.return_value = True
            mock_system_class.return_value = mock_system
            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = "Test transcript"
            mock_transcription_class.return_value = mock_transcription
            mock_completion = Mock()
            mock_completion.complete_text.return_value = "Test completion"
            mock_completion_class.return_value = mock_completion
            mock_prompt = Mock()
            mock_prompt.get_system_message.return_value = "You are a helpful assistant."
            mock_prompt.get_user_message.return_value = (
                "Date: 2024-01-01\nTime: 12:00:00\nUser input: Test transcript"
            )
            mock_prompt.get_messages.return_value = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Date: 2024-01-01\nTime: 12:00:00\nUser input: Test transcript",
                },
            ]
            mock_prompt.should_use_completion.return_value = True
            mock_prompt_class.return_value = mock_prompt
            app = WhisperFlow(mode="command")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(b"fake audio data")
                tmp_file.flush()
                result = app._process_recorded_audio(tmp_file.name)
                assert result is True
                mock_transcription.transcribe_audio.assert_called_once_with(
                    tmp_file.name,
                )
                mock_prompt.should_use_completion.assert_called_once()
                mock_prompt.get_messages.assert_called_once_with("Test transcript")
                mock_completion.complete_text.assert_called_once_with(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": "Date: 2024-01-01\nTime: 12:00:00\nUser input: Test transcript",
                        },
                    ],
                )
                mock_system.paste_text.assert_called_once_with("Test completion")

    def test_process_recorded_audio_transcription_failure(self, mock_config):
        """Test processing recorded audio when transcription fails."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
            patch("whisper_flow.app.CompletionService"),
            patch("whisper_flow.app.PromptManager"),
        ):
            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = None
            mock_transcription_class.return_value = mock_transcription
            app = WhisperFlow()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(b"fake audio data")
                tmp_file.flush()
                result = app._process_recorded_audio(tmp_file.name)
                assert result is False

    def test_validate_api_config_with_key(self, mock_config):
        """Test API configuration validation with API key."""
        mock_config.openai_api_key = "test-key"
        mock_config.transcription_model = "gpt-4o-mini"

        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService"),
            patch("whisper_flow.app.CompletionService"),
            patch("whisper_flow.app.PromptManager"),
        ):
            app = WhisperFlow()
            results = app._validate_api_config()

            assert len(results) == 2
            assert any(
                r["name"] == "OpenAI API Key" and r["status"] == "pass" for r in results
            )
            assert any(
                r["name"] == "Transcription Model" and r["status"] == "pass"
                for r in results
            )

    def test_validate_api_config_without_key(self, mock_config):
        """Test API configuration validation without API key."""
        mock_config.openai_api_key = None
        mock_config.transcription_model = "gpt-4o-mini"

        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService"),
            patch("whisper_flow.app.CompletionService"),
            patch("whisper_flow.app.PromptManager"),
        ):
            app = WhisperFlow()
            results = app._validate_api_config()

            assert len(results) == 2
            assert any(
                r["name"] == "OpenAI API Key" and r["status"] == "fail" for r in results
            )
            assert any(
                r["name"] == "Transcription Model" and r["status"] == "pass"
                for r in results
            )

    def test_audio_speedup_configuration(self, mock_config):
        """Test that audio speedup configuration is properly handled."""
        # Test with speedup disabled (1.0 = normal speed)
        mock_config.speedup_audio = 1.0
        with patch("whisper_flow.app.Config", return_value=mock_config):
            app = WhisperFlow()
            assert app.config.speedup_audio == 1.0

        # Test with speedup enabled (1.5x speed)
        mock_config.speedup_audio = 1.5
        with patch("whisper_flow.app.Config", return_value=mock_config):
            app = WhisperFlow()
            assert app.config.speedup_audio == 1.5

    def test_audio_speedup_processing(self, mock_config):
        """Test that audio speedup processing works correctly."""
        # Test with speedup enabled (1.5x speed)
        mock_config.speedup_audio = 1.5
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder") as mock_audio_class,
            patch("whisper_flow.app.TranscriptionService"),
            patch("whisper_flow.app.CompletionService"),
            patch("whisper_flow.app.PromptManager"),
        ):
            mock_audio = Mock()
            mock_audio.config = mock_config
            mock_audio_class.return_value = mock_audio

            app = WhisperFlow()

            # Verify that the audio recorder has the speedup configuration
            assert app.audio_recorder.config.speedup_audio == 1.5

    def test_logging_configuration(self, mock_config):
        """Test that logging configuration is properly handled."""
        # Test with logging disabled (default)
        mock_config.logging_enabled = False
        with patch("whisper_flow.app.Config", return_value=mock_config):
            app = WhisperFlow()
            assert app.config.logging_enabled is False

        # Test with logging enabled
        mock_config.logging_enabled = True
        with patch("whisper_flow.app.Config", return_value=mock_config):
            app = WhisperFlow()
            assert app.config.logging_enabled is True

    def test_logging_function(self):
        """Test that the logging function works correctly."""
        from whisper_flow.logging import log, set_logging_enabled

        # Test with logging disabled
        set_logging_enabled(False)
        # This should not print anything
        log("This should not appear")

        # Test with logging enabled
        set_logging_enabled(True)
        # This should print
        log("This should appear")
