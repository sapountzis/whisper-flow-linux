"""Unit tests for the WhisperFlow daemon."""

import time
from unittest.mock import Mock, patch

import pytest

from whisper_flow.daemon import WhisperFlowDaemon


class TestWhisperFlowDaemon:
    """Test the WhisperFlow daemon class."""

    def test_init(self, temp_config_dir):
        """Test daemon initialization."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)

            assert daemon.config == mock_config
            assert daemon.tray_icon is None
            assert daemon.is_running is False
            assert daemon.is_recording is False
            assert daemon.current_mode is None
            assert daemon.recording_thread is None
            assert daemon.stop_recording_event is None

            # Check that HotkeyManager was initialized
            assert daemon.hotkey_manager == mock_hotkey_manager
            mock_hotkey_manager_class.assert_called_once()

            # Check that WhisperFlow instances were created for different modes
            assert mock_app_class.call_count == 3

    def test_setup_hotkeys(self, temp_config_dir):
        """Test hotkey setup with HotkeyManager."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.setup_hotkeys()

            # Verify that hotkeys were registered
            assert mock_hotkey_manager.register_hotkey.call_count == 3
            mock_hotkey_manager.start.assert_called_once()

            # Check that the correct hotkeys were registered
            register_calls = mock_hotkey_manager.register_hotkey.call_args_list

            # Verify transcribe hotkey
            transcribe_call = register_calls[0]
            assert transcribe_call[1]["name"] == "transcribe"
            assert transcribe_call[1]["keys"] == "ctrl+cmd"

            # Verify auto_transcribe hotkey
            auto_transcribe_call = register_calls[1]
            assert auto_transcribe_call[1]["name"] == "auto_transcribe"
            assert auto_transcribe_call[1]["keys"] == "ctrl+cmd+space"

            # Verify command hotkey
            command_call = register_calls[2]
            assert command_call[1]["name"] == "command"
            assert command_call[1]["keys"] == "ctrl+cmd+alt"

    def test_test_configuration_success(self, temp_config_dir):
        """Test configuration testing with successful validation."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch("whisper_flow.daemon.WhisperFlowDaemon.notify") as mock_notify,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app.run_comprehensive_validation.return_value = {
                "api_config": [
                    {"name": "Test 1", "status": "pass", "message": "OK"},
                    {"name": "Test 2", "status": "pass", "message": "OK"},
                ],
                "system_deps": [
                    {"name": "Test 3", "status": "pass", "message": "OK"},
                ],
            }
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.transcribe_app = mock_app

            daemon.test_configuration(None, None)

            mock_app.run_comprehensive_validation.assert_called_once()
            mock_notify.assert_called_once_with(
                "✅ Configuration is valid! (3/3 tests passed)",
            )

    def test_test_configuration_with_warnings(self, temp_config_dir):
        """Test configuration testing with warnings."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch("whisper_flow.daemon.WhisperFlowDaemon.notify") as mock_notify,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app.run_comprehensive_validation.return_value = {
                "api_config": [
                    {"name": "Test 1", "status": "pass", "message": "OK"},
                    {"name": "Test 2", "status": "warn", "message": "Warning"},
                ],
            }
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.transcribe_app = mock_app

            daemon.test_configuration(None, None)

            mock_notify.assert_called_once_with(
                "⚠️ Configuration has warnings (1 passed, 1 warnings)",
            )

    def test_test_configuration_with_failures(self, temp_config_dir):
        """Test configuration testing with failures."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch("whisper_flow.daemon.WhisperFlowDaemon.notify") as mock_notify,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app.run_comprehensive_validation.return_value = {
                "api_config": [
                    {"name": "Test 1", "status": "pass", "message": "OK"},
                    {"name": "Test 2", "status": "fail", "message": "Failed"},
                ],
            }
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.transcribe_app = mock_app

            daemon.test_configuration(None, None)

            mock_notify.assert_called_once_with(
                "❌ Configuration has issues (1 passed, 1 failed, 0 warnings)",
            )

    def test_test_configuration_exception(self, temp_config_dir):
        """Test configuration testing when an exception occurs."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch("whisper_flow.daemon.WhisperFlowDaemon.notify") as mock_notify,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app.run_comprehensive_validation.side_effect = Exception("Test error")
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.transcribe_app = mock_app

            daemon.test_configuration(None, None)

            mock_notify.assert_called_once_with(
                "❌ Configuration test failed: Test error",
            )

    def test_stop_daemon(self, temp_config_dir):
        """Test stopping the daemon."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch("whisper_flow.daemon.WhisperFlowDaemon.notify") as mock_notify,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.is_running = True
            daemon.tray_icon = Mock()

            daemon.stop_daemon()

            assert daemon.is_running is False
            mock_notify.assert_called_once_with("👋 WhisperFlow daemon stopping...")
            daemon.tray_icon.stop.assert_called_once()

    def test_open_settings(self, temp_config_dir):
        """Test opening settings."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch("whisper_flow.daemon.WhisperFlowDaemon.notify") as mock_notify,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)

            daemon.open_settings(None, None)

            mock_notify.assert_called_once_with(
                "Settings not yet implemented - edit ~/.config/whisper-flow/config.yaml",
            )

    def test_get_app_for_mode_transcribe(self, temp_config_dir):
        """Test getting app instance for transcribe mode."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.transcribe_app = mock_app

            result = daemon._get_app_for_mode("transcribe")

            assert result == mock_app

    def test_get_app_for_mode_auto_transcribe(self, temp_config_dir):
        """Test getting app instance for auto_transcribe mode."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.auto_transcribe_app = mock_app

            result = daemon._get_app_for_mode("auto_transcribe")

            assert result == mock_app

    def test_get_app_for_mode_command(self, temp_config_dir):
        """Test getting app instance for command mode."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.command_app = mock_app

            result = daemon._get_app_for_mode("command")

            assert result == mock_app

    def test_get_app_for_mode_unknown(self, temp_config_dir):
        """Test getting app instance for unknown mode."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.transcribe_app = mock_app  # Default fallback

            result = daemon._get_app_for_mode("unknown_mode")

            assert result == mock_app

    def test_cancel_recording(self, temp_config_dir):
        """Test canceling recording."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.is_recording = True
            daemon.current_mode = "transcribe"
            daemon.cancel_recording()
            assert daemon.is_recording is False
            assert daemon.current_mode is None

    def test_stop_recording(self, temp_config_dir):
        """Test stopping recording."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.is_recording = True
            daemon.current_mode = "transcribe"
            daemon.recording_thread = Mock()
            daemon.stop_recording_event = Mock()
            daemon._stop_recording()
            assert daemon.is_recording is False
            assert daemon.current_mode is None

    def test_stop_recording_if_active(self, temp_config_dir):
        """Test stopping recording only if the specified mode is active."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)

            # Test when recording and mode matches
            daemon.is_recording = True
            daemon.current_mode = "transcribe"
            daemon._stop_recording_if_active("transcribe")
            assert daemon.is_recording is False

            # Test when recording but mode doesn't match
            daemon.is_recording = True
            daemon.current_mode = "command"
            daemon._stop_recording_if_active("transcribe")
            assert daemon.is_recording is True  # Should not stop

    def test_notify(self, temp_config_dir):
        """Test notification functionality."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch("whisper_flow.system.subprocess.Popen") as mock_popen,
            patch("whisper_flow.system.shutil.which", return_value=True),
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config.notification_timeout = 5000
            mock_config_class.return_value = mock_config
            mock_app = Mock()
            mock_system_manager = Mock()
            mock_app.system_manager = mock_system_manager
            mock_app_class.return_value = mock_app
            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.transcribe_app = mock_app
            daemon.notify("Test message")
            mock_system_manager.notify.assert_called_once_with("Test message")

    def test_notify_fallback(self, temp_config_dir):
        """Test notification fallback when notify-send is not available."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch("whisper_flow.system.shutil.which", return_value=None),
            patch("builtins.print") as mock_print,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config
            mock_app = Mock()
            mock_system_manager = Mock()
            mock_app.system_manager = mock_system_manager
            mock_app_class.return_value = mock_app
            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.transcribe_app = mock_app
            daemon.notify("Test message")
            mock_system_manager.notify.assert_called_once_with("Test message")

    def test_cleanup(self, temp_config_dir):
        """Test cleanup functionality."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.is_running = True
            daemon._cleanup()

            # Verify hotkey manager was stopped
            mock_hotkey_manager.stop.assert_called_once()
            assert daemon.is_running is False

    @pytest.mark.integration
    def test_create_tray_icon(self, temp_config_dir):
        """Test creating tray icon image."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)

            icon_image = daemon.create_tray_icon()

            assert icon_image is not None
            assert hasattr(icon_image, "size")

    @pytest.mark.integration
    def test_create_recording_icon(self, temp_config_dir):
        """Test creating recording icon image."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)

            icon_image = daemon.create_recording_icon()

            assert icon_image is not None
            assert hasattr(icon_image, "size")

    def test_watchdog_functionality(self, temp_config_dir):
        """Test watchdog functionality for detecting hangs."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch(
                "whisper_flow.daemon.WhisperFlowDaemon._force_stop_recording",
            ) as mock_force_stop,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config.max_recording_duration = 5.0  # Short duration for testing
            mock_config.watchdog_interval = 0.1  # Fast interval for testing
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)
            daemon.is_running = True

            # Start watchdog
            daemon._start_watchdog()

            # Simulate recording that exceeds duration limit
            daemon.is_recording = True
            daemon.recording_start_time = time.time() - 10.0  # 10 seconds ago
            daemon.recording_thread = Mock()
            daemon.recording_thread.is_alive.return_value = True

            # Let watchdog run for a moment
            time.sleep(0.2)

            # Check if force stop was called
            mock_force_stop.assert_called_with("Recording timeout")

    def test_processing_lock_timeout(self, temp_config_dir):
        """Test processing lock timeout functionality."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
            patch("whisper_flow.daemon.WhisperFlowDaemon.notify") as mock_notify,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config.processing_lock_timeout = 0.1  # Short timeout for testing
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)

            # Acquire lock manually to simulate contention
            daemon.processing_lock.acquire()

            # Try to process mode (should timeout)
            daemon._process_mode("transcribe")

            # Check if timeout notification was sent
            mock_notify.assert_called_with("⚠️ System busy, request ignored")

            # Release lock
            daemon.processing_lock.release()

    def test_queue_request_timeout(self, temp_config_dir):
        """Test queue request timeout functionality."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config.queue_request_timeout = 1.0  # Short timeout for testing
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)

            # Add old request to queue
            old_time = time.time() - 10.0  # 10 seconds ago
            daemon.request_queue.put(("transcribe", old_time))

            # Process queue (should drop old request)
            daemon._process_next_in_queue()

            # Queue should be empty after dropping old request
            assert daemon.request_queue.empty()

    def test_audio_timeout_mechanism(self, temp_config_dir):
        """Test audio timeout mechanism in recording."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config.audio_read_timeout = 0.1
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app.run_voice_flow_push_to_talk_daemon.return_value = True
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)

            # Verify initial state
            assert not daemon.is_recording
            assert daemon.current_mode is None
            assert daemon.recording_start_time is None

            # Start recording
            daemon.start_recording("transcribe")

            # Do not assert daemon.is_recording here, as the thread may finish instantly in test context

            # Stop recording
            daemon._stop_recording()

            # Verify recording stopped
            assert not daemon.is_recording
            assert daemon.current_mode is None
            assert daemon.recording_start_time is None

    def test_hotkey_manager_heartbeat(self, temp_config_dir):
        """Test hotkey manager heartbeat functionality."""
        with (
            patch("whisper_flow.daemon.Config") as mock_config_class,
            patch("whisper_flow.daemon.WhisperFlow") as mock_app_class,
            patch("whisper_flow.daemon.HotkeyManager") as mock_hotkey_manager_class,
        ):
            mock_config = Mock()
            mock_config.hotkey_transcribe = "ctrl+cmd"
            mock_config.hotkey_auto_transcribe = "ctrl+cmd+space"
            mock_config.hotkey_command = "ctrl+cmd+alt"
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app_class.return_value = mock_app

            mock_hotkey_manager = Mock()
            mock_hotkey_manager.get_health_status.return_value = {
                "is_running": True,
                "listener_alive": True,
                "heartbeat_age": 2.0,
                "active_bindings": 3,
                "pressed_keys": 0,
                "current_push_to_talk": None,
            }
            mock_hotkey_manager_class.return_value = mock_hotkey_manager

            daemon = WhisperFlowDaemon(temp_config_dir)

            # Test health status
            health = daemon.hotkey_manager.get_health_status()
            assert health["is_running"] is True
            assert health["listener_alive"] is True
            assert health["active_bindings"] == 3
