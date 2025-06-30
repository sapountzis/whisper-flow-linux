"""Unit tests for the HotkeyManager class."""

from unittest.mock import Mock, patch

from whisper_flow.hotkey_manager import HotkeyManager, HotkeyMode


class TestHotkeyManager:
    """Test the HotkeyManager class."""

    def test_init(self):
        """Test default initialization."""
        manager = HotkeyManager()
        assert manager.debounce_delay == 0.05
        assert manager.pressed_keys == set()
        assert manager.active_bindings == {}
        assert manager.is_running is False
        assert manager.keyboard_listener is None
        assert manager.active_combination is None
        assert manager.current_push_to_talk is None
        assert manager.processing_callback is None

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        manager = HotkeyManager(debounce_delay=0.1)
        assert manager.debounce_delay == 0.1
        assert manager.pressed_keys == set()
        assert manager.active_bindings == {}
        assert manager.is_running is False

    def test_register_hotkey(self):
        """Test registering hotkeys."""
        manager = HotkeyManager()
        callback = Mock()

        manager.register_hotkey(
            name="test_hotkey",
            keys="ctrl+cmd",
            mode=HotkeyMode.SINGLE_PRESS,
            callback_press=callback,
            priority=1,
            description="Test hotkey",
        )

        assert "test_hotkey" in manager.active_bindings
        binding = manager.active_bindings["test_hotkey"]
        assert binding.keys == {"ctrl", "cmd"}
        assert binding.mode == HotkeyMode.SINGLE_PRESS
        assert binding.callback_press == callback
        assert binding.priority == 1
        assert binding.description == "Test hotkey"

    def test_unregister_hotkey(self):
        """Test unregistering hotkeys."""
        manager = HotkeyManager()
        callback = Mock()

        # Register a hotkey
        manager.register_hotkey(
            name="test_hotkey",
            keys="ctrl+cmd",
            mode=HotkeyMode.SINGLE_PRESS,
            callback_press=callback,
        )

        assert "test_hotkey" in manager.active_bindings

        # Unregister it
        result = manager.unregister_hotkey("test_hotkey")
        assert result is True
        assert "test_hotkey" not in manager.active_bindings

        # Try to unregister non-existent hotkey
        result = manager.unregister_hotkey("non_existent")
        assert result is False

    def test_parse_hotkey_combination(self):
        """Test parsing hotkey combinations."""
        manager = HotkeyManager()

        # Test simple combination
        keys = manager._parse_hotkey_combination("ctrl+cmd")
        assert keys == {"ctrl", "cmd"}

        # Test complex combination
        keys = manager._parse_hotkey_combination("ctrl+cmd+alt+space")
        assert keys == {"ctrl", "cmd", "alt", "space"}

        # Test single key
        keys = manager._parse_hotkey_combination("a")
        assert keys == {"a"}

    def test_get_key_name(self):
        """Test key name standardization."""
        manager = HotkeyManager()

        # Test key with name attribute
        mock_key = Mock()
        mock_key.name = "ctrl_l"
        assert manager._get_key_name(mock_key) == "ctrl"

        # Test basic key mapping functionality
        assert manager.key_mapping.get("ctrl_l", "ctrl_l") == "ctrl"
        assert manager.key_mapping.get("cmd_r", "cmd_r") == "cmd"

    @patch("whisper_flow.hotkey_manager.keyboard.Listener")
    def test_start_stop(self, mock_listener_class):
        """Test starting and stopping the hotkey manager."""
        mock_listener = Mock()
        mock_listener_class.return_value = mock_listener

        manager = HotkeyManager()

        # Test start
        manager.start()
        assert manager.is_running is True
        assert manager.keyboard_listener == mock_listener
        mock_listener_class.assert_called_once()
        mock_listener.start.assert_called_once()

        # Test stop
        manager.stop()
        assert manager.is_running is False
        assert manager.keyboard_listener is None
        mock_listener.stop.assert_called_once()
        assert len(manager.pressed_keys) == 0
        assert manager.active_combination is None
        assert manager.current_push_to_talk is None

    def test_start_already_running(self):
        """Test starting when already running."""
        manager = HotkeyManager()
        manager.is_running = True

        with patch(
            "whisper_flow.hotkey_manager.keyboard.Listener",
        ) as mock_listener_class:
            manager.start()
            mock_listener_class.assert_not_called()

    def test_stop_not_running(self):
        """Test stopping when not running."""
        manager = HotkeyManager()
        manager.is_running = False
        manager.keyboard_listener = None

        # Should not raise any exceptions
        manager.stop()
        assert manager.is_running is False

    def test_is_hotkey_active(self):
        """Test checking if hotkey is active."""
        manager = HotkeyManager()

        # Test when no hotkey is active
        assert manager.is_hotkey_active("test") is False

        # Test when different hotkey is active
        manager.current_push_to_talk = "other"
        assert manager.is_hotkey_active("test") is False

        # Test when target hotkey is active
        manager.current_push_to_talk = "test"
        assert manager.is_hotkey_active("test") is True

    def test_get_active_hotkeys(self):
        """Test getting list of active hotkeys."""
        manager = HotkeyManager()

        # Test empty list
        assert manager.get_active_hotkeys() == []

        # Add some hotkeys
        manager.register_hotkey("hotkey1", "ctrl+a", HotkeyMode.SINGLE_PRESS)
        manager.register_hotkey("hotkey2", "ctrl+b", HotkeyMode.PUSH_TO_TALK)

        hotkeys = manager.get_active_hotkeys()
        assert len(hotkeys) == 2
        assert "hotkey1" in hotkeys
        assert "hotkey2" in hotkeys

    def test_get_hotkey_info(self):
        """Test getting hotkey information."""
        manager = HotkeyManager()

        # Test non-existent hotkey
        info = manager.get_hotkey_info("non_existent")
        assert info is None

        # Register a hotkey and get its info
        manager.register_hotkey(
            name="test_hotkey",
            keys="ctrl+cmd",
            mode=HotkeyMode.SINGLE_PRESS,
            priority=5,
            description="Test description",
        )

        info = manager.get_hotkey_info("test_hotkey")
        assert info is not None
        assert info["name"] == "test_hotkey"
        assert set(info["keys"]) == {"ctrl", "cmd"}
        assert info["mode"] == "single_press"
        assert info["priority"] == 5
        assert info["description"] == "Test description"

    @patch("whisper_flow.hotkey_manager.time.time")
    def test_on_key_press_debouncing(self, mock_time):
        """Test per-key debouncing."""
        manager = HotkeyManager()
        manager.last_key_times["a"] = 100.0
        mock_time.return_value = 100.02  # Within debounce delay for key 'a'

        mock_key = Mock()
        mock_key.name = "a"

        # First press of 'a' should be debounced
        manager._on_key_press(mock_key)
        assert len(manager.pressed_keys) == 0

        # Different key should NOT be debounced
        mock_key.name = "b"
        manager._on_key_press(mock_key)
        assert "b" in manager.pressed_keys

    def test_key_mapping(self):
        """Test key mapping for different key variations."""
        manager = HotkeyManager()

        # Test left/right ctrl mapping
        mock_key = Mock()
        mock_key.name = "ctrl_l"
        assert manager._get_key_name(mock_key) == "ctrl"

        mock_key.name = "ctrl_r"
        assert manager._get_key_name(mock_key) == "ctrl"

        # Test left/right cmd mapping
        mock_key.name = "cmd_l"
        assert manager._get_key_name(mock_key) == "cmd"

        mock_key.name = "cmd_r"
        assert manager._get_key_name(mock_key) == "cmd"

    def test_processing_callback(self):
        """Test processing callback functionality."""
        manager = HotkeyManager()

        # Test without callback
        assert manager.processing_callback is None

        # Register callback
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True
            return True

        manager.register_processing_callback(test_callback)
        assert manager.processing_callback == test_callback

        # Test callback is called
        assert manager.processing_callback() == True
        assert callback_called == True

    def test_ignore_keys_when_processing(self):
        """Test that keys are ignored when system is processing."""
        manager = HotkeyManager()

        # Register a processing callback that returns True
        manager.register_processing_callback(lambda: True)

        # Register a test hotkey
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        manager.register_hotkey(
            "test",
            "ctrl+cmd",
            HotkeyMode.SINGLE_PRESS,
            callback_press=test_callback,
        )

        # Simulate key press when processing - should be ignored
        class MockKey:
            def __init__(self, name):
                self.name = name

        # This should not trigger the callback because processing callback returns True
        manager._on_key_press(MockKey("ctrl"))
        manager._on_key_press(MockKey("cmd"))

        # Callback should not be called because system is processing
        assert not callback_called
        assert manager.active_combination is None

        # Now test without processing callback - should work normally
        manager.register_processing_callback(lambda: False)
        manager.pressed_keys.clear()
        manager._on_key_press(MockKey("ctrl"))
        manager._on_key_press(MockKey("cmd"))

        # Now callback should be called
        assert callback_called
        assert manager.active_combination == "test"

    def test_robust_error_handling(self):
        """Test that errors in key handling are handled gracefully."""
        manager = HotkeyManager()

        # Register a hotkey with a callback that raises an exception
        def bad_callback():
            raise Exception("Test error")

        manager.register_hotkey(
            "test",
            "ctrl+cmd",
            HotkeyMode.SINGLE_PRESS,
            callback_press=bad_callback,
        )

        # This should not crash even though callback raises an exception
        manager.pressed_keys = {"ctrl", "cmd"}
        manager._check_hotkey_combinations()

        # The combination should still be set even if callback fails
        assert manager.active_combination == "test"

        # Test that the system continues to work after callback error
        manager.pressed_keys.clear()
        manager.active_combination = None

        # Register a working hotkey with different keys
        callback_called = False

        def good_callback():
            nonlocal callback_called
            callback_called = True

        manager.register_hotkey(
            "good",
            "ctrl+alt",
            HotkeyMode.SINGLE_PRESS,
            callback_press=good_callback,
        )

        # This should work normally
        manager.pressed_keys = {"ctrl", "alt"}
        manager._check_hotkey_combinations()

        assert callback_called
        assert manager.active_combination == "good"

    def test_invalid_key_input_handling(self):
        """Test that invalid key inputs don't corrupt state."""
        manager = HotkeyManager()

        # Register a test hotkey
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        manager.register_hotkey(
            "test",
            "ctrl+cmd",
            HotkeyMode.SINGLE_PRESS,
            callback_press=test_callback,
        )

        # Test with invalid inputs that could happen in real scenarios
        invalid_inputs = [
            None,
            "invalid_string",
            123,
            [],
            {},
            object(),
        ]

        for invalid_input in invalid_inputs:
            # Reset state
            manager.pressed_keys.clear()
            manager.active_combination = None
            callback_called = False

            # This should not crash or corrupt state
            try:
                manager._on_key_press(invalid_input)
            except Exception as e:
                # Should not raise exceptions
                assert False, f"Invalid input {invalid_input} caused exception: {e}"

            # State should remain clean
            assert len(manager.pressed_keys) == 0, (
                f"Invalid input {invalid_input} corrupted pressed_keys"
            )
            assert manager.active_combination is None, (
                f"Invalid input {invalid_input} corrupted active_combination"
            )
            assert not callback_called, (
                f"Invalid input {invalid_input} triggered callback"
            )

    def test_malformed_key_objects(self):
        """Test handling of malformed key objects that might come from pynput."""
        manager = HotkeyManager()

        # Register a test hotkey
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        manager.register_hotkey(
            "test",
            "ctrl+cmd",
            HotkeyMode.SINGLE_PRESS,
            callback_press=test_callback,
        )

        # Test malformed key objects
        class MalformedKey:
            def __init__(self, name=None, char=None, error_on_access=False):
                self._name = name
                self._char = char
                self._error_on_access = error_on_access

            @property
            def name(self):
                if self._error_on_access:
                    raise Exception("Simulated pynput error")
                return self._name

            @property
            def char(self):
                if self._error_on_access:
                    raise Exception("Simulated pynput error")
                return self._char

        malformed_keys = [
            MalformedKey(),  # No name or char
            MalformedKey(name="", char=""),  # Empty strings
            MalformedKey(error_on_access=True),  # Raises exception on access
        ]

        for key in malformed_keys:
            # Reset state
            manager.pressed_keys.clear()
            manager.active_combination = None
            callback_called = False

            # This should handle gracefully
            try:
                manager._on_key_press(key)
            except Exception as e:
                # Should not raise exceptions
                assert False, f"Malformed key {key} caused exception: {e}"

            # State should remain clean
            assert len(manager.pressed_keys) == 0, (
                f"Malformed key {key} corrupted pressed_keys"
            )
            assert manager.active_combination is None, (
                f"Malformed key {key} corrupted active_combination"
            )
            assert not callback_called, f"Malformed key {key} triggered callback"

    def test_concurrent_key_processing(self):
        """Test that rapid key presses don't cause state corruption."""
        manager = HotkeyManager()

        # Register a test hotkey
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        manager.register_hotkey(
            "test",
            "ctrl+cmd",
            HotkeyMode.SINGLE_PRESS,
            callback_press=test_callback,
        )

        class MockKey:
            def __init__(self, name):
                self.name = name

        # Simulate rapid key presses
        for _ in range(10):
            manager._on_key_press(MockKey("ctrl"))
            manager._on_key_press(MockKey("cmd"))
            manager._on_key_release(MockKey("ctrl"))
            manager._on_key_release(MockKey("cmd"))

        # State should be consistent
        assert len(manager.pressed_keys) == 0
        assert manager.active_combination is None
        assert manager.current_push_to_talk is None
