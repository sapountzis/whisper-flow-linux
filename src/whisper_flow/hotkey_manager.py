"""Advanced hotkey management for whisper-flow with optimized key handling."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from pynput import keyboard

from .logging import log


class HotkeyMode(Enum):
    """Hotkey activation modes."""

    SINGLE_PRESS = "single_press"  # Triggered once on key combination
    PUSH_TO_TALK = "push_to_talk"  # Triggered on press, released on key release


@dataclass
class HotkeyBinding:
    """Configuration for a hotkey binding."""

    keys: set[str]
    mode: HotkeyMode
    callback_press: Callable[[], None] | None = None
    callback_release: Callable[[], None] | None = None
    priority: int = 0  # Higher priority = checked first
    description: str = ""


class HotkeyManager:
    """Simple, robust hotkey manager with minimal complexity."""

    def __init__(self, debounce_delay: float = 0.05):
        """Initialize the hotkey manager.

        Args:
            debounce_delay: Minimum time between key events (seconds)

        """
        self.debounce_delay = debounce_delay

        # State tracking
        self.pressed_keys: set[str] = set()
        self.active_bindings: dict[str, HotkeyBinding] = {}
        self.is_running = False
        self.keyboard_listener: keyboard.Listener | None = None

        # Simple state
        self.last_key_times: dict[str, float] = {}  # Track per-key debouncing
        self.active_combination: str | None = None
        self.current_push_to_talk: str | None = None
        self.processing_callback: Callable[[], bool] | None = None

        # Key mapping for consistent naming
        self.key_mapping = {
            "ctrl_l": "ctrl",
            "ctrl_r": "ctrl",
            "cmd_l": "cmd",
            "cmd_r": "cmd",
            "super_l": "cmd",  # Map super to cmd for consistency
            "super_r": "cmd",  # Map super to cmd for consistency
            "alt_l": "alt",
            "alt_r": "alt",
            "shift_l": "shift",
            "shift_r": "shift",
            "space": "space",
        }

    def register_processing_callback(self, callback: Callable[[], bool]) -> None:
        """Register callback to check if system is processing.

        Args:
            callback: Function that returns True if system is processing

        """
        self.processing_callback = callback

    def register_hotkey(
        self,
        name: str,
        keys: str,
        mode: HotkeyMode,
        callback_press: Callable[[], None] | None = None,
        callback_release: Callable[[], None] | None = None,
        priority: int = 0,
        description: str = "",
    ) -> None:
        """Register a new hotkey binding.

        Args:
            name: Unique identifier for this hotkey
            keys: Key combination string (e.g., "ctrl+cmd+alt")
            mode: Hotkey activation mode
            callback_press: Function to call when hotkey is pressed
            callback_release: Function to call when hotkey is released (push-to-talk only)
            priority: Priority for conflict resolution (higher = checked first)
            description: Human-readable description

        """
        key_set = self._parse_hotkey_combination(keys)
        binding = HotkeyBinding(
            keys=key_set,
            mode=mode,
            callback_press=callback_press,
            callback_release=callback_release,
            priority=priority,
            description=description,
        )
        self.active_bindings[name] = binding
        log(f"Registered hotkey '{name}': {keys} ({mode.value})")

    def unregister_hotkey(self, name: str) -> bool:
        """Unregister a hotkey binding.

        Args:
            name: Name of hotkey to unregister

        Returns:
            True if hotkey was found and removed

        """
        if name in self.active_bindings:
            del self.active_bindings[name]
            log(f"Unregistered hotkey '{name}'")
            return True
        return False

    def start(self) -> None:
        """Start the hotkey listener."""
        if self.is_running:
            log("HotkeyManager is already running")
            return

        try:
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self.keyboard_listener.start()
            self.is_running = True
            log("HotkeyManager started successfully")
        except Exception as e:
            log(f"Failed to start HotkeyManager: {e}")
            raise

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if not self.is_running:
            return

        try:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
            self.is_running = False

            # Reset state
            self.pressed_keys.clear()
            self.last_key_times.clear()
            self.active_combination = None
            self.current_push_to_talk = None

            log("HotkeyManager stopped")
        except Exception as e:
            log(f"Error stopping HotkeyManager: {e}")

    def is_hotkey_active(self, name: str) -> bool:
        """Check if a push-to-talk hotkey is currently active.

        Args:
            name: Name of hotkey to check

        Returns:
            True if the hotkey is currently being held down

        """
        return self.current_push_to_talk == name

    def get_active_hotkeys(self) -> list[str]:
        """Get list of currently registered hotkey names."""
        return list(self.active_bindings.keys())

    def get_hotkey_info(self, name: str) -> dict | None:
        """Get information about a registered hotkey.

        Args:
            name: Name of hotkey

        Returns:
            Dictionary with hotkey information or None if not found

        """
        if name not in self.active_bindings:
            return None

        binding = self.active_bindings[name]
        return {
            "name": name,
            "keys": list(binding.keys),
            "mode": binding.mode.value,
            "priority": binding.priority,
            "description": binding.description,
        }

    def _parse_hotkey_combination(self, hotkey_str: str) -> set[str]:
        """Parse hotkey string into a set of key names.

        Args:
            hotkey_str: Hotkey combination string (e.g., "ctrl+cmd+alt")

        Returns:
            Set of standardized key names

        """
        parts = hotkey_str.lower().split("+")
        key_set = set()

        for part in parts:
            part = part.strip()
            # Add the standardized key name
            key_set.add(part)

        return key_set

    def _get_key_name(self, key) -> str:
        """Get standardized key name from pynput key object.

        Args:
            key: Pynput key object

        Returns:
            Standardized key name string

        """
        try:
            # Validate that this is a proper key object
            if key is None:
                return None

            if hasattr(key, "name"):
                key_name = key.name.lower()
            elif hasattr(key, "char") and key.char:
                key_name = key.char.lower()
            else:
                # Not a valid key object
                return None

            # Validate key name - reject empty or invalid names
            if not key_name or key_name.strip() == "":
                return None

            # Apply key mapping for consistency
            return self.key_mapping.get(key_name, key_name)
        except Exception as e:
            log(f"[HOTKEY] Error in _get_key_name: {e}")
            return None

    def _on_key_press(self, key) -> None:
        """Handle key press events with robust error handling.

        Args:
            key: Pynput key object

        """
        try:
            current_time = time.time()
            key_name = self._get_key_name(key)

            # Validate key name - ignore invalid keys
            if key_name is None:
                log(f"[HOTKEY] Ignoring invalid key: {key}")
                return

            # Check if system is processing - ignore keys if so
            if self.processing_callback and self.processing_callback():
                return

            # Per-key debouncing (only debounce repeated presses of the SAME key)
            if key_name in self.last_key_times:
                if current_time - self.last_key_times[key_name] < self.debounce_delay:
                    return

            # Handle special keys
            if key_name in ["esc", "escape"]:
                self._handle_escape_key()
                return

            # Don't trigger new combinations while one is active
            if self.active_combination is not None:
                return

            self.pressed_keys.add(key_name)

            # Check for hotkey matches
            self._check_hotkey_combinations()

            # Update per-key timing
            self.last_key_times[key_name] = current_time

        except Exception as e:
            log(f"Error in key press handler: {e}")
            # Reset to known good state
            self.pressed_keys.clear()
            self.active_combination = None

    def _on_key_release(self, key) -> None:
        """Handle key release events with robust error handling.

        Args:
            key: Pynput key object

        """
        try:
            key_name = self._get_key_name(key)

            # Validate key name - ignore invalid keys
            if key_name is None:
                return

            self.pressed_keys.discard(key_name)

            # Handle push-to-talk release - only when ALL keys in the combination are released
            if self.current_push_to_talk:
                binding = self.active_bindings[self.current_push_to_talk]
                # Check if ANY key from the combination is still pressed
                if not any(key in self.pressed_keys for key in binding.keys):
                    self._trigger_hotkey_release(self.current_push_to_talk)
                    self.current_push_to_talk = None

            # Reset combination state when all keys released
            if len(self.pressed_keys) == 0:
                self.active_combination = None

        except Exception as e:
            log(f"Error in key release handler: {e}")
            # Reset to known good state
            self.pressed_keys.clear()
            self.active_combination = None
            self.current_push_to_talk = None

    def _check_hotkey_combinations(self) -> None:
        """Check for complete hotkey combinations and trigger callbacks."""
        try:
            # Find all matching combinations
            matching_bindings = []
            for name, binding in self.active_bindings.items():
                if binding.keys.issubset(self.pressed_keys):
                    matching_bindings.append((name, binding))

            if not matching_bindings:
                return

            # Sort by most specific first (most keys), then by priority
            sorted_bindings = sorted(
                matching_bindings,
                key=lambda x: (len(x[1].keys), x[1].priority),
                reverse=True,
            )

            # Trigger the most specific matching combination
            name, binding = sorted_bindings[0]
            self.active_combination = name

            if binding.mode == HotkeyMode.PUSH_TO_TALK:
                self.current_push_to_talk = name

            self._trigger_hotkey_press(name)

        except Exception as e:
            log(f"Error in hotkey combination check: {e}")
            # Reset to known good state
            self.active_combination = None

    def _trigger_hotkey_press(self, name: str) -> None:
        """Trigger hotkey press callback.

        Args:
            name: Name of hotkey to trigger

        """
        try:
            binding = self.active_bindings[name]
            if binding.callback_press:
                binding.callback_press()
        except Exception as e:
            log(f"Error triggering hotkey press '{name}': {e}")

    def _trigger_hotkey_release(self, name: str) -> None:
        """Trigger hotkey release callback.

        Args:
            name: Name of hotkey to trigger

        """
        try:
            binding = self.active_bindings[name]
            if binding.callback_release:
                binding.callback_release()
        except Exception as e:
            log(f"Error triggering hotkey release '{name}': {e}")

    def _handle_escape_key(self) -> None:
        """Handle escape key press for canceling operations."""
        # This can be overridden or configured with a callback
        log("Escape key pressed")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
