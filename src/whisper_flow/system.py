"""System integration functionality for whisper-flow."""

import shutil
import subprocess

from .config import Config


class SystemManager:
    """System integration manager for notifications, clipboard, and window operations."""

    def __init__(self, config: Config):
        """Initialize system manager.

        Args:
            config: Configuration object

        """
        self.config = config

    def notify(self, message: str) -> None:
        """Send desktop notification.

        Args:
            message: Notification message

        """
        if shutil.which("notify-send"):
            subprocess.Popen(
                [
                    "notify-send",
                    f"--expire-time={self.config.notification_timeout}",
                    "Whisper-Flow",
                    message,
                ],
            )
        else:
            # Fallback to console output
            print(f"[Whisper-Flow] {message}")

    def get_active_window_title(self) -> str | None:
        """Get the title of the active window.

        Returns:
            Window title string or None if unable to determine

        """
        try:
            if shutil.which("xdotool"):
                return subprocess.check_output(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    text=True,
                ).strip()
            if shutil.which("wmctrl"):
                # Alternative using wmctrl
                output = subprocess.check_output(["wmctrl", "-a"], text=True)
                # Parse wmctrl output to get active window
                for line in output.split("\n"):
                    if line.strip():
                        parts = line.split(None, 3)
                        if len(parts) >= 4:
                            return parts[3]
            return None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def paste_text(self, text: str) -> bool:
        """Paste text into the current application.

        Args:
            text: Text to paste

        Returns:
            True if successful, False otherwise

        """
        try:
            # Method 1: Copy to clipboard and paste
            if self._copy_to_clipboard(text):
                if shutil.which("xdotool"):
                    subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                        check=False,
                    )
                    return True

            # Method 2: Direct typing fallback
            if shutil.which("xdotool"):
                subprocess.run(["xdotool", "type", "--delay", "0", text], check=False)
                return True

            return False
        except Exception as e:
            print(f"Error pasting text: {e}")
            return False

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard.

        Args:
            text: Text to copy

        Returns:
            True if successful, False otherwise

        """
        try:
            if shutil.which("xclip"):
                p = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                )
                p.communicate(text.encode())
                return p.returncode == 0
            if shutil.which("xsel"):
                p = subprocess.Popen(
                    ["xsel", "--clipboard", "--input"],
                    stdin=subprocess.PIPE,
                )
                p.communicate(text.encode())
                return p.returncode == 0
            return False
        except Exception:
            return False

    def get_clipboard_content(self) -> str | None:
        """Get current clipboard content.

        Returns:
            Clipboard text or None if unable to read

        """
        try:
            if shutil.which("xclip"):
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                return result.stdout if result.returncode == 0 else None
            if shutil.which("xsel"):
                result = subprocess.run(
                    ["xsel", "--clipboard", "--output"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                return result.stdout if result.returncode == 0 else None
            return None
        except Exception:
            return None

    def simulate_keypress(self, key_combination: str) -> bool:
        """Simulate a key press.

        Args:
            key_combination: Key combination (e.g., "ctrl+c", "alt+Tab")

        Returns:
            True if successful, False otherwise

        """
        try:
            if shutil.which("xdotool"):
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", key_combination],
                    check=False,
                )
                return True
            return False
        except Exception:
            return False

    def get_mouse_position(self) -> tuple | None:
        """Get current mouse cursor position.

        Returns:
            (x, y) coordinates or None if unable to determine

        """
        try:
            if shutil.which("xdotool"):
                result = subprocess.run(
                    ["xdotool", "getmouselocation", "--shell"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    x = y = None
                    for line in lines:
                        if line.startswith("X="):
                            x = int(line.split("=")[1])
                        elif line.startswith("Y="):
                            y = int(line.split("=")[1])
                    if x is not None and y is not None:
                        return (x, y)
            return None
        except Exception:
            return None

    def check_dependencies(self) -> dict:
        """Check system dependencies and their availability.

        Returns:
            Dictionary of dependency names and their availability status

        """
        dependencies = {
            "xdotool": shutil.which("xdotool") is not None,
            "xclip": shutil.which("xclip") is not None,
            "xsel": shutil.which("xsel") is not None,
            "notify-send": shutil.which("notify-send") is not None,
            "wmctrl": shutil.which("wmctrl") is not None,
        }
        return dependencies

    def get_system_info(self) -> dict:
        """Get system information relevant to whisper-flow.

        Returns:
            Dictionary containing system information

        """
        deps = self.check_dependencies()
        return {
            "dependencies": deps,
            "clipboard_available": deps["xclip"] or deps["xsel"],
            "window_management_available": deps["xdotool"] or deps["wmctrl"],
            "notifications_available": deps["notify-send"],
            "active_window": self.get_active_window_title(),
            "mouse_position": self.get_mouse_position(),
        }

    def get_highlighted_text(self) -> str | None:
        """Get highlighted text from the current application using the primary selection (X11)."""
        try:
            if shutil.which("xclip"):
                result = subprocess.run(
                    ["xclip", "-selection", "primary", "-o"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    text_out = result.stdout.strip()
                    if text_out:
                        return text_out
            elif shutil.which("xsel"):
                result = subprocess.run(
                    ["xsel", "--primary", "--output"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    text_out = result.stdout.strip()
                    if text_out:
                        return text_out
            return None
        except Exception:
            return None
