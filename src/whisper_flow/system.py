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

    def get_highlighted_text(self) -> str | None:
        """Get currently highlighted/selected text.

        Returns:
            Highlighted text or None if unable to get

        """
        try:
            # Method 1: Use xclip to get primary selection (highlighted text)
            if shutil.which("xclip"):
                result = subprocess.run(
                    ["xclip", "-selection", "primary", "-o"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

            # Method 2: Use xsel as fallback
            if shutil.which("xsel"):
                result = subprocess.run(
                    ["xsel", "--primary", "--output"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

            return None
        except Exception:
            return None
