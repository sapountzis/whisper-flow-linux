"""Background daemon for whisper-flow with system tray and global hotkeys."""

import os
import sys
import threading
from pathlib import Path

import pystray
from PIL import Image, ImageDraw
from pynput import keyboard

from .app import WhisperFlow
from .config import Config


class WhisperFlowDaemon:
    """Background daemon with system tray icon and global hotkey support."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize the daemon."""
        self.config = Config(config_dir=config_dir) if config_dir else Config()
        self.tray_icon = None
        self.is_running = False
        self.is_recording = False
        self.current_mode = None
        self.recording_thread = None
        self.auto_stop_timer = None
        self.hotkey_listener = None
        self.keyboard_listener = None
        self.stop_recording_event = None
        self.pressed_keys = set()

        # Initialize WhisperFlow instances for different modes
        self.transcribe_app = WhisperFlow(config_dir, "transcribe")
        self.auto_transcribe_app = WhisperFlow(config_dir, "auto_transcribe")
        self.command_app = WhisperFlow(config_dir, "command")

    def daemonize(self):
        """Run as background process while preserving desktop session access."""
        # Simple approach: just fork once and redirect output
        # This preserves all environment and session access

        try:
            pid = os.fork()
            if pid > 0:
                # Parent process exits
                sys.exit(0)
        except OSError:
            sys.exit(1)

        # We're now in the child process
        # Don't change session, directory, or umask - preserve everything

        # Only redirect output to avoid cluttering terminal
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect to /dev/null but keep stderr for debugging if needed
        with open("/dev/null") as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open("/dev/null", "a+") as f:
            os.dup2(f.fileno(), sys.stdout.fileno())

        # Write PID file
        pid_file = Path.home() / ".config" / "whisper-flow" / "daemon.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

    def _convert_hotkey_string(self, hotkey_str: str) -> str:
        """Convert hotkey string to pynput format.

        Args:
            hotkey_str: Hotkey string like 'ctrl+cmd+space'

        Returns:
            Pynput formatted hotkey string

        """
        # Convert common key names to pynput format
        key_mapping = {
            "ctrl": "<ctrl>",
            "alt": "<alt>",
            "shift": "<shift>",
            "cmd": "<cmd>",  # Command/Super/Windows key
            "opt": "<cmd>",  # Option key (same as cmd on most systems)
            "super": "<cmd>",  # Super key (alternative name)
            "space": "<space>",
            "escape": "<esc>",
            "enter": "<enter>",
            "tab": "<tab>",
        }

        parts = hotkey_str.lower().split("+")
        converted_parts = []

        for part in parts:
            part = part.strip()
            if part in key_mapping:
                converted_parts.append(key_mapping[part])
            else:
                # Single character keys don't need brackets
                converted_parts.append(part)

        return "+".join(converted_parts)

    def create_tray_icon(self) -> Image.Image:
        """Create the system tray icon."""
        # Create a simple microphone icon
        size = 64
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Draw microphone shape
        # Mic body
        center_x, center_y = size // 2, size // 2
        mic_width, mic_height = 20, 30
        left = center_x - mic_width // 2
        right = center_x + mic_width // 2
        top = center_y - mic_height // 2
        bottom = center_y + mic_height // 2

        # Main mic body (rounded rectangle)
        draw.rounded_rectangle(
            [left, top, right, bottom],
            radius=8,
            fill="white",
            outline="black",
            width=2,
        )

        # Mic stand
        stand_top = bottom + 2
        stand_bottom = stand_top + 10
        draw.line([center_x, stand_top, center_x, stand_bottom], fill="black", width=3)

        # Base
        base_left = center_x - 8
        base_right = center_x + 8
        draw.line(
            [base_left, stand_bottom, base_right, stand_bottom],
            fill="black",
            width=3,
        )

        return image

    def create_recording_icon(self) -> Image.Image:
        """Create the recording state icon (red dot)."""
        size = 64
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Draw red circle
        center = size // 2
        radius = 20
        draw.ellipse(
            [center - radius, center - radius, center + radius, center + radius],
            fill="red",
            outline="darkred",
            width=2,
        )

        return image

    def setup_tray_menu(self):
        """Setup the system tray menu."""
        return pystray.Menu(
            pystray.MenuItem("WhisperFlow Daemon", None, enabled=False),
            pystray.MenuItem(
                f"Transcribe: {self.config.hotkey_transcribe}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                f"Auto-Transcribe: {self.config.hotkey_auto_transcribe}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                f"Command: {self.config.hotkey_command}",
                None,
                enabled=False,
            ),
            pystray.MenuItem("Settings", self.open_settings),
            pystray.MenuItem("Test Configuration", self.test_configuration),
            pystray.MenuItem("Exit", self.stop_daemon),
        )

    def setup_hotkeys(self):
        """Register global hotkeys using pynput with proper push-to-talk support."""
        try:
            # Setup keyboard listener for push-to-talk
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self.keyboard_listener.start()

            # Convert hotkey strings for single-press actions
            auto_transcribe_key = self._convert_hotkey_string(
                self.config.hotkey_auto_transcribe,
            )
            command_key = self._convert_hotkey_string(self.config.hotkey_command)

            # Create hotkey mapping for single-press actions
            hotkeys = {
                auto_transcribe_key: lambda: self._single_press_action(
                    "auto_transcribe",
                ),
                command_key: lambda: self._single_press_action("command"),
                "<esc>": self.cancel_recording,
                "<f1>": self.show_notification_menu,
            }

            # Create and start global hotkey listener
            self.hotkey_listener = keyboard.GlobalHotKeys(hotkeys)
            self.hotkey_listener.start()

        except Exception as e:
            self.notify(f"Error setting up hotkeys: {e}")
            self.hotkey_listener = None
            self.keyboard_listener = None

    def _parse_hotkey_combination(self, hotkey_str: str) -> set:
        """Parse hotkey string into a set of key names."""
        parts = hotkey_str.lower().split("+")
        key_set = set()

        key_mapping = {
            "ctrl": "ctrl",
            "cmd": "cmd",
            "alt": "alt",
            "shift": "shift",
            "space": "space",
        }

        for part in parts:
            part = part.strip()
            if part in key_mapping:
                key_set.add(key_mapping[part])
            else:
                key_set.add(part)

        return key_set

    def _get_key_name(self, key):
        """Get standardized key name from pynput key object."""
        try:
            if hasattr(key, "name"):
                return key.name.lower()
            if hasattr(key, "char") and key.char:
                return key.char.lower()
            return str(key).lower()
        except:
            return str(key).lower()

    def _on_key_press(self, key):
        """Handle key press events for push-to-talk detection."""
        try:
            key_name = self._get_key_name(key)

            # Map pynput key names to our standard names
            key_mapping = {
                "ctrl_l": "ctrl",
                "ctrl_r": "ctrl",
                "cmd_l": "cmd",
                "cmd_r": "cmd",
                "alt_l": "alt",
                "alt_r": "alt",
                "shift_l": "shift",
                "shift_r": "shift",
            }

            key_name = key_mapping.get(key_name, key_name)
            self.pressed_keys.add(key_name)

            # Check if transcribe hotkey combination is pressed
            transcribe_keys = self._parse_hotkey_combination(
                self.config.hotkey_transcribe,
            )
            if transcribe_keys.issubset(self.pressed_keys) and not self.is_recording:
                self.start_recording("transcribe")

        except Exception:
            pass  # Silently ignore key handling errors

    def _on_key_release(self, key):
        """Handle key release events for push-to-talk detection."""
        try:
            key_name = self._get_key_name(key)

            # Map pynput key names to our standard names
            key_mapping = {
                "ctrl_l": "ctrl",
                "ctrl_r": "ctrl",
                "cmd_l": "cmd",
                "cmd_r": "cmd",
                "alt_l": "alt",
                "alt_r": "alt",
                "shift_l": "shift",
                "shift_r": "shift",
            }

            key_name = key_mapping.get(key_name, key_name)
            self.pressed_keys.discard(key_name)

            # Check if transcribe hotkey was released and we're recording transcribe mode
            transcribe_keys = self._parse_hotkey_combination(
                self.config.hotkey_transcribe,
            )
            if self.is_recording and self.current_mode == "transcribe":
                if not transcribe_keys.issubset(self.pressed_keys):
                    self._stop_recording()

        except Exception:
            pass  # Silently ignore key handling errors

    def _single_press_action(self, mode: str):
        """Handle single-press actions (auto-transcribe, command)."""
        if not self.is_recording:
            self.start_recording(mode)

    def start_recording(self, mode: str):
        """Start recording in the specified mode."""
        if self.is_recording:
            return

        self.is_recording = True
        self.current_mode = mode
        self.stop_recording_event = threading.Event()

        # Update tray icon to recording state
        if self.tray_icon:
            self.tray_icon.icon = self.create_recording_icon()

        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._record_audio_thread,
            args=(mode,),
            daemon=True,
        )
        self.recording_thread.start()

    def _record_audio_thread(self, mode: str):
        """Handle audio recording in a separate thread."""
        try:
            app = self._get_app_for_mode(mode)

            if mode == "auto_transcribe":
                # Auto-stop mode: record until silence
                success = app.run_voice_flow_auto_stop(
                    silence_duration=self.config.auto_stop_silence_duration,
                )
            elif mode == "transcribe":
                # Push-to-talk mode: record until stop event is set
                success = app.run_voice_flow_push_to_talk_daemon(
                    stop_key=self.config.hotkey_transcribe,
                    stop_event=self.stop_recording_event,
                )
            else:
                # Command mode: single press, auto-stop on silence
                success = app.run_voice_flow_auto_stop(
                    silence_duration=self.config.auto_stop_silence_duration,
                )

            if not success:
                self.notify("❌ Recording failed")

        except Exception as e:
            self.notify(f"❌ Recording error: {e}")
        finally:
            self._stop_recording()

    def _get_app_for_mode(self, mode: str) -> WhisperFlow:
        """Get the appropriate WhisperFlow instance for the mode."""
        if mode == "transcribe":
            return self.transcribe_app
        if mode == "auto_transcribe":
            return self.auto_transcribe_app
        if mode == "command":
            return self.command_app
        return self.transcribe_app

    def cancel_recording(self):
        """Cancel current recording."""
        if not self.is_recording:
            return

        self._stop_recording()

    def _stop_recording(self):
        """Stop the current recording."""
        if not self.is_recording:
            return

        # Signal the recording thread to stop
        if self.stop_recording_event:
            self.stop_recording_event.set()

        # Reset recording state
        self.is_recording = False
        self.current_mode = None

        # Restore normal tray icon
        if self.tray_icon:
            self.tray_icon.icon = self.create_tray_icon()

        # Cancel auto-stop timer if active
        if self.auto_stop_timer:
            self.auto_stop_timer.cancel()
            self.auto_stop_timer = None

    def notify(self, message: str):
        """Send desktop notification."""
        # Use the system manager from one of our apps
        self.transcribe_app.system_manager.notify(message)

    def open_settings(self, icon, item):
        """Open settings (placeholder for future implementation)."""
        self.notify(
            "Settings not yet implemented - edit ~/.config/whisper-flow/config.yaml",
        )

    def test_configuration(self, icon, item):
        """Test system configuration."""
        try:
            validation = self.transcribe_app.validate_configuration()
            if validation["valid"]:
                self.notify("✅ Configuration is valid!")
            else:
                issues = ", ".join(validation["issues"][:2])  # Show first 2 issues
                self.notify(f"❌ Configuration issues: {issues}")
        except Exception as e:
            self.notify(f"❌ Configuration test failed: {e}")

    def stop_daemon(self, icon=None, item=None):
        """Stop the daemon."""
        try:
            # Show exit notification
            self.notify("👋 WhisperFlow daemon stopping...")
        except Exception:
            pass  # Don't fail if notification doesn't work

        self.is_running = False

        # Stop the tray icon (this will exit the main loop)
        if self.tray_icon:
            self.tray_icon.stop()

    def show_notification_menu(self):
        """Show a notification-based menu when system tray is not available."""
        try:
            # Use notify-send to show menu options
            import subprocess

            menu_text = f"""
🎤 WhisperFlow Daemon Menu

Hotkeys:
• {self.config.hotkey_transcribe} - Transcribe (push-to-talk)
• {self.config.hotkey_auto_transcribe} - Auto-transcribe
• {self.config.hotkey_command} - Command mode
• Escape - Cancel recording

Status: {"Recording" if self.is_recording else "Idle"}
Mode: {self.current_mode if self.is_recording else "None"}

Use 'whisper-flow stop' to exit daemon
            """.strip()

            subprocess.run(
                [
                    "notify-send",
                    "--urgency=normal",
                    "--expire-time=5000",
                    "WhisperFlow Daemon",
                    menu_text,
                ],
                check=False,
            )
        except Exception:
            # Fall back to simple print if notification fails
            print("WhisperFlow menu: type 'help' for commands")

    def run_notification_mode(self):
        """Run in notification mode when tray is not available."""
        print("🎤 WhisperFlow Daemon (CLI Mode)")
        print("Hotkeys active. Press F1 for menu, or type commands:")
        print("Commands: menu, status, test, exit, help")

        try:
            while self.is_running:
                try:
                    command = input("whisper-flow> ").strip().lower()
                    if command == "exit":
                        break
                    if command == "menu":
                        self.show_notification_menu()
                    elif command == "status":
                        status = "Recording" if self.is_recording else "Ready"
                        mode = self.current_mode or "None"
                        print(f"Status: {status}, Mode: {mode}")
                    elif command == "test":
                        self.test_configuration(None, None)
                    elif command == "help":
                        print("Commands: menu, status, test, exit, help")
                    elif command == "":
                        continue
                    else:
                        print(f"Unknown command: {command}")

                except (KeyboardInterrupt, EOFError):
                    break

        except Exception as e:
            print(f"CLI mode error: {e}")
        finally:
            self.stop_daemon()

    def run(self, foreground: bool = False, _worker: bool = False):
        """Run the daemon. This method now handles both launching and running the worker."""
        if not self.config.daemon_enabled:
            if foreground:
                print("Daemon mode is disabled in configuration")
            return

        if foreground or _worker:
            self._run_worker(foreground=foreground)
        else:
            self._launch_worker()

    def _launch_worker(self):
        """Launch the daemon as a background worker process."""
        import subprocess
        import sys
        import time

        args = [sys.argv[0], "daemon", "--_worker"]
        print("Starting WhisperFlow daemon...")

        log_file_path = Path.home() / ".config" / "whisper-flow" / "daemon.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        if log_file_path.exists():
            log_file_path.unlink()

        with open(log_file_path, "w") as log_file:
            subprocess.Popen(
                args,
                stdout=log_file,
                stderr=log_file,
                stdin=subprocess.DEVNULL,
            )

        print("✓ Daemon process launched. Verifying status...")
        time.sleep(2)

        pid_file = Path.home() / ".config" / "whisper-flow" / "daemon.pid"
        if pid_file.exists():
            print("✓ Daemon is running. Tray icon should be visible.")
            log_file_path.unlink(missing_ok=True)
        else:
            log_content = log_file_path.read_text()
            print("\n❌ Daemon failed to start. See error log below:")
            print("-" * 50)
            print(
                log_content or "Log file is empty, the process may have been blocked.",
            )
            print("-" * 50)

        sys.exit(0)

    def _run_worker(self, foreground: bool = False):
        """This is the actual worker logic that runs in the background."""
        self.is_running = True

        pid_file = Path.home() / ".config" / "whisper-flow" / "daemon.pid"
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        self.setup_hotkeys()

        try:
            os.environ["PYSTRAY_BACKEND"] = self.config.pystray_backend
            if foreground:
                print(f"Starting worker with {self.config.pystray_backend} backend...")

            icon_image = self.create_tray_icon()
            menu = self.setup_tray_menu()
            self.tray_icon = pystray.Icon(
                "whisper-flow",
                icon_image,
                "WhisperFlow Daemon",
                menu,
            )

            if not foreground:
                self.notify("🎤 WhisperFlow daemon started")

            self.tray_icon.run()
        except Exception as e:
            if foreground:
                print(f"Tray error: {e}\nFalling back to CLI mode...")
                self.run_notification_mode()
        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleanup resources after running."""
        try:
            if self.hotkey_listener:
                self.hotkey_listener.stop()
                self.hotkey_listener = None
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
        except Exception:
            pass

        # Remove PID file
        try:
            pid_file = Path.home() / ".config" / "whisper-flow" / "daemon.pid"
            if pid_file.exists():
                pid_file.unlink()
        except Exception:
            pass

        self.is_running = False

    def run_headless_mode(self):
        """Run in headless mode when tray is not available in background."""
        # Just keep the daemon running with hotkeys active
        try:
            while self.is_running:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_daemon()


def main():
    """Main entry point for the daemon."""
    daemon = WhisperFlowDaemon()
    daemon.run()


def is_running() -> bool:
    """Check if the daemon is currently running."""
    pid_file = Path.home() / ".config" / "whisper-flow" / "daemon.pid"
    if not pid_file.exists():
        return False

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        # Check if process is still running
        os.kill(pid, 0)
        return True
    except (ValueError, OSError, FileNotFoundError):
        # PID file is invalid or process is dead
        try:
            pid_file.unlink()
        except Exception:
            pass
        return False


def stop_daemon():
    """Stop the running daemon."""
    pid_file = Path.home() / ".config" / "whisper-flow" / "daemon.pid"
    if not pid_file.exists():
        print("Daemon is not running")
        return

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        # Send SIGTERM to the daemon process
        os.kill(pid, 15)  # SIGTERM
        print(f"Sent stop signal to daemon (PID: {pid})")

        # Wait a moment and check if it stopped
        import time

        time.sleep(1)

        if is_running():
            print("Daemon did not stop gracefully, sending SIGKILL...")
            os.kill(pid, 9)  # SIGKILL

        # Clean up PID file
        try:
            pid_file.unlink()
        except Exception:
            pass

    except (ValueError, OSError, FileNotFoundError) as e:
        print(f"Error stopping daemon: {e}")
        # Clean up invalid PID file
        try:
            pid_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
