#!/usr/bin/env python3
"""Test daemon tray icon components."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_daemon_components():
    """Test daemon tray components can be created."""
    # Force GTK backend
    os.environ["PYSTRAY_BACKEND"] = "gtk"

    try:
        import pystray

        from whisper_flow.daemon import WhisperFlowDaemon

        # Test daemon creation
        daemon = WhisperFlowDaemon()
        print("✓ Daemon instance created")

        # Test component creation
        icon_image = daemon.create_tray_icon()
        print("✓ Tray icon image created")

        menu = daemon.setup_tray_menu()
        print("✓ Tray menu created")

        # Test pystray icon creation
        icon = pystray.Icon("test-daemon", icon_image, "Test Daemon", menu)
        print("✓ Pystray icon created")

        print("✓ All daemon tray components working")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_daemon_components()
    sys.exit(0 if success else 1)
