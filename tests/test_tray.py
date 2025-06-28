#!/usr/bin/env python3
"""Test system tray functionality."""

import os
import sys
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_test_icon(color="green"):
    """Create a simple test icon."""
    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    center = size // 2
    radius = 20
    draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        fill=color,
        outline="dark" + color,
        width=2,
    )
    return image


def on_menu_click(icon, item):
    """Handle menu clicks."""
    print(f"✓ Menu clicked: {item}")
    if str(item) == "Exit":
        icon.stop()


def test_tray_menu():
    """Test tray icon with menu."""
    # Force GTK backend
    os.environ["PYSTRAY_BACKEND"] = "gtk"

    try:
        import gi

        print("✓ gi module available")
    except ImportError:
        print("✗ gi module not available")
        return False

    try:
        icon_image = create_test_icon()
        menu = pystray.Menu(
            pystray.MenuItem("Test Item", on_menu_click),
            pystray.MenuItem("Exit", on_menu_click),
        )

        icon = pystray.Icon("test-tray", icon_image, "Test Tray", menu)
        print("✓ Tray icon created - check system tray and right-click to test")
        icon.run()
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_tray_menu()
    sys.exit(0 if success else 1)
