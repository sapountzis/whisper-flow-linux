#!/bin/bash

# WhisperFlow One-Click Setup Script
# This script installs WhisperFlow with system tray support

set -e  # Exit on any error

echo "🎤 WhisperFlow Setup"
echo "===================="

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ This script only supports Linux"
    exit 1
fi

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    echo "❌ Please don't run this script as root"
    exit 1
fi

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 is required but not found"
    echo "Please install Python 3.12 first:"
    echo "  sudo apt update && sudo apt install python3.12 python3.12-venv"
    exit 1
fi

echo "✅ Python 3.12 found"

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt update && sudo apt install -y \
    python3-gi \
    gir1.2-gtk-3.0 \
    gir1.2-appindicator3-0.1 \
    libappindicator3-1 \
    python3-venv \
    python3-pip \
    xdotool \
    xclip \
    libnotify-bin

echo "✅ System dependencies installed"

# Check if we're in the whisper-flow directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "src/whisper_flow/__init__.py" ]]; then
    echo "❌ Please run this script from the whisper-flow directory"
    exit 1
fi

# Create virtual environment with system site packages
echo "🐍 Creating virtual environment..."
python3.12 -m venv .venv --system-site-packages

# Activate environment
echo "🔧 Activating environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -e .

echo "✅ Dependencies installed"

# Test gi module
echo "🧪 Testing system tray support..."
if python -c "import gi; print('✅ gi module available')" 2>/dev/null; then
    echo "✅ System tray support confirmed"
else
    echo "❌ gi module not available - tray icon may not work"
fi

# Test tray functionality
echo "🧪 Testing tray icon..."
if timeout 5s python tests/test_tray.py >/dev/null 2>&1; then
    echo "✅ Tray icon test passed"
else
    echo "⚠️  Tray icon test failed - you may need to restart your desktop environment"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo ""
echo "2. Start the daemon:"
echo "   source .venv/bin/activate"
echo "   whisper-flow daemon --foreground"
echo ""
echo "3. Look for the microphone icon in your system tray"
echo "   Right-click it to access the menu!"
echo ""
echo "For help: whisper-flow --help" 