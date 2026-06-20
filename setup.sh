#!/bin/bash
# One-shot setup for the spatial detection touchscreen app on Raspberry Pi OS 64-bit.
# Idempotent: safe to re-run. Installs system deps, a udev rule for the OAK, a venv,
# and the Python requirements.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Installing system packages (tkinter, Pillow-ImageTk, venv, ffmpeg)..."
sudo apt update
# python3-pil.imagetk provides PIL.ImageTk, which is split out of the base
# Pillow on Debian/Raspberry Pi OS and is needed to blit frames onto the canvas.
sudo apt install -y python3-venv python3-pip python3-tk python3-pil.imagetk ffmpeg

echo "==> Installing udev rule so DepthAI can reach the OAK over USB without sudo..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules >/dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "==> Setting up Python virtual environment..."
if [ ! -d venv ]; then
    python3 -m venv venv
    echo "    Created venv."
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Installing Python dependencies..."
pip install --upgrade pip
# Ensure the full (GUI-capable) OpenCV build wins over depthai's headless pull-in.
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip install -r requirements.txt

echo ""
echo "Done. Run the app with:"
echo "  source venv/bin/activate && python3 app.py"
echo ""
echo "Test the GUI without an OAK attached:"
echo "  source venv/bin/activate && python3 app.py --mock --windowed"
