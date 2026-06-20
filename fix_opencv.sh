#!/bin/bash
set -e
SCRIPT_DIR="$(dirname "$0")"

echo "==> Installing GTK system deps..."
sudo apt install -y libgtk-3-dev libcanberra-gtk3-module python3-venv

echo "==> Setting up venv..."
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    python3 -m venv "$SCRIPT_DIR/venv"
    echo "    Created venv."
fi
source "$SCRIPT_DIR/venv/bin/activate"

echo "==> Installing/reinstalling opencv-python with GUI support..."
pip uninstall opencv-python opencv-python-headless -y 2>/dev/null || true
pip install opencv-python

echo "==> Checking GTK backend..."
GTK_OK=$(python3 -c "
import cv2
info = cv2.getBuildInformation()
print('yes' if 'GTK+:                        YES' in info or 'GTK3:' in info else 'no')
")

if [ "$GTK_OK" = "yes" ]; then
    echo "    GTK: YES - installing remaining deps..."
    pip install depthai numpy
else
    echo "    GTK not found in pip build. Falling back to system OpenCV..."
    pip uninstall opencv-python -y 2>/dev/null || true
    sudo apt install -y python3-opencv
    deactivate
    python3 -m venv "$SCRIPT_DIR/venv" --system-site-packages
    source "$SCRIPT_DIR/venv/bin/activate"
    pip install depthai numpy
    echo "    Switched to system OpenCV."
fi

echo ""
echo "Done. Run the app with:"
echo "  source venv/bin/activate && python3 main.py"
