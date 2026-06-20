#!/bin/bash
set -e

echo "==> Installing GTK system deps..."
sudo apt install -y libgtk-3-dev libcanberra-gtk3-module

echo "==> Reinstalling opencv-python with GUI support..."
source "$(dirname "$0")/venv/bin/activate"
pip uninstall opencv-python opencv-python-headless -y 2>/dev/null || true
pip install opencv-python

echo "==> Checking GTK backend..."
if python3 -c "import cv2; info = cv2.getBuildInformation()" 2>/dev/null | grep -q "GTK+: YES\|GTK3: YES"; then
    echo "    GTK: YES - opencv GUI is working"
else
    echo "    GTK not found in pip build. Falling back to system OpenCV..."
    pip uninstall opencv-python -y 2>/dev/null || true
    sudo apt install -y python3-opencv
    deactivate
    python3 -m venv venv --system-site-packages
    source "$(dirname "$0")/venv/bin/activate"
    pip install depthai numpy
    echo "    Switched to system OpenCV."
fi

echo ""
echo "Done. Run the app with:"
echo "  source venv/bin/activate && python3 main.py"
