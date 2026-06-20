# OAK Spatial Detection

The official DepthAI v3 spatial detection example, running unmodified on a
Luxonis OAK device. It performs on-device object detection (yolov6-nano) and
reports each object's spatial X/Y/Z coordinates using stereo depth.

Source: [luxonis/depthai-core — `examples/python/SpatialDetectionNetwork/spatial_detection.py`](https://github.com/luxonis/depthai-core/blob/main/examples/python/SpatialDetectionNetwork/spatial_detection.py)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The `yolov6-nano` model is fetched from the Luxonis model zoo on first run.

## Run

```bash
python3 spatial_detection.py                        # stereo depth (default)
python3 spatial_detection.py --depthSource neural   # neural depth (OAK4)
```

Two windows open — a color frame with detection boxes + X/Y/Z labels, and a
colorized depth frame. Press `q` to quit.

## Notes

- The cameras run at 20 fps (`STEREO_DEFAULT_FPS`). Three camera streams at
  higher rates can overrun USB bandwidth and cause "communication exception"
  errors — use a USB3 cable and port.
- On Linux, install the udev rules so the device enumerates reliably:
  ```bash
  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
    | sudo tee /etc/udev/rules.d/80-movidius.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```
