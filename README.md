# Greenhouse Capture

A touch-only photo, video, and AI inference app for a **Raspberry Pi 4** with a
**Luxonis OAK-D Lite** camera and the **official 5-inch DSI touchscreen**, built for
battery-powered use in a greenhouse.

## Features

- **Live fullscreen RGB preview.**
- **Photo** — captures a full-resolution still from the 13 MP sensor, saved as JPEG.
- **Record** — toggles H.264/H.265 video via the OAK's hardware `VideoEncoder`,
  remuxed to `.mp4`, with a blinking **REC** indicator and elapsed time.
- **Time-lapse** — captures a still every *N* seconds for dataset building.
- **Model inference** — runs any Luxonis NNArchive (`.tar.xz`) or model zoo model on
  the VPU; bounding boxes and class labels overlaid on the preview in real time.
  Drop models into `models/` and point `model_path` in `config.json`.
- **Stereo depth** — optional colorized disparity map from the OAK-D Lite's left/right
  mono cameras, displayed in a second window.
- **Robustness** — camera-disconnect recovery, recording flush on loss/exit,
  low-storage warning, and optional fullscreen-on-boot.

## Hardware

- Raspberry Pi 4 (Raspberry Pi OS **64-bit**)
- Luxonis OAK-D Lite (connected by USB; use a USB3 port and a good cable)
- Official Raspberry Pi 7" or 5" DSI touchscreen (default config assumes **800×480**)

## Install (Raspberry Pi OS 64-bit)

```bash
# 1. System packages
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg

# 2. udev rules so DepthAI can access the OAK over USB without sudo
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# 3. Project + Python deps
cd ~/rig-code            # wherever you copied this project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> `ffmpeg` is a **system** package (not pip). It is used to wrap the encoded
> H.264/H.265 bitstream into an `.mp4` container. If it is missing, recordings are
> still safe — the raw `.h264`/`.h265` stream is kept and can be muxed later.

## Run

```bash
source venv/bin/activate
python3 main.py                 # fullscreen, uses config.json
python3 main.py --windowed      # windowed (useful when testing over VNC/HDMI)
python3 main.py --note "row 5, morning"   # override the session note
```

Touch controls along the bottom bar: **PHOTO · REC/STOP · TIME-LAPSE · EXIT**.
(An attached keyboard's **q** key also quits — handy during setup.)

## Configuration

Edit `config.json` (all fields optional; defaults are used if absent):

| Field | Meaning | Default |
|---|---|---|
| `note` | Free-text note copied into every `session_meta.json` | `""` |
| `save_root` | Where session folders are created | `~/captures` |
| `preview_size` | Live preview / window size `[w, h]` | `[800, 480]` |
| `video_size` | Recording resolution `[w, h]` | `[1920, 1080]` |
| `photo_size` | Sensor mode / photo resolution `[w, h]` | `[3840, 2160]` |
| `capture_fps` | Frame rate shared by all sensor outputs | `30` |
| `codec` | `h265` (smaller) or `h264` (wider compatibility) | `h265` |
| `photo_jpeg_quality` | JPEG quality 1–100 | `95` |
| `timelapse_interval_sec` | Seconds between time-lapse stills | `30` |
| `fullscreen` | Start fullscreen | `true` |
| `low_storage_mb` | Warn / block captures below this free space (MB) | `500` |
| `model_path` | Path to a local NNArchive (`.tar.xz`) or blob, relative to project root | `""` |
| `model` | Luxonis model zoo ID (used only if `model_path` is empty) | `""` |
| `depth_enabled` | Show colorized stereo depth in a second window | `false` |

## Models

Drop NNArchive files (`.tar.xz`) into the `models/` folder and set `model_path` in
`config.json`:

```json
"model_path": "models/my_model.rvc2.tar.xz"
```

To use a Luxonis model zoo model instead, leave `model_path` empty and set `model`:

```json
"model_path": "",
"model": "yolov6-nano"
```

## Where files are saved

One timestamped folder per launch under `save_root`:

```
~/captures/20260618_142530/
├── session_meta.json                 # date, settings, note
├── photo_20260618_142601_842.jpg     # full-resolution stills (13 MP)
├── vid_20260618_142810_004.mp4       # recorded clips
└── ...
```

If a recording's mux fails (e.g. `ffmpeg` missing) the raw `.h264`/`.h265`
elementary stream is kept instead. Convert it later:

```bash
ffmpeg -framerate 30 -i vid_XXXX.h265 -c copy vid_XXXX.mp4
```

## Start on boot (optional)

```ini
# ~/.config/systemd/user/capture.service
[Unit]
Description=Greenhouse Capture
After=graphical-session.target

[Service]
WorkingDirectory=%h/rig-code
ExecStart=%h/rig-code/venv/bin/python %h/rig-code/main.py
Restart=on-failure
Environment=DISPLAY=:0

[Install]
WantedBy=graphical-session.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now capture.service
```

## Project layout

| Path | Responsibility |
|---|---|
| `main.py` | App: pipeline, capture loop, recording, touch UI, reconnect |
| `config.json` | User-editable settings |
| `models/` | NNArchive / blob model files |
| `requirements.txt` | Python dependencies |

## Troubleshooting

- **Black screen / camera error on startup** — check the USB cable/port and the udev
  rule above; the app retries automatically.
- **Pipeline won't start** — the pipeline runs multiple sensor outputs simultaneously.
  If the OAK-D Lite can't sustain your settings, lower `video_size` (e.g. `[1280, 720]`)
  or `capture_fps` in `config.json`.
- **Model not loading** — check the path in `model_path` is relative to the project
  root, and that the file is a valid NNArchive or blob for RVC2.
- **Recordings are `.h265` not `.mp4`** — install `ffmpeg` (`sudo apt install ffmpeg`).
- **Preview too big/small** — set `preview_size` to your panel's resolution.
- **Touch not registering** — confirm the DSI touchscreen works in the desktop;
  taps are delivered as mouse events.
