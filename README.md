# Greenhouse Capture

A lightweight, touch-only photo & video capture app for a **Raspberry Pi 4** with a
**Luxonis OAK-D Lite** camera and the **official 5-inch DSI touchscreen**, built for
battery-powered use in a greenhouse.

Its only job is to **reliably capture and save media** — plain RGB photos and video.
No depth maps, no object detection. It is deliberately simple and crash-resistant:
a camera disconnect reconnects instead of crashing, in-progress recordings are always
flushed to disk, and low storage is surfaced on screen.

## Features

- **Live fullscreen RGB preview.**
- **Photo** — captures a full-resolution still from the 13 MP sensor (not the
  downscaled preview), saved as JPEG.
- **Record** — toggles H.264/H.265 video via the OAK's hardware `VideoEncoder`,
  remuxed to `.mp4`, with a blinking red **REC** indicator and elapsed time.
- **Time-lapse** — captures a still every *N* seconds for dataset building.
- **One session folder per launch**, sequential timestamped filenames, a
  `session_meta.json` (date, settings, free-text note), and an on-screen capture count.
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
| `preview_fps` | Preview frame rate | `25` |
| `video_size` | Recording resolution `[w, h]` | `[1920, 1080]` |
| `video_fps` | Recording frame rate | `30` |
| `codec` | `h265` (smaller) or `h264` (wider compatibility) | `h265` |
| `photo_jpeg_quality` | JPEG quality 1–100 | `95` |
| `timelapse_interval_sec` | Seconds between time-lapse stills | `30` |
| `fullscreen` | Start fullscreen | `true` |
| `low_storage_mb` | Warn / block captures below this free space (MB) | `500` |

The note is read at launch — set it (e.g. `"greenhouse row 3, afternoon"`) before a
session, or pass `--note` on the command line.

## Where files are saved

One timestamped folder per launch under `save_root`:

```
~/captures/20260618_142530/
├── session_meta.json                 # date, settings, note, live capture counts
├── photo_20260618_142601_842.jpg     # full-resolution stills (13 MP)
├── photo_20260618_142735_119.jpg
├── vid_20260618_142810_004.mp4       # recorded clips (remuxed from the encoder)
└── ...
```

If a recording's mux fails (e.g. `ffmpeg` missing) the raw `.h264`/`.h265`
elementary stream is kept in the same folder instead of the `.mp4`. Convert it later:

```bash
ffmpeg -framerate 30 -i vid_XXXX.h265 -c copy vid_XXXX.mp4
```

## Start on boot (optional)

Create a systemd user service so the app launches fullscreen when the Pi boots into
the desktop. Example `~/.config/systemd/user/capture.service`:

```ini
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

`Restart=on-failure` adds a second safety net on top of the app's own
disconnect recovery.

## How it works (brief)

- **One DepthAI v3 pipeline** drives three outputs from the OAK-D Lite's RGB sensor:
  a downscaled BGR **preview**, a **full-resolution still** gated by an on-device
  `Script` node (so 13 MP frames only cross USB when you tap PHOTO), and a
  continuously-running **`VideoEncoder`**.
- **Recording** is just the host deciding when to write the encoder's output to disk,
  starting at the first keyframe; this makes start/stop instant and leaves a
  crash-recoverable raw stream on disk.
- The whole app is a **single-threaded OpenCV loop** with buttons drawn onto the
  frame, so there is no GUI toolkit and very little to go wrong.

## Project layout

| File | Responsibility |
|---|---|
| `main.py` | Entry point, main loop, input handling, reconnect |
| `camera.py` | DepthAI v3 pipeline: preview, stills, video encoder |
| `storage.py` | Session folders, filenames, metadata, muxing, disk checks |
| `ui.py` | Touch buttons and on-screen status overlays (OpenCV) |
| `config.json` | User-editable settings |
| `requirements.txt` | Python dependencies |

## Troubleshooting

- **Black screen / "CAMERA DISCONNECTED"** — check the USB cable/port and the udev
  rule above; the app retries automatically every couple of seconds.
- **Pipeline won't start (build fails in the console)** — the app runs three sensor
  outputs at once (preview + full-res still + encoder). If the OAK-D Lite can't
  sustain your settings, lower `video_size` (e.g. `[1280, 720]`) and/or `video_fps`
  in `config.json`.
- **Recordings are `.h265` not `.mp4`** — install `ffmpeg` (`sudo apt install ffmpeg`).
- **Preview too big/small for the screen** — set `preview_size` to your panel's
  resolution.
- **Touch not registering** — confirm the DSI touchscreen works in the desktop;
  taps are delivered to the app as mouse events.
