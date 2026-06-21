# Spatial Detector

A **touchscreen GUI** for running **object-detection models with stereo spatial
output** (X / Y / Z per object) on a **Raspberry Pi** with a **Luxonis OAK-D**
camera. The interface is built with **tkinter** and designed for the Pi's DSI
touchscreen — large touch buttons, no keyboard needed.

Pick a detection model from an on-screen list and the DepthAI pipeline rebuilds
on the fly, so you can compare models live. The OAK does all inference and depth
on-device; the Pi just draws the preview and overlays.

## Features

- **Live RGB preview** in a fullscreen tkinter canvas.
- **Spatial object detection** — bounding box, class, confidence, and **X/Y/Z in
  metres** for every detection, via the OAK's `SpatialDetectionNetwork`
  (stereo depth + neural net).
- **Live model switching** — tap **MODEL** to choose any NNArchive in `models/`
  or a Luxonis model-zoo ID; the pipeline rebuilds without restarting the app.
- **Capture suite** — **PHOTO** (annotated JPEG), **REC** (annotated `.mp4`),
  and **TIME-LAPSE** (a still every *N* seconds). Everything saves to a
  timestamped session folder.
- **Robust** — camera-disconnect recovery (it rebuilds and reconnects), a
  low-storage warning, and a mock mode for testing the GUI without hardware.

## Hardware

- Raspberry Pi 4 (Raspberry Pi OS **64-bit**)
- Luxonis OAK-D / OAK-D Lite (USB3 port + a good cable)
- Official Raspberry Pi DSI touchscreen (default config assumes **800×480**)

## Install

```bash
cd ~/rig-code
./setup.sh
```

`setup.sh` installs the system packages (`python3-tk`, `python3-venv`, `ffmpeg`),
adds a udev rule so DepthAI can reach the OAK over USB without `sudo`, creates a
virtual environment, and installs the Python dependencies.

> `python3-tk` provides tkinter and is an **apt** package, not a pip package —
> that is why setup is done via the script rather than `pip install` alone.

## Run

```bash
source venv/bin/activate
python3 app.py                 # fullscreen on the touchscreen
python3 app.py --windowed      # windowed (useful over HDMI/VNC)
python3 app.py --mock          # synthetic frames, no OAK needed (GUI test)
```

Touch bar along the bottom: **MODEL · PHOTO · REC/STOP · TIME-LAPSE · EXIT**.
(`Esc` or `q` also quit — handy during setup.)

## Models

Drop NNArchive files (`*.tar.xz`, RVC2) into the `models/` folder. They appear
automatically in the **MODEL** picker. To also offer Luxonis model-zoo models,
list their IDs under `zoo_models` in `config.json`:

```json
"zoo_models": ["yolov6-nano", "mobilenet-ssd"]
```

Set `default_model` to the display name (a filename or a zoo ID) you want loaded
at startup; leave it empty to use the first discovered model.

A sample model, `models/exp.rvc2.tar.xz`, is included.

## Configuration

Edit `config.json` (all fields optional; defaults used if absent):

| Field | Meaning | Default |
|---|---|---|
| `screen_size` | Touchscreen panel resolution `[w, h]` | `[800, 480]` |
| `fullscreen` | Start fullscreen with the cursor hidden | `true` |
| `capture_fps` | Frame rate shared by all OAK outputs | `30` |
| `default_model` | Model display name to load at startup (empty = first) | `""` |
| `zoo_models` | Extra Luxonis model-zoo IDs to offer in the picker | yolo/ssd |
| `bb_scale` | Fraction of each bbox sampled for depth (0.5 = inner 50%) | `0.5` |
| `depth_lower_mm` / `depth_upper_mm` | Clamp spatial depth range (mm) | `100` / `5000` |
| `save_root` | Where session folders are created | `~/captures` |
| `photo_jpeg_quality` | JPEG quality 1–100 | `95` |
| `timelapse_interval_sec` | Seconds between time-lapse stills | `30` |
| `low_storage_mb` | Warn / block captures below this free space (MB) | `500` |
| `note` | Free-text note copied into `session_meta.json` | `""` |

## Where files are saved

One timestamped folder per launch under `save_root`:

```
~/captures/20260620_142530/
├── session_meta.json                 # date, note, settings
├── photo_20260620_142601_842.jpg     # annotated stills
├── vid_20260620_142810_004.mp4       # recorded clips
└── ...
```

## Project layout

| Path | Responsibility |
|---|---|
| `app.py` | tkinter UI: fullscreen window, video canvas, touch bar, model picker, render loop |
| `camera.py` | `CameraWorker` thread: DepthAI pipeline, frame/detection grab, capture, live model rebuild, reconnect |
| `overlay.py` | Draws detection boxes + X/Y/Z and the status HUD onto frames |
| `recorder.py` | `Recorder` — writes annotated frames to `.mp4` |
| `models.py` | Discovers local NNArchives + zoo IDs for the picker |
| `config.py` | Defaults + `config.json` loading |
| `setup.sh` | One-shot installer |
| `models/` | NNArchive model files |

## Start on boot (optional)

```ini
# ~/.config/systemd/user/spatial-detector.service
[Unit]
Description=Spatial Detector
After=graphical-session.target

[Service]
WorkingDirectory=%h/rig-code
ExecStart=%h/rig-code/venv/bin/python %h/rig-code/app.py
Restart=on-failure
Environment=DISPLAY=:0

[Install]
WantedBy=graphical-session.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now spatial-detector.service
```

## Troubleshooting

- **Black screen / camera error on startup** — check the USB3 cable/port and the
  udev rule; the worker retries and reconnects automatically.
- **Works sometimes, fails other times (intermittent)** — the OAK-D negotiates
  USB3 (SuperSpeed) or USB2 (HighSpeed) at each plug-in, and a flaky cable/port
  can land on USB2, where three camera streams overrun the bus and stall. The
  worker now reads the negotiated link speed on connect and, on USB2,
  automatically reduces FPS and disables extended disparity so it still runs
  (watch the console for `USB link: ... (USB2) -> reduced pipeline`). For full
  frame rate, use a USB3 (blue) port and a data-capable USB3 cable; run
  `python3 diag_camera.py` to see the negotiated speed.
- **Model won't load** — confirm the file in `models/` is a valid RVC2 NNArchive,
  or that the zoo ID is correct and the Pi has internet to download it.
- **Pipeline won't start** — lower `capture_fps` or `screen_size` in `config.json`
  if the OAK can't sustain your settings.
- **No GUI / `ModuleNotFoundError: tkinter`** — run `sudo apt install python3-tk`
  (it is a system package, re-run `./setup.sh`).
- **OpenCV import or window errors** — depthai pulls in the headless OpenCV build;
  `setup.sh` reinstalls `opencv-python` last so the full build wins. If a GTK error
  persists, `sudo apt install -y libgtk-3-dev` and re-run setup.
- **Touch not registering** — confirm the DSI touchscreen works on the desktop;
  taps are delivered as mouse clicks to the tkinter buttons.
