#!/usr/bin/env python3
"""
Greenhouse Capture - Raspberry Pi 4 + OAK-D Lite (DepthAI v3), single file.

Pipeline:
  - RGB preview  → live fullscreen view + photo/record captures.
  - Mono L+R     → StereoDepth → SpatialDetectionNetwork (when model configured).
  - Detections rendered with class, confidence, and X/Y/Z in metres.
  - Photos and recordings capture the annotated frame (what you see = what you save).

Run:
    python3 main.py                 # fullscreen, uses config.json
    python3 main.py --windowed      # windowed (testing over VNC/HDMI)
    python3 main.py --note "row 5"  # override the session note

Touch bar: PHOTO | REC/STOP | TIME-LAPSE | EXIT.   ('q' also quits.)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import depthai as dai
import numpy as np

WINDOW = "capture"
HERE = os.path.dirname(os.path.abspath(__file__))
BAR_H = 96  # touch button-bar height

DEFAULTS = {
    "note": "",
    "save_root": "~/captures",
    "preview_size": [800, 480],
    "capture_fps": 30,
    "photo_jpeg_quality": 95,
    "timelapse_interval_sec": 30,
    "fullscreen": True,
    "low_storage_mb": 500,
    "model": "",        # Luxonis model zoo ID, e.g. "yolov6-nano"; empty = disabled
    "model_path": "",   # local NNArchive (.tar.xz) or blob; overrides "model"
    "depth_enabled": False,   # show colorised depth in a second window
    "depth_lower_mm": 100,    # ignore depth returns below this (mm)
    "depth_upper_mm": 5000,   # ignore depth returns above this (mm)
    "bb_scale": 0.5,          # fraction of bbox used for depth sampling
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_config(path):
    cfg = dict(DEFAULTS)
    try:
        with open(path) as fh:
            cfg.update({k: v for k, v in json.load(fh).items()
                        if not k.startswith("_")})
    except FileNotFoundError:
        print(f"[main] {path} not found, using defaults")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[main] could not read {path} ({exc}); using defaults")
    return cfg


def stamp(millis=False):
    now = datetime.now()
    if millis:
        return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    return now.strftime("%Y%m%d_%H%M%S")


def free_mb(path):
    try:
        import shutil
        return shutil.disk_usage(path).free / (1024 * 1024)
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Video recorder: write annotated preview frames to .mp4 via cv2.VideoWriter.
# --------------------------------------------------------------------------- #

class Recorder:
    def __init__(self, session_dir, fps, size):
        self.session_dir = session_dir
        self.fps = int(fps)
        self.size = size  # (w, h)
        self.active = False
        self._writer = None
        self.mp4 = None
        self.started = None

    @property
    def elapsed(self):
        return time.monotonic() - self.started if self.active and self.started else 0.0

    def start(self):
        path = os.path.join(self.session_dir, f"vid_{stamp(millis=True)}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, self.size)
        if not writer.isOpened():
            return False
        self._writer = writer
        self.mp4 = path
        self.active = True
        self.started = time.monotonic()
        return True

    def write(self, frame):
        if self.active and self._writer:
            self._writer.write(frame)

    def stop(self):
        if not self.active:
            return None
        self.active = False
        if self._writer:
            self._writer.release()
            self._writer = None
        return self.mp4 if self.mp4 and os.path.exists(self.mp4) else None


# --------------------------------------------------------------------------- #
# Touch UI
# --------------------------------------------------------------------------- #

_W, _BK, _RED, _GRN, _GRY, _AMB = ((255, 255, 255), (0, 0, 0), (40, 40, 220),
                                    (70, 180, 70), (60, 60, 60), (40, 170, 230))


class UI:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.pending = []
        self._toast = ""
        self._toast_until = self._flash_until = 0.0
        keys = [("photo", "PHOTO"), ("record", "REC"),
                ("timelapse", "TIME-LAPSE"), ("exit", "EXIT")]
        bw = w // len(keys)
        self.buttons = []
        for i, (k, lbl) in enumerate(keys):
            x1 = i * bw
            x2 = w if i == len(keys) - 1 else (i + 1) * bw
            self.buttons.append((k, lbl, x1, x2))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and y >= self.h - BAR_H:
            for k, _, x1, x2 in self.buttons:
                if x1 <= x <= x2:
                    self.pending.append(k)
                    return

    def take_events(self):
        ev, self.pending = self.pending, []
        return ev

    def flash(self):
        self._flash_until = time.monotonic() + 0.12

    def toast(self, msg, secs=2.0):
        self._toast, self._toast_until = msg, time.monotonic() + secs

    def draw(self, frame, st):
        now = time.monotonic()
        if now < self._flash_until:
            white = frame.copy()
            white[:] = 255
            cv2.addWeighted(frame, 0.4, white, 0.6, 0, frame)

        if st["recording"]:
            if int(now * 2) % 2 == 0:
                cv2.circle(frame, (24, 28), 12, _RED, -1)
            s = int(st["rec_elapsed"])
            _text(frame, f"REC {s // 60:02d}:{s % 60:02d}", (44, 36), _RED, 0.8)
        cnt = f"Photos:{st['photos']}  Clips:{st['clips']}"
        (tw, _), _ = cv2.getTextSize(cnt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        _text(frame, cnt, (self.w - tw - 12, 30), _W, 0.6)
        if st["timelapse"]:
            _text(frame, "TIMELAPSE ON", (self.w // 2 - 90, 30), _AMB, 0.7)
        if st["low_storage"]:
            f = st["free_mb"]
            msg = "LOW STORAGE" + (f" ({int(f)} MB)" if f is not None else "")
            _text(frame, msg, (12, self.h - BAR_H - 16), _RED, 0.8)

        for k, lbl, x1, x2 in self.buttons:
            on = (k == "record" and st["recording"]) or (k == "timelapse" and st["timelapse"])
            fill = _RED if (k == "record" and st["recording"]) else (_GRN if on else _GRY)
            cv2.rectangle(frame, (x1, self.h - BAR_H), (x2 - 2, self.h), fill, -1)
            cv2.rectangle(frame, (x1, self.h - BAR_H), (x2 - 2, self.h), _BK, 2)
            label = "STOP" if k == "record" and st["recording"] else lbl
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            _text(frame, label, (x1 + (x2 - x1 - tw) // 2,
                                 self.h - BAR_H + (BAR_H + th) // 2), _W, 0.8)

        if now < self._toast_until and self._toast:
            (tw, th), _ = cv2.getTextSize(self._toast, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x, y = (self.w - tw) // 2, self.h - BAR_H - 40
            cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), _BK, -1)
            _text(frame, self._toast, (x, y), _W, 0.8)


def _text(frame, s, org, color, scale):
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BK, 4, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def info_screen(w, h, lines):
    frame = np.zeros((h, w, 3), dtype="uint8")
    y = h // 2 - len(lines) * 16
    for ln in lines:
        (tw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, ln, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, _W, 2, cv2.LINE_AA)
        y += 40
    return frame


# --------------------------------------------------------------------------- #
# Detection overlay
# --------------------------------------------------------------------------- #

def draw_detections(frame, detections, labels, w, h):
    for det in detections:
        x1 = int(det.xmin * w)
        y1 = int(det.ymin * h)
        x2 = int(det.xmax * w)
        y2 = min(int(det.ymax * h), h - BAR_H - 2)
        if x1 >= x2 or y1 >= y2:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), _GRN, 2)

        lbl = labels[det.label] if det.label < len(labels) else str(det.label)
        conf_text = f"{lbl} {det.confidence:.0%}"

        try:
            sc = det.spatialCoordinates
            x_m = sc.x / 1000.0
            y_m = sc.y / 1000.0
            z_m = sc.z / 1000.0
            coord_text = f"X:{x_m:+.2f} Y:{y_m:+.2f} Z:{z_m:.2f}m"
        except AttributeError:
            coord_text = None

        ty = max(y1 - 6, 14)
        _text(frame, conf_text, (x1, ty), _GRN, 0.5)
        if coord_text:
            _text(frame, coord_text, (x1, max(ty - 18, 14)), _AMB, 0.45)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def build_pipeline(pipeline, cfg):
    """Wire preview + optional SpatialDetectionNetwork (stereo + model).
    Returns (preview_q, spatial_q, label_map, depth_q)."""
    fps = int(cfg["capture_fps"])
    pw, ph = cfg["preview_size"]

    cam = pipeline.create(dai.node.Camera).build()
    preview_q = cam.requestOutput((pw, ph), fps=fps).createOutputQueue()
    print(f"[camera] preview OK ({pw}x{ph}@{fps})")

    spatial_q = depth_q = None
    label_map = []

    model_path = cfg.get("model_path", "").strip()
    model_name = cfg.get("model", "").strip()

    if model_path or model_name:
        try:
            # Stereo depth (required for spatial detection)
            mono_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth)
            mono_left.requestFullResolutionOutput().link(stereo.left)
            mono_right.requestFullResolutionOutput().link(stereo.right)
            stereo.setRectification(True)
            stereo.setLeftRightCheck(True)
            try:
                stereo.setDefaultProfilePreset(
                    dai.node.StereoDepth.PresetType.HIGH_DENSITY)
            except Exception:
                pass

            if cfg.get("depth_enabled", False):
                depth_q = stereo.disparity.createOutputQueue(maxSize=4, blocking=False)

            # Model descriptor
            if model_path:
                path = os.path.expanduser(model_path)
                if not os.path.isabs(path):
                    path = os.path.join(HERE, path)
                model_desc = dai.NNArchive(path)
                tag = os.path.basename(path)
            else:
                model_desc = dai.NNModelDescription(model_name)
                tag = model_name

            # Spatial detection network
            spatial_net = (pipeline.create(dai.node.SpatialDetectionNetwork)
                           .build(cam, model_desc))
            stereo.depth.link(spatial_net.inputDepth)

            try:
                spatial_net.setBoundingBoxScaleFactor(float(cfg.get("bb_scale", 0.5)))
                spatial_net.setDepthLowerThreshold(int(cfg.get("depth_lower_mm", 100)))
                spatial_net.setDepthUpperThreshold(int(cfg.get("depth_upper_mm", 5000)))
            except Exception:
                pass

            spatial_q = spatial_net.out.createOutputQueue(maxSize=4, blocking=False)
            label_map = spatial_net.getClasses()
            print(f"[camera] spatial detection OK ({tag}, {len(label_map)} classes)")

        except Exception as exc:
            print(f"[camera] spatial detection unavailable: {exc}")

    return preview_q, spatial_q, label_map, depth_q


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Greenhouse capture app")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--note")
    ap.add_argument("--windowed", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.note is not None:
        cfg["note"] = args.note
    if args.windowed:
        cfg["fullscreen"] = False

    w, h = cfg["preview_size"]
    jpeg_q = int(cfg["photo_jpeg_quality"])
    low_mb = int(cfg["low_storage_mb"])
    tl_interval = max(1, int(cfg["timelapse_interval_sec"]))

    session_dir = os.path.join(os.path.expanduser(cfg["save_root"]), stamp())
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, "session_meta.json"), "w") as fh:
        json.dump({"started": datetime.now().isoformat(timespec="seconds"),
                   "note": cfg["note"], "settings": cfg}, fh, indent=2)
    print(f"[main] saving to {session_dir}")

    ui = UI(w, h)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if cfg["fullscreen"]:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW, ui.on_mouse)

    if cfg.get("depth_enabled", False):
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    _depth_cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    _depth_cmap[0] = [0, 0, 0]  # zero-disparity pixels stay black

    rec = Recorder(session_dir, cfg["capture_fps"], (w, h))
    photos = clips = 0
    timelapse = False
    next_tl = next_disk = 0.0
    low_storage, fmb = False, None
    current_detections = []
    quit_app = False

    while not quit_app:
        try:
            with dai.Pipeline() as pipeline:
                pq, sq, labels, dq = build_pipeline(pipeline, cfg)
                pipeline.start()
                print("[camera] pipeline started")
                ui.toast("Camera ready", 2.0)

                while pipeline.isRunning() and not quit_app:
                    frame = pq.get().getCvFrame()

                    # Pull latest spatial detections (non-blocking; keep last if none)
                    if sq is not None:
                        msg = sq.tryGet()
                        if msg is not None:
                            current_detections = msg.detections

                    # Optional depth colormap window
                    if dq is not None:
                        disp_msg = dq.tryGet()
                        if disp_msg is not None:
                            nd = disp_msg.getFrame()
                            max_d = max(1, int(nd.max()))
                            colored = cv2.applyColorMap(
                                ((nd / max_d) * 255).astype(np.uint8), _depth_cmap)
                            cv2.imshow("depth", colored)

                    # Draw detections (bounding box + label + X/Y/Z)
                    draw_detections(frame, current_detections, labels, w, h)

                    # Record annotated frame (no UI bar) — matches live view
                    if rec.active:
                        rec.write(frame)

                    now = time.monotonic()
                    if now >= next_disk:
                        next_disk = now + 5.0
                        fmb = free_mb(session_dir)
                        low_storage = fmb is not None and fmb < low_mb

                    if timelapse and now >= next_tl:
                        next_tl = now + tl_interval
                        if not low_storage:
                            path = os.path.join(session_dir,
                                                f"photo_{stamp(millis=True)}.jpg")
                            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
                            photos += 1

                    for key in ui.take_events():
                        if key == "photo":
                            if low_storage:
                                ui.toast("LOW STORAGE - not capturing")
                            else:
                                path = os.path.join(session_dir,
                                                    f"photo_{stamp(millis=True)}.jpg")
                                if cv2.imwrite(path, frame,
                                               [cv2.IMWRITE_JPEG_QUALITY, jpeg_q]):
                                    photos += 1
                                    ui.flash()
                                    ui.toast(f"Photo saved ({w}x{h})")
                                else:
                                    ui.toast("PHOTO SAVE FAILED")
                        elif key == "record":
                            if rec.active:
                                ui.toast("Saving clip...")
                                mp4 = rec.stop()
                                clips += 1 if mp4 else 0
                                ui.toast("Clip saved" if mp4 else "Save failed")
                            elif low_storage:
                                ui.toast("LOW STORAGE - not recording")
                            elif rec.start():
                                ui.toast("Recording")
                            else:
                                ui.toast("Could not start recording")
                        elif key == "timelapse":
                            timelapse = not timelapse
                            next_tl = now
                            ui.toast(f"Timelapse every {tl_interval}s"
                                     if timelapse else "Timelapse off")
                        elif key == "exit":
                            quit_app = True

                    ui.draw(frame, {"recording": rec.active, "rec_elapsed": rec.elapsed,
                                    "timelapse": timelapse, "photos": photos,
                                    "clips": clips, "low_storage": low_storage,
                                    "free_mb": fmb})
                    cv2.imshow(WINDOW, frame)
                    if cv2.waitKey(1) == ord("q"):
                        quit_app = True

                pipeline.stop()

        except Exception as exc:
            print(f"[main] camera error: {exc}")
            if rec.active:
                mp4 = rec.stop()
                clips += 1 if mp4 else 0
                ui.toast("Saved clip (camera lost)")
            msg = str(exc)[:60]
            cv2.imshow(WINDOW, info_screen(w, h, ["CAMERA ERROR", msg,
                                                   "Retrying... (q to quit)"]))
            if cv2.waitKey(2000) == ord("q"):
                quit_app = True

    print("[main] shutting down")
    if rec.active:
        rec.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] interrupted")
        sys.exit(0)
