#!/usr/bin/env python3
"""
Greenhouse Capture - Raspberry Pi 4 + OAK-D Lite (DepthAI v3), single file.

Built up from the official Luxonis camera example that we confirmed works on
this device:
    https://docs.luxonis.com/software-v3/depthai/examples/camera/camera_output

STEP 1 (this file): live preview at the configured size + a touch UI with PHOTO
and EXIT. PHOTO saves the current preview frame as a JPEG into a per-launch
session folder. One camera output only - same shape as the working example.

Later steps add a full-resolution still stream, then video recording + timelapse.

Run:
    python3 main.py                 # fullscreen, uses config.json
    python3 main.py --windowed      # windowed (for testing over VNC/HDMI)
    python3 main.py --note "row 5"  # override the session note

('q' on a keyboard also quits.)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import depthai as dai

WINDOW = "capture"
HERE = os.path.dirname(os.path.abspath(__file__))
BAR_H = 96  # touch button-bar height

DEFAULTS = {
    "note": "",
    "save_root": "~/captures",
    "preview_size": [800, 480],
    "preview_fps": 25,
    "photo_jpeg_quality": 95,
    "fullscreen": True,
}


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


# --------------------------------------------------------------------------- #
# Touch UI: a bottom button bar + transient toast, drawn onto the frame.
# --------------------------------------------------------------------------- #

_W, _BK, _GRY = (255, 255, 255), (0, 0, 0), (60, 60, 60)


class UI:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.pending = []
        self._toast = ""
        self._toast_until = self._flash_until = 0.0
        keys = [("photo", "PHOTO"), ("exit", "EXIT")]
        bw = w // len(keys)
        self.buttons = []  # (key, label, x1, x2)
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

    def draw(self, frame, photos):
        now = time.monotonic()
        if now < self._flash_until:  # capture flash
            white = frame.copy()
            white[:] = 255
            cv2.addWeighted(frame, 0.4, white, 0.6, 0, frame)

        cnt = f"Photos:{photos}"
        (tw, _), _ = cv2.getTextSize(cnt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        _text(frame, cnt, (self.w - tw - 12, 30), _W, 0.6)

        for k, lbl, x1, x2 in self.buttons:
            cv2.rectangle(frame, (x1, self.h - BAR_H), (x2 - 2, self.h), _GRY, -1)
            cv2.rectangle(frame, (x1, self.h - BAR_H), (x2 - 2, self.h), _BK, 2)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            _text(frame, lbl, (x1 + (x2 - x1 - tw) // 2,
                               self.h - BAR_H + (BAR_H + th) // 2), _W, 0.8)

        if now < self._toast_until and self._toast:
            (tw, th), _ = cv2.getTextSize(self._toast, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x, y = (self.w - tw) // 2, self.h - BAR_H - 40
            cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), _BK, -1)
            _text(frame, self._toast, (x, y), _W, 0.8)


def _text(frame, s, org, color, scale):
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BK, 4, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


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

    photos = 0

    # --- pipeline: exactly the working example, just at the configured size ---
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        previewQ = cam.requestOutput((w, h)).createOutputQueue()
        pipeline.start()
        print("[camera] pipeline started")
        ui.toast("Camera ready", 2.0)

        while pipeline.isRunning():
            videoIn = previewQ.get()
            assert isinstance(videoIn, dai.ImgFrame)
            frame = videoIn.getCvFrame()

            for key in ui.take_events():
                if key == "photo":
                    path = os.path.join(session_dir, f"photo_{stamp(millis=True)}.jpg")
                    if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q]):
                        photos += 1
                        ui.flash()
                        ui.toast(f"Photo saved ({frame.shape[1]}x{frame.shape[0]})")
                    else:
                        ui.toast("PHOTO SAVE FAILED")
                elif key == "exit":
                    pipeline.stop()

            ui.draw(frame, photos)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) == ord("q"):
                break

    print("[main] shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] interrupted")
        sys.exit(0)
