#!/usr/bin/env python3
"""
Greenhouse Capture - single-file photo/video app for a Raspberry Pi 4 with an
OAK-D Lite (DepthAI v3) and a touchscreen.

One file, single-threaded, touch-only. The whole session runs inside
`with dai.Pipeline() as pipeline:` while `pipeline.isRunning()` - exactly like
the official v3 examples and tracker.py. An outer loop re-enters that block to
reconnect if the camera drops; leaving the block always releases the device, so
a rebuild never gets stuck on "no available sensor".

Run:
    python3 main.py                 # fullscreen, uses config.json
    python3 main.py --windowed      # windowed (for testing over VNC/HDMI)
    python3 main.py --note "row 5"  # override the session note

Touch buttons (bottom bar): PHOTO | REC/STOP | TIME-LAPSE | EXIT.
('q' on a keyboard also quits.)
"""

import argparse
import json
import os
import subprocess
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
    "video_size": [1920, 1080],
    "video_fps": 30,
    "codec": "h265",
    "photo_jpeg_quality": 95,
    "timelapse_interval_sec": 30,
    "fullscreen": True,
    "low_storage_mb": 500,
}

# On-device Script: keep the latest full-res frame, send it to the host only when
# a trigger arrives (so 13 MP frames don't stream over USB continuously).
STILL_SCRIPT = """
latest = None
while True:
    f = node.inputs['in'].tryGet()
    if f is not None:
        latest = f
    trig = node.inputs['trigger'].tryGet()
    if trig is not None and latest is not None:
        node.io['still'].send(latest)
"""


# --------------------------------------------------------------------------- #
# Config + small helpers
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


def is_keyframe(frame):
    try:
        return frame.getFrameType() == dai.EncodedFrame.FrameType.I
    except Exception:
        return True  # fall back: clip may start mid-GOP but stays valid


# --------------------------------------------------------------------------- #
# Video recorder: writes the encoder bitstream to disk, then remuxes to .mp4.
# --------------------------------------------------------------------------- #

class Recorder:
    def __init__(self, session_dir, codec, fps):
        self.session_dir = session_dir
        self.ext = "h265" if codec == "h265" else "h264"
        self.fps = int(fps)
        self.active = False
        self._fh = None
        self.raw = self.mp4 = None
        self.started = None
        self._kf_seen = False

    @property
    def elapsed(self):
        return time.monotonic() - self.started if self.active and self.started else 0.0

    def start(self):
        base = os.path.join(self.session_dir, f"vid_{stamp(millis=True)}")
        self.raw, self.mp4 = f"{base}.{self.ext}", f"{base}.mp4"
        try:
            self._fh = open(self.raw, "wb")
        except OSError as exc:
            print(f"[rec] cannot open {self.raw}: {exc}")
            return False
        self.active, self.started, self._kf_seen = True, time.monotonic(), False
        return True

    def write(self, frames):
        if not self.active or self._fh is None:
            return
        try:
            for fr in frames:
                if not self._kf_seen:
                    if is_keyframe(fr):
                        self._kf_seen = True
                    else:
                        continue  # skip P-frames before the first keyframe
                fr.getData().tofile(self._fh)
            self._fh.flush()  # minimise loss on power-off
        except (OSError, ValueError) as exc:
            print(f"[rec] write error: {exc}")

    def stop(self):
        """Close + remux. Returns mp4 path on success, else None."""
        if not self.active:
            return None
        self.active = False
        if self._fh:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        if not self.raw or not os.path.exists(self.raw) or os.path.getsize(self.raw) == 0:
            return None
        cmd = ["ffmpeg", "-y", "-framerate", str(self.fps), "-i", self.raw,
               "-c", "copy", self.mp4]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[rec] ffmpeg remux failed ({exc}); keeping raw {self.raw}")
            return None
        try:
            os.remove(self.raw)
        except OSError:
            pass
        return self.mp4


# --------------------------------------------------------------------------- #
# Touch UI
# --------------------------------------------------------------------------- #

_W, _BK, _RED, _GRN, _GRY, _AMB = ((255, 255, 255), (0, 0, 0), (40, 40, 220),
                                   (70, 180, 70), (60, 60, 60), (40, 170, 230))


class UI:
    """Bottom button bar + status overlays drawn straight onto the frame."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.pending = []
        self._flash_until = self._toast_until = 0.0
        self._toast = ""
        keys = [("photo", "PHOTO"), ("record", "REC"),
                ("timelapse", "TIME-LAPSE"), ("exit", "EXIT")]
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
            if k == "record" and st["recording"]:
                lbl = "STOP"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            _text(frame, lbl, (x1 + (x2 - x1 - tw) // 2, self.h - BAR_H + (BAR_H + th) // 2), _W, 0.8)

        if now < self._toast_until and self._toast:
            (tw, th), _ = cv2.getTextSize(self._toast, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x, y = (self.w - tw) // 2, self.h - BAR_H - 40
            cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), _BK, -1)
            _text(frame, self._toast, (x, y), _W, 0.8)


def _text(frame, s, org, color, scale):
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BK, 4, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def info_screen(w, h, lines):
    import numpy as np
    frame = np.zeros((h, w, 3), dtype="uint8")
    y = h // 2 - len(lines) * 16
    for ln in lines:
        (tw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, ln, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, _W, 2, cv2.LINE_AA)
        y += 40
    return frame


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def build_pipeline(pipeline, cfg):
    """Wire CAM_A -> preview + encoder + full-res still on `pipeline` (before
    start). Returns (preview_q, encoded_q, still_q, trigger_q); a queue is None
    if that feature couldn't be created on this device."""
    pw, ph = cfg["preview_size"]
    vw, vh = cfg["video_size"]
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    preview = cam.requestOutput((pw, ph), type=dai.ImgFrame.Type.NV12,
                                fps=cfg["preview_fps"])
    preview_q = preview.createOutputQueue(maxSize=4, blocking=False)
    print("[camera] preview OK")

    encoded_q = None
    try:
        video = cam.requestOutput((vw, vh), type=dai.ImgFrame.Type.NV12,
                                  fps=cfg["video_fps"])
        profile = (dai.VideoEncoderProperties.Profile.H265_MAIN
                   if cfg["codec"].lower() == "h265"
                   else dai.VideoEncoderProperties.Profile.H264_MAIN)
        enc = pipeline.create(dai.node.VideoEncoder).build(
            video, frameRate=cfg["video_fps"], profile=profile)
        try:
            enc.setKeyframeFrequency(int(cfg["video_fps"]))
        except Exception:
            pass
        encoded_q = enc.out.createOutputQueue(maxSize=int(cfg["video_fps"]) * 2,
                                              blocking=False)
        print("[camera] encoder OK")
    except Exception as exc:
        print(f"[camera] encoder unavailable, video disabled: {exc}")

    still_q = trigger_q = None
    try:
        full = cam.requestFullResolutionOutput(fps=2)  # on-demand only
        script = pipeline.create(dai.node.Script)
        try:
            script.inputs["in"].setBlocking(False)
            script.inputs["in"].setMaxSize(1)
        except Exception:
            pass
        full.link(script.inputs["in"])
        script.setScript(STILL_SCRIPT)
        still_q = script.outputs["still"].createOutputQueue(maxSize=2, blocking=False)
        trigger_q = script.inputs["trigger"].createInputQueue()
        print("[camera] still OK")
    except Exception as exc:
        print(f"[camera] still unavailable, photos disabled: {exc}")

    return preview_q, encoded_q, still_q, trigger_q


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
    low_mb = int(cfg["low_storage_mb"])
    tl_interval = max(1, int(cfg["timelapse_interval_sec"]))
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

    rec = Recorder(session_dir, cfg["codec"].lower(), cfg["video_fps"])
    photos = clips = 0
    timelapse = False
    next_tl = next_disk = 0.0
    low_storage, fmb = False, None
    last_preview = None
    quit_app = False

    while not quit_app:
        try:
            with dai.Pipeline() as pipeline:
                pq, eq, sq, tq = build_pipeline(pipeline, cfg)
                pipeline.start()
                print("[camera] pipeline started")
                ui.toast("Camera ready", 2.0)

                while pipeline.isRunning() and not quit_app:
                    fr = pq.tryGetAll()
                    if fr:
                        last_preview = fr[-1].getCvFrame()
                    frame = last_preview
                    if frame is None:
                        if (cv2.waitKey(15) & 0xFF) == ord("q"):
                            quit_app = True
                        continue

                    if eq is not None and rec.active:
                        rec.write(eq.tryGetAll())
                    elif eq is not None:
                        eq.tryGetAll()  # keep the queue drained when idle

                    if sq is not None:
                        still = sq.tryGet()
                        if still is not None:
                            img = still.getCvFrame()
                            path = os.path.join(session_dir, f"photo_{stamp(millis=True)}.jpg")
                            if cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q]):
                                photos += 1
                                ui.flash()
                                ui.toast(f"Photo saved ({img.shape[1]}x{img.shape[0]})")
                            else:
                                ui.toast("PHOTO SAVE FAILED")

                    now = time.monotonic()
                    if now >= next_disk:
                        next_disk = now + 5.0
                        fmb = free_mb(session_dir)
                        low_storage = fmb is not None and fmb < low_mb

                    if timelapse and now >= next_tl:
                        next_tl = now + tl_interval
                        if tq is not None and not low_storage:
                            tq.send(dai.Buffer())

                    for key in ui.take_events():
                        if key == "photo":
                            if tq is None:
                                ui.toast("Stills unavailable on this device")
                            elif low_storage:
                                ui.toast("LOW STORAGE - not capturing")
                            else:
                                tq.send(dai.Buffer())
                        elif key == "record":
                            if eq is None and not rec.active:
                                ui.toast("Video unavailable on this device")
                            elif rec.active:
                                ui.toast("Saving clip...")
                                mp4 = rec.stop()
                                clips += 1 if mp4 else 0
                                ui.toast("Clip saved" if mp4 else "Saved raw (mux failed)")
                            elif low_storage:
                                ui.toast("LOW STORAGE - not recording")
                            elif rec.start():
                                ui.toast("Recording")
                            else:
                                ui.toast("Could not start recording")
                        elif key == "timelapse":
                            timelapse = not timelapse
                            next_tl = now
                            ui.toast(f"Timelapse every {tl_interval}s" if timelapse else "Timelapse off")
                        elif key == "exit":
                            quit_app = True

                    ui.draw(frame, {"recording": rec.active, "rec_elapsed": rec.elapsed,
                                    "timelapse": timelapse, "photos": photos, "clips": clips,
                                    "low_storage": low_storage, "free_mb": fmb})
                    cv2.imshow(WINDOW, frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        quit_app = True

        except Exception as exc:  # device dropped / bring-up failed
            print(f"[main] camera error: {exc}")
            if rec.active:
                mp4 = rec.stop()
                clips += 1 if mp4 else 0
                ui.toast("Saved clip (camera lost)")
            cv2.imshow(WINDOW, info_screen(w, h, ["CAMERA DISCONNECTED",
                                                  "Check the OAK-D Lite cable / power.",
                                                  "Reconnecting..."]))
            if (cv2.waitKey(2000) & 0xFF) == ord("q"):
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
