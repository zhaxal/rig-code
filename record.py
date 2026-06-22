#!/usr/bin/env python3
"""record.py — minimal 3-camera recording mode.

Records the main RGB camera and both stereo mono cameras simultaneously to
separate MP4 files.  No detection, no model picker, no timelapse, no dataset.

Run:
    python3 record.py              # fullscreen
    python3 record.py --windowed   # windowed (testing over HDMI/VNC)
    python3 record.py --mock       # synthetic frames, no OAK required

Controls: REC/STOP · PHOTO · EXIT   (Esc / q quit.)
"""

import argparse
import math
import os
import queue
import shutil
import threading
import time

import cv2
import numpy as np
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

from config import load_config
from overlay import letterbox
from recorder import Recorder, stamp

HERE = os.path.dirname(os.path.abspath(__file__))
BAR_H = 90
MONO_W, MONO_H = 640, 400


def _free_mb(path):
    try:
        return shutil.disk_usage(path).free / (1024 * 1024)
    except OSError:
        return None


class RecordWorker(threading.Thread):
    def __init__(self, cfg, frame_q, mock=False):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.frame_q = frame_q
        self.mock = mock
        self.cmd_q = queue.Queue()

        self.w, self.h = cfg["screen_size"]
        self.fps = int(cfg["capture_fps"])
        self.jpeg_q = int(cfg["photo_jpeg_quality"])
        self.low_mb = int(cfg["low_storage_mb"])

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._loading = False
        self._fps_val = 0.0
        self._toast = ""
        self._toast_until = 0.0
        self._build_fps = self.fps

        self.session_dir = os.path.join(os.path.expanduser(cfg["save_root"]), stamp())
        os.makedirs(self.session_dir, exist_ok=True)

        # RGB output resolution: native 16:9 crop from the sensor
        self.rgb_w = self.w
        self.rgb_h = max(1, round(self.w * 9 / 16))

        self.photos = 0
        self.clips = 0
        self.low_storage = False
        self.free_mb = None

        self.rec_main  = Recorder(self.session_dir, self.fps, (self.rgb_w, self.rgb_h))
        self.rec_left  = Recorder(self.session_dir, self.fps, (MONO_W, MONO_H))
        self.rec_right = Recorder(self.session_dir, self.fps, (MONO_W, MONO_H))

        self._last_raw = None  # latest clean RGB frame for photo saves

    # ------------------------------------------------------------------ #
    # Public API (called from UI thread)

    def send(self, *cmd):
        self.cmd_q.put(cmd)

    def stop(self):
        self._stop.set()
        self.cmd_q.put(("quit",))

    def get_status(self):
        with self._lock:
            return {
                "recording": self.rec_main.active,
                "rec_elapsed": self.rec_main.elapsed,
                "photos": self.photos,
                "clips": self.clips,
                "loading": self._loading,
            }

    # ------------------------------------------------------------------ #
    # Internals

    def _toast_msg(self, msg, secs=2.0):
        self._toast = msg
        self._toast_until = time.monotonic() + secs

    def _push(self, frame):
        try:
            self.frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_q.put_nowait(frame)
        except queue.Full:
            pass

    def _save_photo(self, frame):
        if self.low_storage:
            self._toast_msg("LOW STORAGE")
            return
        path = os.path.join(self.session_dir, f"photo_{stamp(millis=True)}.jpg")
        if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q]):
            with self._lock:
                self.photos += 1
            self._toast_msg("Photo saved")
        else:
            self._toast_msg("PHOTO SAVE FAILED")

    def _start_recording(self):
        if self.low_storage:
            self._toast_msg("LOW STORAGE - not recording")
            return
        ok = self.rec_main.start() and self.rec_left.start() and self.rec_right.start()
        if ok:
            self._toast_msg("Recording — 3 cameras")
        else:
            for r in (self.rec_main, self.rec_left, self.rec_right):
                if r.active:
                    r.stop()
            self._toast_msg("Could not start recording")

    def _stop_recording(self):
        self._toast_msg("Saving clips...")
        saved = sum(1 for r in (self.rec_main, self.rec_left, self.rec_right)
                    if r.stop() is not None)
        with self._lock:
            if saved:
                self.clips += 1
        self._toast_msg(f"Saved {saved}/3 clips" if saved < 3 else "Clips saved (3 cameras)")

    def _handle_command(self, cmd, raw_frame):
        action = cmd[0]
        if action == "quit":
            self._stop.set()
        elif action == "photo":
            self._save_photo(raw_frame)
        elif action == "record":
            if self.rec_main.active:
                self._stop_recording()
            else:
                self._start_recording()

    def _drain_commands(self, raw_frame):
        try:
            while True:
                cmd = self.cmd_q.get_nowait()
                self._handle_command(cmd, raw_frame)
        except queue.Empty:
            pass

    def _tune_for_usb(self, device):
        self._build_fps = self.fps
        try:
            speed = device.getUsbSpeed()
        except Exception as exc:
            print(f"[record] could not read USB speed ({exc})")
            return
        if "SUPER" in str(speed).upper():
            print(f"[record] USB link: {speed} -> {self._build_fps} fps")
        else:
            self._build_fps = min(self.fps, 10)
            print(f"[record] USB2 link -> reduced to {self._build_fps} fps")
            self._toast_msg("USB2 link - reduced FPS", 3.0)

    def _build_pipeline(self, pipeline):
        import depthai as dai
        fps = self._build_fps

        cam_a = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=fps)
        cam_b = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B, sensorFps=fps)
        cam_c = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C, sensorFps=fps)

        preview_q = cam_a.requestOutput(
            (self.rgb_w, self.rgb_h), dai.ImgFrame.Type.BGR888p
        ).createOutputQueue(maxSize=4, blocking=False)

        left_q = cam_b.requestOutput(
            (MONO_W, MONO_H)
        ).createOutputQueue(maxSize=4, blocking=False)

        right_q = cam_c.requestOutput(
            (MONO_W, MONO_H)
        ).createOutputQueue(maxSize=4, blocking=False)

        print(f"[record] pipeline: RGB {self.rgb_w}x{self.rgb_h} "
              f"+ mono {MONO_W}x{MONO_H} @ {fps} fps")
        return preview_q, left_q, right_q

    _STALE_TIMEOUT = 15.0

    def run(self):
        if self.mock:
            self._run_mock()
            return

        while not self._stop.is_set():
            try:
                import depthai as dai
                with self._lock:
                    self._loading = True
                with dai.Device() as device:
                    self._tune_for_usb(device)
                    with dai.Pipeline(device) as pipeline:
                        pq, lq, rq = self._build_pipeline(pipeline)
                        pipeline.start()
                        with self._lock:
                            self._loading = False
                        self._toast_msg("Ready")
                        self._loop(pipeline, pq, lq, rq)
            except Exception as exc:
                print(f"[record] error: {exc}")
                for r in (self.rec_main, self.rec_left, self.rec_right):
                    if r.active:
                        r.stop()
                with self._lock:
                    self._loading = False
                self._error_frame(str(exc)[:48])
                if self._stop.wait(5.0):
                    break
        print("[record] worker stopped")

    def _loop(self, pipeline, pq, lq, rq):
        next_disk = 0.0
        ema = None
        last = time.monotonic()
        last_frame = time.monotonic()

        while pipeline.isRunning() and not self._stop.is_set():
            frame_msg = pq.tryGet()
            if frame_msg is None:
                if time.monotonic() - last_frame > self._STALE_TIMEOUT:
                    raise RuntimeError("camera stalled (no frames)")
                time.sleep(0.005)
                continue
            last_frame = time.monotonic()

            raw = cv2.rotate(frame_msg.getCvFrame(), cv2.ROTATE_180)
            self._last_raw = raw

            left_msg = lq.tryGet()
            left_raw = (cv2.rotate(left_msg.getCvFrame(), cv2.ROTATE_180)
                        if left_msg is not None else None)

            right_msg = rq.tryGet()
            right_raw = (cv2.rotate(right_msg.getCvFrame(), cv2.ROTATE_180)
                         if right_msg is not None else None)

            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0:
                inst = 1.0 / dt
                ema = inst if ema is None else (0.9 * ema + 0.1 * inst)
                with self._lock:
                    self._fps_val = ema

            if now >= next_disk:
                next_disk = now + 5.0
                self.free_mb = _free_mb(self.session_dir)
                self.low_storage = (self.free_mb is not None
                                    and self.free_mb < self.low_mb)

            if self.rec_main.active:
                self.rec_main.write(raw)
                if left_raw is not None:
                    left_bgr = (cv2.cvtColor(left_raw, cv2.COLOR_GRAY2BGR)
                                if left_raw.ndim == 2 else left_raw)
                    self.rec_left.write(left_bgr)
                if right_raw is not None:
                    right_bgr = (cv2.cvtColor(right_raw, cv2.COLOR_GRAY2BGR)
                                 if right_raw.ndim == 2 else right_raw)
                    self.rec_right.write(right_bgr)

            self._drain_commands(raw)

            display, _ = letterbox(raw, self.w, self.h)
            hud = display.copy()
            self._draw_hud(hud)
            self._push(hud)

    # ------------------------------------------------------------------ #
    # HUD overlay

    def _draw_hud(self, frame):
        with self._lock:
            fps = self._fps_val
            recording = self.rec_main.active
            elapsed = self.rec_main.elapsed
            photos = self.photos
            clips = self.clips
            toast = self._toast if time.monotonic() < self._toast_until else ""
            low = self.low_storage
            free_mb = self.free_mb

        lines = []
        if recording:
            m, s = divmod(int(elapsed), 60)
            lines.append(f"REC {m:02d}:{s:02d}")
        if low:
            lines.append("LOW STORAGE")
        lines.append(toast if toast else f"FPS {fps:.1f}  P:{photos}  C:{clips}")
        if free_mb is not None:
            lines.append(f"{free_mb:.0f} MB free")

        y = 24
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x = self.w - tw - 8
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28

    def _error_frame(self, msg):
        frame = np.zeros((self.h, self.w, 3), dtype="uint8")
        for i, ln in enumerate(["CAMERA ERROR", msg, "Retrying..."]):
            (tw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, ln, ((self.w - tw) // 2, self.h // 2 - 30 + i * 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        self._push(frame)

    # ------------------------------------------------------------------ #
    # Mock loop (no hardware)

    def _run_mock(self):
        t0 = time.monotonic()
        next_disk = 0.0
        while not self._stop.is_set():
            t = time.monotonic() - t0
            cx = int(self.rgb_w * (0.3 + 0.2 * (1 + math.sin(t)) / 2))

            raw = np.full((self.rgb_h, self.rgb_w, 3), 25, dtype="uint8")
            cv2.circle(raw, (cx, self.rgb_h // 2), 30, (80, 160, 240), -1)
            cv2.putText(raw, "MOCK RGB", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cx_m = cx * MONO_W // self.rgb_w
            left_raw = np.full((MONO_H, MONO_W), 60, dtype="uint8")
            cv2.circle(left_raw, (cx_m, MONO_H // 2), 20, 180, -1)
            right_raw = np.full((MONO_H, MONO_W), 40, dtype="uint8")
            cv2.circle(right_raw, (cx_m + 10, MONO_H // 2), 20, 200, -1)

            self._last_raw = raw
            now = time.monotonic()
            with self._lock:
                self._fps_val = float(self.fps)

            if now >= next_disk:
                next_disk = now + 5.0
                self.free_mb = _free_mb(self.session_dir)
                self.low_storage = (self.free_mb is not None
                                    and self.free_mb < self.low_mb)

            if self.rec_main.active:
                self.rec_main.write(raw)
                self.rec_left.write(cv2.cvtColor(left_raw, cv2.COLOR_GRAY2BGR))
                self.rec_right.write(cv2.cvtColor(right_raw, cv2.COLOR_GRAY2BGR))

            self._drain_commands(raw)

            display, _ = letterbox(raw, self.w, self.h)
            hud = display.copy()
            self._draw_hud(hud)
            self._push(hud)
            time.sleep(1.0 / max(1, self.fps))

        for r in (self.rec_main, self.rec_left, self.rec_right):
            if r.active:
                r.stop()
        print("[record] mock worker stopped")


# --------------------------------------------------------------------------- #
# UI

class RecordApp:
    def __init__(self, root, cfg, mock=False):
        self.root = root
        self.cfg = cfg
        self.W, self.H = cfg["screen_size"]
        self.video_h = self.H - BAR_H

        worker_cfg = dict(cfg)
        worker_cfg["screen_size"] = [self.W, self.video_h]
        self.frame_q = queue.Queue(maxsize=1)
        self.worker = RecordWorker(worker_cfg, self.frame_q, mock=mock)

        self._photo_ref = None
        self._last_btn_state = None
        self._build_ui()

        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("q", lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.worker.start()
        self._render()

    def _build_ui(self):
        self.root.configure(bg="black")
        self.canvas = tk.Canvas(self.root, width=self.W, height=self.video_h,
                                bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw")

        bar = tk.Frame(self.root, bg="black", height=BAR_H)
        bar.pack(side="bottom", fill="x")
        font = tkfont.Font(family="DejaVu Sans", size=14, weight="bold")

        self.buttons = {}
        specs = [
            ("record", "REC",   "#3a3a3a", lambda: self.worker.send("record")),
            ("photo",  "PHOTO", "#3a3a3a", lambda: self.worker.send("photo")),
            ("exit",   "EXIT",  "#3a3a3a", self.quit),
        ]
        for i, (key, label, color, cmd) in enumerate(specs):
            b = tk.Button(bar, text=label, command=cmd, font=font,
                          fg="white", bg=color, activebackground="#555",
                          activeforeground="white", bd=0, relief="flat",
                          highlightthickness=0)
            b.place(relx=i / len(specs), rely=0,
                    relwidth=1 / len(specs), relheight=1.0)
            self.buttons[key] = b

    def _render(self):
        try:
            frame = self.frame_q.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.itemconfig(self.canvas_img, image=img)
            self._photo_ref = img

        self._refresh_buttons()
        self.root.after(15, self._render)

    def _refresh_buttons(self):
        st = self.worker.get_status()
        state = (st["recording"], st["loading"])
        if state == self._last_btn_state:
            return
        self._last_btn_state = state
        self.buttons["record"].config(
            text="STOP" if st["recording"] else "REC",
            bg="#a33" if st["recording"] else "#3a3a3a")

    def quit(self):
        try:
            self.worker.stop()
        finally:
            self.root.after(150, self.root.destroy)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="3-camera recording mode")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--windowed", action="store_true",
                    help="windowed mode (for HDMI/VNC testing)")
    ap.add_argument("--mock", action="store_true",
                    help="synthetic frames, no OAK required")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.windowed:
        cfg["fullscreen"] = False

    root = tk.Tk()
    root.title("3-Camera Recorder")
    w, h = cfg["screen_size"]
    if cfg.get("fullscreen", True) and not args.windowed:
        root.attributes("-fullscreen", True)
        root.config(cursor="none")
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
        cfg["screen_size"] = [root.winfo_screenwidth(), root.winfo_screenheight()]
    else:
        root.geometry(f"{w}x{h}")

    RecordApp(root, cfg, mock=args.mock)
    root.mainloop()


if __name__ == "__main__":
    main()
