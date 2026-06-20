#!/usr/bin/env python3
"""CameraWorker: the DepthAI pipeline running in a background thread.

The worker owns the OAK device. Each loop it grabs an RGB frame and the latest
spatial detections, draws the overlay, optionally records / saves stills, and
pushes the annotated frame to the UI through a size-1 queue (newest wins).

The tkinter UI talks to the worker only through thread-safe queues:
  * commands in  : ("photo",) ("record",) ("timelapse",) ("switch", entry) ("quit",)
  * frames out   : annotated BGR numpy arrays (queue maxsize=1)
  * status       : get_status() -> dict snapshot for button state

Live model switching reuses the outer-loop / inner-pipeline pattern: a "switch"
command breaks the inner loop and the pipeline is rebuilt with the new model.
This also yields camera-disconnect recovery for free.

A ``mock=True`` worker generates synthetic frames + detections so the GUI can be
exercised without an OAK attached.
"""

import json
import math
import os
import queue
import shutil
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from overlay import draw_detections, draw_hud
from recorder import Recorder, stamp

HERE = os.path.dirname(os.path.abspath(__file__))


def _free_mb(path):
    try:
        return shutil.disk_usage(path).free / (1024 * 1024)
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Mock detection objects (mirror DepthAI's SpatialImgDetection attributes)
# --------------------------------------------------------------------------- #

class _FakeSC:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeDet:
    def __init__(self, xmin, ymin, xmax, ymax, label, conf, sc):
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        self.label, self.confidence = label, conf
        self.spatialCoordinates = sc


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

class CameraWorker(threading.Thread):
    def __init__(self, cfg, model_entry, frame_q, mock=False):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.frame_q = frame_q
        self.mock = mock
        self.cmd_q = queue.Queue()

        self.w, self.h = cfg["screen_size"]
        self.fps = int(cfg["capture_fps"])
        self.jpeg_q = int(cfg["photo_jpeg_quality"])
        self.low_mb = int(cfg["low_storage_mb"])
        self.tl_interval = max(1, int(cfg["timelapse_interval_sec"]))

        self._model = model_entry            # current ModelEntry (may be None)
        self._desired_model = model_entry    # set by "switch" command
        self._stop = threading.Event()

        # Capture session
        self.session_dir = os.path.join(
            os.path.expanduser(cfg["save_root"]), stamp())
        os.makedirs(self.session_dir, exist_ok=True)
        with open(os.path.join(self.session_dir, "session_meta.json"), "w") as fh:
            json.dump({"started": datetime.now().isoformat(timespec="seconds"),
                       "note": cfg.get("note", ""), "settings": cfg}, fh, indent=2)

        self.rec = Recorder(self.session_dir, self.fps, (self.w, self.h))
        self.timelapse = False
        self._next_tl = 0.0  # next time-lapse capture time (monotonic)
        self.photos = 0
        self.clips = 0
        self.low_storage = False
        self.free_mb = None

        self._fps_val = 0.0
        self._toast = ""
        self._toast_until = 0.0
        self._lock = threading.Lock()
        self._loading = False  # pipeline (re)building

    # --- public API (called from the UI thread) ------------------------- #

    def send(self, *cmd):
        self.cmd_q.put(cmd)

    def stop(self):
        self._stop.set()
        self.cmd_q.put(("quit",))

    def get_status(self):
        with self._lock:
            return {
                "recording": self.rec.active,
                "rec_elapsed": self.rec.elapsed,
                "timelapse": self.timelapse,
                "photos": self.photos,
                "clips": self.clips,
                "model": self._model.name if self._model else "(no model)",
                "loading": self._loading,
            }

    # --- internals ------------------------------------------------------ #

    def _toast_msg(self, msg, secs=2.0):
        self._toast = msg
        self._toast_until = time.monotonic() + secs

    def _hud_state(self):
        with self._lock:
            toast = self._toast if time.monotonic() < self._toast_until else ""
            return {
                "model": self._model.name if self._model else "(no model)",
                "fps": self._fps_val,
                "recording": self.rec.active,
                "rec_elapsed": self.rec.elapsed,
                "timelapse": self.timelapse,
                "photos": self.photos,
                "clips": self.clips,
                "low_storage": self.low_storage,
                "free_mb": self.free_mb,
                "toast": toast,
            }

    def _push(self, frame):
        """Drop any stale frame and enqueue the newest one (non-blocking)."""
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
            self._toast_msg("LOW STORAGE - not capturing")
            return
        path = os.path.join(self.session_dir, f"photo_{stamp(millis=True)}.jpg")
        if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q]):
            with self._lock:
                self.photos += 1
            self._toast_msg(f"Photo saved ({self.w}x{self.h})")
        else:
            self._toast_msg("PHOTO SAVE FAILED")

    def _handle_command(self, cmd, clean_frame):
        """Process one UI command. Returns True if the inner loop should break
        (model switch) so the pipeline can be rebuilt."""
        action = cmd[0]
        if action == "quit":
            self._stop.set()
        elif action == "photo":
            self._save_photo(clean_frame)
        elif action == "record":
            if self.rec.active:
                self._toast_msg("Saving clip...")
                mp4 = self.rec.stop()
                if mp4:
                    with self._lock:
                        self.clips += 1
                self._toast_msg("Clip saved" if mp4 else "Save failed")
            elif self.low_storage:
                self._toast_msg("LOW STORAGE - not recording")
            elif self.rec.start():
                self._toast_msg("Recording")
            else:
                self._toast_msg("Could not start recording")
        elif action == "timelapse":
            with self._lock:
                self.timelapse = not self.timelapse
            if self.timelapse:
                # First still after one full interval, not immediately.
                self._next_tl = time.monotonic() + self.tl_interval
            self._toast_msg(f"Timelapse every {self.tl_interval}s"
                            if self.timelapse else "Timelapse off")
        elif action == "switch":
            self._desired_model = cmd[1]
            return True
        return False

    # --- pipeline build (real hardware) --------------------------------- #

    def _build_pipeline(self, pipeline):
        """Wire RGB preview + (if a model is set) stereo + SpatialDetectionNetwork.
        Returns (preview_q, spatial_q, label_map).

        When a model is active the displayed/recorded frame is the detection
        network's *passthrough* (the exact frame the net ran on), so the overlay
        boxes line up with the objects. Without a model we fall back to a plain
        RGB camera output.
        """
        import depthai as dai

        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        spatial_q = None
        label_map = []
        entry = self._model
        if entry is not None:
            mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth)
            mono_left.requestOutput((640, 400)).link(stereo.left)
            mono_right.requestOutput((640, 400)).link(stereo.right)
            stereo.setLeftRightCheck(True)
            stereo.setDefaultProfilePreset(
                dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

            if entry.kind == "archive":
                model_desc = dai.NNArchive(entry.ref)
            else:
                model_desc = dai.NNModelDescription(entry.ref)

            # v3 SpatialDetectionNetwork.build wires depth itself: (rgb, stereo, model).
            spatial_net = (pipeline.create(dai.node.SpatialDetectionNetwork)
                           .build(cam, stereo, model_desc, fps=self.fps))
            try:
                spatial_net.setBoundingBoxScaleFactor(float(self.cfg["bb_scale"]))
                spatial_net.setDepthLowerThreshold(int(self.cfg["depth_lower_mm"]))
                spatial_net.setDepthUpperThreshold(int(self.cfg["depth_upper_mm"]))
            except Exception:
                pass

            spatial_q = spatial_net.out.createOutputQueue(maxSize=4, blocking=False)
            preview_q = spatial_net.passthrough.createOutputQueue(maxSize=4,
                                                                  blocking=False)
            label_map = spatial_net.getClasses() or []
            print(f"[camera] spatial detection OK "
                  f"({entry.name}, {len(label_map)} classes)")
        else:
            preview_q = cam.requestOutput((self.w, self.h),
                                          dai.ImgFrame.Type.BGR888p,
                                          fps=self.fps).createOutputQueue()
            print(f"[camera] preview OK ({self.w}x{self.h}@{self.fps})")

        return preview_q, spatial_q, label_map

    # --- main loops ----------------------------------------------------- #

    def run(self):
        if self.mock:
            self._run_mock()
            return

        while not self._stop.is_set():
            try:
                import depthai as dai
                with self._lock:
                    self._model = self._desired_model
                    self._loading = True
                with dai.Pipeline() as pipeline:
                    pq, sq, labels = self._build_pipeline(pipeline)
                    pipeline.start()
                    with self._lock:
                        self._loading = False
                    self._toast_msg("Camera ready")
                    self._loop(pipeline, pq, sq, labels)
            except Exception as exc:
                print(f"[camera] error: {exc}")
                if self.rec.active:
                    self.rec.stop()
                with self._lock:
                    self._loading = False
                self._error_frame(str(exc)[:48])
                if self._stop.wait(2.0):
                    break
        print("[camera] worker stopped")

    _STALE_TIMEOUT = 15.0  # no frames for this long -> force a pipeline rebuild

    def _loop(self, pipeline, pq, sq, labels):
        current_detections = []
        next_disk = 0.0
        if self.timelapse:
            self._next_tl = time.monotonic() + self.tl_interval
        ema = None
        last = time.monotonic()
        last_frame = time.monotonic()

        while pipeline.isRunning() and not self._stop.is_set():
            # Non-blocking grab so we stay responsive to stop/switch and can
            # detect a stalled camera instead of hanging on a blocking get().
            frame_msg = pq.tryGet()
            if frame_msg is None:
                if time.monotonic() - last_frame > self._STALE_TIMEOUT:
                    raise RuntimeError("camera stalled (no frames)")
                time.sleep(0.005)
                continue
            last_frame = time.monotonic()
            frame = frame_msg.getCvFrame()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            if frame.shape[1] != self.w or frame.shape[0] != self.h:
                frame = cv2.resize(frame, (self.w, self.h))

            if sq is not None:
                msg = sq.tryGet()
                if msg is not None:
                    current_detections = msg.detections

            draw_detections(frame, current_detections, labels, self.w, self.h)

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

            if self.rec.active:
                self.rec.write(frame)

            if self.timelapse and now >= self._next_tl:
                self._next_tl = now + self.tl_interval
                if not self.low_storage:
                    self._save_photo(frame)

            if self._drain_commands(frame):
                break  # model switch -> rebuild pipeline

            hud_frame = frame.copy()
            draw_hud(hud_frame, self._hud_state())
            self._push(hud_frame)

    def _drain_commands(self, clean_frame):
        switch = False
        try:
            while True:
                cmd = self.cmd_q.get_nowait()
                if self._handle_command(cmd, clean_frame):
                    switch = True
        except queue.Empty:
            pass
        return switch

    def _error_frame(self, msg):
        frame = np.zeros((self.h, self.w, 3), dtype="uint8")
        for i, ln in enumerate(["CAMERA ERROR", msg, "Retrying..."]):
            (tw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, ln, ((self.w - tw) // 2, self.h // 2 - 30 + i * 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        self._push(frame)

    # --- mock loop (no hardware) ---------------------------------------- #

    def _run_mock(self):
        with self._lock:
            self._model = self._desired_model
        t0 = time.monotonic()
        next_disk = 0.0
        while not self._stop.is_set():
            t = time.monotonic() - t0
            frame = np.full((self.h, self.w, 3), 30, dtype="uint8")
            cv2.rectangle(frame, (0, 0), (self.w, self.h), (60, 40, 20), 8)

            # A box drifting horizontally with a plausible spatial reading
            cx = 0.3 + 0.2 * (1 + math.sin(t)) / 2
            dets = [_FakeDet(cx, 0.35, cx + 0.18, 0.7, 0, 0.92,
                             _FakeSC(200 * math.sin(t), -120, 1500 + 400 * math.cos(t)))]
            draw_detections(frame, dets, ["object"], self.w, self.h)

            now = time.monotonic()
            with self._lock:
                self._fps_val = float(self.fps)
            if now >= next_disk:
                next_disk = now + 5.0
                self.free_mb = _free_mb(self.session_dir)
                self.low_storage = (self.free_mb is not None
                                    and self.free_mb < self.low_mb)
            if self.rec.active:
                self.rec.write(frame)
            if self.timelapse and now >= self._next_tl:
                self._next_tl = now + self.tl_interval
                if not self.low_storage:
                    self._save_photo(frame)

            if self._drain_commands(frame):
                with self._lock:
                    self._model = self._desired_model
                self._toast_msg(f"Switched to {self._model.name}"
                                if self._model else "No model")

            hud = frame.copy()
            draw_hud(hud, self._hud_state())
            self._push(hud)
            time.sleep(1.0 / max(1, self.fps))

        if self.rec.active:
            self.rec.stop()
        print("[camera] mock worker stopped")
