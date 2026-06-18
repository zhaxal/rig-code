#!/usr/bin/env python3
"""
main.py - Greenhouse photo/video capture app for Raspberry Pi 4 + OAK-D Lite.

Single-threaded, touch-only, fullscreen. Its only job is to reliably capture and
save media, so it favours simplicity and graceful recovery over features.

Structure mirrors the official DepthAI v3 examples: the whole session runs
inside `with dai.Pipeline() as pipeline:` while `pipeline.isRunning()`. An outer
loop re-enters that block to reconnect if the camera drops - leaving the `with`
block always releases the device, so a rebuild never hits "no available sensor".

Run:
    python3 main.py                 # uses config.json next to this file
    python3 main.py --note "row 5"  # override the session note
    python3 main.py --windowed      # don't go fullscreen (handy for testing)

Touch controls (bottom bar): PHOTO | REC/STOP | TIME-LAPSE | EXIT.
"""

import argparse
import json
import os
import sys
import time

import cv2
import depthai as dai

from camera import build_pipeline
from storage import SessionStorage, VideoRecorder
from ui import TouchUI

WINDOW = "capture"
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
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


def load_config(path):
    """Merge config.json over the defaults; tolerate a missing/broken file."""
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(path) as fh:
            user = json.load(fh)
        cfg.update({k: v for k, v in user.items() if not k.startswith("_")})
    except FileNotFoundError:
        print(f"[main] {path} not found, using defaults")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[main] could not read {path} ({exc}); using defaults")
    return cfg


def setup_window(fullscreen):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)


def info_screen(width, height, lines):
    """A black frame with centred text, for status/error displays."""
    import numpy as np
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    y = height // 2 - (len(lines) * 16)
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, line, ((width - tw) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        y += 40
    return frame


class App:
    """Holds the state that must survive a camera reconnect."""

    def __init__(self, cfg, storage, ui):
        self.cfg = cfg
        self.storage = storage
        self.ui = ui
        self.width, self.height = cfg["preview_size"]
        self.recorder = VideoRecorder(storage, cfg["video_fps"])
        self.timelapse_on = False
        self.timelapse_interval = max(1, int(cfg["timelapse_interval_sec"]))
        self.next_timelapse = 0.0
        self.next_storage_check = 0.0
        self.low_storage = False
        self.free_mb = None

    # ---- one connected session -------------------------------------------

    def run_session(self, pipeline, streams):
        """Run the capture loop while the pipeline is alive.

        Returns True if the user chose to quit, False if the camera dropped
        (so the caller reconnects). Raises on a device error mid-loop, which
        the caller also treats as "reconnect"."""
        ui, storage, recorder = self.ui, self.storage, self.recorder
        ui.toast("Camera ready", 2.0)

        while pipeline.isRunning():
            frame = streams.get_preview()
            if frame is None:  # pipeline still warming up
                if _q_pressed(15):
                    return True
                continue

            # Drain the encoder every loop; write only while recording.
            encoded = streams.poll_encoded()
            if recorder.active:
                recorder.write(encoded)

            # Save any full-res still that arrived.
            still = streams.poll_still()
            if still is not None:
                self._save_still(still)

            now = time.monotonic()
            if now >= self.next_storage_check:
                self.next_storage_check = now + 5.0
                self.free_mb = storage.free_mb()
                self.low_storage = storage.low_storage()

            if self.timelapse_on and now >= self.next_timelapse:
                self.next_timelapse = now + self.timelapse_interval
                if not self.low_storage:  # don't keep filling a near-full disk
                    streams.trigger_still()

            for key in ui.take_events():
                if self._handle_event(key, streams, now):
                    return True  # exit requested

            ui.draw(frame, self._state())
            cv2.imshow(WINDOW, frame)
            if _q_pressed(1):  # keyboard escape hatch for developers
                return True

        return False  # pipeline stopped -> reconnect

    def _save_still(self, still):
        path = self.storage.new_photo_path()
        ok = cv2.imwrite(path, still,
                         [cv2.IMWRITE_JPEG_QUALITY, self.storage.jpeg_quality])
        if ok:
            self.storage.count_photo()
            self.ui.flash()
            self.ui.toast(f"Photo saved ({still.shape[1]}x{still.shape[0]})")
        else:
            self.ui.toast("PHOTO SAVE FAILED")

    def _handle_event(self, key, streams, now):
        """Act on a button press. Returns True if the app should exit."""
        ui, recorder = self.ui, self.recorder
        if key == "photo":
            if not streams.has_stills:
                ui.toast("Stills unavailable on this device")
            elif self.low_storage:
                ui.toast("LOW STORAGE - not capturing")
            else:
                streams.trigger_still()
        elif key == "record":
            if not streams.has_video and not recorder.active:
                ui.toast("Video unavailable on this device")
            elif recorder.active:
                ui.toast("Saving clip...")
                mp4 = recorder.stop()
                ui.toast("Clip saved" if mp4 else "Saved raw (mux failed)")
            elif self.low_storage:
                ui.toast("LOW STORAGE - not recording")
            elif recorder.start():
                ui.toast("Recording")
            else:
                ui.toast("Could not start recording")
        elif key == "timelapse":
            self.timelapse_on = not self.timelapse_on
            if self.timelapse_on:
                self.next_timelapse = now  # fire immediately
                ui.toast(f"Timelapse every {self.timelapse_interval}s")
            else:
                ui.toast("Timelapse off")
        elif key == "exit":
            return True
        return False

    def _state(self):
        return {
            "recording": self.recorder.active,
            "rec_elapsed": self.recorder.elapsed,
            "timelapse": self.timelapse_on,
            "photo_count": self.storage.photo_count,
            "video_count": self.storage.video_count,
            "low_storage": self.low_storage,
            "free_mb": self.free_mb,
        }


def _q_pressed(wait_ms):
    return (cv2.waitKey(wait_ms) & 0xFF) == ord("q")


def main():
    parser = argparse.ArgumentParser(description="Greenhouse capture app")
    parser.add_argument("--config", default=os.path.join(HERE, "config.json"))
    parser.add_argument("--note", help="override session note")
    parser.add_argument("--windowed", action="store_true",
                        help="run windowed instead of fullscreen")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.note is not None:
        cfg["note"] = args.note
    if args.windowed:
        cfg["fullscreen"] = False

    width, height = cfg["preview_size"]
    storage = SessionStorage(cfg)
    print(f"[main] saving to {storage.session_dir}")

    ui = TouchUI(width, height)
    setup_window(cfg["fullscreen"])
    cv2.setMouseCallback(WINDOW, ui.on_mouse)
    app = App(cfg, storage, ui)

    quit_app = False
    while not quit_app:
        try:
            # Whole session lives inside the pipeline context; leaving it (on
            # exit OR error) always releases the device for a clean rebuild.
            with dai.Pipeline() as pipeline:
                streams = build_pipeline(pipeline, cfg)
                pipeline.start()
                print("[camera] pipeline started")
                quit_app = app.run_session(pipeline, streams)
        except Exception as exc:  # device dropped / bring-up failed
            print(f"[main] camera error: {exc}")
            if app.recorder.active:
                mp4 = app.recorder.stop()  # flush whatever we recorded
                ui.toast("Saved clip (camera lost)" if mp4 else "Clip recovered")
            frame = info_screen(width, height,
                                ["CAMERA DISCONNECTED",
                                 "Check the OAK-D Lite cable.",
                                 "Reconnecting..."])
            cv2.imshow(WINDOW, frame)
            if _q_pressed(2000):  # pause before retry; q to give up
                quit_app = True

    print("[main] shutting down")
    if app.recorder.active:
        app.recorder.stop()  # flush any in-progress recording
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] interrupted")
        sys.exit(0)
