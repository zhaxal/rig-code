#!/usr/bin/env python3
"""
main.py - Greenhouse photo/video capture app for Raspberry Pi 4 + OAK-D Lite.

Single-threaded loop, touch-only, fullscreen. Its only job is to reliably
capture and save media, so it favours simplicity and graceful recovery over
features: a camera disconnect rebuilds the pipeline instead of crashing, an
in-progress recording is always flushed to disk, and low storage is surfaced.

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

from capture import CameraError, CameraManager
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
    """A simple black frame with centred text, for status/error displays."""
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

    camera = CameraManager(cfg)
    recorder = VideoRecorder(storage, cfg["video_fps"])

    timelapse_on = False
    timelapse_interval = max(1, int(cfg["timelapse_interval_sec"]))
    next_timelapse = 0.0
    next_storage_check = 0.0
    low_storage = False
    free_mb = None
    last_reconnect_attempt = 0.0

    running = True
    try:
        camera.build()
        ui.toast("Camera ready", 2.0)
    except CameraError as exc:
        print(f"[main] initial camera build failed: {exc}")

    while running:
        # ---- ensure the camera is up, rebuild if it dropped --------------
        if not camera.is_alive():
            if recorder.active:
                # Flush whatever we recorded before losing the camera.
                mp4 = recorder.stop()
                ui.toast("Saved clip (camera lost)" if mp4 else "Clip recovered")
            now = time.monotonic()
            if now - last_reconnect_attempt > 2.0:
                last_reconnect_attempt = now
                camera.close()
                try:
                    camera.build()
                    ui.toast("Camera reconnected", 2.0)
                except CameraError as exc:
                    print(f"[main] reconnect failed: {exc}")
            frame = info_screen(width, height,
                                ["CAMERA DISCONNECTED",
                                 "Check the OAK-D Lite cable.",
                                 "Reconnecting..."])
            cv2.imshow(WINDOW, frame)
            if (cv2.waitKey(200) & 0xFF) == ord("q"):
                break
            continue

        # ---- pull a preview frame ---------------------------------------
        try:
            frame = camera.get_preview()
        except CameraError as exc:
            print(f"[main] preview lost: {exc}")
            continue  # loop back into the reconnect path above
        if frame is None:
            # No frame yet (pipeline still warming up); keep the UI responsive.
            if (cv2.waitKey(15) & 0xFF) == ord("q"):
                break
            continue

        # ---- service the encoder every loop -----------------------------
        # Always drain the encoded queue; write it only while recording.
        try:
            encoded = camera.poll_encoded()
        except CameraError:
            encoded = []
        if recorder.active:
            recorder.write(encoded)

        # ---- handle a captured still ------------------------------------
        try:
            still = camera.poll_still()
        except CameraError:
            still = None
        if still is not None:
            path = storage.new_photo_path()
            ok = cv2.imwrite(path, still,
                             [cv2.IMWRITE_JPEG_QUALITY, storage.jpeg_quality])
            if ok:
                storage.count_photo()
                ui.flash()
                ui.toast(f"Photo saved ({still.shape[1]}x{still.shape[0]})")
            else:
                ui.toast("PHOTO SAVE FAILED")

        # ---- periodic disk-space check ----------------------------------
        now = time.monotonic()
        if now >= next_storage_check:
            next_storage_check = now + 5.0
            free_mb = storage.free_mb()
            low_storage = storage.low_storage()

        # ---- timelapse --------------------------------------------------
        if timelapse_on and now >= next_timelapse:
            next_timelapse = now + timelapse_interval
            if not low_storage:  # don't keep filling a near-full disk
                try:
                    camera.trigger_still()
                except CameraError:
                    pass

        # ---- process touch input ----------------------------------------
        for key in ui.take_events():
            if key == "photo":
                if low_storage:
                    ui.toast("LOW STORAGE - not capturing")
                else:
                    try:
                        camera.trigger_still()
                    except CameraError:
                        ui.toast("Capture failed")
            elif key == "record":
                if recorder.active:
                    ui.toast("Saving clip...")
                    mp4 = recorder.stop()
                    ui.toast("Clip saved" if mp4 else "Saved raw (mux failed)")
                elif low_storage:
                    ui.toast("LOW STORAGE - not recording")
                elif recorder.start():
                    ui.toast("Recording")
                else:
                    ui.toast("Could not start recording")
            elif key == "timelapse":
                timelapse_on = not timelapse_on
                if timelapse_on:
                    next_timelapse = now  # fire immediately
                    ui.toast(f"Timelapse every {timelapse_interval}s")
                else:
                    ui.toast("Timelapse off")
            elif key == "exit":
                running = False

        # ---- draw & present ---------------------------------------------
        state = {
            "recording": recorder.active,
            "rec_elapsed": recorder.elapsed,
            "timelapse": timelapse_on,
            "photo_count": storage.photo_count,
            "video_count": storage.video_count,
            "low_storage": low_storage,
            "free_mb": free_mb,
        }
        ui.draw(frame, state)
        cv2.imshow(WINDOW, frame)

        # 'q' on an attached keyboard is a developer escape hatch.
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            running = False

    # ---- clean shutdown -------------------------------------------------
    print("[main] shutting down")
    if recorder.active:
        recorder.stop()  # flush any in-progress recording
    camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] interrupted")
        sys.exit(0)
