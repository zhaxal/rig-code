#!/usr/bin/env python3
"""Video recorder: writes annotated preview frames to an .mp4 via cv2.VideoWriter.

What you see on the preview (boxes + X/Y/Z overlay) is exactly what gets recorded.
"""

import os
import time
from datetime import datetime

import cv2


def stamp(millis=False):
    now = datetime.now()
    if millis:
        return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    return now.strftime("%Y%m%d_%H%M%S")


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
