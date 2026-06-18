"""
ui.py - Touch UI drawn directly onto the preview frame with OpenCV.

No GUI toolkit: a fullscreen OpenCV window plus large drawn buttons keeps the
dependency surface tiny and the app single-threaded and crash-resistant. On the
official DSI touchscreen, taps arrive as OpenCV mouse-down events, so the same
hit-testing works for touch and mouse.
"""

import time

import cv2

# Colours are BGR (OpenCV convention).
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_RED = (40, 40, 220)
_GREEN = (70, 180, 70)
_GREY = (60, 60, 60)
_AMBER = (40, 170, 230)
_BAR_H = 96  # button bar height in pixels


class Button:
    def __init__(self, key, label, x1, x2, y1, y2):
        self.key = key
        self.label = label
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class TouchUI:
    """Lays out a bottom button bar and draws status overlays."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        self._pending = []          # queued button keys from taps
        self._flash_until = 0.0     # white capture-flash effect timer
        self._toast = ""            # transient on-screen message
        self._toast_until = 0.0

        # Four equal buttons across the bottom: Photo, Record, Timelapse, Exit.
        keys = [("photo", "PHOTO"), ("record", "REC"),
                ("timelapse", "TIME-LAPSE"), ("exit", "EXIT")]
        bw = width // len(keys)
        y1, y2 = height - _BAR_H, height
        self.buttons = []
        for i, (key, label) in enumerate(keys):
            x1 = i * bw
            x2 = (i + 1) * bw if i < len(keys) - 1 else width
            self.buttons.append(Button(key, label, x1, x2, y1, y2))

    # ---- input ------------------------------------------------------------

    def on_mouse(self, event, x, y, flags, param):
        """OpenCV mouse/touch callback. Registers a tap on a button."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in self.buttons:
                if btn.contains(x, y):
                    self._pending.append(btn.key)
                    return

    def take_events(self):
        """Return and clear queued button presses since last call."""
        events, self._pending = self._pending, []
        return events

    # ---- feedback ---------------------------------------------------------

    def flash(self):
        """Briefly whiten the screen to confirm a still capture."""
        self._flash_until = time.monotonic() + 0.12

    def toast(self, message, seconds=2.0):
        self._toast = message
        self._toast_until = time.monotonic() + seconds

    # ---- drawing ----------------------------------------------------------

    def draw(self, frame, state):
        """Overlay buttons + status onto frame (modified in place)."""
        now = time.monotonic()

        # Capture flash: briefly blend the frame toward white.
        if now < self._flash_until:
            white = frame.copy()
            white[:] = 255
            cv2.addWeighted(frame, 0.4, white, 0.6, 0, frame)

        self._draw_topbar(frame, state)
        self._draw_buttons(frame, state)
        self._draw_toast(frame, now)
        return frame

    def _draw_topbar(self, frame, state):
        # Recording indicator + elapsed time (left).
        if state.get("recording"):
            blink_on = int(time.monotonic() * 2) % 2 == 0
            if blink_on:
                cv2.circle(frame, (24, 28), 12, _RED, -1)
            secs = int(state.get("rec_elapsed", 0))
            txt = f"REC {secs // 60:02d}:{secs % 60:02d}"
            _text(frame, txt, (44, 36), _RED, scale=0.8)

        # Capture count (right).
        count = f"Photos:{state.get('photo_count', 0)}  Clips:{state.get('video_count', 0)}"
        (tw, _), _ = cv2.getTextSize(count, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        _text(frame, count, (self.w - tw - 12, 30), _WHITE, scale=0.6)

        # Timelapse active badge (centre-top).
        if state.get("timelapse"):
            _text(frame, "TIMELAPSE ON", (self.w // 2 - 90, 30), _AMBER, scale=0.7)

        # Low-storage warning (just above the button bar).
        if state.get("low_storage"):
            free = state.get("free_mb")
            msg = "LOW STORAGE" + (f" ({int(free)} MB)" if free is not None else "")
            _text(frame, msg, (12, self.h - _BAR_H - 16), _RED, scale=0.8)

    def _draw_buttons(self, frame, state):
        for btn in self.buttons:
            # Highlight active toggles.
            active = ((btn.key == "record" and state.get("recording")) or
                      (btn.key == "timelapse" and state.get("timelapse")))
            fill = _RED if (btn.key == "record" and state.get("recording")) else (
                _GREEN if active else _GREY)
            cv2.rectangle(frame, (btn.x1, btn.y1), (btn.x2 - 2, btn.y2), fill, -1)
            cv2.rectangle(frame, (btn.x1, btn.y1), (btn.x2 - 2, btn.y2), _BLACK, 2)

            label = btn.label
            if btn.key == "record" and state.get("recording"):
                label = "STOP"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cx = btn.x1 + (btn.x2 - btn.x1 - tw) // 2
            cy = btn.y1 + (_BAR_H + th) // 2
            _text(frame, label, (cx, cy), _WHITE, scale=0.8)

    def _draw_toast(self, frame, now):
        if now < self._toast_until and self._toast:
            (tw, th), _ = cv2.getTextSize(self._toast, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = (self.w - tw) // 2
            y = self.h - _BAR_H - 40
            cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12),
                          _BLACK, -1)
            _text(frame, self._toast, (x, y), _WHITE, scale=0.8)


def _text(frame, text, org, color, scale=0.7, thickness=2):
    """Draw text with a black outline so it reads over any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK,
                thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)
