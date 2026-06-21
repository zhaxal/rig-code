#!/usr/bin/env python3
"""Frame overlays: detection boxes with class/confidence/X-Y-Z, and a status HUD.

All drawing is done on the BGR numpy frame with OpenCV before it is handed to the
tkinter Canvas, so the recorded video matches the live preview.
"""

import time

import cv2
import numpy as np

# BGR colours
_W = (255, 255, 255)
_BK = (0, 0, 0)
_RED = (40, 40, 220)
_GRN = (70, 180, 70)
_AMB = (40, 170, 230)


def _text(frame, s, org, color, scale):
    """Draw text with a black outline so it stays readable over any frame."""
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BK, 4, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def letterbox(frame, w, h):
    """Fit ``frame`` into a ``w``×``h`` canvas preserving its aspect ratio.

    Returns ``(canvas, placement)`` where ``placement`` is ``(ox, oy, sw, sh)``:
    the pixel rectangle the (scaled) frame occupies, centred with black bars
    filling the remainder. Detection coordinates are normalised to the original
    frame, so mapping them through ``placement`` keeps boxes aligned instead of
    stretching the image to fill a mismatched aspect ratio.
    """
    fh, fw = frame.shape[:2]
    if fw == 0 or fh == 0:
        return frame, (0, 0, w, h)
    scale = min(w / fw, h / fh)
    sw, sh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (sw, sh))
    ox, oy = (w - sw) // 2, (h - sh) // 2
    canvas = np.zeros((h, w, 3), dtype=frame.dtype)
    canvas[oy:oy + sh, ox:ox + sw] = resized
    return canvas, (ox, oy, sw, sh)


def draw_detections(frame, detections, labels, placement, rotate180=False):
    """Draw each detection's bounding box, label, confidence and spatial X/Y/Z.

    ``placement`` is ``(ox, oy, sw, sh)`` — the pixel rectangle the source image
    occupies in ``frame`` (see :func:`letterbox`). Detection coordinates are
    normalised to that source image, so they are mapped into the rectangle here.
    ``rotate180`` mirrors the coordinates to match a frame that was rotated 180°
    for display (camera mounted upside-down), keeping boxes on their objects.
    """
    ox, oy, sw, sh = placement
    for det in detections:
        x1n, y1n, x2n, y2n = det.xmin, det.ymin, det.xmax, det.ymax
        if rotate180:
            x1n, x2n = 1.0 - x2n, 1.0 - x1n
            y1n, y2n = 1.0 - y2n, 1.0 - y1n
        x1 = int(ox + x1n * sw)
        y1 = int(oy + y1n * sh)
        x2 = int(ox + x2n * sw)
        y2 = int(oy + y2n * sh)
        if x1 >= x2 or y1 >= y2:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), _GRN, 2)

        lbl = labels[det.label] if det.label < len(labels) else str(det.label)
        conf_text = f"{lbl} {det.confidence:.0%}"

        try:
            sc = det.spatialCoordinates
            # X/Y are measured in the (upside-down) sensor frame; the 180°
            # rotation flips their sign relative to the upright display. Z is
            # distance along the optical axis and is unaffected by the rotation.
            sx, sy = (-sc.x, -sc.y) if rotate180 else (sc.x, sc.y)
            coord_text = (f"X:{sx / 1000.0:+.2f} "
                          f"Y:{sy / 1000.0:+.2f} "
                          f"Z:{sc.z / 1000.0:.2f}m")
        except AttributeError:
            coord_text = None

        ty = max(y1 - 6, 14)
        _text(frame, conf_text, (x1, ty), _GRN, 0.5)
        if coord_text:
            _text(frame, coord_text, (x1, max(ty - 18, 14)), _AMB, 0.45)


def draw_hud(frame, st):
    """Top-of-frame status: model, FPS, REC timer, time-lapse, counts, warnings."""
    now = time.monotonic()
    w = frame.shape[1]

    # Top-left: model name + FPS
    _text(frame, f"{st['model']}  {st['fps']:.0f} FPS", (12, 26), _W, 0.55)

    # Recording indicator (blinking dot + elapsed)
    if st["recording"]:
        if int(now * 2) % 2 == 0:
            cv2.circle(frame, (24, 50), 10, _RED, -1)
        s = int(st["rec_elapsed"])
        _text(frame, f"REC {s // 60:02d}:{s % 60:02d}", (44, 56), _RED, 0.7)

    if st["timelapse"]:
        _text(frame, "TIMELAPSE ON", (w // 2 - 90, 26), _AMB, 0.6)

    # Top-right: photo / clip counts
    cnt = f"Photos:{st['photos']}  Clips:{st['clips']}"
    (tw, _), _ = cv2.getTextSize(cnt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    _text(frame, cnt, (w - tw - 12, 26), _W, 0.55)

    if st["low_storage"]:
        f = st["free_mb"]
        msg = "LOW STORAGE" + (f" ({int(f)} MB)" if f is not None else "")
        _text(frame, msg, (12, frame.shape[0] - 14), _RED, 0.7)

    # Transient toast, centred near the bottom
    toast = st.get("toast")
    if toast:
        (tw, th), _ = cv2.getTextSize(toast, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x, y = (w - tw) // 2, frame.shape[0] - 40
        cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), _BK, -1)
        _text(frame, toast, (x, y), _W, 0.7)
