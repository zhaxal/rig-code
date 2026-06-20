#!/usr/bin/env python3
"""Configuration loading and defaults for the spatial detection app.

All settings live in ``config.json`` next to this file. Every field is optional;
a missing field (or a missing/invalid file) falls back to the defaults below.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    # --- Display --------------------------------------------------------
    "screen_size": [800, 480],   # DSI touchscreen resolution [w, h]
    "fullscreen": True,          # borderless fullscreen + hidden cursor
    "capture_fps": 30,           # frame rate shared by all OAK outputs

    # --- Model ----------------------------------------------------------
    # default_model: a display name from models.discover_models(). When empty,
    # the first discovered model is used. Switch live from the GUI.
    "default_model": "",
    "bb_scale": 0.5,             # fraction of bbox sampled for depth
    "depth_lower_mm": 100,       # ignore spatial depth below this (mm)
    "depth_upper_mm": 5000,      # ignore spatial depth above this (mm)

    # --- Extra model-zoo IDs to offer in the picker ---------------------
    # Local NNArchives in models/ are always listed; these are appended.
    "zoo_models": ["yolov6-nano", "mobilenet-ssd"],

    # --- Capture --------------------------------------------------------
    "save_root": "~/captures",
    "photo_jpeg_quality": 95,
    "timelapse_interval_sec": 30,
    "low_storage_mb": 500,       # warn / block captures below this free space
    "note": "",                  # free-text note copied into session_meta.json
}


def load_config(path):
    """Return DEFAULTS merged with the JSON at ``path`` (keys starting with
    ``_`` are treated as comments and ignored)."""
    cfg = dict(DEFAULTS)
    try:
        with open(path) as fh:
            cfg.update({k: v for k, v in json.load(fh).items()
                        if not k.startswith("_")})
    except FileNotFoundError:
        print(f"[config] {path} not found, using defaults")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[config] could not read {path} ({exc}); using defaults")
    return cfg
