#!/usr/bin/env python3
"""Model discovery for the GUI picker.

A *model* is described by a small ``ModelEntry``:
  - ``name``   : display name shown in the picker
  - ``ref``    : absolute path to the NNArchive file

The camera worker turns ``ref`` into a DepthAI model descriptor when it (re)builds
the pipeline, so this module stays import-light and usable without DepthAI present.
"""

import glob
import os
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

ModelEntry = namedtuple("ModelEntry", ["name", "ref"])


def discover_models():
    """Return a list of ModelEntry for every ``*.tar.xz`` in models/."""
    return [
        ModelEntry(name=os.path.basename(path), ref=path)
        for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.tar.xz")))
    ]


def pick_default(entries, default_name=""):
    """Choose the starting model: the configured ``default_name`` if present,
    else the first discovered entry, else None."""
    default_name = (default_name or "").strip()
    if default_name:
        for e in entries:
            if e.name == default_name:
                return e
    return entries[0] if entries else None
