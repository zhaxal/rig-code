#!/usr/bin/env python3
"""Model discovery for the GUI picker.

A *model* is described by a small ``ModelEntry``:
  - ``name``   : display name shown in the picker
  - ``kind``   : "archive" (local NNArchive file) or "zoo" (Luxonis model-zoo ID)
  - ``ref``    : absolute path (archive) or zoo ID string (zoo)

The camera worker turns ``ref`` into a DepthAI model descriptor when it (re)builds
the pipeline, so this module stays import-light and usable without DepthAI present.
"""

import glob
import os
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

ModelEntry = namedtuple("ModelEntry", ["name", "kind", "ref"])


def discover_models(zoo_models=None):
    """Return a list of ModelEntry: every ``*.tar.xz`` in models/, then the
    configured model-zoo IDs. Order defines the picker order."""
    entries = []

    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.tar.xz"))):
        entries.append(ModelEntry(name=os.path.basename(path),
                                  kind="archive", ref=path))

    seen = {e.name for e in entries}
    for zoo_id in (zoo_models or []):
        zoo_id = zoo_id.strip()
        if zoo_id and zoo_id not in seen:
            entries.append(ModelEntry(name=zoo_id, kind="zoo", ref=zoo_id))
            seen.add(zoo_id)

    return entries


def pick_default(entries, default_name=""):
    """Choose the starting model: the configured ``default_name`` if present,
    else the first discovered entry, else None."""
    default_name = (default_name or "").strip()
    if default_name:
        for e in entries:
            if e.name == default_name:
                return e
    return entries[0] if entries else None
