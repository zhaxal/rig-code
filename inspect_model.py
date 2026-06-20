#!/usr/bin/env python3
"""Print NNArchive output tensor names and shapes. Run before implementing parser."""

import os
import sys
import depthai as dai

HERE = os.path.dirname(os.path.abspath(__file__))

path = sys.argv[1] if len(sys.argv) > 1 else "models/exp.rvc2.tar.xz"
if not os.path.isabs(path):
    path = os.path.join(HERE, path)

print(f"Inspecting: {path}\n")
archive = dai.NNArchive(path)

try:
    config = archive.getConfig()
    print(f"Model type : {config.model.metadata.get('type', 'unknown')}")
    print(f"Classes    : {config.model.metadata.get('classes', [])}\n")
except Exception:
    pass

try:
    for head in archive.getConfig().model.heads:
        print(f"Head: {head}")
except Exception:
    pass

print("Output tensors:")
try:
    for output in archive.getOutputsInfo():
        print(f"  name={output.name!r}  shape={output.dims}  dtype={output.dataType}")
except Exception as exc:
    print(f"  (could not read via getOutputsInfo: {exc})")

print("\nRaw config JSON:")
try:
    import json
    print(json.dumps(archive.getConfig().model.__dict__, indent=2, default=str))
except Exception as exc:
    print(f"  ({exc})")
