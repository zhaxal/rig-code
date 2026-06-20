#!/usr/bin/env python3
"""Camera connection diagnostic for the OAK device (DepthAI v3).

Runs a series of escalating checks so we can see exactly *where* the connection
breaks instead of guessing. Run it on the Pi with the OAK attached:

    python3 diag_camera.py

Each stage prints PASS/FAIL. The first FAIL is the thing to fix.
"""

import sys
import time
import traceback


def hr(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def stage_import():
    hr("1. Import depthai + version")
    import depthai as dai
    print(f"   depthai version: {dai.__version__}")
    major = int(dai.__version__.split(".")[0])
    if major < 3:
        print(f"   FAIL: this app needs depthai>=3.0.0, found {dai.__version__}")
        return None
    print("   PASS")
    return dai


def stage_enumerate(dai):
    hr("2. Enumerate connected devices")
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("   FAIL: no OAK devices found.")
        print("   - Check USB cable (must be data-capable, ideally USB3 blue port)")
        print("   - Try a powered USB hub; OAK-D can brown out on Pi ports")
        print("   - On Linux, install udev rules:")
        print("       echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"03e7\", MODE=\"0666\"' "
              "| sudo tee /etc/udev/rules.d/80-movidius.rules")
        print("       sudo udevadm control --reload-rules && sudo udevadm trigger")
        return False
    for d in devices:
        print(f"   found: {d.getMxId()}  state={d.state}  protocol={d.protocol}")
    print("   PASS")
    return True


def stage_connect(dai):
    hr("3. Open device + check USB speed")
    try:
        with dai.Device() as device:
            try:
                speed = device.getUsbSpeed()
                print(f"   USB speed: {speed}")
                if str(speed).endswith("HIGH") or "SUPER" not in str(speed).upper():
                    if "SUPER" not in str(speed).upper():
                        print("   WARN: running at USB2 speed. Stereo + NN streams")
                        print("         can exceed USB2 bandwidth -> 'communication")
                        print("         exception' / stalls. Use a USB3 cable+port.")
            except Exception as exc:
                print(f"   (could not read USB speed: {exc})")
            print(f"   connected cameras: {device.getConnectedCameras()}")
        print("   PASS")
        return True
    except Exception as exc:
        print(f"   FAIL: {exc}")
        traceback.print_exc()
        return False


def stage_rgb(dai):
    hr("4. Minimal RGB-only pipeline (10 frames)")
    try:
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            q = cam.requestOutput((640, 400), dai.ImgFrame.Type.BGR888p,
                                  fps=30).createOutputQueue()
            pipeline.start()
            got = 0
            t0 = time.monotonic()
            while got < 10 and time.monotonic() - t0 < 10:
                if q.tryGet() is not None:
                    got += 1
            print(f"   received {got}/10 frames in {time.monotonic()-t0:.1f}s")
            if got == 0:
                print("   FAIL: camera connects but produces no RGB frames.")
                return False
        print("   PASS")
        return True
    except Exception as exc:
        print(f"   FAIL: {exc}")
        traceback.print_exc()
        return False


def stage_stereo(dai):
    hr("5. Stereo depth pipeline (10 depth frames)")
    try:
        with dai.Pipeline() as pipeline:
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=20)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=20)
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setExtendedDisparity(True)
            left.requestOutput((640, 400)).link(stereo.left)
            right.requestOutput((640, 400)).link(stereo.right)
            q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
            pipeline.start()
            got = 0
            t0 = time.monotonic()
            while got < 10 and time.monotonic() - t0 < 15:
                if q.tryGet() is not None:
                    got += 1
            print(f"   received {got}/10 depth frames in {time.monotonic()-t0:.1f}s")
            if got == 0:
                print("   FAIL: stereo cameras (CAM_B/CAM_C) not producing depth.")
                print("   - This board may not have stereo pair, or preset is wrong.")
                return False
        print("   PASS")
        return True
    except Exception as exc:
        print(f"   FAIL: {exc}")
        traceback.print_exc()
        return False


def stage_spatial(dai, model_ref="yolov6-nano"):
    hr(f"6. Full SpatialDetectionNetwork ({model_ref})")
    try:
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=20)
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=20)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=20)
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setExtendedDisparity(True)
            left.requestOutput((640, 400)).link(stereo.left)
            right.requestOutput((640, 400)).link(stereo.right)
            net = (pipeline.create(dai.node.SpatialDetectionNetwork)
                   .build(cam, stereo, dai.NNModelDescription(model_ref)))
            pq = net.passthrough.createOutputQueue(maxSize=4, blocking=False)
            print(f"   model loaded, classes: {len(net.getClasses() or [])}")
            pipeline.start()
            got = 0
            t0 = time.monotonic()
            while got < 10 and time.monotonic() - t0 < 30:
                if pq.tryGet() is not None:
                    got += 1
            print(f"   received {got}/10 passthrough frames in "
                  f"{time.monotonic()-t0:.1f}s")
            if got == 0:
                print("   FAIL: pipeline starts but no frames (the app's symptom).")
                return False
        print("   PASS")
        return True
    except Exception as exc:
        print(f"   FAIL: {exc}")
        traceback.print_exc()
        return False


def main():
    model_ref = sys.argv[1] if len(sys.argv) > 1 else "yolov6-nano"

    dai = stage_import()
    if dai is None:
        return 1
    if not stage_enumerate(dai):
        return 1
    if not stage_connect(dai):
        return 1
    if not stage_rgb(dai):
        return 1
    if not stage_stereo(dai):
        return 1
    if not stage_spatial(dai, model_ref):
        return 1

    hr("ALL STAGES PASSED")
    print("Camera + stereo + spatial detection all work in isolation.")
    print("If the app still fails, the issue is in the app loop, not the camera.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
