#!/usr/bin/env python3

# Official Luxonis DepthAI v3 example, verbatim:
# https://docs.luxonis.com/software-v3/depthai/examples/camera/camera_output
# Minimal "does the camera work" test. If this prints frames, the OAK-D Lite
# and its sensor are healthy and we build the app back up on top of it.

import cv2
import depthai as dai

# Create pipeline
with dai.Pipeline() as pipeline:
    # Define source and output
    cam = pipeline.create(dai.node.Camera).build()
    videoQueue = cam.requestOutput((640, 400)).createOutputQueue()

    # Connect to device and start pipeline
    pipeline.start()
    while pipeline.isRunning():
        videoIn = videoQueue.get()
        assert isinstance(videoIn, dai.ImgFrame)
        cv2.imshow("video", videoIn.getCvFrame())

        if cv2.waitKey(1) == ord("q"):
            break
