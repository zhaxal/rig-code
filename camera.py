"""
camera.py - DepthAI v3 pipeline for the OAK-D Lite, built the way the official
v3 examples and the proven tracker.py do it: every node and queue is created on
a pipeline that the caller owns inside a `with dai.Pipeline() as pipeline:`
block, queues are created *before* pipeline.start(), and the host loop runs
while pipeline.isRunning().

`build_pipeline(pipeline, cfg)` wires three things off the single 13 MP RGB
sensor (CAM_A) and returns a Streams handle the host loop uses each frame:

  1. A small preview stream for the live screen.
  2. A continuously-running H.264/H.265 VideoEncoder (host decides when to
     write its output to disk - see storage.VideoRecorder).
  3. A full-resolution still, captured on demand: the full-res frames feed an
     on-device Script node that only forwards the latest frame to the host when
     we send it a trigger, so we are not streaming 13 MP frames continuously.

Keeping the device lifecycle (the `with` block) in the caller is what makes
reconnect reliable: leaving the block always releases the device, so a rebuild
never hits "no available sensor" from a leaked connection.
"""

import depthai as dai

# Runs on-device in the Script node: remember the most recent full-res frame and
# forward it to the host only when a trigger arrives.
_STILL_SCRIPT = """
latest = None
while True:
    f = node.inputs['in'].tryGet()
    if f is not None:
        latest = f
    trig = node.inputs['trigger'].tryGet()
    if trig is not None and latest is not None:
        node.io['still'].send(latest)
"""


class Streams:
    """Per-frame access to the running pipeline's queues."""

    def __init__(self, preview_q, encoded_q, still_q, trigger_q):
        self._preview_q = preview_q
        self._encoded_q = encoded_q
        self._still_q = still_q
        self._trigger_q = trigger_q
        self._last_preview = None

    @property
    def has_video(self):
        return self._encoded_q is not None

    @property
    def has_stills(self):
        return self._still_q is not None and self._trigger_q is not None

    def get_preview(self):
        """Freshest preview frame as a BGR numpy array, or the previous one if
        no new frame arrived this loop (returns None until the first frame)."""
        frames = self._preview_q.tryGetAll()
        if frames:
            self._last_preview = frames[-1].getCvFrame()  # newest, drop stale
        return self._last_preview

    def poll_encoded(self):
        """Encoded frames available this loop (possibly empty)."""
        if self._encoded_q is None:
            return []
        return self._encoded_q.tryGetAll()

    def trigger_still(self):
        """Ask the device to send back the latest full-resolution frame."""
        if self.has_stills:
            self._trigger_q.send(dai.Buffer())

    def poll_still(self):
        """A full-res still as a BGR numpy array if one arrived, else None."""
        if self._still_q is None:
            return None
        frame = self._still_q.tryGet()
        return frame.getCvFrame() if frame is not None else None


def build_pipeline(pipeline, cfg):
    """Create the nodes and queues on `pipeline` and return a Streams handle.

    Call this inside `with dai.Pipeline() as pipeline:` and *before*
    `pipeline.start()`. The encoder and still streams degrade gracefully: if the
    device can't provide one, that feature is disabled (the UI hides it) instead
    of the whole app failing."""
    pw, ph = cfg.get("preview_size", [800, 480])
    preview_fps = cfg.get("preview_fps", 25)
    vw, vh = cfg.get("video_size", [1920, 1080])
    video_fps = cfg.get("video_fps", 30)

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    # 1. Preview: small NV12 output downscaled on-device; getCvFrame() -> BGR.
    preview = cam.requestOutput((pw, ph), type=dai.ImgFrame.Type.NV12,
                                fps=preview_fps)
    preview_q = preview.createOutputQueue(maxSize=4, blocking=False)
    print("[camera] preview output OK")

    # 2. Continuous video encoder (host gates writing to disk).
    encoded_q = None
    try:
        video = cam.requestOutput((vw, vh), type=dai.ImgFrame.Type.NV12,
                                  fps=video_fps)
        profile = (dai.VideoEncoderProperties.Profile.H265_MAIN
                   if cfg.get("codec", "h265").lower() == "h265"
                   else dai.VideoEncoderProperties.Profile.H264_MAIN)
        encoder = pipeline.create(dai.node.VideoEncoder).build(
            video, frameRate=video_fps, profile=profile)
        try:
            encoder.setKeyframeFrequency(int(video_fps))  # ~1 keyframe/sec
        except Exception:
            pass
        encoded_q = encoder.out.createOutputQueue(
            maxSize=int(video_fps) * 2, blocking=False)
        print("[camera] video encoder OK")
    except Exception as exc:
        print(f"[camera] encoder unavailable, video disabled: {exc}")

    # 3. Full-resolution still through an on-device trigger Script node.
    still_q = trigger_q = None
    try:
        # Low fps: we only need stills on demand, so don't stream 13 MP fast.
        full = cam.requestFullResolutionOutput(fps=2)
        script = pipeline.create(dai.node.Script)
        try:
            script.inputs["in"].setBlocking(False)
            script.inputs["in"].setMaxSize(1)
        except Exception:
            pass
        full.link(script.inputs["in"])
        script.setScript(_STILL_SCRIPT)
        still_q = script.outputs["still"].createOutputQueue(
            maxSize=2, blocking=False)
        trigger_q = script.inputs["trigger"].createInputQueue()
        print("[camera] full-res still OK")
    except Exception as exc:
        print(f"[camera] full-res still unavailable, stills disabled: {exc}")

    return Streams(preview_q, encoded_q, still_q, trigger_q)
