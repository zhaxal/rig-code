"""
capture.py - DepthAI v3 pipeline wrapper for the OAK-D Lite.

One CameraManager owns a single pipeline that produces three things at once
from the 13 MP RGB sensor (CAM_A):

  1. A downscaled BGR preview stream for the live screen.
  2. A full-resolution still, captured on demand via a Script node that only
     forwards a frame to the host when we send it a trigger (so we are not
     streaming 13 MP frames over USB continuously).
  3. A continuously-running H.264/H.265 VideoEncoder; the host decides when to
     write its output to disk (see storage.VideoRecorder).

All device interaction is wrapped so a camera disconnect raises CameraError
instead of taking down the app; main.py catches it and rebuilds the pipeline.
"""

import depthai as dai


class CameraError(RuntimeError):
    """Raised when the camera disconnects or the pipeline fails."""


# Script run on-device: keep only the latest full-res frame and forward it to
# the host only when a trigger arrives. Avoids constant 13 MP USB traffic.
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


def _out_queue(output, max_size=4, blocking=False):
    """createOutputQueue with non-blocking defaults, tolerant of API changes."""
    try:
        return output.createOutputQueue(maxSize=max_size, blocking=blocking)
    except TypeError:
        return output.createOutputQueue()


class CameraManager:
    def __init__(self, config):
        self.cfg = config
        self.pipeline = None
        self._preview_q = None
        self._still_q = None
        self._trigger_q = None
        self._encoded_q = None
        self._last_preview = None  # reused if no fresh frame this loop

    # ---- lifecycle --------------------------------------------------------

    def build(self):
        """Create and start the pipeline. Raises CameraError on failure.

        Built in named stages so a failure reports exactly which output the
        OAK-D Lite rejected. The full three-stream config (preview + full-res
        still + 1080p encode) is more than the single-output docs examples, so
        if the device can't sustain it we fall back to a 2-stream config and,
        last, a preview-only config rather than failing outright."""
        stage = "create pipeline"
        try:
            # Enter the pipeline's context just like tracker.py's
            # `with dai.Pipeline() as pipeline:`. In DepthAI v3 the context
            # owns the device connection/run-state; a bare pipeline can
            # start() yet report isRunning() == False, which lands main.py in
            # its "camera disconnected" reconnect loop.
            pipeline = dai.Pipeline().__enter__()
            stage = "build Camera(CAM_A)"
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

            stage = "preview requestOutput"
            self._build_preview(cam)
            print("[camera] preview output OK")

            stage = "video encoder"
            try:
                self._build_encoder(cam, pipeline)
                print("[camera] video encoder OK")
            except Exception as exc:
                print(f"[camera] encoder unavailable, video disabled: {exc}")

            stage = "full-res still"
            try:
                self._build_still(cam, pipeline)
                print("[camera] full-res still OK")
            except Exception as exc:
                print(f"[camera] full-res still unavailable, "
                      f"stills disabled: {exc}")

            stage = "pipeline start"
            pipeline.start()
            self.pipeline = pipeline
            print("[camera] pipeline started")
        except Exception as exc:  # any DepthAI/XLink error during bring-up
            self.close()
            raise CameraError(f"pipeline build failed at [{stage}]: {exc}") from exc

    def _build_preview(self, cam):
        """Preview output. Tries interleaved BGR, falls back to planar."""
        pw, ph = self.cfg.get("preview_size", [800, 480])
        fps = self.cfg.get("preview_fps", 25)
        last = None
        for t in (dai.ImgFrame.Type.BGR888i, dai.ImgFrame.Type.BGR888p):
            try:
                preview = cam.requestOutput(size=(pw, ph), type=t, fps=fps)
                self._preview_q = _out_queue(preview, max_size=4)
                return
            except Exception as exc:
                last = exc
        raise last

    def _build_encoder(self, cam, pipeline):
        """Continuous video encoder (host gates writing to disk)."""
        vw, vh = self.cfg.get("video_size", [1920, 1080])
        fps = self.cfg.get("video_fps", 30)
        video = cam.requestOutput(size=(vw, vh), type=dai.ImgFrame.Type.NV12, fps=fps)
        profile = (dai.VideoEncoderProperties.Profile.H265_MAIN
                   if self.cfg.get("codec", "h265").lower() == "h265"
                   else dai.VideoEncoderProperties.Profile.H264_MAIN)
        encoder = pipeline.create(dai.node.VideoEncoder).build(
            video, frameRate=fps, profile=profile)
        # A keyframe every second => recording starts quickly & cleanly.
        try:
            encoder.setKeyframeFrequency(int(fps))
        except Exception:
            pass
        self._encoded_q = _out_queue(encoder.out, max_size=int(fps) * 2)

    def _build_still(self, cam, pipeline):
        """Full-resolution still through a trigger Script node."""
        full = cam.requestFullResolutionOutput(useHighestResolution=True)
        script = pipeline.create(dai.node.Script)
        # Don't let unconsumed full-res frames back-pressure the sensor.
        try:
            script.inputs["in"].setBlocking(False)
            script.inputs["in"].setMaxSize(1)
        except Exception:
            pass
        full.link(script.inputs["in"])
        script.setScript(_STILL_SCRIPT)
        self._still_q = _out_queue(script.outputs["still"], max_size=2)
        self._trigger_q = script.inputs["trigger"].createInputQueue()

    def is_alive(self):
        try:
            return self.pipeline is not None and self.pipeline.isRunning()
        except Exception:
            return False

    def close(self):
        """Stop and release the pipeline; safe to call repeatedly."""
        if self.pipeline is not None:
            try:
                # Mirror leaving the `with dai.Pipeline()` block: __exit__
                # stops the pipeline and releases the device.
                self.pipeline.__exit__(None, None, None)
            except Exception:
                pass
        self.pipeline = None
        self._preview_q = self._still_q = None
        self._trigger_q = self._encoded_q = None

    # ---- per-loop operations ---------------------------------------------

    def get_preview(self):
        """Latest preview frame as a BGR numpy array, or the previous one if no
        new frame is ready. Raises CameraError if the device dropped."""
        if not self.is_alive():
            raise CameraError("device not running")
        try:
            frames = self._preview_q.tryGetAll()
            if frames:
                self._last_preview = frames[-1].getCvFrame()  # freshest, drop stale
        except Exception as exc:
            raise CameraError(f"preview read failed: {exc}") from exc
        return self._last_preview

    # Capability flags so the UI/loop can skip disabled features.
    @property
    def has_stills(self):
        return self._trigger_q is not None and self._still_q is not None

    @property
    def has_video(self):
        return self._encoded_q is not None

    def trigger_still(self):
        """Ask the device to send back the latest full-resolution frame."""
        if not self.has_stills:
            return  # stills disabled (full-res output unavailable)
        try:
            self._trigger_q.send(dai.Buffer())
        except Exception as exc:
            raise CameraError(f"still trigger failed: {exc}") from exc

    def poll_still(self):
        """Return a full-res still as a BGR numpy array if one arrived, else None."""
        if self._still_q is None:
            return None
        try:
            frame = self._still_q.tryGet()
            return frame.getCvFrame() if frame is not None else None
        except Exception as exc:
            raise CameraError(f"still read failed: {exc}") from exc

    def poll_encoded(self):
        """Return a list of encoded frames available this loop (possibly empty)."""
        if self._encoded_q is None:
            return []
        try:
            return self._encoded_q.tryGetAll()
        except Exception as exc:
            raise CameraError(f"encoded read failed: {exc}") from exc
