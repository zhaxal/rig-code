"""
storage.py - Session folders, filenames, metadata, video muxing and disk checks.

Layout produced per launch:

    <save_root>/<YYYYMMDD_HHMMSS>/        one session folder per launch
        session_meta.json                 settings + free-text note + counts
        photo_YYYYMMDD_HHMMSS_mmm.jpg      sequential timestamped stills
        vid_YYYYMMDD_HHMMSS_mmm.mp4        recorded clips
        vid_YYYYMMDD_HHMMSS_mmm.h265       raw bitstream (kept until remuxed)

Everything here is deliberately defensive: a greenhouse capture rig should never
lose data or crash because of a full disk or a failed mux. None of it touches
the camera, so it's plain Python and independently testable.
"""

import json
import os
import shutil
import subprocess
import time
from datetime import datetime


def _timestamp(millis=False):
    """Filesystem-safe timestamp. millis=True adds _mmm for unique filenames."""
    now = datetime.now()
    if millis:
        return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    return now.strftime("%Y%m%d_%H%M%S")


class SessionStorage:
    """Owns one session folder and hands out timestamped file paths."""

    def __init__(self, config):
        self.cfg = config
        self.codec = config.get("codec", "h265").lower()
        self.jpeg_quality = int(config.get("photo_jpeg_quality", 95))
        self.low_storage_mb = int(config.get("low_storage_mb", 500))

        save_root = os.path.expanduser(config.get("save_root", "~/captures"))
        self.session_dir = os.path.join(save_root, _timestamp())
        os.makedirs(self.session_dir, exist_ok=True)

        self.photo_count = 0
        self.video_count = 0
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._write_meta()

    # ---- metadata ---------------------------------------------------------

    def _write_meta(self):
        """(Re)write session_meta.json atomically, at start and after each
        capture, so the file on disk always reflects reality."""
        meta = {
            "session_started": self.started_at,
            "session_updated": datetime.now().isoformat(timespec="seconds"),
            "note": self.cfg.get("note", ""),
            "settings": {
                "video_size": self.cfg.get("video_size", [1920, 1080]),
                "video_fps": self.cfg.get("video_fps", 30),
                "codec": self.codec,
                "preview_size": self.cfg.get("preview_size", [800, 480]),
                "timelapse_interval_sec": self.cfg.get("timelapse_interval_sec", 30),
            },
            "photo_count": self.photo_count,
            "video_count": self.video_count,
        }
        path = os.path.join(self.session_dir, "session_meta.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(meta, fh, indent=2)
            os.replace(tmp, path)  # rename so the JSON is never half-written
        except OSError as exc:
            print(f"[storage] could not write session_meta.json: {exc}")

    # ---- file paths -------------------------------------------------------

    def new_photo_path(self):
        return os.path.join(self.session_dir, f"photo_{_timestamp(millis=True)}.jpg")

    def new_video_paths(self):
        """Return (raw_bitstream_path, mp4_path) sharing the same timestamp."""
        base = f"vid_{_timestamp(millis=True)}"
        ext = "h265" if self.codec == "h265" else "h264"
        raw = os.path.join(self.session_dir, f"{base}.{ext}")
        mp4 = os.path.join(self.session_dir, f"{base}.mp4")
        return raw, mp4

    # ---- counters ---------------------------------------------------------

    def count_photo(self):
        self.photo_count += 1
        self._write_meta()

    def count_video(self):
        self.video_count += 1
        self._write_meta()

    # ---- disk space -------------------------------------------------------

    def free_mb(self):
        """Free megabytes on the volume holding the session folder."""
        try:
            return shutil.disk_usage(self.session_dir).free / (1024 * 1024)
        except OSError:
            return None  # unknown - don't block, but the UI warns

    def low_storage(self):
        free = self.free_mb()
        return free is not None and free < self.low_storage_mb


class VideoRecorder:
    """Writes the encoder's raw bitstream to disk, then remuxes to .mp4.

    The encoder runs continuously in the pipeline; this class only decides
    *when* to write frames. Writing the raw bitstream straight to disk (and
    flushing each batch) means a crash or power loss mid-recording still leaves
    a recoverable file we can mux later.
    """

    def __init__(self, storage, fps):
        self.storage = storage
        self.fps = int(fps)
        self.active = False
        self._fh = None
        self.raw_path = None
        self.mp4_path = None
        self.started_at = None
        self._keyframe_seen = False

    @property
    def elapsed(self):
        if not self.active or self.started_at is None:
            return 0.0
        return time.monotonic() - self.started_at

    def start(self):
        self.raw_path, self.mp4_path = self.storage.new_video_paths()
        try:
            self._fh = open(self.raw_path, "wb")
        except OSError as exc:
            print(f"[recorder] cannot open {self.raw_path}: {exc}")
            self._fh = None
            return False
        self.active = True
        self.started_at = time.monotonic()
        self._keyframe_seen = False  # wait for first keyframe -> clean start
        return True

    def write(self, encoded_frames):
        """Write a list of encoded frames (from Streams.poll_encoded())."""
        if not self.active or self._fh is None:
            return
        try:
            for frame in encoded_frames:
                if not self._keyframe_seen:
                    if _is_keyframe(frame):
                        self._keyframe_seen = True
                    else:
                        continue  # skip leading P-frames before first keyframe
                frame.getData().tofile(self._fh)
            self._fh.flush()  # minimise data loss on unexpected power-off
        except (OSError, ValueError) as exc:
            print(f"[recorder] write error: {exc}")

    def stop(self):
        """Close the raw file and remux to .mp4. Returns the mp4 path or None."""
        if not self.active:
            return None
        self.active = False
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
            self._fh = None

        mp4 = _remux_to_mp4(self.raw_path, self.mp4_path, self.fps)
        if mp4:
            self.storage.count_video()
        return mp4


def _is_keyframe(frame):
    """True if this encoded frame is an I-frame (keyframe). Falls back to True
    if the API is unavailable, so a clip just starts mid-GOP but stays valid.
    Imports depthai lazily so storage stays testable without the hardware."""
    try:
        import depthai as dai
        return frame.getFrameType() == dai.EncodedFrame.FrameType.I
    except Exception:
        return True


def _remux_to_mp4(raw_path, mp4_path, fps):
    """Wrap the raw H.264/H.265 elementary stream in an .mp4 container with a
    lossless stream copy. On success the raw bitstream is deleted; on failure it
    is kept so nothing is lost."""
    if not raw_path or not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        print("[recorder] no data captured; nothing to mux")
        return None
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", raw_path,
           "-c", "copy", mp4_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # FileNotFoundError => ffmpeg not installed. Keep the raw stream.
        print(f"[recorder] ffmpeg remux failed ({exc}); keeping raw {raw_path}")
        return None
    try:
        os.remove(raw_path)  # mux succeeded, raw no longer needed
    except OSError:
        pass
    return mp4_path
