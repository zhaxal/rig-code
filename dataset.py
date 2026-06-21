#!/usr/bin/env python3
"""COCO dataset writer: turn live detections into a training dataset on disk.

Each *capture* saves a **clean** image (no boxes drawn) plus the model's current
bounding boxes as COCO object-detection annotations. One ``CocoWriter`` owns one
model's dataset; because different models have different class sets, every model
gets its own dataset directory so their ``categories`` never collide.

On-disk layout (``dataset_dir`` per model)::

    dataset_dir/
      images/<timestamp>.jpg
      annotations.json        # {images, annotations, categories} - standard COCO

The writer resumes an existing ``annotations.json`` (ids continue from the max
already present), so captures accumulate across app restarts. ``annotations.json``
is rewritten atomically after every capture to survive a kill mid-write.

Detection boxes are normalised (0-1) to the frame the network ran on; ``rotate180``
mirrors them to match a 180°-rotated (upside-down camera) image, exactly like the
overlay does, so the saved boxes line up with the saved pixels.

Bounding boxes use the COCO convention ``bbox = [x, y, width, height]`` in absolute
pixels with a top-left origin. ``category_id`` is the model's own class index.
``score`` (the detection confidence) is an extra, non-standard field kept so weak
pseudo-labels can be filtered later; standard COCO loaders ignore it.
"""

import json
import os

import cv2

from recorder import stamp


class CocoWriter:
    def __init__(self, dataset_dir, label_map, rotate180=True,
                 min_conf=0.5, jpeg_quality=95):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.json_path = os.path.join(dataset_dir, "annotations.json")
        self.rotate180 = rotate180
        self.min_conf = float(min_conf)
        self.jpeg_quality = int(jpeg_quality)
        os.makedirs(self.images_dir, exist_ok=True)

        self._images = []
        self._annotations = []
        self._next_img_id = 1
        self._next_ann_id = 1
        self._load_existing()

        # Categories come from this model's class list. Preserve any already on
        # disk (a resumed dataset keeps its original category set).
        if not self._categories:
            self._categories = [
                {"id": i, "name": name, "supercategory": "object"}
                for i, name in enumerate(label_map or [])
            ]

    @property
    def count(self):
        """Number of images (captures) in the dataset."""
        return len(self._images)

    def _load_existing(self):
        self._categories = []
        if not os.path.exists(self.json_path):
            return
        try:
            with open(self.json_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return  # treat an unreadable file as a fresh dataset
        self._images = data.get("images", [])
        self._annotations = data.get("annotations", [])
        self._categories = data.get("categories", [])
        if self._images:
            self._next_img_id = max(im["id"] for im in self._images) + 1
        if self._annotations:
            self._next_ann_id = max(a["id"] for a in self._annotations) + 1

    def _coco_box(self, det, w, h):
        """Map a normalised detection to a COCO ``[x, y, w, h]`` pixel box, or
        ``None`` if it is degenerate. Mirrors the overlay's rotate180 flip."""
        x1n, y1n, x2n, y2n = det.xmin, det.ymin, det.xmax, det.ymax
        if self.rotate180:
            x1n, x2n = 1.0 - x2n, 1.0 - x1n
            y1n, y2n = 1.0 - y2n, 1.0 - y1n
        x1 = min(max(x1n * w, 0.0), w)
        y1 = min(max(y1n * h, 0.0), h)
        x2 = min(max(x2n * w, 0.0), w)
        y2 = min(max(y2n * h, 0.0), h)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return None
        return [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)]

    def add_sample(self, image_bgr, detections):
        """Save one clean image and append its annotations. Returns the number of
        boxes written (0 is a valid negative sample). Raises on image-write
        failure so the caller can surface it."""
        h, w = image_bgr.shape[:2]
        name = f"{stamp(millis=True)}.jpg"
        path = os.path.join(self.images_dir, name)
        if not cv2.imwrite(path, image_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
            raise IOError(f"could not write {path}")

        img_id = self._next_img_id
        self._next_img_id += 1
        self._images.append({
            "id": img_id,
            "file_name": os.path.join("images", name),
            "width": w,
            "height": h,
        })

        boxes = 0
        for det in detections or []:
            if det.confidence < self.min_conf:
                continue
            bbox = self._coco_box(det, w, h)
            if bbox is None:
                continue
            self._annotations.append({
                "id": self._next_ann_id,
                "image_id": img_id,
                "category_id": int(det.label),
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 2),
                "iscrowd": 0,
                "score": round(float(det.confidence), 4),
            })
            self._next_ann_id += 1
            boxes += 1

        self._flush()
        return boxes

    def _flush(self):
        """Atomically rewrite annotations.json (temp file + os.replace)."""
        data = {
            "images": self._images,
            "annotations": self._annotations,
            "categories": self._categories,
        }
        tmp = self.json_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, self.json_path)
