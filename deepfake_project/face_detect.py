"""
face_detect.py
==============
YOLO face detection + context cropping for multi-face images.

This module is **routing only**: it decides *which image* DeepfakeReasoningModel
sees. The model, its weights and its 448/224 preprocessing are untouched, so a
single face still goes through the whole-image path the model was validated on.

Two layers, deliberately separated:

  1. Pure geometry (no torch, no ultralytics): ``expand_box``, ``square_box``,
     ``overflow_fraction``, ``crop_with_context``, ``filter_faces``. These are
     unit-tested without model weights -- see test_face_pipeline.py.
  2. ``FaceDetector``: a thin ultralytics wrapper, imported lazily so the
     geometry layer stays dependency-free.

Crop contract (plan section 3):
    detector box -> expand by ``margin`` on every side -> square it about its
    centre -> clamp to the image, reflecting at most ``REFLECT_MAX_FRAC`` of the
    side -> else fall back to the clamped rectangle.

The square step exists because the model's transform is
``Resize((448, 448))`` + ``CenterCrop``: a non-square crop is *distorted*, not
letterboxed. Reflecting is capped because synthetic mirrored skin becomes
high-frequency "evidence" for the FFT branch, which the model never saw in
training. Beyond the cap we accept mild aspect distortion instead.

Usage:
    from face_detect import FaceDetector, filter_faces, crop_with_context

    det = FaceDetector("./models/yolov8n-face.pt")
    boxes = det.detect(image)
    kept, skipped = filter_faces(boxes, image.size, min_face_px=96, max_faces=8)
    crop, meta = crop_with_context(image, kept[0], margin=0.30)
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ── Defaults (mirrored by the face_pipeline: block in config.yaml) ────────────

DEFAULT_MARGIN = 0.30            # "a little background" around the detector box
DEFAULT_TARGET_SIZE = 448        # model image_size, used only for the upscale report
DEFAULT_MIN_FACE_PX = 96         # shorter box side, in ORIGINAL pixels
DEFAULT_MAX_FACES = 8            # top-K by area
REFLECT_MAX_FRAC = 0.10          # max overflow, as a fraction of the square side
DEFAULT_DETECTOR_IMGSZ = 640
DEFAULT_DETECTOR_CONF = 0.25
DEFAULT_DETECTOR_IOU = 0.50
DEFAULT_MAX_DETECT_PIXELS = 4096  # downscale huge inputs before detection


# ── Box model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FaceBox:
    """Axis-aligned face box in ORIGINAL image pixels."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"FaceBox needs x2>=x1 and y2>=y1, got "
                f"({self.x1}, {self.y1}, {self.x2}, {self.y2})")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def clamped(self, img_w: int, img_h: int) -> "FaceBox":
        """Clip to the image rectangle."""
        return FaceBox(
            min(max(self.x1, 0.0), float(img_w)),
            min(max(self.y1, 0.0), float(img_h)),
            min(max(self.x2, 0.0), float(img_w)),
            min(max(self.y2, 0.0), float(img_h)),
            self.score,
        )

    def as_list(self) -> List[float]:
        return [round(self.x1, 2), round(self.y1, 2),
                round(self.x2, 2), round(self.y2, 2)]

    def to_dict(self) -> Dict[str, Any]:
        return {"box": self.as_list(), "score": round(float(self.score), 4)}


# ── Pure geometry ────────────────────────────────────────────────────────────

def expand_box(box: FaceBox, margin: float) -> FaceBox:
    """Pad every side by ``margin`` x that side's length.

    margin=0.30 keeps a ring of background around the face, which is what the
    training crops contain (~256px head-and-shoulders) and what makes the
    ear/hairline/jaw blending seams -- a named telemetry region -- visible.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    return FaceBox(box.x1 - margin * box.width,
                   box.y1 - margin * box.height,
                   box.x2 + margin * box.width,
                   box.y2 + margin * box.height,
                   box.score)


def square_box(box: FaceBox) -> FaceBox:
    """Grow the shorter side about the box centre until the box is square."""
    side = max(box.width, box.height)
    half = side / 2.0
    return FaceBox(box.cx - half, box.cy - half, box.cx + half, box.cy + half, box.score)


def overflow_fraction(box: FaceBox, img_w: int, img_h: int) -> float:
    """Largest distance the box extends outside the image, / its longest side."""
    side = max(box.width, box.height)
    if side <= 0:
        return 0.0
    over = max(0.0, -box.x1, -box.y1, box.x2 - img_w, box.y2 - img_h)
    return over / side


def _reflect_pads(core: Tuple[int, int, int, int],
                  x0: int, y0: int, side: int) -> Optional[Tuple[int, int, int, int]]:
    """Padding needed to lift ``core`` to the square canvas, or None if impossible.

    numpy's reflect mode requires each pad to be smaller than the axis it
    mirrors. On a very small image the pad can exceed the clamped core, so this
    returns None and the caller degrades to a plain clamped crop (``stretch``)
    instead of raising -- a tiny face must never abort the run.
    """
    left, top = core[0] - x0, core[1] - y0
    right = side - left - (core[2] - core[0])
    bottom = side - top - (core[3] - core[1])
    core_w, core_h = core[2] - core[0], core[3] - core[1]
    if min(left, top, right, bottom) < 0 or core_w < 1 or core_h < 1:
        return None
    if left >= core_w or right >= core_w or top >= core_h or bottom >= core_h:
        return None
    return left, top, right, bottom


def _pad_reflect(arr: np.ndarray,
                 left: int, top: int, right: int, bottom: int) -> np.ndarray:
    """Reflect-pad an HWC array. numpy requires each pad < that axis length."""
    h, w = arr.shape[:2]
    if left >= w or right >= w or top >= h or bottom >= h:
        raise ValueError(
            f"reflect pad {left, top, right, bottom} too large for {w}x{h} crop")
    pad = ((top, bottom), (left, right)) + ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, pad, mode="reflect")


def crop_with_context(image: Image.Image,
                      box: FaceBox,
                      margin: float = DEFAULT_MARGIN,
                      target_size: int = DEFAULT_TARGET_SIZE,
                      reflect_max_frac: float = REFLECT_MAX_FRAC
                      ) -> Tuple[Image.Image, Dict[str, Any]]:
    """Crop one face with its background ring. No resizing happens here.

    Returns ``(crop, meta)`` where ``meta["pad_mode"]`` is:

      ``"none"``    the square crop was fully inside the image
      ``"reflect"`` overflow within ``reflect_max_frac`` -> mirrored padding
      ``"stretch"`` overflow beyond the cap -> clamped rectangle, downstream
                    ``Resize((448, 448))`` accepts mild distortion rather than
                    inventing mirrored skin for the FFT branch to read

    ``meta["reflect_rejected"]`` is True when reflect was *wanted* (overflow
    within the cap) but the image was too small to mirror that far; the crop
    still succeeds, it just does not get to be square.
    """
    img_w, img_h = image.size
    expanded = expand_box(box, margin)
    squared = square_box(expanded)
    side = squared.width
    if side < 1.0:
        raise ValueError(f"degenerate crop box (side={side:.2f}) for {box}")

    x0, y0 = int(round(squared.x1)), int(round(squared.y1))
    x1, y1 = int(round(squared.x2)), int(round(squared.y2))
    side_i = max(x1 - x0, y1 - y0, 1)
    x1, y1 = x0 + side_i, y0 + side_i          # exact square after rounding

    frac = overflow_fraction(squared, img_w, img_h)
    core = (max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1))
    if core[2] <= core[0] or core[3] <= core[1]:
        raise ValueError(
            f"box does not intersect the {img_w}x{img_h} image: {box.as_list()}")

    reflect_rejected = False
    if x0 >= 0 and y0 >= 0 and x1 <= img_w and y1 <= img_h:
        crop, pad_mode = image.crop(core), "none"
    elif frac <= reflect_max_frac and (pads := _reflect_pads(core, x0, y0, side_i)):
        arr = _pad_reflect(np.asarray(image.crop(core)), *pads)
        crop, pad_mode = Image.fromarray(arr), "reflect"
    else:
        # Either too much overflow (synthetic mirrored skin would feed the FFT
        # branch) or the pad is wider than a small image can mirror. Both fall
        # back to the clamped rectangle rather than failing the run.
        crop, pad_mode = image.crop(core), "stretch"
        reflect_rejected = frac <= reflect_max_frac

    w, h = crop.size
    meta = {
        **box.to_dict(),
        "margin": margin,
        "pad_mode": pad_mode,
        "reflect_rejected": reflect_rejected,
        "overflow_frac": round(frac, 4),
        "crop_size": [w, h],
        "face_frac": round(box.area / float(img_w * img_h), 4),
        # >2.0 means the face was upscaled into 448: reported, never hidden
        "upscale": round(target_size / max(1, min(w, h)), 3),
    }
    return crop, meta


def filter_faces(boxes: Sequence[FaceBox],
                 image_size: Tuple[int, int],
                 min_face_px: int = DEFAULT_MIN_FACE_PX,
                 max_faces: int = DEFAULT_MAX_FACES
                 ) -> Tuple[List[FaceBox], List[Dict[str, Any]]]:
    """Clip, drop faces smaller than ``min_face_px``, keep the ``max_faces`` largest.

    Returns ``(kept, skipped)``; every rejection carries a machine-readable
    reason so callers can report "detected but not judged" instead of silently
    returning a clean verdict for a face it never evaluated.
    """
    img_w, img_h = image_size
    kept: List[FaceBox] = []
    skipped: List[Dict[str, Any]] = []

    for b in boxes:
        cb = b.clamped(img_w, img_h)
        if cb.width < min_face_px or cb.height < min_face_px:
            skipped.append({
                **b.to_dict(),
                "reason": "face_too_small",
                "size": [round(cb.width, 1), round(cb.height, 1)],
            })
            continue
        kept.append(cb)

    kept.sort(key=lambda b: b.area, reverse=True)
    if max_faces >= 0 and len(kept) > max_faces:
        skipped.extend({**b.to_dict(), "reason": "over_max_faces"}
                       for b in kept[max_faces:])
        kept = kept[:max_faces]
    return kept, skipped


def draw_boxes(image: Image.Image,
               boxes: Sequence[FaceBox],
               color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """Return a copy with boxes drawn on -- for eyeballing detections."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(2, int(round(max(out.size) / 400)))
    for i, b in enumerate(boxes):
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=width)
        draw.text((b.x1 + width, max(0.0, b.y1 - 14)),
                  f"#{i} {b.score:.2f}", fill=color)
    return out


# ── Detector (ultralytics, imported lazily) ──────────────────────────────────

@contextmanager
def _float32_default():
    """Pin torch's default dtype to float32 inside the block, then restore it.

    DeepfakeReasoningModel is built in bfloat16, and building it leaves the
    process defaulting to bfloat16. ultralytics' fuse_conv_and_bn creates a zero
    bias via ``torch.zeros(...)`` for bias-free Conv layers, which inherits that
    default -- so fusion ends up multiplying a bf16 zero-bias by an fp32 weight
    matrix and raises "expected m1 and m2 to have the same dtype". Detection has
    no reason to inherit the classifier's dtype, so it pins its own.
    """
    import torch  # local import keeps the geometry layer usable without torch

    prev = torch.get_default_dtype()
    if prev == torch.float32:
        yield
        return
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

class FaceDetector:
    """YOLO face detector. Heavy imports stay inside ``__init__``."""

    def __init__(self,
                 weights: str = "./models/yolov8n-face.pt",
                 imgsz: int = DEFAULT_DETECTOR_IMGSZ,
                 conf: float = DEFAULT_DETECTOR_CONF,
                 iou: float = DEFAULT_DETECTOR_IOU,
                 device: Optional[str] = None,
                 max_detect_pixels: int = DEFAULT_MAX_DETECT_PIXELS):
        try:
            from ultralytics import YOLO  # noqa: WPS433 (lazy by design)
        except ImportError as exc:  # pragma: no cover - environment guidance
            raise ImportError(
                "ultralytics is required for FaceDetector "
                "(pip install 'ultralytics==8.3.*' --no-deps)") from exc

        self.weights = weights
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.max_detect_pixels = int(max_detect_pixels)
        with _float32_default():
            self.model = YOLO(weights)
            self.model.model.float()

    def _prepare(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        """Downscale oversized inputs; return the array and the box rescale factor."""
        img = image.convert("RGB")
        scale = 1.0
        longest = max(img.size)
        if self.max_detect_pixels and longest > self.max_detect_pixels:
            scale = self.max_detect_pixels / float(longest)
            new_size = (max(1, int(round(img.width * scale))),
                        max(1, int(round(img.height * scale))))
            img = img.resize(new_size, Image.LANCZOS)
        return np.asarray(img), scale

    def detect(self, image: Image.Image) -> List[FaceBox]:
        """All detected faces, highest score first, in ORIGINAL pixel coordinates."""
        arr, scale = self._prepare(image)
        kwargs: Dict[str, Any] = dict(source=arr, imgsz=self.imgsz, conf=self.conf,
                                      iou=self.iou, verbose=False)
        if self.device is not None:
            kwargs["device"] = self.device
        # predict() fuses conv+bn; fusion builds a zero bias with torch.zeros(),
        # which follows the *default* dtype. Pin fp32 for the call and restore.
        with _float32_default():
            results = self.model.predict(**kwargs)
        if not results:
            return []
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        scores = res.boxes.conf.detach().cpu().numpy()
        inv = 1.0 / scale if scale else 1.0
        boxes = [FaceBox(float(x1) * inv, float(y1) * inv,
                         float(x2) * inv, float(y2) * inv, float(s))
                 for (x1, y1, x2, y2), s in zip(xyxy, scores)]
        boxes.sort(key=lambda b: b.score, reverse=True)
        return boxes
