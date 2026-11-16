"""
test_passthrough_parity.py
==========================
Pins the plan's section 5.1 guarantee: a single face covering the frame must be
classified **exactly** as inference.py classifies it.

Two layers, because they fail differently:

  1. TestPassthroughContract (always runs, no weights, no GPU)
     Records the tensors inference_faces.classify sends to the model and
     compares them bitwise with the tensors inference.py's own loop builds
     (build_transform(448) + compute_fft_features(224), unsqueeze'd, batch of 1).
     This is what catches a future refactor that resizes, re-crops, pads or
     otherwise reshapes the passthrough input -- the kind of change that leaves
     verdicts plausible and no longer identical.

  2. TestPassthroughBitwiseWithModel (opt-in: RUN_MODEL_PARITY_TESTS=1)
     Runs the real DeepfakeReasoningModel on real single-face images and asserts
     torch.equal against the reference forward. Slow (loads ~4.4 GB), so it is
     not part of the default run.

Run:
    python test_passthrough_parity.py
    RUN_MODEL_PARITY_TESTS=1 python test_passthrough_parity.py
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import compute_fft_features
from face_detect import FaceBox
from inference import build_transform
from inference_faces import _autocast_ctx, classify, route_image

IMAGE_SIZE = 448
FFT_SIZE = 224
CHECKPOINT = "./runs/intern_fused_ground/checkpoints/best.pth"
BASE_MODEL = "./models/internvl3_2b_lr_fused"


def make_image(w=400, h=400):
    """Deterministic gradient -- asymmetric so any shift/crop changes values."""
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ys = np.linspace(0, 255, h, dtype=np.uint8)
    arr = np.stack([np.tile(xs, (h, 1)), np.tile(ys[:, None], (1, w)),
                    np.full((h, w), 90, dtype=np.uint8)], axis=-1)
    return Image.fromarray(arr)


def make_args():
    return SimpleNamespace(margin=0.30, min_face_px=96, max_faces=8,
                           image_size=IMAGE_SIZE, passthrough_min_face_frac=0.05)


class StubDetector:
    def __init__(self, boxes):
        self.boxes = list(boxes)

    def detect(self, image):
        return list(self.boxes)


class RecordingModel:
    """Captures the tensors it is called with and returns deterministic logits."""

    def __init__(self):
        self.calls = []

    def __call__(self, pixel_values, freq_features=None, **kwargs):
        self.calls.append({"pv": pixel_values.detach().clone(),
                           "freq": None if freq_features is None
                                   else freq_features.detach().clone()})
        # Deterministic, input-dependent: identical inputs -> identical logits.
        means = pixel_values.mean(dim=(1, 2, 3))
        stds = pixel_values.std(dim=(1, 2, 3))
        return {"cls_logits": torch.stack([means, stds], dim=-1)}


class TestPassthroughContract(unittest.TestCase):
    """The passthrough tensor must be the one inference.py builds."""

    def setUp(self):
        self.image = make_image()
        # 280x280 face in a 400x400 frame -> 49% of the frame -> passthrough
        self.route = route_image(self.image, StubDetector([FaceBox(60, 60, 340, 340)]),
                                 make_args())
        self.transform = build_transform(IMAGE_SIZE)

    def reference_tensors(self):
        """Exactly what inference.py's loop does."""
        pv = self.transform(self.image).unsqueeze(0)
        freq = compute_fft_features(self.image, target_size=FFT_SIZE).unsqueeze(0)
        return pv, freq

    def test_route_is_passthrough(self):
        self.assertEqual(self.route["path"], "passthrough")
        self.assertEqual(len(self.route["items"]), 1)
        self.assertEqual(self.route["items"][0]["kind"], "whole")

    def test_pixel_values_are_bitwise_identical_to_inference_py(self):
        model = RecordingModel()
        _, _, pv = classify(model, [self.route["items"][0]["image"]],
                            self.transform, torch.device("cpu"))
        ref_pv, _ = self.reference_tensors()
        self.assertEqual(tuple(pv.shape), (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertTrue(torch.equal(pv.cpu(), ref_pv),
                        "passthrough pixel_values drifted from inference.py's preprocessing")

    def test_fft_features_are_bitwise_identical_to_inference_py(self):
        model = RecordingModel()
        _, freq, _ = classify(model, [self.route["items"][0]["image"]],
                              self.transform, torch.device("cpu"))
        _, ref_freq = self.reference_tensors()
        self.assertEqual(tuple(freq.shape), (1, 3, FFT_SIZE, FFT_SIZE))
        self.assertTrue(torch.equal(freq, ref_freq),
                        "passthrough FFT features drifted from inference.py's preprocessing")

    def test_model_receives_exactly_those_tensors(self):
        model = RecordingModel()
        classify(model, [self.route["items"][0]["image"]], self.transform,
                 torch.device("cpu"))
        self.assertEqual(len(model.calls), 1)
        ref_pv, ref_freq = self.reference_tensors()
        self.assertTrue(torch.equal(model.calls[0]["pv"], ref_pv))
        self.assertTrue(torch.equal(model.calls[0]["freq"], ref_freq))

    def test_probabilities_match_a_reference_forward(self):
        model = RecordingModel()
        probs, _, _ = classify(model, [self.route["items"][0]["image"]],
                               self.transform, torch.device("cpu"))
        ref_pv, ref_freq = self.reference_tensors()
        with torch.no_grad():
            ref_logits = model(ref_pv, freq_features=ref_freq)["cls_logits"].float()
        expected = F.softmax(ref_logits, dim=-1)
        self.assertTrue(torch.equal(probs, expected))

    def test_the_guard_actually_discriminates(self):
        """A crop path must NOT look like the passthrough tensor.

        Without this, a bug that routed everything through passthrough (or
        cropped everything) could pass the tests above.
        """
        two = route_image(self.image,
                          StubDetector([FaceBox(20, 20, 150, 200), FaceBox(250, 60, 380, 240)]),
                          make_args())
        self.assertEqual(two["path"], "per_face")
        crop_pv, _ = self.reference_tensors()
        crop = self.transform(two["items"][0]["image"]).unsqueeze(0)
        self.assertEqual(tuple(crop.shape), tuple(crop_pv.shape))
        self.assertFalse(torch.equal(crop, crop_pv))


@unittest.skipUnless(
    os.environ.get("RUN_MODEL_PARITY_TESTS") == "1" and os.path.isfile(CHECKPOINT),
    "set RUN_MODEL_PARITY_TESTS=1 (needs the checkpoint) to run the full-model parity test")
class TestPassthroughBitwiseWithModel(unittest.TestCase):
    """Full proof: the real model, the real single-face images, torch.equal."""

    @classmethod
    def setUpClass(cls):
        from adapter_loader import load_model
        from face_detect import FaceDetector

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model, _, _ = load_model(base_model=BASE_MODEL, checkpoint=CHECKPOINT,
                                     device=str(cls.device))
        cls.detector = FaceDetector("./models/yolov8n-face.pt")
        cls.transform = build_transform(IMAGE_SIZE)

    IMAGES = ["test_images/real/161022.jpg", "test_images/real/310151.png",
              "test_images/omi.jpg"]

    def test_single_face_images_are_bitwise_identical(self):
        for path in self.IMAGES:
            if not os.path.isfile(path):
                self.skipTest(f"missing {path}")
            image = Image.open(path).convert("RGB")
            route = route_image(image, self.detector, make_args())
            self.assertEqual(route["path"], "passthrough", f"{path} did not take passthrough")

            # reference: inference.py's loop, INCLUDING its bf16 autocast around
            # the classifier forward (comparing without it introduces a real
            # numerical difference that is not a bug)
            pv = self.transform(image).unsqueeze(0).to(self.device)
            freq = compute_fft_features(image, target_size=FFT_SIZE).unsqueeze(0).to(self.device)
            with torch.no_grad(), _autocast_ctx(self.device):
                ref = F.softmax(self.model(pv, freq_features=freq)["cls_logits"].float(),
                                dim=-1).cpu()

            probs, _, _ = classify(self.model, [it["image"] for it in route["items"]],
                                   self.transform, self.device)
            self.assertTrue(torch.equal(ref, probs),
                            f"{path}: passthrough output differs from inference.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
