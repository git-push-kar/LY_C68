"""
test_face_routing.py
====================
Routing-contract tests for inference_faces.route_image: which image the model
is shown for a given set of detections.

Uses a stub detector, so these run without ultralytics, weights or a GPU. They
encode the plan's section 5.1 decisions, including the ordering bug this file
was written to pin: a lone face that fills the frame must take the passthrough
path even when its box is smaller than --min_face_px, because passthrough crops
nothing.

Run:  python test_face_routing.py
"""

import unittest
from types import SimpleNamespace

from PIL import Image

from face_detect import FaceBox
from inference_faces import route_image


def make_args(**overrides):
    base = dict(margin=0.30, min_face_px=96, max_faces=8, image_size=448,
                passthrough_min_face_frac=0.05)
    base.update(overrides)
    return SimpleNamespace(**base)


class StubDetector:
    """Stands in for FaceDetector; returns pre-set boxes."""

    def __init__(self, boxes):
        self.boxes = list(boxes)

    def detect(self, image):
        return list(self.boxes)


def image(w=200, h=200):
    return Image.new("RGB", (w, h), (120, 120, 120))


class TestRouting(unittest.TestCase):

    def test_portrait_at_measured_fraction_takes_passthrough(self):
        # omi.jpg: 417x530 face in 1644x1534 -> 0.0876, inside the portrait range
        route = route_image(image(1644, 1534),
                            StubDetector([FaceBox(584, 279, 1001, 809)]), make_args())
        self.assertEqual(route["path"], "passthrough")

    def test_training_style_crop_takes_passthrough_despite_small_box(self):
        # The real test_images/real/* case: 200x200 image, face box ~80x110.
        # min_face_px=96 would reject this box for CROPPING, but passthrough
        # must still win because the whole image is handed over uncropped.
        route = route_image(image(200, 200), StubDetector([FaceBox(60, 45, 140, 155)]),
                            make_args())
        self.assertEqual(route["path"], "passthrough")
        self.assertEqual(len(route["skipped"]), 0)
        self.assertEqual(route["items"][0]["kind"], "whole")

    def test_lone_small_face_in_a_large_scene_is_not_passthrough(self):
        # 400x600 face in a 4000x3000 scene = 2% of the frame, well under the 0.05
        # floor: the measured range for faces in wide scenes (chick.jpg: 0.021).
        route = route_image(image(4000, 3000), StubDetector([FaceBox(1000, 900, 1400, 1500)]),
                            make_args())
        self.assertEqual(route["path"], "per_face")
        self.assertGreater(len(route["items"]), 0)
        self.assertEqual(len(route["items"]), 1)
        self.assertEqual(route["items"][0]["kind"], "face")
        self.assertIn("margin", route["items"][0]["meta"])

    def test_lone_face_below_min_face_px_in_a_large_scene_falls_back(self):
        route = route_image(image(4000, 3000), StubDetector([FaceBox(1000, 900, 1080, 1000)]),
                            make_args())
        self.assertEqual(route["path"], "no_face")
        self.assertEqual(route["skipped"][0]["reason"], "face_too_small")
        self.assertEqual(route["items"][0]["kind"], "whole")

    def test_two_faces_route_per_face_with_crops(self):
        # Both boxes sit far enough inside the frame that the 30% margin does not
        # leave it, so these crops must not need padding at all.
        boxes = [FaceBox(300, 200, 600, 600), FaceBox(1200, 300, 1500, 700)]
        route = route_image(image(2000, 1000), StubDetector(boxes), make_args())
        self.assertEqual(route["path"], "per_face")
        self.assertEqual(len(route["items"]), 2)
        for i, item in enumerate(route["items"]):
            self.assertEqual(item["kind"], "face")
            self.assertEqual(item["id"], i)
            self.assertEqual(item["image"].size[0], item["image"].size[1])  # square
            self.assertEqual(item["meta"]["pad_mode"], "none")
            self.assertGreater(item["image"].size[0], item["box"].width)    # margin added

    def test_no_detections_route_to_no_face(self):
        route = route_image(image(), StubDetector([]), make_args())
        self.assertEqual(route["path"], "no_face")
        self.assertEqual(len(route["items"]), 1)
        self.assertEqual(route["items"][0]["kind"], "whole")

    def test_max_faces_truncates_and_reports_skips(self):
        boxes = [FaceBox(i * 100, 0, i * 100 + 200, 200) for i in range(4)]
        route = route_image(image(1000, 400), StubDetector(boxes),
                            make_args(max_faces=2, min_face_px=50))
        self.assertEqual(route["path"], "per_face")
        self.assertEqual(len(route["items"]), 2)
        self.assertEqual(sum(1 for s in route["skipped"] if s["reason"] == "over_max_faces"), 2)

    def test_no_detect_uses_the_whole_image(self):
        route = route_image(image(), None, make_args())
        self.assertEqual(route["path"], "no_detect")
        self.assertEqual(route["items"][0]["kind"], "whole")
        self.assertEqual(route["boxes"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
