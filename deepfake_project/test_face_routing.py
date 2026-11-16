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
from inference_faces import (aggregate_verdict, find_disagreements,
                             lone_face_fills_frame, route_image,
                             select_reason_targets)


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

    def test_lone_face_rule_is_shared_with_the_floor_sweep(self):
        self.assertTrue(lone_face_fills_frame([0.05], 0.05))     # floor is inclusive
        self.assertFalse(lone_face_fills_frame([0.049], 0.05))
        self.assertFalse(lone_face_fills_frame([0.60, 0.55], 0.05))  # two faces, never
        self.assertFalse(lone_face_fills_frame([], 0.05))

    def test_whole_image_rows_never_have_a_null_id(self):
        # flags.reasoned_faces is built from rows[i]["id"], so a null id on the
        # passthrough row surfaced in the JSON as reasoned_faces: [null].
        routes = [
            route_image(image(200, 200), StubDetector([FaceBox(60, 45, 140, 155)]), make_args()),
            route_image(image(), StubDetector([]), make_args()),
            route_image(image(), None, make_args()),
        ]
        for route in routes:
            self.assertIsInstance(route["items"][0]["id"], int, route["path"])
        self.assertEqual([r["items"][0]["id"] for r in routes], [0, 0, 0])


class TestAggregation(unittest.TestCase):
    """Document-verdict rules (plan section 5)."""

    ROWS = [{"p_fake": 0.10}, {"p_fake": 0.72}, {"p_fake": 0.40}]

    def test_max_takes_the_worst_face(self):
        self.assertEqual(aggregate_verdict(self.ROWS, "max", 0.50), "FAKE")
        self.assertEqual(aggregate_verdict(self.ROWS, "any_fake", 0.50), "FAKE")

    def test_mean_can_differ_from_max(self):
        # mean = 0.407 -> REAL at 0.50 although one face is FAKE
        self.assertEqual(aggregate_verdict(self.ROWS, "mean", 0.50), "REAL")

    def test_per_face_has_no_document_verdict(self):
        self.assertIsNone(aggregate_verdict(self.ROWS, "per_face", 0.50))

    def test_threshold_is_respected(self):
        rows = [{"p_fake": 0.42}]
        self.assertEqual(aggregate_verdict(rows, "max", 0.42), "FAKE")
        self.assertEqual(aggregate_verdict(rows, "max", 0.50), "REAL")

    def test_empty_rows_have_no_verdict(self):
        self.assertIsNone(aggregate_verdict([], "max", 0.50))


class TestDisagreement(unittest.TestCase):
    """LLM/classifier conflict detection (INTEGRATION_GUIDE section 9)."""

    @staticmethod
    def row(verdict, answer):
        return {"verdict": verdict, "p_fake": 0.9,
                "reasoning": {"generated_answer": answer}}

    def test_agreement_is_not_a_disagreement(self):
        rows = [self.row("FAKE", "fake"), self.row("REAL", "real")]
        self.assertEqual(find_disagreements(rows, [0, 1]), [])

    def test_contradiction_is_flagged(self):
        rows = [self.row("FAKE", "real"), self.row("REAL", "fake")]
        self.assertEqual(find_disagreements(rows, [0, 1]), [0, 1])

    def test_unparsed_answer_is_not_a_contradiction(self):
        rows = [self.row("FAKE", "unknown")]
        self.assertEqual(find_disagreements(rows, [0]), [])

    def test_unreasoned_rows_are_ignored(self):
        rows = [{"verdict": "FAKE", "p_fake": 0.9, "reasoning": None}]
        self.assertEqual(find_disagreements(rows, []), [])


class TestReasonTargets(unittest.TestCase):
    """Default = reason about the highest-risk face only."""

    ROWS = [{"p_fake": 0.04}, {"p_fake": 0.97}, {"p_fake": 0.31}]

    def test_top1_picks_the_highest_p_fake(self):
        self.assertEqual(select_reason_targets(self.ROWS, "top1"), [1])

    def test_top1_ties_break_deterministically(self):
        rows = [{"p_fake": 0.5}, {"p_fake": 0.5}]
        self.assertEqual(select_reason_targets(rows, "top1"), [0])
        self.assertEqual(select_reason_targets(rows, "all"), [0, 1])

    def test_all_reasons_every_face_worst_first(self):
        self.assertEqual(select_reason_targets(self.ROWS, "all"), [1, 2, 0])

    def test_none_reasons_about_nothing(self):
        self.assertEqual(select_reason_targets(self.ROWS, "none"), [])

    def test_empty_rows(self):
        self.assertEqual(select_reason_targets([], "top1"), [])

    def test_single_passthrough_row_is_the_top1(self):
        self.assertEqual(select_reason_targets([{"p_fake": 0.16}], "top1"), [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
