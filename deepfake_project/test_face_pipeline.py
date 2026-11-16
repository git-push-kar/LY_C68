"""
test_face_pipeline.py
=====================
Geometry tests for face_detect.py. No torch, no ultralytics, no weights:
these cover the arithmetic that decides what the model is shown, so they run
in a second and catch the failures that would otherwise surface as silently
misframed faces.

Run:  python test_face_pipeline.py
"""

import unittest

import numpy as np
from PIL import Image

from face_detect import (
    FaceBox,
    _reflect_pads,
    crop_with_context,
    draw_boxes,
    expand_box,
    filter_faces,
    overflow_fraction,
    square_box,
)


def make_image(w=400, h=300):
    """Deterministic gradient image; distinct pixels make crop offsets visible."""
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ys = np.linspace(0, 255, h, dtype=np.uint8)
    arr = np.stack([np.tile(xs, (h, 1)), np.tile(ys[:, None], (1, w)),
                    np.full((h, w), 128, dtype=np.uint8)], axis=-1)
    return Image.fromarray(arr)


class TestBoxMath(unittest.TestCase):

    def test_expand_box_pads_each_side_by_margin(self):
        b = FaceBox(100, 100, 200, 200)          # 100x100
        e = expand_box(b, 0.30)
        self.assertAlmostEqual(e.x1, 70.0)
        self.assertAlmostEqual(e.y1, 70.0)
        self.assertAlmostEqual(e.x2, 230.0)
        self.assertAlmostEqual(e.y2, 230.0)
        self.assertAlmostEqual(e.score, b.score)

    def test_expand_box_is_proportional_for_wide_boxes(self):
        b = FaceBox(100, 100, 300, 200)          # 200x100
        e = expand_box(b, 0.25)
        self.assertAlmostEqual(e.width, 300.0)   # +25% of 200 on both sides
        self.assertAlmostEqual(e.height, 150.0)  # +25% of 100 on both sides

    def test_expand_box_rejects_negative_margin(self):
        with self.assertRaises(ValueError):
            expand_box(FaceBox(0, 0, 10, 10), -0.1)

    def test_square_box_grows_short_side_about_centre(self):
        b = FaceBox(100, 100, 300, 200)          # 200x100, centre (200,150)
        s = square_box(b)
        self.assertAlmostEqual(s.width, 200.0)
        self.assertAlmostEqual(s.height, 200.0)
        self.assertAlmostEqual(s.cx, 200.0)
        self.assertAlmostEqual(s.cy, 150.0)

    def test_square_box_is_identity_when_already_square(self):
        b = FaceBox(10, 20, 110, 120)
        s = square_box(b)
        self.assertEqual((s.x1, s.y1, s.x2, s.y2), (10.0, 20.0, 110.0, 120.0))

    def test_box_rejects_inverted_coordinates(self):
        with self.assertRaises(ValueError):
            FaceBox(100, 0, 50, 10)

    def test_overflow_fraction(self):
        img_w, img_h = 400, 300
        inside = FaceBox(100, 100, 200, 200)
        self.assertEqual(overflow_fraction(inside, img_w, img_h), 0.0)
        # 10 px past the left edge on a 100 px square -> 0.10
        self.assertAlmostEqual(overflow_fraction(FaceBox(-10, 100, 90, 200), img_w, img_h), 0.10)
        # 30 px past the bottom edge on a 100x160 box -> measured against the
        # LONGEST side (160), so 30/160 = 0.1875, not 30/100
        self.assertAlmostEqual(overflow_fraction(FaceBox(100, 170, 200, 330), img_w, img_h), 0.1875)


class TestCropWithContext(unittest.TestCase):

    def test_centred_face_takes_plain_crop(self):
        img = make_image()
        crop, meta = crop_with_context(img, FaceBox(150, 100, 250, 200), margin=0.30, target_size=448)
        self.assertEqual(meta["pad_mode"], "none")
        self.assertEqual(crop.size, (160, 160))          # 100 * 1.6, square
        self.assertAlmostEqual(meta["margin"], 0.30)
        self.assertAlmostEqual(meta["upscale"], 448 / 160, places=3)
        self.assertAlmostEqual(meta["face_frac"], round(100 * 100 / (400 * 300), 4), places=4)

    def test_aspect_ratio_of_the_crop_never_depends_on_the_face(self):
        img = make_image(600, 600)
        wide, _ = crop_with_context(img, FaceBox(100, 250, 500, 350))   # 400x100 face
        tall, _ = crop_with_context(img, FaceBox(250, 100, 350, 500))   # 100x400 face
        self.assertEqual(wide.size[0], wide.size[1])
        self.assertEqual(tall.size[0], tall.size[1])
        self.assertEqual(wide.size, tall.size)

    def test_near_edge_face_uses_reflect_padding(self):
        img = make_image()                               # 400x300
        # 80 px face at x=16: expanded box starts at -8 -> 6.25% overflow
        crop, meta = crop_with_context(img, FaceBox(16, 100, 96, 180), margin=0.30)
        self.assertEqual(meta["pad_mode"], "reflect")
        self.assertEqual(crop.size[0], crop.size[1])
        self.assertEqual(crop.size, (128, 128))
        self.assertLessEqual(meta["overflow_frac"], 0.10)

    def test_reflect_padding_mirrors_rather_than_filling_black(self):
        img = make_image()
        crop, meta = crop_with_context(img, FaceBox(16, 100, 96, 180), margin=0.30)
        self.assertEqual(meta["pad_mode"], "reflect")
        arr = np.asarray(crop)
        # Left 8 px are mirrored from the image edge, never a constant fill.
        pad = arr[:, :8, :].reshape(-1, 3)
        self.assertGreater(len(np.unique(pad, axis=0)), 1, "pad is flat fill, not mirrored")

    def test_far_edge_face_falls_back_to_clamped_rectangle(self):
        img = make_image(200, 150)
        # 100 px face flush at the origin: overflow 30/160 = 18.75% > 10% cap
        crop, meta = crop_with_context(img, FaceBox(0, 0, 100, 100), margin=0.30)
        self.assertEqual(meta["pad_mode"], "stretch")
        self.assertGreater(meta["overflow_frac"], 0.10)
        self.assertEqual(crop.size, (130, 130))          # clamped, no invention

    def test_crop_never_exceeds_the_image(self):
        img = make_image(120, 90)
        for box in (FaceBox(0, 0, 120, 90), FaceBox(110, 80, 120, 90), FaceBox(0, 80, 10, 90)):
            crop, meta = crop_with_context(img, box, margin=0.30)
            self.assertLessEqual(crop.size[0], 120, meta)
            self.assertLessEqual(crop.size[1], 90, meta)

    def test_margin_increases_upscale_report(self):
        img = make_image(1000, 1000)
        _, tight = crop_with_context(img, FaceBox(400, 400, 600, 600), margin=0.0)
        _, loose = crop_with_context(img, FaceBox(400, 400, 600, 600), margin=0.5)
        self.assertLess(loose["upscale"], tight["upscale"])

    def test_degenerate_box_is_rejected(self):
        with self.assertRaises(ValueError):
            crop_with_context(make_image(), FaceBox(50, 50, 50, 50), margin=0.3)

    def test_box_outside_the_image_is_rejected_clearly(self):
        with self.assertRaises(ValueError) as ctx:
            crop_with_context(make_image(100, 100), FaceBox(200, 200, 260, 260), margin=0.3)
        self.assertIn("does not intersect", str(ctx.exception))

    def test_meta_always_reports_whether_reflect_was_rejected(self):
        _, meta = crop_with_context(make_image(), FaceBox(150, 100, 250, 200), margin=0.30)
        self.assertIn("reflect_rejected", meta)
        self.assertFalse(meta["reflect_rejected"])


class TestReflectFallback(unittest.TestCase):
    """Small images must degrade to a clamped crop instead of raising."""

    def test_pads_accepted_when_within_the_core(self):
        # 8 px past the left edge of a 120x128 core -> mirroring 8 px is fine
        self.assertEqual(_reflect_pads((0, 76, 120, 204), -8, 76, 128), (8, 0, 0, 0))

    def test_pads_rejected_when_wider_than_the_core(self):
        # a 4x4 core cannot be mirrored by 6 px on either side
        self.assertIsNone(_reflect_pads((0, 0, 4, 4), -6, -6, 16))

    def test_pads_rejected_for_degenerate_core(self):
        self.assertIsNone(_reflect_pads((5, 5, 5, 5), 0, 0, 8))

    def test_pads_rejected_on_negative_padding(self):
        self.assertIsNone(_reflect_pads((0, 0, 10, 10), 2, 2, 8))

    def test_tiny_images_never_raise(self):
        for size in ((1, 1), (2, 2), (3, 5), (7, 4), (12, 9), (16, 16), (32, 3), (200, 4)):
            img = make_image(*size)
            w, h = size
            for box in (FaceBox(0, 0, max(1, w - 1), max(1, h - 1)),
                        FaceBox(0, 0, max(1, w // 2), max(1, h // 2))):
                with self.subTest(size=size, box=box.as_list()):
                    crop, meta = crop_with_context(img, box, margin=0.30)
                    self.assertGreaterEqual(crop.size[0], 1)
                    self.assertGreaterEqual(crop.size[1], 1)
                    self.assertIn(meta["pad_mode"], ("none", "reflect", "stretch"))
                    self.assertIn("reflect_rejected", meta)


class TestFilterFaces(unittest.TestCase):

    def test_small_face_is_skipped_with_reason(self):
        boxes = [FaceBox(10, 10, 60, 60),          # 50 px -> too small
                 FaceBox(100, 100, 300, 300)]      # 200 px -> kept
        kept, skipped = filter_faces(boxes, (400, 400), min_face_px=96)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].width, 200.0)
        self.assertEqual(skipped[0]["reason"], "face_too_small")
        self.assertEqual(skipped[0]["size"], [50.0, 50.0])

    def test_kept_faces_are_sorted_by_area_not_score(self):
        big_low = FaceBox(0, 0, 300, 300, score=0.30)
        small_high = FaceBox(320, 0, 390, 70, score=0.99)
        kept, _ = filter_faces([small_high, big_low], (400, 400), min_face_px=50, max_faces=2)
        self.assertEqual([b.width for b in kept], [300.0, 70.0])

    def test_max_faces_truncates_largest_first(self):
        boxes = [FaceBox(0, 0, 100, 100), FaceBox(110, 0, 310, 200),
                 FaceBox(0, 200, 200, 400)]        # 200x200, fully inside 400x400
        kept, skipped = filter_faces(boxes, (400, 400), min_face_px=50, max_faces=2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "over_max_faces")
        self.assertAlmostEqual(min(b.area for b in kept), 200 * 200)

    def test_boxes_are_clipped_to_the_image(self):
        kept, _ = filter_faces([FaceBox(-50, -50, 150, 150)], (400, 400), min_face_px=50)
        self.assertEqual(kept[0].as_list(), [0.0, 0.0, 150.0, 150.0])

    def test_empty_input_is_not_an_error(self):
        kept, skipped = filter_faces([], (400, 400))
        self.assertEqual((kept, skipped), ([], []))


class TestDrawBoxes(unittest.TestCase):

    def test_draw_returns_a_copy_and_does_not_mutate_input(self):
        img = make_image(200, 200)
        before = np.asarray(img).copy()
        out = draw_boxes(img, [FaceBox(50, 50, 150, 150, score=0.9)])
        np.testing.assert_array_equal(np.asarray(img), before)
        self.assertFalse(np.array_equal(np.asarray(out), before))

    def test_draw_handles_no_boxes(self):
        img = make_image(80, 80)
        out = draw_boxes(img, [])
        np.testing.assert_array_equal(np.asarray(out), np.asarray(img))


if __name__ == "__main__":
    unittest.main(verbosity=2)
