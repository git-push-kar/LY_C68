"""
test_calibration.py
===================
Tests for the M5 calibration analysis (face_calibration.py). Pure numpy: no
model, no detector, no dataset, so they run in well under a second.

Run:  python test_calibration.py
"""

import unittest

import numpy as np

from face_calibration import (best_threshold, doc_probs, face_probs, frac_sweep,
                              roc_auc, route_at_frac, sweep_threshold)


def record(label, whole_p, faces, margin_ps):
    """One frame record shaped like the scorer's cache lines."""
    return {
        "rel": f"frame_{label}_{whole_p}.png", "label": label, "set": "ffiw",
        "w": 256, "h": 256,
        "faces": [{"box": [0, 0, 100, 100], "side": side, "frac": frac}
                  for side, frac in faces],
        "whole_p": whole_p,
        "margins": {"0.3": [{"p": p, "upscale": 2.0, "pad_mode": "none"}
                            for p in margin_ps]},
    }


class TestAuc(unittest.TestCase):

    def test_perfect_separation_is_one(self):
        self.assertEqual(roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])), 1.0)

    def test_inverted_scores_are_zero(self):
        self.assertEqual(roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_ties_are_half_credit(self):
        # every score identical -> AUC 0.5, not 0 or 1
        self.assertEqual(roc_auc(np.array([0, 1, 0, 1]), np.array([0.5] * 4)), 0.5)

    def test_a_single_class_is_nan(self):
        self.assertTrue(np.isnan(roc_auc(np.array([1, 1]), np.array([0.2, 0.8]))))


class TestAggregation(unittest.TestCase):

    def test_doc_prob_is_the_max_over_faces(self):
        recs = [record(1, 0.10, [(120, 0.3), (110, 0.3)], [0.20, 0.85])]
        _, s = doc_probs(recs, 0.3)
        self.assertAlmostEqual(s[0], 0.85)

    def test_frame_without_a_usable_face_falls_back_to_whole_image(self):
        recs = [record(0, 0.07, [(120, 0.3)], [None])]
        y, s = doc_probs(recs, 0.3)
        self.assertEqual(y[0], 0)
        self.assertAlmostEqual(s[0], 0.07)

    def test_face_probs_skip_none_entries(self):
        recs = [record(1, 0.1, [(120, 0.3), (110, 0.2)], [0.9, None])]
        y, s = face_probs(recs, 0.3)
        self.assertEqual(len(s), 1)
        self.assertEqual(y.tolist(), [1])


class TestSweep(unittest.TestCase):

    def test_threshold_moves_operating_point(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.2, 0.4, 0.6, 0.8])
        rows = {r["threshold"]: r for r in sweep_threshold(y, s, 0.3, 0.7, 0.1)}
        self.assertEqual(rows[0.5]["acc"], 1.0)
        self.assertEqual(rows[0.7]["tpr"], 0.5)      # 0.6 no longer flagged
        self.assertEqual(rows[0.3]["tnr"], 0.5)      # 0.4 now flagged

    def test_best_threshold_prefers_youden_then_closest_to_half(self):
        rows = [{"threshold": 0.40, "youden": 0.5, "acc": 0.8},
                {"threshold": 0.50, "youden": 0.5, "acc": 0.8},
                {"threshold": 0.60, "youden": 0.4, "acc": 0.7}]
        self.assertEqual(best_threshold(rows)["threshold"], 0.50)


class TestRoutingAtFrac(unittest.TestCase):

    def test_lone_face_above_floor_takes_passthrough(self):
        rec = record(1, 0.30, [(133, 0.38)], [0.90])
        self.assertEqual(route_at_frac(rec, 0.05, 0.3, 96)["path"], "passthrough")

    def test_lone_face_below_floor_is_cropped(self):
        rec = record(1, 0.30, [(133, 0.03)], [0.90])
        self.assertEqual(route_at_frac(rec, 0.05, 0.3, 96)["path"], "per_face")

    def test_two_faces_are_never_passthrough(self):
        rec = record(1, 0.30, [(133, 0.60), (120, 0.55)], [0.90, 0.10])
        self.assertEqual(route_at_frac(rec, 0.05, 0.3, 96)["path"], "per_face")

    def test_faces_below_min_face_px_fall_back_to_the_whole_image(self):
        rec = record(1, 0.30, [(80, 0.03)], [0.90])
        routed = route_at_frac(rec, 0.05, 0.3, 96)
        self.assertEqual(routed["path"], "no_face")
        self.assertEqual(routed["p"], 0.30)

    def test_frac_sweep_on_single_large_faces_is_flat(self):
        # All FFIW-style frames: 1 face at 0.38 of the frame. Any floor up to
        # 0.30 routes them identically, which is why FFIW cannot constrain it.
        recs = [record(1, 0.10, [(133, 0.38)], [0.80]),
                record(0, 0.80, [(133, 0.42)], [0.20])]
        rows = frac_sweep(recs, 0.3, 0.5, 96, candidates=(0.02, 0.05, 0.30))
        self.assertEqual([r["passthrough_share"] for r in rows], [1.0, 1.0, 1.0])
        self.assertEqual(len({r["auc"] for r in rows}), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
