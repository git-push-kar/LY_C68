"""
test_adapter_loader.py
======================
Path-identity tests for adapter_loader's base-model guard.

The guard exists to stop adapters being attached to the wrong foundation. It
must not fire on the *same* base merely passed differently (relative vs
absolute vs Windows casing), and it must still fire for vanilla-vs-fused.

Run:  python test_adapter_loader.py
"""

import os
import tempfile
import unittest

from adapter_loader import AdapterLoadError, _check_base, same_model_reference


class TestSameModelReference(unittest.TestCase):

    def test_relative_and_absolute_paths_to_same_dir_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "internvl3_2b_lr_fused")
            os.makedirs(base)
            rel = os.path.relpath(base)
            self.assertTrue(same_model_reference(rel, base), f"{rel} vs {base}")

    def test_absolute_path_against_dot_slash_form_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "base")
            os.makedirs(base)
            dotted = os.path.join(".", os.path.relpath(base))
            self.assertTrue(same_model_reference(dotted, os.path.abspath(base)))

    def test_trailing_separator_and_dot_segments_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "base")
            os.makedirs(base)
            variant = os.path.join(tmp, "nested", "..", "base") + os.sep
            self.assertTrue(same_model_reference(base, variant))

    def test_case_variants_match(self):
        # normcase lowercases on Windows, so this is a real regression guard there
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "InternVL3-2B")
            os.makedirs(base)
            self.assertTrue(same_model_reference(base, base.upper()))

    def test_different_existing_dirs_do_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            vanilla = os.path.join(tmp, "InternVL3-2B")
            fused = os.path.join(tmp, "internvl3_2b_lr_fused")
            os.makedirs(vanilla)
            os.makedirs(fused)
            self.assertFalse(same_model_reference(vanilla, fused))

    def test_missing_paths_fall_back_to_basename(self):
        self.assertTrue(same_model_reference(
            "/mnt/elsewhere/models/internvl3_2b_lr_fused",
            "./models/internvl3_2b_lr_fused"))

    def test_missing_paths_with_different_basenames_do_not_match(self):
        self.assertFalse(same_model_reference(
            "/mnt/elsewhere/models/InternVL3-2B",
            "./models/internvl3_2b_lr_fused"))


class TestCheckBase(unittest.TestCase):

    def test_absolute_path_of_recorded_base_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "internvl3_2b_lr_fused")
            os.makedirs(base)
            report = {"provenance": {"base_model": os.path.relpath(base)}}
            _check_base("<tmp>", report, base)          # must not raise

    def test_wrong_base_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            vanilla = os.path.join(tmp, "InternVL3-2B")
            fused = os.path.join(tmp, "internvl3_2b_lr_fused")
            os.makedirs(vanilla)
            os.makedirs(fused)
            report = {"provenance": {"base_model": vanilla}}
            with self.assertRaises(AdapterLoadError):
                _check_base("<tmp>", report, fused)

    def test_missing_report_is_not_a_failure(self):
        _check_base("<tmp>", {}, "./models/internvl3_2b_lr_fused")


if __name__ == "__main__":
    unittest.main(verbosity=2)
