"""
test_adapter_loader.py
======================
Strict-loading tests for adapter_loader.

The guards exist to stop adapters being attached to the wrong foundation, or
attached only partially. They must not fire on the *same* base merely passed
differently (relative vs absolute vs Windows casing), and they must fire for
vanilla-vs-fused, for a dropped key and for an unknown key.

Run:  python test_adapter_loader.py
"""

import os
import tempfile
import unittest

from adapter_loader import (AdapterLoadError, _check_base, check_adapter_keys,
                            check_base_model, same_model_reference)


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


class TestCheckAdapterKeys(unittest.TestCase):
    """Strictness that keeps a partial adapter from loading silently.

    These are the three shapes a broken package takes: a key missing from the
    file (dropped during extraction), a key the model has no target for (wrong
    architecture or a stale file), and a mismatch in both directions at once.
    """

    LIVE = {"model.layers.0.q_proj.lora_A.weight",
            "model.layers.0.q_proj.lora_B.weight"}

    def test_identical_key_sets_pass(self):
        check_adapter_keys(self.LIVE, self.LIVE, "llm_adapter")      # must not raise

    def test_dropped_key_is_refused(self):
        dropped = set(self.LIVE) - {"model.layers.0.q_proj.lora_B.weight"}
        with self.assertRaises(AdapterLoadError) as ctx:
            check_adapter_keys(dropped, self.LIVE, "llm_adapter")
        message = str(ctx.exception)
        self.assertIn("llm_adapter", message)
        self.assertIn("no value in the file", message)
        self.assertIn("partially-adapted", message)

    def test_unknown_key_is_refused(self):
        extra = set(self.LIVE) | {"model.layers.9.mlp.lora_A.weight"}
        with self.assertRaises(AdapterLoadError) as ctx:
            check_adapter_keys(extra, self.LIVE, "vision_adapter")
        message = str(ctx.exception)
        self.assertIn("vision_adapter", message)
        self.assertIn("no target in the model", message)

    def test_mismatch_in_both_directions_is_reported_together(self):
        file_keys = {"model.layers.0.q_proj.lora_A.weight", "ghost.key"}
        with self.assertRaises(AdapterLoadError) as ctx:
            check_adapter_keys(file_keys, self.LIVE, "custom_heads")
        message = str(ctx.exception)
        self.assertIn("ghost.key", message)
        self.assertIn("lora_B.weight", message)

    def test_any_iterable_is_accepted(self):
        check_adapter_keys(list(self.LIVE), tuple(self.LIVE), "llm_adapter")


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


class TestCheckBaseModel(unittest.TestCase):
    """The pure comparison behind _check_base, tested without any files."""

    def test_unrecorded_provenance_is_not_an_error(self):
        check_base_model(None, "./models/internvl3_2b_lr_fused")
        check_base_model("", "./models/internvl3_2b_lr_fused")

    def test_same_base_is_accepted(self):
        check_base_model("./models/internvl3_2b_lr_fused",
                         "./models/internvl3_2b_lr_fused")

    def test_different_base_is_refused(self):
        with self.assertRaises(AdapterLoadError) as ctx:
            check_base_model("./models/InternVL3-2B", "./models/internvl3_2b_lr_fused",
                             "runs/x/adapters")
        message = str(ctx.exception)
        self.assertIn("InternVL3-2B", message)
        self.assertIn("different base", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
