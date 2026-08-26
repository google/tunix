#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class StableClipContractTest(unittest.TestCase):

  def test_only_registered_full_profile_family_enables_p63(self):
    profiles = ROOT / "canon-zero-tim/cluster/profiles"
    gsm = (profiles / "qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env").read_text()
    frozen = (
        profiles / "qwen3-8b-dp8-tp8-frozenlake-v1-hp.env"
    ).read_text()
    diagnostic = (
        profiles / "qwen3-1p7b-dp16-tp4-gsm8k-p62-debug.env"
    ).read_text()
    deepswe = (
        profiles / "qwen3-4b-dp8-tp8-deepswe-v1-hp.env"
    ).read_text()
    self.assertIn("export CANON_P63_OVERFLOW_SAFE_CLIP=1", gsm)
    self.assertIn("export CANON_P63_OVERFLOW_SAFE_CLIP=1", frozen)
    self.assertIn("export CANON_P63_OVERFLOW_SAFE_CLIP=1", deepswe)
    self.assertNotIn("CANON_P63_OVERFLOW_SAFE_CLIP", diagnostic)

  def test_exact_python_readers_use_hybrid_transform(self):
    gsm = (ROOT / "examples/math_gsm8k/qwen3_grpo_demo.py").read_text()
    frozen = (
        ROOT / "examples/frozenlake/train_frozenlake_qwen3.py"
    ).read_text()
    deepswe = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    trainer = (ROOT / "tunix/sft/peft_trainer.py").read_text()
    for text in (gsm, frozen, deepswe):
      self.assertIn("canonical_overflow_safe_clip_max_norm", text)
      self.assertIn("overflow_safe_clip_by_global_norm", text)
      self.assertIn("[P63.STABLE_CLIP] configured", text)
    self.assertIn("hybrid_global_norm", trainer)
    self.assertIn('commit_evidence["overflow_safe_clip"]', trainer)
    self.assertIn("[P63.STABLE_CLIP] ", trainer)

  def test_postflight_and_exact_image_require_p63(self):
    classifier = (
        ROOT
        / "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/"
        "classify_full_recipe.py"
    ).read_text()
    exact_image = (
        ROOT / "canon-zero-tim/tests/v1_phase4/run_exact_image.sh"
    ).read_text()
    self.assertIn('"CANON_P63_OVERFLOW_SAFE_CLIP": "1"', classifier)
    self.assertIn("p63_gsm8k_fallback_not_observed", classifier)
    self.assertIn("p63_clip=1", exact_image)


if __name__ == "__main__":
  unittest.main()
