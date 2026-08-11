"""Regression tests for workload-specific sampler importance sampling."""

from __future__ import annotations

from pathlib import Path
import unittest

from tunix.rl.agentic import agentic_grpo_learner


_REPO_ROOT = Path(__file__).resolve().parents[3]
_sampler_is_valid = getattr(
    agentic_grpo_learner, "_canonical_alignment_sampler_is_valid"
)


class SamplerIsContractTest(unittest.TestCase):

  def test_gsm8k_direct_rollout_logprobs_allow_no_sampler_correction(self):
    self.assertTrue(_sampler_is_valid(None, "gsm8k"))

  def test_frozenlake_rejects_missing_token_sampler_correction(self):
    self.assertFalse(_sampler_is_valid(None, "frozenlake"))

  def test_frozenlake_accepts_token_sampler_correction(self):
    self.assertTrue(_sampler_is_valid("token", "frozenlake"))

  def test_frozenlake_recipe_pins_token_sampler_correction(self):
    source = (
        _REPO_ROOT / "examples/frozenlake/train_frozenlake_qwen3.py"
    ).read_text(encoding="utf-8")
    self.assertIn('sampler_is="token",', source)
    self.assertNotIn(
        'sampler_is=None if CANON_P32_WORKLOAD else "token",', source
    )

  def test_gsm8k_recipe_keeps_direct_rollout_logprobs(self):
    source = (
        _REPO_ROOT / "examples/math_gsm8k/qwen3_grpo_demo.py"
    ).read_text(encoding="utf-8")
    self.assertIn(
        '"sampler_is": None if CANON_P32_WORKLOAD else "token",', source
    )

  def test_p41_benchmark_only_uses_serial_engine_seed(self):
    source = (
        _REPO_ROOT / "examples/math_gsm8k/qwen3_grpo_demo.py"
    ).read_text(encoding="utf-8")
    self.assertIn("if CANON_P41_OPTIMIZER_BENCH:", source)
    self.assertIn(
        'vllm_rollout_dict["rollout_vllm_kwargs"]["seed"] = SEED', source
    )
    self.assertIn(
        'P41 optimizer benchmark requires max_concurrency=1', source
    )
    self.assertNotIn('"seed": SEED,', source)


if __name__ == "__main__":
  unittest.main()
