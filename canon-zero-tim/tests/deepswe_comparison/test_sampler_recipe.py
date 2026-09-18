#!/usr/bin/env python3
"""P58 native-raw versus native token-IS recipe controls."""

from __future__ import annotations

import ast
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
names = {
    "_p58_native_sampler_recipe",
    "_validate_p58_native_sampler_recipe",
}
functions = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in names
]
if {node.name for node in functions} != names:
  raise RuntimeError("cannot isolate P58 sampler recipe helpers")


class AlignmentGateError(RuntimeError):
  pass


namespace = {
    "alignment": types.SimpleNamespace(AlignmentGateError=AlignmentGateError),
}
exec(
    compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
    namespace,
)
recipe_kind = namespace["_p58_native_sampler_recipe"]
validate_recipe = namespace["_validate_p58_native_sampler_recipe"]


def env(disable_sampler: str, disable_tis: str) -> dict[str, str]:
  return {
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": "native",
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
      ),
      "CANON_P34_DISABLE_SAMPLER_IS": disable_sampler,
      "CANON_P34_DISABLE_TIS": disable_tis,
  }


class P58SamplerRecipeTest(unittest.TestCase):

  def test_deepswe_cli_exposes_only_the_registered_sampler_controls(self):
    script = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn('"--sampler_is"', script)
    self.assertIn('choices=["none", "token"]', script)
    self.assertIn('default="none"', script)
    self.assertIn('"--sampler_is_threshold"', script)
    self.assertIn('"sampler_is": SAMPLER_IS', script)
    self.assertIn('"sampler_is_threshold": SAMPLER_IS_THRESHOLD', script)
    self.assertNotIn('"--group_clip_filter_threshold"', script)
    self.assertIn("P58_FIXED_SEED = bool(", script)
    self.assertIn(
        'vllm_rollout_dict["rollout_vllm_kwargs"]["seed"] = SEED',
        script,
    )
    self.assertNotIn('base_rollout_dict["seed"] = SEED', script)
    self.assertIn("scope=engine-global", script)
    self.assertIn("[P58.SEED] PASS", script)

  def test_postflight_requires_exactly_one_signed_recipe_marker(self):
    postflight = (
        ROOT / "canon-zero-tim/cluster/steps/90_run.sh"
    ).read_text()
    self.assertIn("n_p58_recipe_raw", postflight)
    self.assertIn("n_p58_recipe_is", postflight)
    self.assertIn("n_p58_seed", postflight)
    self.assertIn("n_p58_seed_route", postflight)
    self.assertIn("VLLM.JAX_SEED", postflight)
    self.assertIn("request_seed=none scope=engine-global", postflight)
    self.assertIn("P58 fixed-seed marker contract failed", postflight)
    self.assertIn("P58 native-raw recipe marker contract failed", postflight)
    self.assertIn("P58 native-is recipe marker contract failed", postflight)
    self.assertIn("n_deepswe_tito_admission", postflight)
    self.assertIn("n_deepswe_tito_continuation", postflight)
    self.assertIn("DeepSWE TiTO admission receipt contract failed", postflight)
    self.assertIn("P58 did not exercise a multi-turn TiTO continuation", postflight)

  def test_deepswe_tito_is_common_to_native_and_zero(self):
    script = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn("deepswe_exact_token_continuity", script)
    self.assertIn("[DEEPSWE.TITO] ADMISSION_PASS", script)
    self.assertIn("mode=token-in-token-out retokenize_sampled_tokens=0", script)
    self.assertNotIn("legacy-text-control", script)

  def test_native_raw_contract(self):
    recipe = recipe_kind(env("1", "1"))
    self.assertEqual(recipe, "native-raw")
    validate_recipe(
        recipe=recipe,
        sampler_is=None,
        sampler_is_threshold=2.0,
        use_rollout_logps=True,
        rollout_logps_present=True,
        trainer_logps_present=True,
        old_logps_are_rollout=True,
        old_logps_are_trainer=False,
        sampler_is_weights_present=False,
    )

  def test_native_is_contract(self):
    recipe = recipe_kind(env("0", "0"))
    self.assertEqual(recipe, "native-is")
    validate_recipe(
        recipe=recipe,
        sampler_is="token",
        sampler_is_threshold=2.0,
        use_rollout_logps=True,
        rollout_logps_present=True,
        trainer_logps_present=True,
        old_logps_are_rollout=False,
        old_logps_are_trainer=True,
        sampler_is_weights_present=True,
    )

  def test_partial_environment_tuple_is_rejected(self):
    with self.assertRaisesRegex(AlignmentGateError, "exact 1/1 raw or 0/0"):
      recipe_kind(env("0", "1"))

  def test_native_is_threshold_drift_is_rejected(self):
    with self.assertRaisesRegex(AlignmentGateError, "threshold"):
      validate_recipe(
          recipe="native-is",
          sampler_is="token",
          sampler_is_threshold=2.5,
          use_rollout_logps=True,
          rollout_logps_present=True,
          trainer_logps_present=True,
          old_logps_are_rollout=False,
          old_logps_are_trainer=True,
          sampler_is_weights_present=True,
      )

  def test_native_is_missing_weights_is_rejected(self):
    with self.assertRaisesRegex(AlignmentGateError, "tis_weights=absent"):
      validate_recipe(
          recipe="native-is",
          sampler_is="token",
          sampler_is_threshold=2.0,
          use_rollout_logps=True,
          rollout_logps_present=True,
          trainer_logps_present=True,
          old_logps_are_rollout=False,
          old_logps_are_trainer=True,
          sampler_is_weights_present=False,
      )


if __name__ == "__main__":
  unittest.main()
