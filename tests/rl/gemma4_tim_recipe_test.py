"""Default-recipe and arm-isolation tests; no JAX or engine dependency."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from examples.frozenlake.gemma4_tim import recipe


class RecipeTest(unittest.TestCase):

  def test_default_workload_and_actual_model_inheritance(self):
    resolved = recipe.resolve("native")
    cfg = resolved["config"]
    self.assertEqual(cfg["batch_size"], 64)
    self.assertEqual(cfg["agentic_grpo_config"]["num_generations"], 8)
    self.assertEqual(cfg["num_batches"], 5)
    self.assertEqual(cfg["rl_training_config"]["max_steps"], 5)
    self.assertEqual(cfg["env_kwargs"]["max_steps"], 8)
    for role in ("actor", "reference", "rollout"):
      self.assertEqual(cfg[role + "_model_config"]["model_id"], recipe.MODEL_ID)
    self.assertEqual(cfg["actor_model_config"]["load_dtype"], "float32")
    self.assertEqual(cfg["reference_model_config"]["same_mesh_as"], "actor")
    self.assertIsNone(cfg["reference_model_config"]["mesh"])
    self.assertEqual(cfg["vllm_config"]["tensor_parallel_size"], 4)

  def test_only_arm_difference_is_sampler_is(self):
    native = recipe.resolve("native")["config"]
    tis = recipe.resolve("tis")["config"]
    zero = recipe.resolve("zero")["config"]
    self.assertEqual(native, zero)
    self.assertEqual(tis["agentic_grpo_config"]["sampler_is"], "token")
    tis["agentic_grpo_config"]["sampler_is"] = None
    self.assertEqual(native, tis)

  def test_mutations_and_extra_keys_are_not_admitted(self):
    for mutate in (
        lambda c: c["config"].update(batch_size=1),
        lambda c: c.update(target_status="PASS"),
        lambda c: c.update(unregistered=True),
        lambda c: c["config"]["agentic_grpo_config"].update(sampler_is="token"),
    ):
      contract = recipe.resolve("native")
      mutate(contract)
      with self.assertRaises(recipe.RecipeError):
        recipe.validate_resolved(contract)
    recipe.validate_resolved(recipe.resolve("tis"))

  def test_absent_empty_and_zero_foreign_flags(self):
    recipe.require_isolated_environment({"JAX_PLATFORMS": "cpu"})
    for key in ("CANON_ENGINE_MODULE_C", "CANON_GEMMA4_E4B_P1_ADMISSION",
                "FL_TRAINER_MESH", "T_BATCH_SIZE", "ROLLOUT_ENGINE"):
      for value in ("", "0", "1"):
        with self.assertRaises(recipe.RecipeError):
          recipe.require_isolated_environment({key: value})

  def test_unknown_arm_stage_and_proxy_rejected(self):
    for arm, stage in (("standard", "train"), ("zero", "stock-admission"),
                       ("native", "p45")):
      with self.assertRaises(recipe.RecipeError):
        recipe.resolve(arm, stage=stage)
    with self.assertRaises(recipe.RecipeError):
      recipe.require_isolated_environment({"JAX_PLATFORMS": "proxy"})

  def test_yaml_drift_is_detected(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      for name in recipe.SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((recipe.REPO / name).read_bytes() + b"\n")
      with self.assertRaisesRegex(recipe.RecipeError, "source drift"):
        recipe.resolve("native", repo=root)

  def test_path_binding_does_not_mutate_recipe(self):
    cfg = recipe.resolve("tis")["config"]
    original = copy.deepcopy(cfg)
    bound = recipe.bind_paths(cfg, snapshot=Path("/model"), data=Path("/data"),
                              output=Path("/output"))
    self.assertEqual(cfg, original)
    self.assertEqual(bound["vllm_config"]["model_version"], "/model")
    self.assertEqual(bound["rl_training_config"]["max_steps"], 5)

  def test_plan_import_does_not_load_jax(self):
    result = subprocess.run([
        sys.executable, "-c",
        "from examples.frozenlake.gemma4_tim import recipe; "
        "recipe.resolve('zero'); import sys; assert 'jax' not in sys.modules",
    ], cwd=recipe.REPO, capture_output=True, text=True, check=False)
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_cli_duplicate_arm_and_unknown_override_refused(self):
    for arguments in (("--arm", "tis"), ("--batch_size", "1")):
      result = subprocess.run([sys.executable, "-m", "examples.frozenlake.gemma4_tim.run",
                                "--arm", "native", *arguments], cwd=recipe.REPO,
                               capture_output=True, text=True, check=False)
      self.assertEqual(result.returncode, 2)
      self.assertNotIn("PLAN_PASS", result.stdout)

  def test_clean_execution_source_gate(self):
    from examples.frozenlake.gemma4_tim import run
    with self.assertRaises(recipe.RecipeError):
      run.require_clean_source("79fe3572")
    for responses in (("a" * 40, " M file"), ("b" * 40, "")):
      with mock.patch.object(run.subprocess, "check_output", side_effect=responses):
        with self.assertRaises(recipe.RecipeError):
          run.require_clean_source("a" * 40)
    with mock.patch.object(run.subprocess, "check_output", side_effect=("a" * 40, "")):
      run.require_clean_source("a" * 40)


if __name__ == "__main__":
  unittest.main()
