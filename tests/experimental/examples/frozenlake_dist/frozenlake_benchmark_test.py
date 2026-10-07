# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests of the FrozenLake benchmark workload; no TPU/JAX/pytest.

Run directly:
python3 tests/experimental/examples/frozenlake_dist/frozenlake_benchmark_test.py
"""

import argparse
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]


def load_module(path):
  spec = importlib.util.spec_from_file_location(path.stem, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


benchmark = load_module(
    ROOT / "tunix/experimental/examples/rl_efficiency_benchmark/benchmark.py"
)
workload = benchmark.load_workload()
RECIPE = Path(workload.__file__).parent


def shell_defaults():
  """Returns the literal script defaults; run_gemma4_e2b.sh sets them first."""
  defaults = {}
  for name in ("run_gemma4_e2b.sh", "launcher.sh"):
    text = (RECIPE / name).read_text()
    for variable, value in re.findall(
        r"\$\{([A-Z_][A-Z0-9_]*):?-([^}]*)\}", text
    ):
      defaults.setdefault(variable, value)
  return defaults


def parse(text, like):
  """Parses a shell setting as the type of the Python value LIKE."""
  if isinstance(like, bool):
    return text in ("1", "true", "True")
  return type(like)(text)


def function_calls(target):
  """Returns the callees of module-level function TARGET, or None."""
  module, function = target.rsplit(".", 1)
  path = ROOT / (module.replace(".", "/") + ".py")
  for node in ast.parse(path.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name == function:
      return {
          ast.unparse(call.func)
          for call in ast.walk(node)
          if isinstance(call, ast.Call)
      }
  return None


class FrozenLakeBenchmarkTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.path = Path(self.temp.name)
    self.w = {
        **workload.DEFAULTS,
        "batch": 8,
        "steps": 3,
        "warmup": 1,
        "concurrency": 64,
    }

  def dist_settings(self):
    """Returns the overrides and the effective literal recipe settings."""
    command, env = workload.dist_launch(self.w, self.path, self.path)
    self.assertEqual(command, ["bash", str(RECIPE / "run_gemma4_e2b.sh")])
    return env, {**shell_defaults(), **env}

  def test_dist_launch_overrides_only_benchmark_settings(self):
    env, settings = self.dist_settings()
    # The scripts would silently ignore a misspelled override.
    self.assertLessEqual(env.keys(), shell_defaults().keys())
    for name, value in {
        "BATCH_SIZE": "8",
        "MINI_BATCH_SIZE": "8",
        "NUM_BATCHES": "3",
        "NUM_EPOCHS": "1",
        "MAX_STEPS": "3",
        "ROLLOUT_MAX_CONCURRENCY": "64",
        "MODEL_DTYPE": "bfloat16",
        "MODEL_LOAD_DTYPE": "float32",
        "CHECKPOINT_SAVE_INTERVAL_STEPS": "0",
        "LOG_DIR": "",
        "TRAJECTORY_LOG_DIR": "",
    }.items():
      self.assertEqual(env[name], value, name)
    self.assertEqual(settings["MODEL_ID"], workload.MODEL_ID)
    # The runtime hooks record Raiden weight installation.
    self.assertEqual(settings["WEIGHT_SYNC_MODE"], "raiden")

  def test_agentic_recipe_matches_distributed_settings(self):
    _, settings = self.dist_settings()
    config = workload.agentic_config(self.w, self.path)
    model = config["model_config"]
    train = config["rl_training_config"]
    optimizer = train["actor_optimizer_config"]
    rollout = config["rollout_config"]
    vllm = config["vllm_config"]
    grpo = config["agentic_grpo_config"]
    for name, value in (
        ("MODEL_NAME", model["model_name"]),
        ("MODEL_ID", model["model_id"]),
        ("MODEL_DTYPE", model["dtype"]),
        ("MODEL_LOAD_DTYPE", config["actor_model_config"]["load_dtype"]),
        ("SEED", model["rng_seed"]),
        ("FLASH_ATTENTION_BLOCK_SIZE", model["flash_attention_block_size"]),
        ("BATCH_SIZE", config["batch_size"]),
        ("NUM_BATCHES", config["num_batches"]),
        ("NUM_EPOCHS", config["num_train_epochs"]),
        ("MINI_BATCH_SIZE", train["mini_batch_size"]),
        ("MAX_STEPS", train["max_steps"]),
        ("COMPUTE_LOGPS_CHUNK_SIZE", train["compute_logps_chunk_size"]),
        ("LEARNING_RATE", optimizer["learning_rate"]),
        ("ADAM_B1", optimizer["b1"]),
        ("ADAM_B2", optimizer["b2"]),
        ("WEIGHT_DECAY", optimizer["weight_decay"]),
        ("OPT_CHAIN_TYPE", optimizer["opt_chain_type"]),
        ("MAX_GRAD_NORM", optimizer["chain_kwargs"]["max_norm"]),
        ("TEMPERATURE", rollout["temperature"]),
        ("TOP_P", rollout["top_p"]),
        ("TOP_K", rollout["top_k"]),
        ("MAX_PROMPT_LENGTH", rollout["max_prompt_length"]),
        ("MAX_RESPONSE_LENGTH", rollout["total_generation_steps"]),
        ("USE_ROLLOUT_LOGPS", rollout["return_logprobs"]),
        ("VLLM_HBM_UTILIZATION", vllm["hbm_utilization"]),
        ("VLLM_MAX_NUM_SEQS", vllm["max_num_seqs"]),
        ("VLLM_MAX_NUM_BATCHED_TOKENS", vllm["max_num_batched_tokens"]),
        ("ROLLOUT_TP", vllm["tensor_parallel_size"]),
        ("NUM_GENERATIONS", grpo["num_generations"]),
        ("NUM_ITERATIONS", grpo["num_iterations"]),
        ("BETA", grpo["beta"]),
        ("EPSILON", grpo["epsilon"]),
        ("EPSILON_HIGH", grpo["epsilon_high"]),
        ("LOSS_ALGO", grpo["loss_algo"]),
        ("LOSS_AGG_MODE", grpo["loss_agg_mode"]),
        ("KL_LOSS_MODE", grpo["kl_loss_mode"]),
        ("ADVANTAGE_ESTIMATOR", grpo["advantage_estimator"]),
        ("SAMPLER_IS", grpo["sampler_is"]),
        ("SAMPLER_IS_THRESHOLD", grpo["sampler_is_threshold"]),
        ("ROLLOUT_MAX_CONCURRENCY", grpo["max_concurrency"]),
        ("OFF_POLICY_STEPS", grpo["off_policy_steps"]),
        ("EPISODE_TIMEOUT_SECS", grpo["episode_timeout"]),
        ("MAX_TURNS", config["env_kwargs"]["max_steps"]),
        ("IS_SLIPPERY", config["env_kwargs"]["is_slippery"]),
        (
            "USE_MULTISTEP_PROMPT",
            config["agent_kwargs"]["use_multistep_prompt"],
        ),
        ("DATASET_SIZE", config["data_config"]["size"]),
        ("SEED", config["data_config"]["seed"]),
    ):
      with self.subTest(name=name):
        self.assertEqual(parse(settings[name], value), value)
    # Both stacks shuffle the same generated maps with the run seed.
    self.assertEqual(settings["SHUFFLE"], "1")
    self.assertEqual(
        settings["VLLM_MAX_MODEL_LEN"],
        "$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256))",
    )
    self.assertEqual(
        rollout["kv_cache_size"], self.w["prompt"] + self.w["response"] + 256
    )

  def test_agentic_mesh_and_microbatch_match_distributed(self):
    env, settings = self.dist_settings()
    config = workload.agentic_config(self.w, self.path)
    train = config["rl_training_config"]
    # Both stacks train micro_groups prompt groups per microbatch.
    expected = self.w["micro_groups"] * self.w["generations"]
    self.assertEqual(int(env["TRAIN_MICRO_BATCH_SIZE"]), expected)
    self.assertEqual(int(env["COMPUTE_LOGPS_MICRO_BATCH_SIZE"]), expected)
    self.assertEqual(train["train_micro_batch_size"], self.w["micro_groups"])
    self.assertEqual(
        train["compute_logps_micro_batch_size"], self.w["micro_groups"]
    )
    trainer = f"({settings['TRAINER_FSDP']},{settings['TRAINER_TP']})"
    rollout = f"({settings['ROLLOUT_FSDP']},{settings['ROLLOUT_TP']})"
    self.assertEqual(config["model_config"]["mesh"]["shape"], trainer)
    self.assertEqual(config["actor_model_config"]["mesh"]["shape"], trainer)
    self.assertEqual(config["reference_model_config"]["same_mesh_as"], "actor")
    self.assertEqual(config["rollout_model_config"]["mesh"]["shape"], rollout)
    self.assertEqual(config["vllm_config"]["data_parallel_size"], 1)
    self.assertIsNone(train["checkpoint_root_directory"])
    self.assertIsNone(train["metrics_logging_options"])
    self.assertTrue(config["agentic_grpo_config"]["exact_token_continuity"])

  def test_benchmark_entry_points_wrap_the_recipe(self):
    env, _ = self.dist_settings()
    defaults = shell_defaults()
    launcher = (RECIPE / "launcher.sh").read_text()
    for name in (
        "TRAINER_PROCESS_MAIN",
        "ROLLOUT_PROCESS_MAIN",
        "ORCHESTRATOR_PROCESS_MAIN",
    ):
      with self.subTest(name=name):
        self.assertIn(f'--process_main="${name}"', launcher)
        self.assertIsNotNone(function_calls(defaults[name]))
        calls = function_calls(env[name])
        # For example, run_trainer_node.main wraps ...common.run_trainer_node.
        self.assertIn(".".join(defaults[name].split(".")[-2:]), calls)
        self.assertTrue(calls & {"install", "runtime.install"})
    self.assertEqual(
        ROOT / (workload.MODULE.replace(".", "/") + ".py"),
        Path(workload.__file__),
    )
    self.assertEqual(
        workload.agentic_config(self.w, self.path)["data_module"],
        workload.MODULE,
    )
    self.assertLessEqual(
        {"runtime.install", "app.run"},
        function_calls(workload.MODULE + ".agentic_main"),
    )
    self.assertLessEqual(
        {"frozenlake.create_dataset", "runtime.record_dataset"},
        function_calls(workload.MODULE + ".create_dataset"),
    )

  def test_plans_both_stacks_with_one_training_contract(self):
    model = self.path / "model"
    model.mkdir()
    (model / "weights.safetensors").touch()
    (model / "config.json").write_text("{}")
    manifests = {}
    for mode in ("dist", "agentic"):
      output = self.path / mode
      args = argparse.Namespace(
          **self.w,
          mode=mode,
          output=output,
          model_dir=model,
          cache_root=self.path / "cache",
          hardware="v5p-4",
          timing_mode="stages",
          execute=False,
          timeout=60,
      )
      with contextlib.redirect_stdout(io.StringIO()):
        benchmark.prepare(args, workload)
      self.assertFalse((output / "launcher.log").exists())
      manifests[mode] = json.loads((output / "manifest.json").read_text())
    dist, agentic = manifests["dist"], manifests["agentic"]
    self.assertEqual(dist["training_contract"], agentic["training_contract"])
    self.assertEqual(
        dist["command"], ["bash", str(RECIPE / "run_gemma4_e2b.sh")]
    )
    self.assertEqual(
        agentic["command"][:3], [sys.executable, "-m", workload.MODULE]
    )
    self.assertEqual(
        dist["environment"]["TUNIX_BENCHMARK_RESPONSE_RESERVE"], "512"
    )

  def test_validate_rejects_unrunnable_workloads(self):
    workload.validate(self.w)
    rows = self.w["steps"] * self.w["batch"] - 1
    with self.assertRaisesRegex(ValueError, "dataset rows"):
      workload.validate({**self.w, "dataset_size": rows})
    with self.assertRaisesRegex(ValueError, "positive"):
      workload.validate({**self.w, "turns": 0})


if __name__ == "__main__":
  unittest.main()
