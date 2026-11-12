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

They read the YAML configs with PyYAML. Run directly:
python3 tests/dist_rl/examples/frozenlake_dist/frozenlake_benchmark_test.py
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

import yaml

ROOT = Path(__file__).resolve().parents[4]


def load_module(path):
  spec = importlib.util.spec_from_file_location(path.stem, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


benchmark = load_module(
    ROOT / "tunix/dist_rl/examples/rl_efficiency_benchmark/benchmark.py"
)
workload = benchmark.load_workload()
EXAMPLE = Path(workload.__file__).parent
RECIPE = EXAMPLE.parent / "recipes" / "benchmark_frozenlake_gemma4_e2b.sh"
LAUNCH = ROOT / "tunix/dist_rl/launch"
_ROLES = (
    "actor_model_config",
    "reference_model_config",
    "rollout_model_config",
)


def _deep_merge(base, override):
  merged = dict(base)
  for key, value in override.items():
    if isinstance(value, dict) and isinstance(merged.get(key), dict):
      merged[key] = _deep_merge(merged[key], value)
    else:
      merged[key] = value
  return merged


def _flatten(mapping, prefix=""):
  flat = {}
  for key, value in mapping.items():
    dotted = f"{prefix}.{key}" if prefix else str(key)
    if isinstance(value, dict) and value:
      flat.update(_flatten(value, dotted))
    else:
      flat[dotted] = value
  return flat


def yaml_defaults():
  """Returns the flattened `base_dist_config.yaml` + `frozenlake.yaml`."""
  base = yaml.safe_load((LAUNCH / "common/base_dist_config.yaml").read_text())
  task = yaml.safe_load((EXAMPLE / "frozenlake.yaml").read_text())
  config = _deep_merge(base, task)
  # Role configs inherit the task's `model_config`.
  for role in _ROLES:
    config[role] = _deep_merge(config["model_config"], task.get(role, {}))
  return _flatten(config)


def parse_overrides(args):
  """Parses `section.key=value` arguments, dropping one level of quotes."""
  overrides = {}
  for arg in args:
    key, value = arg.split("=", 1)
    overrides[key] = value.strip('"')
  return overrides


def recipe_overrides():
  """Returns the `section.key=value` arguments of the recipe script."""
  text = RECIPE.read_text()
  args = re.findall(r"^\s+([a-z_][a-z0-9_.]*=\S*?)\s*\\$", text, re.MULTILINE)
  # Fails loud if a formatting change makes the pattern above miss one.
  assignments = re.findall(r"^\s+[a-z_][a-z0-9_.]*=", text, re.MULTILINE)
  if len(args) != len(assignments):
    raise ValueError(
        f"Parsed {len(args)} of the {len(assignments)} `section.key=value`"
        f" lines in {RECIPE.name}; keep one override per line, with no"
        " spaces in its value, each ending in ` \\`."
    )
  return parse_overrides(args)


def legacy_launcher_env_vars():
  """Returns the old launchers' environment variables, which launch.sh rejects."""
  text = (LAUNCH / "common/legacy_env.sh").read_text()
  match = re.search(r"LEGACY_LAUNCHER_ENV_VARS=\((.*?)\)", text, re.DOTALL)
  return set(match.group(1).split())


def parse(text, like):
  """Parses a setting as the type of the Python value LIKE."""
  if isinstance(like, bool):
    return str(text) in ("1", "true", "True")
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
    """Returns the overrides, the environment, and the effective settings."""
    command, env = workload.dist_launch(self.w, self.path, self.path)
    self.assertEqual(command[:2], ["bash", str(RECIPE)])
    overrides = parse_overrides(command[2:])
    explicit = {**recipe_overrides(), **overrides}
    settings = {**yaml_defaults(), **explicit}
    # `model_config` overrides reach every role config not set explicitly.
    for key, value in explicit.items():
      if key.startswith("model_config."):
        suffix = key.removeprefix("model_config.")
        for role in _ROLES:
          if f"{role}.{suffix}" not in explicit:
            settings[f"{role}.{suffix}"] = value
    return overrides, env, settings

  def test_dist_launch_overrides_only_benchmark_settings(self):
    overrides, env, settings = self.dist_settings()
    # The resolver rejects unknown keys; catch misspellings here too. An
    # override may set a whole section (`metrics_logging_options=null`).
    known = yaml_defaults().keys()
    for key in overrides:
      self.assertTrue(
          any(name == key or name.startswith(f"{key}.") for name in known), key
      )
    for name, value in {
        "batch_size": "8",
        "rl_training_config.mini_batch_size": "8",
        "rl_training_config.max_steps": "3",
        "data_config.limit": "24",
        "agentic_grpo_config.max_concurrency": "64",
        "model_config.dtype": "bfloat16",
        "model_config.load_dtype": "float32",
        "rl_training_config.checkpointing_options.save_interval_steps": "0",
        "rl_training_config.metrics_logging_options": "null",
        "dist_training_config.trajectory_log_dir": "",
    }.items():
      self.assertEqual(overrides[name], value, name)
    # launch.sh fails on the old launchers' environment variables.
    self.assertEqual(env, {"LOG_ROOT": str(self.path)})
    self.assertFalse(env.keys() & legacy_launcher_env_vars())
    self.assertEqual(settings["actor_model_config.model_id"], workload.MODEL_ID)
    # The runtime hooks record Raiden weight installation.
    self.assertEqual(settings["weight_sync_config.mode"], "raiden")

  def test_agentic_recipe_matches_distributed_settings(self):
    _, _, settings = self.dist_settings()
    config = workload.agentic_config(self.w, self.path)
    model = config["model_config"]
    train = config["rl_training_config"]
    optimizer = train["actor_optimizer_config"]
    rollout = config["rollout_config"]
    vllm = config["vllm_config"]
    grpo = config["agentic_grpo_config"]
    for name, value in (
        ("actor_model_config.model_name", model["model_name"]),
        ("actor_model_config.model_id", model["model_id"]),
        ("actor_model_config.dtype", model["dtype"]),
        (
            "actor_model_config.use_flash_attention",
            model["use_flash_attention"],
        ),
        (
            "actor_model_config.flash_attention_block_size",
            model["flash_attention_block_size"],
        ),
        (
            "actor_model_config.load_dtype",
            config["actor_model_config"]["load_dtype"],
        ),
        ("batch_size", config["batch_size"]),
        ("rl_training_config.mini_batch_size", train["mini_batch_size"]),
        ("rl_training_config.max_steps", train["max_steps"]),
        (
            "rl_training_config.compute_logps_chunk_size",
            train["compute_logps_chunk_size"],
        ),
        (
            "rl_training_config.actor_optimizer_config.learning_rate",
            optimizer["learning_rate"],
        ),
        ("rl_training_config.actor_optimizer_config.b1", optimizer["b1"]),
        ("rl_training_config.actor_optimizer_config.b2", optimizer["b2"]),
        (
            "rl_training_config.actor_optimizer_config.weight_decay",
            optimizer["weight_decay"],
        ),
        (
            "rl_training_config.actor_optimizer_config.opt_chain_type",
            optimizer["opt_chain_type"],
        ),
        (
            "rl_training_config.actor_optimizer_config.chain_kwargs.max_norm",
            optimizer["chain_kwargs"]["max_norm"],
        ),
        ("rollout_config.temperature", rollout["temperature"]),
        ("rollout_config.top_p", rollout["top_p"]),
        ("rollout_config.max_prompt_length", rollout["max_prompt_length"]),
        (
            "rollout_config.total_generation_steps",
            rollout["total_generation_steps"],
        ),
        ("rollout_config.return_logprobs", rollout["return_logprobs"]),
        ("vllm_config.hbm_utilization", vllm["hbm_utilization"]),
        ("vllm_config.max_num_seqs", vllm["max_num_seqs"]),
        ("vllm_config.max_num_batched_tokens", vllm["max_num_batched_tokens"]),
        ("vllm_config.max_model_len", rollout["kv_cache_size"]),
        ("agentic_grpo_config.num_generations", grpo["num_generations"]),
        ("agentic_grpo_config.num_iterations", grpo["num_iterations"]),
        ("agentic_grpo_config.beta", grpo["beta"]),
        ("agentic_grpo_config.epsilon", grpo["epsilon"]),
        ("agentic_grpo_config.epsilon_high", grpo["epsilon_high"]),
        ("agentic_grpo_config.loss_algo", grpo["loss_algo"]),
        ("agentic_grpo_config.loss_agg_mode", grpo["loss_agg_mode"]),
        ("agentic_grpo_config.kl_loss_mode", grpo["kl_loss_mode"]),
        (
            "agentic_grpo_config.advantage_estimator",
            grpo["advantage_estimator"],
        ),
        ("agentic_grpo_config.sampler_is", grpo["sampler_is"]),
        (
            "agentic_grpo_config.sampler_is_threshold",
            grpo["sampler_is_threshold"],
        ),
        ("agentic_grpo_config.max_concurrency", grpo["max_concurrency"]),
        ("agentic_grpo_config.off_policy_steps", grpo["off_policy_steps"]),
        ("agentic_grpo_config.episode_timeout", grpo["episode_timeout"]),
        ("agentic_grpo_config.overlong_filter", grpo["overlong_filter"]),
        ("agentic_grpo_config.max_turns", config["env_kwargs"]["max_steps"]),
        ("env_kwargs.max_steps", config["env_kwargs"]["max_steps"]),
        ("env_kwargs.is_slippery", config["env_kwargs"]["is_slippery"]),
        (
            "agent_kwargs.use_multistep_prompt",
            config["agent_kwargs"]["use_multistep_prompt"],
        ),
        ("data_config.size", config["data_config"]["size"]),
        ("data_config.seed", config["data_config"]["seed"]),
        ("data_config.limit", config["data_config"]["limit"]),
    ):
      with self.subTest(name=name):
        self.assertEqual(parse(settings[name], value), value)
    self.assertEqual(
        settings["actor_model_config.remat_config"],
        model["remat_config"].lower(),
    )
    # Both stacks disable top-k sampling.
    self.assertLessEqual(settings["rollout_config.top_k"], 0)
    self.assertEqual(rollout["top_k"], 0)
    # Both stacks shuffle the same generated maps with the run seed.
    self.assertIs(settings["data_config.shuffle"], True)
    self.assertEqual(
        rollout["kv_cache_size"], self.w["prompt"] + self.w["response"] + 256
    )

  def test_agentic_mesh_and_microbatch_match_distributed(self):
    overrides, _, settings = self.dist_settings()
    config = workload.agentic_config(self.w, self.path)
    train = config["rl_training_config"]
    # Both stacks train micro_groups prompt groups per microbatch.
    expected = self.w["micro_groups"] * self.w["generations"]
    self.assertEqual(
        int(overrides["rl_training_config.train_micro_batch_size"]), expected
    )
    self.assertEqual(
        int(overrides["rl_training_config.compute_logps_micro_batch_size"]),
        expected,
    )
    self.assertEqual(train["train_micro_batch_size"], self.w["micro_groups"])
    self.assertEqual(
        train["compute_logps_micro_batch_size"], self.w["micro_groups"]
    )
    trainer = settings["actor_model_config.mesh.shape"]
    rollout = settings["rollout_model_config.mesh.shape"]
    self.assertEqual(config["model_config"]["mesh"]["shape"], trainer)
    self.assertEqual(config["actor_model_config"]["mesh"]["shape"], trainer)
    self.assertEqual(config["reference_model_config"]["same_mesh_as"], "actor")
    self.assertEqual(config["rollout_model_config"]["mesh"]["shape"], rollout)
    # The rollout TP is the rollout mesh `tp` axis.
    self.assertEqual(
        settings["rollout_model_config.mesh.axis_names"], "('fsdp','tp')"
    )
    self.assertEqual(
        config["vllm_config"]["tensor_parallel_size"],
        int(rollout.strip("()").split(",")[1]),
    )
    self.assertEqual(config["vllm_config"]["data_parallel_size"], 1)
    self.assertIsNone(train["checkpoint_root_directory"])
    # Neither stack creates metrics backends.
    self.assertIsNone(train["metrics_logging_options"])
    self.assertEqual(
        overrides["rl_training_config.metrics_logging_options"], "null"
    )

  def test_benchmark_entry_points_wrap_the_launch_mains(self):
    overrides, _, _ = self.dist_settings()
    defaults = yaml_defaults()
    local_launcher = (LAUNCH / "common/local.sh").read_text()
    for key, bash_var in (
        ("cluster_config.trainer_main", "TRAINER_MAIN"),
        ("cluster_config.rollout_main", "ROLLOUT_MAIN"),
        ("cluster_config.orchestrator_main", "ORCHESTRATOR_MAIN"),
    ):
      with self.subTest(key=key):
        self.assertIn(f'--process_main="${{{bash_var}}}"', local_launcher)
        self.assertIsNotNone(function_calls(defaults[key]))
        calls = function_calls(overrides[key])
        # For example, runtime.trainer_main wraps ...launch.nodes.trainer.main.
        self.assertIn(".".join(defaults[key].split(".")[-2:]), calls)
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
        {"frozenlake.generate_entries", "runtime.record_dataset"},
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
    self.assertEqual(dist["command"][:2], ["bash", str(RECIPE)])
    self.assertEqual(
        agentic["command"][:3], [sys.executable, "-m", workload.MODULE]
    )
    self.assertEqual(
        dist["environment"]["TUNIX_BENCHMARK_RESPONSE_RESERVE"], "512"
    )
    self.assertFalse(dist["environment"].keys() & legacy_launcher_env_vars())

  def test_validate_rejects_unrunnable_workloads(self):
    workload.validate(self.w)
    rows = self.w["steps"] * self.w["batch"] - 1
    with self.assertRaisesRegex(ValueError, "dataset rows"):
      workload.validate({**self.w, "dataset_size": rows})
    with self.assertRaisesRegex(ValueError, "positive"):
      workload.validate({**self.w, "turns": 0})


if __name__ == "__main__":
  unittest.main()
