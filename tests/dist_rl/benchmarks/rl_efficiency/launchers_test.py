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

"""Dry-runs the FrozenLake workload's distributed launch.

`frozenlake_workload.dist_launch` runs
`examples/recipes/benchmark_frozenlake_gemma4_e2b.sh` with the benchmark's
overrides; `examples/recipes/recipes_test.py` checks the recipe itself.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types

from absl.testing import absltest
from tunix.dist_rl.launch.common import launcher_testing
from tunix.dist_rl.launch.nodes import orchestrator

# The recipe that the benchmark's distributed stack runs, relative to
# `launcher_testing.EXAMPLES_DIR`.
_GEMMA4_RECIPE = "recipes/benchmark_frozenlake_gemma4_e2b.sh"


def _load_workload() -> types.ModuleType:
  """Loads `frozenlake_workload.py` by path, as the benchmark harness does."""
  # Located through `launcher_testing` (not `__file__`), so the path holds
  # wherever the OSS export puts the test code.
  path = (
      launcher_testing.EXAMPLES_DIR.parent
      / "benchmarks/rl_efficiency/frozenlake_workload.py"
  )
  spec = importlib.util.spec_from_file_location("frozenlake_workload", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class LaunchersTest(launcher_testing.LauncherTestCase):

  def test_dist_launch_resolves(self):
    workload = _load_workload()
    output = pathlib.Path(self.create_tempdir().full_path)
    model_dir = output / "model"
    command, env = workload.dist_launch(workload.DEFAULTS, output, model_dir)
    self.assertEqual(
        pathlib.Path(command[1]).resolve(),
        (launcher_testing.EXAMPLES_DIR / _GEMMA4_RECIPE).resolve(),
    )
    # `launch.sh` would reject the old launchers' environment variables.
    self.assertEqual(env, {"LOG_ROOT": str(output)})
    hp = self.dry_run(_GEMMA4_RECIPE, *command[2:]).resolve()
    config = hp.config
    cluster = config["cluster_config"]
    self.assertEqual(
        cluster["orchestrator_main"], f"{workload.MODULE}.dist_main"
    )
    runtime = "tunix.dist_rl.benchmarks.rl_efficiency.runtime"
    self.assertEqual(cluster["trainer_main"], f"{runtime}.trainer_main")
    self.assertEqual(cluster["rollout_main"], f"{runtime}.rollout_main")
    self.assertEqual(
        config["tokenizer_config"]["tokenizer_path"], str(model_dir)
    )
    for role in ("actor_model_config", "rollout_model_config"):
      self.assertEqual(config[role]["model_download_path"], str(model_dir))
      self.assertEqual(config[role]["dtype"], "bfloat16")
      self.assertEqual(config[role]["load_dtype"], "float32")
    rl_train = config["rl_training_config"]
    self.assertEqual(
        rl_train["checkpointing_options"]["save_interval_steps"], 0
    )
    # No metrics backends and no trajectory logs.
    self.assertIsNone(rl_train["metrics_logging_options"])
    self.assertIsNone(orchestrator.build_metrics_logging_options(hp))
    self.assertEqual(config["dist_training_config"]["trajectory_log_dir"], "")
    self.assertEqual(
        config["env_kwargs"]["max_steps"],
        config["agentic_grpo_config"]["max_turns"],
    )
    vllm = config["vllm_config"]
    self.assertEqual(
        (vllm["max_num_seqs"], vllm["max_num_batched_tokens"]), (32, 8192)
    )
    self.assertEqual(vllm["max_model_len"], 2048 + 2048 + 256)
    self.assertEqual(config["data_config"]["limit"], 150 * 64)


if __name__ == "__main__":
  absltest.main()
