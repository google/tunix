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

"""Dry-runs the FrozenLake `launcher.sh` and `run_gemma4_e2b.sh`.

`run_gemma4_e2b.sh` is also dry-run with the benchmark's overrides
(`frozenlake_benchmark.dist_launch`).
"""

from __future__ import annotations

import importlib.util
import pathlib
import types

from absl.testing import absltest
from absl.testing import parameterized
from tunix.cli import config as cli_config
from tunix.dist_rl.launch.common import launcher_testing
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import orchestrator

_QWEN3 = launcher_testing.Expected(
    name="qwen3_8b",
    script="frozenlake_dist/launcher.sh",
    example="frozenlake_dist",
    module="frozenlake",
    target="local",
    batch_size=64,
    mini_batch_size=64,
    num_generations=8,
    max_steps=450,
    actor_mesh=(1, 4),
    rollout_mesh=(1, 4),
    model_id="Qwen/Qwen3-8B",
)
_GEMMA4 = launcher_testing.Expected(
    name="gemma4_e2b",
    script="frozenlake_dist/run_gemma4_e2b.sh",
    example="frozenlake_dist",
    module="frozenlake",
    target="local",
    batch_size=64,
    mini_batch_size=64,
    num_generations=8,
    max_steps=450,
    actor_mesh=(2, 1),
    rollout_mesh=(1, 2),
    model_id="google/gemma-4-E2B-it",
)


def _load_benchmark_workload() -> types.ModuleType:
  """Loads `frozenlake_benchmark.py` by path, as the benchmark harness does."""
  path = (
      launcher_testing.EXAMPLES_DIR / "frozenlake_dist/frozenlake_benchmark.py"
  )
  spec = importlib.util.spec_from_file_location("frozenlake_benchmark", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class LaunchersTest(launcher_testing.LauncherTestCase):

  @parameterized.named_parameters(
      (expected.name, expected) for expected in (_QWEN3, _GEMMA4)
  )
  def test_recipe_keeps_the_previous_launcher_settings(
      self, expected: launcher_testing.Expected
  ):
    run, hp = self.assert_launcher(expected)
    # `agentic_grpo_config.beta: 0.0`: no reference inference worker.
    self.assertNotIn("Inference command:", run.stdout)
    self.assertFalse(hp.config_mapping("cluster_config")["run_inference_node"])
    rollout = hp.config_mapping("rollout_config")
    # The tokenizer's EOS token stops a turn, not the base config's Qwen ids.
    self.assertEqual(rollout["eos_tokens"], "")
    self.assertEqual(
        hp.config_mapping("chat_parser_config"),
        {"type": "auto", "enable_thinking": False},
    )
    grpo = hp.config_mapping("agentic_grpo_config")
    self.assertEqual(grpo["max_turns"], 8)
    self.assertEqual(
        hp.config_mapping("env_kwargs"),
        {"max_steps": grpo["max_turns"], "is_slippery": False},
    )
    self.assertEqual(
        hp.config_mapping("agent_kwargs"), {"use_multistep_prompt": True}
    )
    vllm = hp.config_mapping("vllm_config")
    self.assertEqual(
        vllm["max_model_len"],
        rollout["max_prompt_length"] + rollout["total_generation_steps"] + 256,
    )
    self.assertEqual(vllm["max_num_seqs"], 512)
    self.assertEqual(vllm["max_num_batched_tokens"], 32768)
    self.assertEqual(vllm["hbm_utilization"], 0.75)
    rl_train = hp.config_mapping("rl_training_config")
    optimizer = rl_train["actor_optimizer_config"]
    # A constant learning rate, not a schedule.
    self.assertEqual(cli_config.create_learning_rate(optimizer), 1e-6)
    self.assertEqual((optimizer["b1"], optimizer["b2"]), (0.9, 0.95))
    self.assertEqual(optimizer["weight_decay"], 0.0)
    self.assertEqual(optimizer["chain_kwargs"], {"max_norm": 100.0})
    self.assertTrue(
        rl_train["checkpoint_root_directory"].endswith(
            "/checkpoints/frozenlake"
        )
    )
    self.assertEqual(grpo["reward_mode"], "env")
    self.assertEqual(hp.config["reward_functions"], [])
    self.assertEqual(hp.config_mapping("weight_sync_config")["mode"], "raiden")
    self.assertEqual(
        hp.config_mapping("data_config"),
        {
            "split": "train",
            "shuffle": True,
            "seed": 42,
            "size": 10000,
            "limit": 9600,
            "grid_size_range": [2, 9],
        },
    )
    self._assert_orchestrator_settings(hp)

  def _assert_orchestrator_settings(
      self, hp: resolve_config.DistHyperParameters
  ) -> None:
    """Checks the orchestrator's algorithm and sampling settings."""
    grpo = orchestrator.build_grpo_config(hp)
    self.assertEqual(grpo.num_generations, 8)
    self.assertEqual(grpo.num_iterations, 1)
    self.assertEqual(grpo.beta, 0.0)
    self.assertEqual((grpo.epsilon, grpo.epsilon_high), (0.003, 0.005))
    self.assertEqual(grpo.loss_algo, "gspo-token")
    self.assertEqual(grpo.loss_agg_mode, "sequence-mean-token-mean")
    self.assertEqual(grpo.kl_loss_mode, "low_var_kl")
    self.assertEqual(grpo.advantage_estimator, "rloo")
    self.assertEqual(grpo.sampler_is, "token")
    self.assertEqual(grpo.sampler_is_threshold, 2.0)
    self.assertTrue(grpo.use_rollout_logps)
    self.assertEqual(grpo.reward_mode, "env")
    generation_args = orchestrator.build_generation_args(hp)
    self.assertEqual(generation_args.temperature, 0.7)
    self.assertEqual(generation_args.top_p, 1.0)
    self.assertEqual(generation_args.top_k, 0)
    batch_config = orchestrator.build_batch_config(hp, pad_id=0, eos_id=1)
    self.assertEqual(batch_config.max_prompt_length, 2048)
    self.assertEqual(batch_config.max_response_length, 2048)
    self.assertIsNone(batch_config.max_seq_token_per_tpu)
    options = orchestrator.build_metrics_logging_options(hp)
    self.assertEqual(options.project_name, "tunix-frozenlake")
    self.assertEqual(options.log_dir, "/tmp/trellis_frozenlake")

  def test_qwen3_8b(self):
    _, hp = self.assert_launcher(_QWEN3)
    actor = hp.config_mapping("actor_model_config")
    self.assertEqual(actor["model_name"], "Qwen3-8B")
    self.assertEqual((actor["dtype"], actor["load_dtype"]), ("float32",) * 2)
    self.assertEqual(actor["remat_config"], "decoder")
    self.assertTrue(actor["use_flash_attention"])
    self.assertEqual(actor["flash_attention_block_size"], 256)
    rl_train = hp.config_mapping("rl_training_config")
    self.assertEqual(rl_train["train_micro_batch_size"], 4)
    self.assertEqual(rl_train["compute_logps_micro_batch_size"], 4)
    self.assertEqual(rl_train["compute_logps_chunk_size"], 0)
    checkpointing = rl_train["checkpointing_options"]
    self.assertEqual(checkpointing["save_interval_steps"], 1000000000)
    self.assertEqual(checkpointing["max_to_keep"], 1)
    self.assertEqual(
        hp.config_mapping("agentic_grpo_config")["max_concurrency"], 256
    )
    cluster = hp.config_mapping("cluster_config")
    self.assertTrue(cluster["artifact_root"].endswith("/artifacts/frozenlake"))
    local = cluster["local"]
    self.assertEqual(local["trainer_tpu_chips"], "0,1,2,3")
    self.assertEqual(local["rollout_tpu_chips"], "4,5,6,7")
    self.assertEqual(local["tpu_chips_per_host_bounds"], "1,4,1")
    self.assertEqual(local["tpu_host_bounds"], "1,1,1")
    batch_config = orchestrator.build_batch_config(hp, pad_id=0, eos_id=1)
    self.assertEqual(batch_config.num_data_shards, 1)

  def test_gemma4_e2b(self):
    _, hp = self.assert_launcher(_GEMMA4)
    for role in ("actor_model_config", "rollout_model_config"):
      self.assertEqual(hp.config_mapping(role)["model_name"], "gemma-4-e2b")
    rl_train = hp.config_mapping("rl_training_config")
    self.assertEqual(rl_train["train_micro_batch_size"], 2)
    self.assertEqual(rl_train["compute_logps_micro_batch_size"], 2)
    self.assertEqual(rl_train["compute_logps_chunk_size"], 2048)
    self.assertEqual(
        hp.config_mapping("agentic_grpo_config")["max_concurrency"], 512
    )
    cluster = hp.config_mapping("cluster_config")
    self.assertTrue(
        cluster["artifact_root"].endswith(
            "/artifacts/gemma4_e2b_dist_frozenlake"
        )
    )
    local = cluster["local"]
    self.assertEqual(local["trainer_tpu_chips"], "0,1")
    self.assertEqual(local["rollout_tpu_chips"], "2,3")
    self.assertEqual(local["tpu_chips_per_host_bounds"], "1,2,1")
    batch_config = orchestrator.build_batch_config(hp, pad_id=0, eos_id=1)
    self.assertEqual(batch_config.train_micro_batch_size, 2)
    # The trainer's FSDP 2 shards each batch two ways.
    self.assertEqual(batch_config.num_data_shards, 2)

  def test_rpc_deadline_follows_the_episode_timeout(self):
    # As in the previous launcher: at least 1800s, longer when the weight-sync
    # phases need it.
    for episode_timeout, rpc_timeout_s in ((600, 1800.0), (1800, 2100.0)):
      with self.subTest(episode_timeout=episode_timeout):
        run = self.dry_run(
            _QWEN3.script,
            f"agentic_grpo_config.episode_timeout={episode_timeout}",
        )
        _, derived = orchestrator.resolve_weight_sync_timeouts(run.resolve())
        self.assertEqual(derived, rpc_timeout_s)

  def test_user_args_override_launcher_values(self):
    run = self.dry_run(
        _GEMMA4.script,
        "rl_training_config.checkpoint_root_directory=/tmp/frozenlake_ckpt",
        "cluster_config.local.trainer_tpu_chips=4,5",
    )
    config = run.resolve().config
    self.assertEqual(
        config["rl_training_config"]["checkpoint_root_directory"],
        "/tmp/frozenlake_ckpt",
    )
    self.assertEqual(
        config["cluster_config"]["local"]["trainer_tpu_chips"], "4,5"
    )

  def test_benchmark_launch_resolves(self):
    workload = _load_benchmark_workload()
    output = pathlib.Path(self.create_tempdir().full_path)
    model_dir = output / "model"
    command, env = workload.dist_launch(workload.DEFAULTS, output, model_dir)
    self.assertEqual(
        pathlib.Path(command[1]).resolve(),
        (launcher_testing.EXAMPLES_DIR / _GEMMA4.script).resolve(),
    )
    # `launch.sh` would reject the old launchers' environment variables.
    self.assertEqual(env, {"LOG_ROOT": str(output)})
    config = self.dry_run(_GEMMA4.script, *command[2:]).resolve().config
    cluster = config["cluster_config"]
    self.assertEqual(
        cluster["orchestrator_main"], f"{workload.MODULE}.dist_main"
    )
    runtime = "tunix.dist_rl.examples.rl_efficiency_benchmark.runtime"
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
    self.assertEqual(rl_train["metrics_logging_options"]["log_dir"], "")
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
