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

"""Dry-runs every distributed RL recipe script."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.launch.common import launcher_testing

_CI_SMOKE = launcher_testing.Expected(
    name="ci_smoke_gsm8k_qwen3_0p6b",
    script="recipes/ci_smoke_gsm8k_qwen3_0p6b.sh",
    example="math_gsm8k_dist",
    module="gsm8k",
    target="local",
    batch_size=2,
    mini_batch_size=2,
    num_generations=2,
    max_steps=2,
    actor_mesh=(1, 4),
    rollout_mesh=(1, 4),
    model_id="Qwen/Qwen3-0.6B",
)
_CI_SMOKE_MAXTEXT = launcher_testing.Expected(
    name="ci_smoke_gsm8k_qwen3_0p6b_maxtext",
    script="recipes/ci_smoke_gsm8k_qwen3_0p6b_maxtext.sh",
    example="math_gsm8k_dist",
    module="gsm8k",
    target="local",
    batch_size=2,
    mini_batch_size=2,
    num_generations=2,
    max_steps=2,
    actor_mesh=(1, 4),
    rollout_mesh=(1, 4),
    model_id="Qwen/Qwen3-0.6B",
)
_BENCHMARK_FROZENLAKE_GEMMA4 = launcher_testing.Expected(
    name="benchmark_frozenlake_gemma4_e2b",
    script="recipes/benchmark_frozenlake_gemma4_e2b.sh",
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
# Copybara replaces `k8s_launcher.sh`'s internal default image with
# `$TUNIX_IMAGE` (`copy.bara.sky`), so the GKE dry runs set one.
_K8S_IMAGE = "cluster_config.k8s.tunix_image=test-image"
_TRELLIS = launcher_testing.Expected(
    name="trellis_gsm8k_qwen3p5_35b",
    script="recipes/trellis_gsm8k_qwen3p5_35b.sh",
    example="math_gsm8k_dist",
    module="gsm8k",
    target="k8s",
    batch_size=4,
    mini_batch_size=4,
    num_generations=2,
    max_steps=2,
    actor_mesh=(4, 2, 1),
    rollout_mesh=(1, 8),
    model_id="Qwen/Qwen3.5-35B-A3B",
    extra_args=(_K8S_IMAGE,),
)


class RecipesTest(launcher_testing.LauncherTestCase):

  @parameterized.named_parameters(
      (expected.name, expected)
      for expected in (
          _CI_SMOKE,
          _CI_SMOKE_MAXTEXT,
          _BENCHMARK_FROZENLAKE_GEMMA4,
          _TRELLIS,
      )
  )
  def test_recipe_contract(self, expected: launcher_testing.Expected):
    self.assert_launcher(expected)

  def test_ci_smoke(self):
    _, hp = self.assert_launcher(_CI_SMOKE)
    config = hp.config
    self.assertEqual(config["rollout_engine"], "inprocess_vllm")
    self.assertEqual(config["rollout_config"]["max_prompt_length"], 512)
    self.assertEqual(config["rollout_config"]["total_generation_steps"], 512)
    self.assertEqual(config["weight_sync_config"]["mode"], "raiden")
    self.assertEqual(config["data_config"]["split"], "train[:16]")
    self.assertFalse(config["data_config"]["shuffle"])
    self.assertEqual(config["agentic_grpo_config"]["beta"], 0.0)
    rl_train = config["rl_training_config"]
    self.assertEqual(rl_train["eval_every_n_steps"], 999)
    self.assertEqual(rl_train["checkpointing_options"]["max_to_keep"], 2)
    self.assertEqual(
        rl_train["metrics_logging_options"]["project_name"], "ci-smoke-gsm8k"
    )
    cluster = hp.config_mapping("cluster_config")
    self.assertEqual(cluster["wait_timeout_secs"], 900)
    expected_local = {
        "auto_detect_pci_tpus": True,
        "trainer_tpu_chips": "0,1,2,3",
        "rollout_tpu_chips": "4,5,6,7",
        "tpu_chips_per_host_bounds": "2,2,1",
        "tpu_host_bounds": "1,1,1",
        "allow_multiple_libtpu_load": True,
    }
    self.assertEqual(
        {key: cluster["local"][key] for key in expected_local}, expected_local
    )
    # `model_download_path` derives from `cluster_config.artifact_root`.
    self.assertTrue(
        cluster["artifact_root"].endswith("artifacts/qwen3_dist_gsm8k")
    )
    for role in ("actor_model_config", "rollout_model_config"):
      self.assertTrue(
          config[role]["model_download_path"].endswith(
              "artifacts/qwen3_dist_gsm8k/models/Qwen3-0.6B"
          ),
          msg=role,
      )

  def test_ci_smoke_maxtext_uses_maxtext_actor_and_rollout(self):
    _, hp = self.assert_launcher(_CI_SMOKE_MAXTEXT)
    config = hp.config
    self.assertTrue(
        config["cluster_config"]["artifact_root"].endswith(
            "artifacts/qwen3_dist_gsm8k"
        )
    )
    actor = config["actor_model_config"]
    self.assertEqual(actor["model_source"], "maxtext")
    self.assertTrue(
        actor["model_path"].endswith(
            "artifacts/qwen3_dist_gsm8k/maxtext_models/qwen3-0.6b/0/items"
        )
    )
    self.assertEqual(actor["kwargs"]["maxtext_model_name"], "qwen3-0.6b")
    # Raiden syncs MaxText weights into a MaxText (`MaxTextForCausalLM`)
    # rollout.
    rollout = config["rollout_model_config"]
    self.assertEqual(rollout["model_source"], "maxtext")
    self.assertEqual(rollout["kwargs"]["maxtext_model_name"], "qwen3-0.6b")
    self.assertEqual(config["rollout_engine"], "vllm")
    # Only the orchestrator skips the logprob check; the rollout node rejects
    # `rollout_config.return_logprobs: false`.
    self.assertTrue(config["rollout_config"]["return_logprobs"])
    self.assertFalse(config["agentic_grpo_config"]["use_rollout_logps"])
    weight_sync_cfg = config["weight_sync_config"]
    self.assertEqual(weight_sync_cfg["mode"], "raiden")
    self.assertTrue(weight_sync_cfg["use_weight_converter"])
    self.assertTrue(weight_sync_cfg["verify_weights"])
    self.assertTrue(
        config["rl_training_config"]["checkpoint_root_directory"].endswith(
            "artifacts/qwen3_dist_gsm8k/checkpoints_maxtext"
        )
    )

  def test_benchmark_frozenlake_gemma4_e2b(self):
    _, hp = self.assert_launcher(_BENCHMARK_FROZENLAKE_GEMMA4)
    for role in ("actor_model_config", "rollout_model_config"):
      self.assertEqual(hp.config_mapping(role)["model_name"], "gemma-4-e2b")
    # The trainer's FSDP 2 shards each batch two ways (the orchestrator's
    # `num_data_shards` is `fsdp * dp * expert`).
    self.assertEqual(
        hp.mesh_axes("actor_model_config"),
        {"fsdp": 2, "tp": 1, "expert": 1, "dp": 1},
    )
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
    # The rest comes from `frozenlake.yaml` (see `frozenlake_dist/README.md`).
    actor = hp.config_mapping("actor_model_config")
    self.assertEqual((actor["dtype"], actor["load_dtype"]), ("float32",) * 2)
    self.assertEqual(actor["remat_config"], "decoder")
    self.assertTrue(actor["use_flash_attention"])
    self.assertEqual(actor["flash_attention_block_size"], 256)
    # The tokenizer's EOS token stops a turn, not the base config's Qwen ids.
    self.assertEqual(hp.config_mapping("rollout_config")["eos_tokens"], "")
    self.assertEqual(
        hp.config_mapping("chat_parser_config"),
        {"type": "auto", "enable_thinking": False},
    )
    vllm = hp.config_mapping("vllm_config")
    self.assertEqual(
        (vllm["max_num_seqs"], vllm["max_num_batched_tokens"]), (512, 32768)
    )

  def test_trellis(self):
    run, hp = self.assert_launcher(_TRELLIS)
    config = hp.config
    self.assertEqual(config["rl_training_config"]["train_micro_batch_size"], 8)
    self.assertEqual(config["rollout_engine"], "vllm")
    self.assertFalse(config["vllm_config"]["enable_prefix_caching"])
    actor = config["actor_model_config"]
    self.assertEqual(actor["model_source"], "maxtext")
    self.assertEqual(
        actor["model_path"],
        "/app/artifacts/math_gsm8k_dist/checkpoints/qwen3.5-35b-a3b/0/items",
    )
    self.assertEqual(config["rollout_model_config"]["model_source"], "maxtext")
    self.assertEqual(
        config["model_config"]["kwargs"],
        {
            "maxtext_model_name": "qwen3.5-35b-a3b",
            "maxtext_output_dir": "/app/artifacts/math_gsm8k_dist/maxtext",
        },
    )
    weight_sync_cfg = config["weight_sync_config"]
    self.assertEqual(weight_sync_cfg["raiden_use_ffi"], 1)
    self.assertTrue(weight_sync_cfg["prefuse_moe_weights"])
    self.assertTrue(weight_sync_cfg["verify_weights"])
    k8s = config["cluster_config"]["k8s"]
    self.assertEqual(k8s["trainer_extra_env"], "")
    self.assertEqual(k8s["trainer"]["jobset_yaml"], "jobset.pathways.yaml")
    self.assertEqual(k8s["trainer"]["tpu_slice"], "tpuv5:2x2x2")
    self.assertEqual(k8s["rollout"]["jobset_yaml"], "jobset.tpu.yaml")
    self.assertEqual(k8s["rollout"]["tpu_slice"], "tpuv5:2x2x2")
    self.assertEqual(config["cluster_config"]["rollout_replicas"], 2)
    # Ports come from `math_gsm8k_dist/k8s_launcher.sh`; sequence lengths from
    # `math_gsm8k_dist/gsm8k.yaml`.
    self.assertEqual(config["cluster_config"]["ports"]["trainer"], 20002)
    self.assertEqual(config["rollout_config"]["max_prompt_length"], 1024)
    self.assertEqual(config["rollout_config"]["total_generation_steps"], 1024)
    self.assertIn("RAIDEN_USE_FFI=1", run.stdout)
    self.assertIn("RAIDEN_USE_FFI=0", run.stdout)
    self.assertIn("testuser-roll-1", run.stdout)

  def test_user_args_override_recipe_values(self):
    run = self.dry_run(
        "recipes/trellis_gsm8k_qwen3p5_35b.sh",
        _K8S_IMAGE,
        "rl_training_config.max_steps=7",
        "cluster_config.rollout_replicas=3",
    )
    config = run.resolve().config
    self.assertEqual(config["rl_training_config"]["max_steps"], 7)
    self.assertEqual(config["cluster_config"]["rollout_replicas"], 3)
    self.assertIn("testuser-roll-2", run.stdout)


if __name__ == "__main__":
  absltest.main()
