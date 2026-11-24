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

"""Unit tests for distributed RL `resolve_config` built on `HyperParameters`."""

from __future__ import annotations

import json
import os
import re
import subprocess
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import omegaconf
from tunix.cli import config as cli_config
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.common import gcs_cache
from tunix.dist_rl.launch.common import resolve_config


class ResolveConfigTest(absltest.TestCase):

  def test_rejects_program_name_or_base_yaml(self):
    for argv in (
        ["resolve_config", "batch_size=8"],
        [str(resolve_config.DEFAULT_BASE_CONFIG_PATH), "batch_size=8"],
        ["batch_size=8", "bare_token"],
    ):
      with self.subTest(argv=argv):
        with self.assertRaisesRegex(ValueError, "no program name"):
          resolve_config.initialize(argv)

  def test_main_drops_program_name(self):
    with mock.patch("sys.stdout") as stdout:
      self.assertEqual(
          resolve_config.main(["any_prog", "cluster_config.run_id=main_run"]),
          0,
      )
    rendered = "".join(c.args[0] for c in stdout.write.call_args_list)
    self.assertIn("export RUN_ID=main_run\n", rendered)

  def test_main_requires_program_name(self):
    for argv in ([], ["cluster_config.run_id=main_run"]):
      with self.subTest(argv=argv):
        with self.assertRaisesRegex(ValueError, "must start with the program"):
          resolve_config.main(argv)

  def test_subclasses_hyperparameters_and_matches_launcher_defaults(self):
    hp = resolve_config.initialize([])
    self.assertIsInstance(hp, cli_config.HyperParameters)
    self.assertEqual(hp.config["model_config"]["model_name"], "Qwen3-1.7B")
    self.assertEqual(hp.config["model_config"]["model_id"], "Qwen/Qwen3-1.7B")
    self.assertEqual(hp.config["model_config"]["dtype"], "float32")
    self.assertEqual(hp.config["model_config"]["load_dtype"], "float32")
    self.assertEqual(hp.config["model_config"]["mesh"]["shape"], "(1,2)")
    self.assertEqual(
        hp.config["model_config"]["mesh"]["axis_names"], "('fsdp','tp')"
    )
    self.assertEqual(hp.config["batch_size"], 4)
    self.assertEqual(hp.config["rl_training_config"]["mini_batch_size"], 4)
    self.assertEqual(hp.config["rl_training_config"]["max_steps"], 1)
    self.assertEqual(hp.config["rollout_config"]["max_prompt_length"], 1024)
    self.assertEqual(
        hp.config["rollout_config"]["total_generation_steps"], 1024
    )
    self.assertEqual(hp.config["agentic_grpo_config"]["num_generations"], 8)
    self.assertEqual(hp.config["agentic_grpo_config"]["beta"], 0.0)
    self.assertEqual(hp.config["agentic_grpo_config"]["epsilon"], 0.2)
    self.assertEqual(hp.config["weight_sync_config"]["mode"], "none")
    self.assertEqual(
        hp.config["cluster_config"]["local"]["trainer_devices"], "0,1"
    )
    self.assertEqual(
        hp.config["cluster_config"]["local"]["rollout_devices"], "2,3"
    )
    # The default beta is 0, so no reference inference node is started.
    self.assertFalse(hp.config["cluster_config"]["run_inference_node"])

  def test_model_config_inheritance_and_role_specific_override(self):
    hp = resolve_config.initialize([
        "model_config.model_name=Qwen3-0.6B",
        "model_config.model_id=Qwen/Qwen3-0.6B",
        "model_config.dtype=bfloat16",
        "model_config.load_dtype=bfloat16",
        "actor_model_config.dtype=float32",
        "model_config.mesh.shape=(2,2)",
        "rollout_model_config.mesh.shape=(1,4)",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertEqual(
        hp.config["actor_model_config"]["model_name"], "Qwen3-0.6B"
    )
    self.assertEqual(
        hp.config["rollout_model_config"]["model_name"], "Qwen3-0.6B"
    )
    self.assertEqual(hp.config["actor_model_config"]["dtype"], "float32")
    self.assertEqual(hp.config["rollout_model_config"]["dtype"], "bfloat16")
    self.assertEqual(
        hp.mesh_axes("actor_model_config"),
        {"fsdp": 2, "tp": 2, "expert": 1, "dp": 1},
    )
    self.assertEqual(
        hp.mesh_axes("rollout_model_config"),
        {"fsdp": 1, "tp": 4, "expert": 1, "dp": 1},
    )
    self.assertFalse(hp.config["cluster_config"]["run_inference_node"])

  def test_override_config_file_merges_and_cli_wins(self):
    tmp_dir = self.create_tempdir()
    recipe_yaml = tmp_dir.create_file(
        "gsm8k_verl.yaml",
        """\
model_config:
  model_name: "Qwen3-4B"
  model_id: "Qwen/Qwen3-4B"
  dtype: "bfloat16"
batch_size: 8
rl_training_config:
  mini_batch_size: 4
  max_steps: 20
weight_sync_config:
  mode: "raiden"
""",
    )
    hp = resolve_config.initialize([
        f"override_config_file={recipe_yaml.full_path}",
        "rl_training_config.max_steps=5",
    ])
    self.assertEqual(hp.config["actor_model_config"]["model_name"], "Qwen3-4B")
    self.assertEqual(hp.config["batch_size"], 8)
    self.assertEqual(hp.config["rl_training_config"]["mini_batch_size"], 4)
    self.assertEqual(hp.config["rl_training_config"]["max_steps"], 5)
    # Sibling keys in rl_training_config are preserved (not wiped).
    self.assertEqual(
        hp.config["rl_training_config"]["train_micro_batch_size"], 1
    )
    self.assertEqual(hp.config["weight_sync_config"]["mode"], "raiden")

  def test_tokenizer_uses_base_cli_validation(self):
    # A Hugging Face hub id or local path works without HF_TOKEN.
    env_without_token = {
        k: v for k, v in os.environ.items() if k != "HF_TOKEN"
    }
    with mock.patch.dict(os.environ, env_without_token, clear=True):
      hp_hub = resolve_config.initialize([])
      self.assertEqual(
          hp_hub.config["tokenizer_config"]["tokenizer_path"],
          "Qwen/Qwen3-1.7B",
      )
      hp = resolve_config.initialize(
          ["tokenizer_config.tokenizer_path=/tmp/models/qwen"]
      )
    self.assertEqual(
        hp.config["tokenizer_config"]["tokenizer_path"], "/tmp/models/qwen"
    )
    with self.assertRaisesRegex(
        ValueError, "tokenizer_path must be specified"
    ):
      resolve_config.initialize(["tokenizer_config.tokenizer_path="])
    with self.assertRaisesRegex(ValueError, "tokenizer_type"):
      resolve_config.initialize(["tokenizer_config.tokenizer_type=bpe"])

  def test_unknown_key_fails_loud(self):
    with self.assertRaisesRegex(
        ValueError, "rl_training_config.nonexistent_flag"
    ):
      resolve_config.initialize(["rl_training_config.nonexistent_flag=123"])

  def test_run_inference_node_defaults_to_kl_penalty(self):
    cluster = resolve_config.initialize([]).config["cluster_config"]
    self.assertFalse(cluster["run_inference_node"])
    kl_penalty = "agentic_grpo_config.beta=0.04"
    cluster = resolve_config.initialize([kl_penalty]).config["cluster_config"]
    self.assertTrue(cluster["run_inference_node"])
    # A reference worker started separately registers through discovery; there
    # is no address to configure.
    cluster = resolve_config.initialize(
        [kl_penalty, "cluster_config.run_inference_node=false"]
    ).config["cluster_config"]
    self.assertFalse(cluster["run_inference_node"])
    with self.assertRaisesRegex(ValueError, "cluster_config.inference_addr"):
      resolve_config.initialize(["cluster_config.inference_addr=host:20002"])

  def test_artifact_root_and_model_download_path_defaults(self):
    local = resolve_config.initialize([
        "cluster_config.artifact_root=artifacts/my_task",
        "model_config.model_name=Qwen3-0.6B",
    ])
    root = resolve_config.REPO_ROOT / "artifacts" / "my_task"
    self.assertEqual(local.config["cluster_config"]["artifact_root"], str(root))
    self.assertEqual(
        local.config["actor_model_config"]["model_download_path"],
        str(root / "models" / "Qwen3-0.6B"),
    )

    k8s = resolve_config.initialize([
        "cluster_config.target=k8s",
        "cluster_config.artifact_root=artifacts/my_task",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertEqual(
        k8s.config["cluster_config"]["artifact_root"], "artifacts/my_task"
    )
    self.assertEqual(
        k8s.config["rollout_model_config"]["model_download_path"],
        "artifacts/my_task/models/Qwen3-1.7B",
    )

    with self.assertRaisesRegex(ValueError, "artifact_root must not be empty"):
      resolve_config.initialize(["cluster_config.artifact_root="])

  def test_artifact_root_rejects_uris(self):
    # `pathlib` would turn the URI into the relative path `gs:/bucket/run`.
    for target in ("local", "k8s", "external"):
      with self.subTest(target=target):
        with self.assertRaisesRegex(
            ValueError,
            "cluster_config.artifact_root must be a filesystem path, got"
            " 'gs://bucket/run'",
        ):
          resolve_config.initialize([
              f"cluster_config.target={target}",
              "agentic_grpo_config.beta=0.0",
              "cluster_config.artifact_root=gs://bucket/run",
          ])

  def test_output_paths_can_be_uris(self):
    hp = resolve_config.initialize([
        "cluster_config.target=k8s",
        "agentic_grpo_config.beta=0.0",
        "rl_training_config.checkpoint_root_directory=gs://bucket/ckpt",
        "rl_training_config.metrics_logging_options.log_dir=gs://bucket/logs",
        "trajectory_store_config.enable=true",
        "trajectory_store_config.dir=gs://bucket/store",
    ])
    rl_train = hp.config_mapping("rl_training_config")
    self.assertEqual(rl_train["checkpoint_root_directory"], "gs://bucket/ckpt")
    self.assertEqual(
        rl_train["metrics_logging_options"]["log_dir"], "gs://bucket/logs"
    )
    self.assertEqual(
        hp.config_mapping("trajectory_store_config")["dir"], "gs://bucket/store"
    )

  def test_null_sandbox_namespace_follows_the_gke_namespace(self):
    hp = resolve_config.initialize([
        "cluster_config.target=k8s",
        "cluster_config.k8s.namespace=team-ns",
        "kubernetes_config.namespace=null",
    ])
    self.assertEqual(hp.config["kubernetes_config"]["namespace"], "team-ns")
    self.assertIn(
        "export SANDBOX_NAMESPACE=team-ns",
        resolve_config.render_bash_exports(hp),
    )
    # An explicit sandbox namespace wins.
    hp = resolve_config.initialize([
        "cluster_config.target=k8s",
        "cluster_config.k8s.namespace=team-ns",
        "kubernetes_config.namespace=swe-ns",
    ])
    self.assertEqual(hp.config["kubernetes_config"]["namespace"], "swe-ns")
    with self.assertRaisesRegex(ValueError, "needs cluster_config.target=k8s"):
      resolve_config.initialize(["kubernetes_config.namespace=null"])

  def test_file_trajectory_store_dir_defaults_under_artifact_root(self):
    hp = resolve_config.initialize([
        "cluster_config.target=external",
        "cluster_config.artifact_root=/artifacts/run",
        "trajectory_store_config.enable=true",
    ])
    self.assertEqual(
        hp.config_mapping("trajectory_store_config")["dir"],
        "/artifacts/run/trajectory_store",
    )
    hp = resolve_config.initialize([
        "cluster_config.target=external",
        "cluster_config.artifact_root=/artifacts/run",
        "trajectory_store_config.dir=/stores/run",
    ])
    self.assertEqual(
        hp.config_mapping("trajectory_store_config")["dir"], "/stores/run"
    )

  def test_cluster_target_choices(self):
    with self.assertRaisesRegex(
        ValueError,
        r"cluster_config.target must be one of \('local', 'k8s', 'external'\),"
        r" got 'borg'",
    ):
      resolve_config.initialize(["cluster_config.target=borg"])

  def test_external_target_requires_an_absolute_artifact_root(self):
    cluster = resolve_config.initialize([
        "cluster_config.target=external",
        "cluster_config.artifact_root=/artifacts/run_1",
        "agentic_grpo_config.beta=0.04",
    ]).config_mapping("cluster_config")
    self.assertEqual(cluster["target"], "external")
    # Never joined to the repo root.
    self.assertEqual(cluster["artifact_root"], "/artifacts/run_1")
    # Only `target: k8s` rejects a reference worker.
    self.assertTrue(cluster["run_inference_node"])
    # Each process would resolve a relative root against its own working
    # directory, including the base config's.
    for overrides in (["cluster_config.artifact_root=artifacts/run_1"], []):
      with self.subTest(overrides=overrides):
        with self.assertRaisesRegex(
            ValueError,
            "cluster_config.target=external requires an absolute"
            " cluster_config.artifact_root",
        ):
          resolve_config.initialize(
              ["cluster_config.target=external", *overrides]
          )

  def test_rollout_replicas(self):
    k8s = ("cluster_config.target=k8s", "agentic_grpo_config.beta=0.0")
    external = (
        "cluster_config.target=external",
        "cluster_config.artifact_root=/artifacts/run",
    )

    def replicas(*overrides: str) -> object:
      cluster = resolve_config.initialize(list(overrides)).config_mapping(
          "cluster_config"
      )
      return cluster["rollout_replicas"]

    self.assertEqual(replicas(), 1)
    self.assertEqual(replicas(*k8s, "cluster_config.rollout_replicas=2"), 2)
    self.assertEqual(
        replicas(*external, "cluster_config.rollout_replicas=16"), 16
    )
    # `local.sh` starts one rollout.
    with self.assertRaisesRegex(
        ValueError, "cluster_config.rollout_replicas must be 1, got 2"
    ):
      replicas("cluster_config.rollout_replicas=2")
    for value in ("0", "-1", "1.5", "two", "true"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(
            ValueError, "cluster_config.rollout_replicas must be an int >= 1"
        ):
          replicas(*k8s, f"cluster_config.rollout_replicas={value}")
    # The k8s-only key it replaces is unknown now.
    with self.assertRaisesRegex(
        ValueError,
        "Unknown configuration key 'cluster_config.k8s.rollout.replicas'",
    ):
      replicas(*k8s, "cluster_config.k8s.rollout.replicas=2")
    # `k8s.sh` deploys `ROLLOUT_REPLICAS` rollouts.
    exports = resolve_config.render_bash_exports(
        resolve_config.initialize([*k8s, "cluster_config.rollout_replicas=2"])
    )
    self.assertIn("export ROLLOUT_REPLICAS=2\n", exports)
    self.assertNotIn("K8S_ROLLOUT_REPLICAS", exports)

  def test_min_rollout_replicas(self):
    k8s = ("cluster_config.target=k8s", "agentic_grpo_config.beta=0.0")
    external = (
        "cluster_config.target=external",
        "cluster_config.artifact_root=/artifacts/run",
    )

    def min_replicas(*overrides: str) -> object:
      cluster = resolve_config.initialize(list(overrides)).config_mapping(
          "cluster_config"
      )
      return cluster["min_rollout_replicas"]

    self.assertIsNone(min_replicas())
    self.assertEqual(min_replicas("cluster_config.min_rollout_replicas=1"), 1)
    self.assertIsNone(
        min_replicas(
            *k8s,
            "cluster_config.rollout_replicas=4",
            "cluster_config.min_rollout_replicas=null",
        )
    )
    self.assertEqual(
        min_replicas(
            *k8s,
            "cluster_config.rollout_replicas=4",
            "cluster_config.min_rollout_replicas=2",
        ),
        2,
    )
    self.assertEqual(
        min_replicas(
            *k8s,
            "cluster_config.rollout_replicas=4",
            "cluster_config.min_rollout_replicas=4",
        ),
        4,
    )
    self.assertEqual(
        min_replicas(
            *external,
            "cluster_config.rollout_replicas=16",
            "cluster_config.min_rollout_replicas=4",
        ),
        4,
    )
    for value in ("0", "-1", "1.5", "two", "true", "false"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(
            ValueError,
            "cluster_config.min_rollout_replicas must be null or an int >= 1",
        ):
          min_replicas(
              *k8s,
              "cluster_config.rollout_replicas=4",
              f"cluster_config.min_rollout_replicas={value}",
          )
    with self.assertRaisesRegex(
        ValueError,
        r"cluster_config\.min_rollout_replicas \(3\) must be <="
        r" cluster_config\.rollout_replicas \(2\)",
    ):
      min_replicas(
          *k8s,
          "cluster_config.rollout_replicas=2",
          "cluster_config.min_rollout_replicas=3",
      )
    with self.assertRaisesRegex(
        ValueError,
        r"cluster_config\.min_rollout_replicas \(2\) must be <="
        r" cluster_config\.rollout_replicas \(1\)",
    ):
      min_replicas("cluster_config.min_rollout_replicas=2")
    exports = resolve_config.render_bash_exports(
        resolve_config.initialize([
            *k8s,
            "cluster_config.rollout_replicas=4",
            "cluster_config.min_rollout_replicas=2",
        ])
    )
    self.assertIn("export ROLLOUT_REPLICAS=4\n", exports)
    self.assertNotIn("MIN_ROLLOUT_REPLICAS", exports)

  def test_output_dirs_default_under_artifact_root(self):
    def output_dirs(*overrides: str) -> tuple[object, object]:
      rl_train = resolve_config.initialize(list(overrides)).config_mapping(
          "rl_training_config"
      )
      return (
          rl_train["checkpoint_root_directory"],
          rl_train["metrics_logging_options"]["log_dir"],
      )

    root = resolve_config.REPO_ROOT / "artifacts" / "my_task"
    self.assertEqual(
        output_dirs("cluster_config.artifact_root=artifacts/my_task"),
        (str(root / "checkpoints"), str(root / "metrics")),
    )
    # A relative k8s root stays relative to the container working directory.
    self.assertEqual(
        output_dirs(
            "cluster_config.target=k8s",
            "agentic_grpo_config.beta=0.0",
            "cluster_config.artifact_root=artifacts/my_task",
        ),
        ("artifacts/my_task/checkpoints", "artifacts/my_task/metrics"),
    )
    self.assertEqual(
        output_dirs(
            "cluster_config.target=external",
            "cluster_config.artifact_root=/artifacts/run_1",
        ),
        ("/artifacts/run_1/checkpoints", "/artifacts/run_1/metrics"),
    )
    # Explicit directories are kept.
    self.assertEqual(
        output_dirs(
            "rl_training_config.checkpoint_root_directory=/ckpt",
            "rl_training_config.metrics_logging_options.log_dir=/logs",
        ),
        ("/ckpt", "/logs"),
    )
    # Null metrics logging options log no metrics; nothing is derived.
    rl_train = resolve_config.initialize(
        ["rl_training_config.metrics_logging_options=null"]
    ).config_mapping("rl_training_config")
    self.assertIsNone(rl_train["metrics_logging_options"])
    with self.assertRaisesRegex(
        TypeError,
        "rl_training_config.metrics_logging_options must be a mapping or null",
    ):
      resolve_config.initialize(
          ["rl_training_config.metrics_logging_options=off"]
      )

  def test_trajectory_logging(self):
    dist_train = resolve_config.initialize([]).config_mapping(
        "dist_training_config"
    )
    self.assertIs(dist_train["trajectory_logging"], True)
    self.assertEqual(dist_train["trajectory_log_dir"], "")
    dist_train = resolve_config.initialize(
        ["dist_training_config.trajectory_logging=false"]
    ).config_mapping("dist_training_config")
    self.assertIs(dist_train["trajectory_logging"], False)
    # A directory for trajectories that are never logged is contradictory.
    with self.assertRaisesRegex(
        ValueError,
        "trajectory_logging=false logs no trajectories, but"
        " dist_training_config.trajectory_log_dir is set",
    ):
      resolve_config.initialize([
          "dist_training_config.trajectory_logging=false",
          "dist_training_config.trajectory_log_dir=/logs/trajectories",
      ])
    for value in ("yes_please", "0"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(
            TypeError, "dist_training_config.trajectory_logging must be a bool"
        ):
          resolve_config.initialize(
              [f"dist_training_config.trajectory_logging={value}"]
          )

  def test_rollout_logps_settings(self):
    rs_band = [
        "agentic_grpo_config.sampler_rs_min=0.999",
        "agentic_grpo_config.sampler_rs_max=1.002",
    ]
    grpo = resolve_config.initialize([
        "agentic_grpo_config.sampler_rs=geometric",
        *rs_band,
        "agentic_grpo_config.seq_logprob_error_threshold=2.0",
    ]).config_mapping("agentic_grpo_config")
    self.assertIs(grpo["use_rollout_logps"], True)
    self.assertEqual(grpo["sampler_rs"], "geometric")
    # `sampler_is` keeps its "token" default without rollout log-probabilities.
    grpo = resolve_config.initialize(
        ["agentic_grpo_config.use_rollout_logps=false"]
    ).config_mapping("agentic_grpo_config")
    self.assertEqual(grpo["sampler_is"], "token")
    for overrides, ignored in (
        (["agentic_grpo_config.sampler_rs=geometric", *rs_band], "sampler_rs"),
        (
            ["agentic_grpo_config.seq_logprob_error_threshold=2.0"],
            "seq_logprob_error_threshold",
        ),
    ):
      with self.subTest(ignored=ignored):
        with self.assertRaisesRegex(
            ValueError,
            rf"agentic_grpo_config sets \['{ignored}'\], which only take"
            " effect with agentic_grpo_config.use_rollout_logps=true",
        ):
          resolve_config.initialize(
              ["agentic_grpo_config.use_rollout_logps=false", *overrides]
          )
    with self.assertRaisesRegex(
        TypeError, "agentic_grpo_config.use_rollout_logps must be a bool"
    ):
      resolve_config.initialize(["agentic_grpo_config.use_rollout_logps=0"])

  def test_invariants_fail_loud(self):
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      resolve_config.initialize(
          ["batch_size=5", "rl_training_config.mini_batch_size=2"]
      )

    with self.assertRaisesRegex(ValueError, "must be divisible by.*train_micro"):
      resolve_config.initialize([
          "batch_size=2",
          "rl_training_config.mini_batch_size=1",
          "agentic_grpo_config.num_generations=3",
          "rl_training_config.train_micro_batch_size=2",
      ])

    with self.assertRaisesRegex(ValueError, "requires.*model_path"):
      resolve_config.initialize(
          ["model_config.model_source=maxtext", "model_config.model_path="]
      )

    # GRPO needs at least two generations per prompt.
    with self.assertRaisesRegex(ValueError, "num_generations must be > 1"):
      resolve_config.initialize(["agentic_grpo_config.num_generations=1"])

  def test_rollout_random_init_defaults_to_raiden_weight_sync(self):
    hp = resolve_config.initialize([])
    self.assertIsNone(hp.config["vllm_config"]["init_with_random_weights"])
    self.assertIs(resolve_config.rollout_init_with_random_weights(hp), False)
    for mode, init, expected in (
        # Null: random weights only when Raiden's bootstrap sync replaces them.
        ("raiden", "null", True),
        ("none", "null", False),
        ("in_memory", "null", False),
        # Explicit values are kept; any mode may load the checkpoint.
        ("raiden", "true", True),
        ("raiden", "false", False),
        ("none", "false", False),
    ):
      with self.subTest(mode=mode, init=init):
        hp = resolve_config.initialize([
            f"weight_sync_config.mode={mode}",
            f"vllm_config.init_with_random_weights={init}",
        ])
        self.assertIs(
            resolve_config.rollout_init_with_random_weights(hp), expected
        )

  def test_rollout_random_init_without_raiden_fails_loud(self):
    for mode in ("none", "in_memory"):
      with self.subTest(mode=mode):
        with self.assertRaisesRegex(
            ValueError,
            f"weight_sync_config.mode='{mode}' never loads the trainer's"
            " weights.*sample from random weights for the whole run",
        ):
          resolve_config.initialize([
              f"weight_sync_config.mode={mode}",
              "vllm_config.init_with_random_weights=true",
          ])
    # Only the in-process vLLM rollout reads it.
    for rollout_engine in ("vllm", "vanilla"):
      with self.subTest(rollout_engine=rollout_engine):
        resolve_config.initialize([
            f"rollout_engine={rollout_engine}",
            "vllm_config.init_with_random_weights=true",
        ])
    for init in ("1", "yes_please"):
      with self.subTest(init=init):
        with self.assertRaisesRegex(
            TypeError, "init_with_random_weights must be a bool or null"
        ):
          resolve_config.initialize(
              [f"vllm_config.init_with_random_weights={init}"]
          )

  def test_maxtext_vllm_rollout_requires_raiden(self):
    maxtext = [
        "model_config.model_source=maxtext",
        "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
        "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
    ]
    for rollout_engine in ("inprocess_vllm", "vllm"):
      # vLLM's `MaxTextForCausalLM` never loads the checkpoint itself.
      for mode in ("none", "in_memory"):
        with self.subTest(rollout_engine=rollout_engine, mode=mode):
          with self.assertRaisesRegex(
              ValueError,
              f"A MaxText {rollout_engine} rollout.* needs"
              f" weight_sync_config.mode=raiden, got '{mode}'",
          ):
            resolve_config.initialize([
                *maxtext,
                f"rollout_engine={rollout_engine}",
                f"weight_sync_config.mode={mode}",
            ])
      with self.subTest(rollout_engine=rollout_engine, mode="raiden"):
        hp = resolve_config.initialize([
            *maxtext,
            f"rollout_engine={rollout_engine}",
            "weight_sync_config.mode=raiden",
        ])
        self.assertEqual(
            hp.config["rollout_model_config"]["model_source"], "maxtext"
        )

  def test_k8s_target_resolves_reference_worker(self):
    # `k8s.sh` rejects the reference worker only for the commands that start
    # nodes, so `stop` and `otel` resolve a KL penalty on k8s too.
    kl_penalty = "agentic_grpo_config.beta=0.04"
    hp = resolve_config.initialize(["cluster_config.target=k8s", kl_penalty])
    self.assertTrue(hp.config["cluster_config"]["run_inference_node"])
    self.assertIn(
        "export RUN_INFERENCE_NODE=true\n",
        resolve_config.render_bash_exports(hp),
    )
    # A reference worker started separately joins through discovery.
    hp = resolve_config.initialize([
        "cluster_config.target=k8s",
        kl_penalty,
        "cluster_config.run_inference_node=false",
    ])
    self.assertEqual(hp.config["agentic_grpo_config"]["beta"], 0.04)
    self.assertFalse(hp.config["cluster_config"]["run_inference_node"])

  def test_k8s_worker_jobset_settings_fail_loud_and_are_exported(self):
    # `k8s.sh` applies the orchestrator before it renders the workers, so the
    # resolver rejects what the YAML generator would reject for a worker.
    k8s = ["cluster_config.target=k8s", "agentic_grpo_config.beta=0.0"]

    # Scalar type and range validation.
    with self.assertRaisesRegex(
        TypeError, "cluster_config.k8s.fail_fast must be a bool"
    ):
      resolve_config.initialize([*k8s, "cluster_config.k8s.fail_fast=1"])
    with self.assertRaisesRegex(
        TypeError, "cluster_config.k8s.preemptible must be a bool"
    ):
      resolve_config.initialize([*k8s, "cluster_config.k8s.preemptible=1"])
    for str_override in (
        "priority_class=123",
        "service_account=true",
        "cpu_nodepool=123",
        "head_nodepool=true",
    ):
      key = str_override.split("=")[0]
      with self.subTest(str_key=key):
        with self.assertRaisesRegex(
            TypeError, rf"cluster_config\.k8s\.{key} must be a str"
        ):
          resolve_config.initialize([
              *k8s,
              "cluster_config.k8s.rollout.jobset_yaml=jobset.pathways.yaml",
              "cluster_config.k8s.use_dynamic_slicing=true",
              f"cluster_config.k8s.{str_override}",
          ])
    with self.assertRaisesRegex(
        ValueError, "startup_retries must be an int >= 0"
    ):
      resolve_config.initialize(
          [*k8s, "cluster_config.k8s.startup_retries=true"]
      )
    with self.assertRaisesRegex(
        ValueError, "termination_grace_seconds must be an int >= 0 or null"
    ):
      resolve_config.initialize(
          [*k8s, "cluster_config.k8s.termination_grace_seconds=-1"]
      )
    with self.assertRaisesRegex(
        ValueError, "termination_grace_seconds must be an int >= 0 or null"
    ):
      resolve_config.initialize(
          [*k8s, "cluster_config.k8s.termination_grace_seconds=true"]
      )

    with self.assertRaisesRegex(
        ValueError, "startup_retries must be an int >= 0"
    ):
      resolve_config.initialize([*k8s, "cluster_config.k8s.startup_retries=-1"])
    with self.assertRaisesRegex(
        ValueError, "head_nodepool requires .*use_dynamic_slicing=true"
    ):
      resolve_config.initialize([*k8s, "cluster_config.k8s.head_nodepool=np"])
    with self.assertRaisesRegex(
        TypeError, "head_excluded_nodepools must be a list of str"
    ):
      resolve_config.initialize(
          [*k8s, "cluster_config.k8s.head_excluded_nodepools=cpu-np"]
      )
    with self.assertRaisesRegex(
        ValueError, "head_excluded_nodepools: Invalid .*'cpu np'"
    ):
      resolve_config.initialize(
          [*k8s, "cluster_config.k8s.head_excluded_nodepools=['cpu np']"]
      )
    with self.assertRaisesRegex(
        ValueError, "head_excluded_nodepools requires .*use_dynamic_slicing"
    ):
      resolve_config.initialize([
          *k8s,
          "cluster_config.k8s.rollout.jobset_yaml=jobset.pathways.yaml",
          "cluster_config.k8s.use_dynamic_slicing=true",
          "cluster_config.k8s.head_excluded_nodepools=[cpu-np]",
      ])

    # teardown_gang_on_exit requires run_id to be a valid K8s label value.
    with self.assertRaisesRegex(
        ValueError,
        "cluster_config.run_id must be a valid Kubernetes label value",
    ):
      resolve_config.initialize([
          *k8s,
          "cluster_config.run_id=bad/run",
          "cluster_config.k8s.teardown_gang_on_exit=true",
          "cluster_config.k8s.rollout.jobset_yaml=jobset.mcjax.ray.yaml",
      ])
    # run_id=bad/run remains allowed when teardown_gang_on_exit=false.
    resolve_config.initialize([
        *k8s,
        "cluster_config.run_id=bad/run",
        "cluster_config.k8s.teardown_gang_on_exit=false",
    ])

    # JobSet-only settings with the default LeaderWorkerSet rollout template.
    for jobset_key in (
        "fail_fast=true",
        "teardown_gang_on_exit=true",
        "use_dynamic_slicing=true",
    ):
      with self.subTest(jobset_key=jobset_key):
        with self.assertRaisesRegex(
            ValueError,
            rf"cluster_config\.k8s\.{jobset_key.split('=')[0]} need JobSet"
            r" worker templates \(jobset\.\*\.yaml\), got"
            r" cluster_config\.k8s\.rollout\.jobset_yaml="
            r"leaderworkerset\.mcjax\.ray\.yaml",
        ):
          resolve_config.initialize([*k8s, f"cluster_config.k8s.{jobset_key}"])

    # Shared scheduling settings succeed with the default rollout.
    resolve_config.initialize([
        *k8s,
        "cluster_config.k8s.kueue_queue_name=q",
        "cluster_config.k8s.preemptible=true",
        "cluster_config.k8s.priority_class=p",
        "cluster_config.k8s.service_account=sa",
        "cluster_config.k8s.termination_grace_seconds=60",
    ])

    # `local.sh` deploys no JobSets.
    resolve_config.initialize(["cluster_config.k8s.fail_fast=true"])

    exports = resolve_config.render_bash_exports(
        resolve_config.initialize([
            *k8s,
            "cluster_config.k8s.rollout.jobset_yaml=jobset.mcjax.ray.yaml",
            "cluster_config.k8s.fail_fast=true",
            "cluster_config.k8s.startup_retries=5",
            "cluster_config.k8s.use_dynamic_slicing=true",
            "cluster_config.k8s.head_nodepool=head-np",
        ])
    )
    self.assertIn("export K8S_FAIL_FAST=true\n", exports)
    self.assertIn("export K8S_STARTUP_RETRIES=5\n", exports)
    self.assertIn("export K8S_USE_DYNAMIC_SLICING=true\n", exports)
    self.assertIn("export K8S_HEAD_NODEPOOL=head-np\n", exports)
    self.assertIn("export K8S_HEAD_EXCLUDED_NODEPOOLS=''\n", exports)
    # `k8s.sh` omits the flag, so each template keeps its own default.
    self.assertIn("export K8S_TERMINATION_GRACE_SECONDS=''\n", exports)
    exports = resolve_config.render_bash_exports(
        resolve_config.initialize([
            *k8s,
            "cluster_config.k8s.head_excluded_nodepools=[cpu-np,sandbox-pool]",
        ])
    )
    self.assertIn(
        "export K8S_HEAD_EXCLUDED_NODEPOOLS=cpu-np,sandbox-pool\n", exports
    )

  def test_orchestrator_choices_fail_loud(self):
    hp = resolve_config.initialize([
        "agentic_grpo_config.trajectory_group_order=prompt_arrival",
        "weight_sync_config.mode=in_memory",
    ])
    self.assertEqual(
        hp.config["agentic_grpo_config"]["trajectory_group_order"],
        "prompt_arrival",
    )
    self.assertEqual(hp.config["weight_sync_config"]["mode"], "in_memory")
    with self.assertRaisesRegex(
        ValueError, "agentic_grpo_config.trajectory_group_order must be one of"
    ):
      resolve_config.initialize(
          ["agentic_grpo_config.trajectory_group_order=random"]
      )
    with self.assertRaisesRegex(
        ValueError, "weight_sync_config.mode must be one of"
    ):
      resolve_config.initialize(["weight_sync_config.mode=gloo"])
    with self.assertRaisesRegex(
        ValueError, "agentic_grpo_config.max_version_span must be non-negative"
    ):
      resolve_config.initialize(["agentic_grpo_config.max_version_span=-1"])
    with self.assertRaisesRegex(
        ValueError, "agentic_grpo_config.max_version_span is only supported"
    ):
      resolve_config.initialize([
          "agentic_grpo_config.max_version_span=2",
          "agentic_grpo_config.trajectory_group_order=prompt_arrival",
      ])

  def test_max_seq_tokens_per_device_syncs_with_legacy_tpu_field(self):
    hp = resolve_config.initialize([
        "batch_size=6",
        "rl_training_config.max_seq_tokens_per_device=4096",
        "rl_training_config.mini_batch_size=3",
        "rl_training_config.train_micro_batch_size=4",
    ])
    self.assertEqual(
        hp.config["rl_training_config"]["max_seq_token_per_tpu"], 4096
    )
    self.assertEqual(
        hp.config["rl_training_config"]["max_seq_tokens_per_device"], 4096
    )

    # Setting max_seq_tokens_per_device=0 disables packing, so MaxText
    # micro-batch FSDP divisibility validation still fires.
    with self.assertRaisesRegex(
        ValueError, "must be divisible by actor_model_config.mesh fsdp"
    ):
      resolve_config.initialize([
          "actor_model_config.model_source=maxtext",
          "actor_model_config.model_path=/ckpt/qwen3-1.7b/0/items",
          "actor_model_config.kwargs.maxtext_model_name=qwen3-1.7b",
          "actor_model_config.mesh.shape=(2,1)",
          "rl_training_config.max_seq_tokens_per_device=0",
          "rl_training_config.train_micro_batch_size=1",
      ])

    with self.assertRaisesRegex(ValueError, "must match when both are set"):
      resolve_config.initialize([
          "rl_training_config.max_seq_tokens_per_device=4096",
          "rl_training_config.max_seq_token_per_tpu=8192",
      ])

  def test_weight_sync_requires_matching_maxtext_sources(self):
    maxtext_actor = [
        "actor_model_config.model_source=maxtext",
        "actor_model_config.model_path=/ckpt/qwen3-1.7b/0/items",
        "actor_model_config.kwargs.maxtext_model_name=qwen3-1.7b",
    ]
    with self.assertRaisesRegex(
        ValueError,
        "must both be 'maxtext' or both not; got 'maxtext' and 'huggingface'",
    ):
      resolve_config.initialize(
          ["weight_sync_config.mode=raiden", *maxtext_actor]
      )
    with self.assertRaisesRegex(ValueError, "got 'huggingface' and 'maxtext'"):
      resolve_config.initialize([
          "weight_sync_config.mode=in_memory",
          "rollout_model_config.model_source=maxtext",
          "rollout_model_config.kwargs.maxtext_model_name=qwen3-1.7b",
      ])
    # Without weight sync the two sources are independent.
    hp = resolve_config.initialize(maxtext_actor)
    self.assertEqual(
        hp.config["rollout_model_config"]["model_source"], "huggingface"
    )
    # `model_config.model_source` sets both roles.
    hp = resolve_config.initialize([
        "weight_sync_config.mode=raiden",
        "model_config.model_source=maxtext",
        "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
        "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
    ])
    self.assertEqual(hp.config["actor_model_config"]["model_source"], "maxtext")
    self.assertEqual(
        hp.config["rollout_model_config"]["model_source"], "maxtext"
    )

  def test_null_sampler_is_resolves_to_none(self):
    grpo = resolve_config.initialize([]).config["agentic_grpo_config"]
    self.assertEqual(grpo["sampler_is"], "token")
    # YAML null on a string key resolves to "" in `tunix.cli.config`.
    grpo = resolve_config.initialize(
        ["agentic_grpo_config.sampler_is=null"]
    ).config["agentic_grpo_config"]
    self.assertIsNone(grpo["sampler_is"])

  def test_reward_fn_mode_requires_reward_functions(self):
    # The base config scores with the environment rewards and sets no reward
    # functions.
    hp = resolve_config.initialize([])
    self.assertEqual(hp.config["agentic_grpo_config"]["reward_mode"], "env")
    self.assertEqual(hp.config["reward_functions"], [])
    with self.assertRaisesRegex(ValueError, "needs at least one entry"):
      resolve_config.initialize(["agentic_grpo_config.reward_mode=reward_fn"])
    hp = resolve_config.initialize([
        "agentic_grpo_config.reward_mode=reward_fn",
        "reward_functions=['tunix/cli/reward_fn/gsm8k.py']",
    ])
    self.assertEqual(
        hp.config["reward_functions"], ["tunix/cli/reward_fn/gsm8k.py"]
    )

  def test_warmup_steps_derived_from_warmup_ratio(self):
    opt_key = "rl_training_config.actor_optimizer_config"

    def warmup_steps(*overrides: str) -> int:
      hp = resolve_config.initialize(list(overrides))
      return hp.config_mapping("rl_training_config")["actor_optimizer_config"][
          "warmup_steps"
      ]

    # The default ratio (0.1) matches the previous ceil(decay_steps / 10).
    for decay_steps in (1, 9, 10, 11, 500, 1001):
      self.assertEqual(
          warmup_steps(f"{opt_key}.decay_steps={decay_steps}"),
          (decay_steps + 9) // 10,
      )
    self.assertEqual(
        warmup_steps(
            f"{opt_key}.decay_steps=10", f"{opt_key}.warmup_ratio=0.25"
        ),
        3,
    )
    self.assertEqual(
        warmup_steps(
            f"{opt_key}.decay_steps=10",
            f"{opt_key}.warmup_ratio=0.25",
            f"{opt_key}.warmup_steps=7",
        ),
        7,
    )
    with self.assertRaisesRegex(ValueError, "warmup_ratio must be in"):
      warmup_steps(f"{opt_key}.decay_steps=10", f"{opt_key}.warmup_ratio=1.5")

  def test_typed_vllm_rollout_settings(self):
    vllm = resolve_config.initialize([]).config["vllm_config"]
    self.assertFalse(vllm["return_routed_experts"])
    self.assertEqual(vllm["routed_experts_encode_mode"], "auto")
    self.assertIsNone(vllm["routed_experts_num_experts"])
    self.assertTrue(vllm["detokenize"])

    vllm = resolve_config.initialize([
        "vllm_config.return_routed_experts=true",
        "vllm_config.routed_experts_encode_mode=uint10",
        "vllm_config.routed_experts_num_experts=256",
        "vllm_config.detokenize=false",
    ]).config["vllm_config"]
    self.assertIs(vllm["return_routed_experts"], True)
    self.assertEqual(vllm["routed_experts_encode_mode"], "uint10")
    self.assertEqual(vllm["routed_experts_num_experts"], 256)
    self.assertIs(vllm["detokenize"], False)

  def test_invalid_vllm_rollout_settings_fail_loud(self):
    with self.assertRaisesRegex(ValueError, "routed_experts_encode_mode"):
      resolve_config.initialize(["vllm_config.routed_experts_encode_mode=int8"])
    with self.assertRaisesRegex(ValueError, "routed_experts_num_experts"):
      resolve_config.initialize(["vllm_config.routed_experts_num_experts=0"])
    with self.assertRaisesRegex(ValueError, "needs a vLLM rollout_engine"):
      resolve_config.initialize([
          "rollout_engine=vanilla",
          "vllm_config.return_routed_experts=true",
      ])
    with self.assertRaisesRegex(TypeError, "vllm_config.detokenize"):
      resolve_config.initialize(["vllm_config.detokenize=yes_please"])
    for bad_ft in ("yes_please", "0", "null"):
      with self.assertRaisesRegex(
          TypeError, r"cluster_config\.fault_tolerance\.enabled"
      ):
        resolve_config.initialize(
            [f"cluster_config.fault_tolerance.enabled={bad_ft}"]
        )

  def test_maxtext_skip_step_kwargs_validation(self):
    with self.assertRaisesRegex(
        ValueError, "require actor_model_config.model_source='maxtext'"
    ):
      resolve_config.initialize(
          ["actor_model_config.kwargs.skip_step_on_nan=false"]
      )
    maxtext_base = [
        "actor_model_config.model_source=maxtext",
        "actor_model_config.model_path=/ckpt/qwen3-0.6b/0/items",
        "actor_model_config.kwargs.maxtext_model_name=qwen3-0.6b",
        "rollout_model_config.model_source=maxtext",
        "rollout_model_config.kwargs.maxtext_model_name=qwen3-0.6b",
        "weight_sync_config.mode=raiden",
    ]
    with self.assertRaisesRegex(ValueError, "must be a bool"):
      resolve_config.initialize(
          [*maxtext_base, 'actor_model_config.kwargs.skip_step_on_nan="false"']
      )
    with self.assertRaisesRegex(ValueError, "must be a positive int"):
      resolve_config.initialize(
          [*maxtext_base, "actor_model_config.kwargs.skip_step_interval=0"]
      )
    with self.assertRaisesRegex(ValueError, "must be a positive finite number"):
      resolve_config.initialize([
          *maxtext_base,
          "actor_model_config.kwargs.skip_step_scaling_factor=-1.0",
      ])

  def test_rollout_fault_tolerance_settings_and_validation(self):
    def ft_values(*overrides: str) -> dict[str, object]:
      cluster = resolve_config.initialize(list(overrides)).config_mapping(
          "cluster_config"
      )
      return dict(cluster["fault_tolerance"])

    self.assertEqual(
        ft_values(),
        {
            "enabled": True,
            "max_task_retries": 1,
            "poll_retry_budget_s": None,
            "min_active_rollout_ratio": 0.5,
            "min_active_rollout_grace_s": 900.0,
        },
    )
    self.assertEqual(
        ft_values(
            "cluster_config.fault_tolerance.enabled=false",
            "cluster_config.fault_tolerance.max_task_retries=3",
            "cluster_config.fault_tolerance.poll_retry_budget_s=45.5",
            "cluster_config.fault_tolerance.min_active_rollout_ratio=null",
            "cluster_config.fault_tolerance.min_active_rollout_grace_s=0",
        ),
        {
            "enabled": False,
            "max_task_retries": 3,
            "poll_retry_budget_s": 45.5,
            "min_active_rollout_ratio": None,
            "min_active_rollout_grace_s": 0,
        },
    )
    self.assertEqual(
        ft_values(
            "cluster_config.fault_tolerance.max_task_retries=0",
            "cluster_config.fault_tolerance.poll_retry_budget_s=0",
            "cluster_config.fault_tolerance.min_active_rollout_ratio=1",
        ),
        {
            "enabled": True,
            "max_task_retries": 0,
            "poll_retry_budget_s": 0,
            "min_active_rollout_ratio": 1,
            "min_active_rollout_grace_s": 900.0,
        },
    )

    for arg in (
        "cluster_config.fault_tolerance.max_task_retries=true",
        "cluster_config.fault_tolerance.max_task_retries=1.5",
        "cluster_config.fault_tolerance.max_task_retries=null",
        "cluster_config.fault_tolerance.max_task_retries=abc",
        "cluster_config.fault_tolerance.poll_retry_budget_s=true",
        "cluster_config.fault_tolerance.poll_retry_budget_s=abc",
        "cluster_config.fault_tolerance.min_active_rollout_ratio=false",
        "cluster_config.fault_tolerance.min_active_rollout_ratio=abc",
        "cluster_config.fault_tolerance.min_active_rollout_grace_s=true",
        "cluster_config.fault_tolerance.min_active_rollout_grace_s=null",
        "cluster_config.fault_tolerance.min_active_rollout_grace_s=abc",
    ):
      key = arg.split("=", 1)[0]
      with self.subTest(arg=arg):
        with self.assertRaisesRegex(TypeError, re.escape(key)):
          resolve_config.initialize([arg])

    for arg in (
        "cluster_config.fault_tolerance.max_task_retries=-1",
        "cluster_config.fault_tolerance.poll_retry_budget_s=-0.1",
        "cluster_config.fault_tolerance.poll_retry_budget_s=.inf",
        "cluster_config.fault_tolerance.poll_retry_budget_s=-.inf",
        "cluster_config.fault_tolerance.poll_retry_budget_s=.nan",
        "cluster_config.fault_tolerance.min_active_rollout_ratio=0.0",
        "cluster_config.fault_tolerance.min_active_rollout_ratio=-0.5",
        "cluster_config.fault_tolerance.min_active_rollout_ratio=1.1",
        "cluster_config.fault_tolerance.min_active_rollout_ratio=.nan",
        "cluster_config.fault_tolerance.min_active_rollout_ratio=.inf",
        "cluster_config.fault_tolerance.min_active_rollout_grace_s=-1.0",
        "cluster_config.fault_tolerance.min_active_rollout_grace_s=.inf",
        "cluster_config.fault_tolerance.min_active_rollout_grace_s=-.inf",
        "cluster_config.fault_tolerance.min_active_rollout_grace_s=.nan",
    ):
      key = arg.split("=", 1)[0]
      with self.subTest(arg=arg):
        with self.assertRaisesRegex(ValueError, re.escape(key)):
          resolve_config.initialize([arg])

  def test_vllm_config_kwargs_is_removed(self):
    # Nothing read it; vLLM engine arguments go in `vllm_config.engine_kwargs`.
    with self.assertRaisesRegex(
        ValueError, "Unknown configuration key 'vllm_config.kwargs'"
    ):
      resolve_config.initialize(["vllm_config.kwargs.attention=flash"])

  def test_maxtext_roles_require_one_maxtext_model_name(self):
    maxtext = [
        "model_config.model_source=maxtext",
        "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
    ]
    for argv in (
        (),
        ("model_config.kwargs.maxtext_model_name=null",),
        ("model_config.kwargs.maxtext_model_name=",),
    ):
      with self.subTest(argv=argv):
        with self.assertRaisesRegex(
            ValueError, "actor_model_config.kwargs.maxtext_model_name"
        ):
          resolve_config.initialize([*maxtext, *argv])
    # The rollout needs it too: there is no fallback to `model_name`.
    with self.assertRaisesRegex(
        ValueError, "rollout_model_config.kwargs.maxtext_model_name"
    ):
      resolve_config.initialize(["rollout_model_config.model_source=maxtext"])
    with self.assertRaisesRegex(ValueError, "must match"):
      resolve_config.initialize([
          *maxtext,
          "weight_sync_config.mode=raiden",
          "actor_model_config.kwargs.maxtext_model_name=qwen3-1.7b",
          "rollout_model_config.kwargs.maxtext_model_name=qwen3-0.6b",
      ])
    # Without weight sync the names may differ, but the MaxText rollout then
    # never gets real weights.
    with self.assertRaisesRegex(ValueError, "needs weight_sync_config.mode"):
      resolve_config.initialize([
          *maxtext,
          "weight_sync_config.mode=none",
          "actor_model_config.kwargs.maxtext_model_name=qwen3-1.7b",
          "rollout_model_config.kwargs.maxtext_model_name=qwen3-0.6b",
      ])
    # A null kwargs (how merge mode clears a mapping) fails with context.
    with self.assertRaisesRegex(
        TypeError, "actor_model_config.kwargs must be a mapping"
    ):
      resolve_config.initialize([*maxtext, "model_config.kwargs=null"])
    # Non-string and untrimmed names fail here, not later in MaxText.
    for name in ("true", "' qwen3-0.6b '"):
      with self.subTest(name=name):
        with self.assertRaisesRegex(ValueError, "non-empty string"):
          resolve_config.initialize(
              [*maxtext, f"model_config.kwargs.maxtext_model_name={name}"]
          )
    # `local.sh` converts the checkpoint with the exported name.
    hp = resolve_config.initialize([
        *maxtext,
        "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
        "weight_sync_config.mode=raiden",
    ])
    self.assertIn(
        "export ACTOR_MAXTEXT_MODEL_NAME=qwen3-1.7b\n",
        resolve_config.render_bash_exports(hp),
    )
    self.assertIn(
        "export ACTOR_MAXTEXT_MODEL_NAME=''\n",
        resolve_config.render_bash_exports(resolve_config.initialize([])),
    )

  _MAXTEXT_TRAINER = (
      "model_config.model_source=maxtext",
      "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
      "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
      "weight_sync_config.mode=raiden",
  )
  _EXTRA = "dist_training_config.maxtext_extra_config"

  def test_maxtext_extra_config_needs_a_maxtext_trainer(self):
    entries = (
        f"{self._EXTRA}.remat_policy=custom",
        f"{self._EXTRA}.sa_block_q=1024",
        f"{self._EXTRA}.packing=true",
    )
    # A Tunix trainer would silently ignore them.
    with self.assertRaisesRegex(
        ValueError, "maxtext_extra_config configures a MaxText trainer"
    ):
      resolve_config.initialize(list(entries))
    hp = resolve_config.initialize([*self._MAXTEXT_TRAINER, *entries])
    # The values keep their YAML types.
    self.assertEqual(
        hp.config_mapping("dist_training_config")["maxtext_extra_config"],
        {"remat_policy": "custom", "sa_block_q": 1024, "packing": True},
    )

  def test_maxtext_extra_config_rejects_non_scalar_values(self):
    with self.assertRaisesRegex(TypeError, "must be a mapping"):
      resolve_config.initialize([*self._MAXTEXT_TRAINER, f"{self._EXTRA}=3"])
    # Each value becomes one MaxText `key=value` argument.
    for entry in (
        f"{self._EXTRA}.remat_policy=null",
        f"{self._EXTRA}.remat_policy=[full]",
        f"{self._EXTRA}.remat_policy.decoder=full",
    ):
      with self.subTest(entry=entry):
        with self.assertRaisesRegex(
            TypeError, "values must be a str, int, float or bool"
        ):
          resolve_config.initialize([*self._MAXTEXT_TRAINER, entry])

  def test_maxtext_extra_config_rejects_keys_that_are_not_config_names(self):
    # "learning_rate=1e-6" would reach MaxText as `learning_rate=1e-6=1`.
    for key in ("learning_rate=1e-6", "remat policy", ""):
      with self.subTest(key=key):
        with self.assertRaisesRegex(
            ValueError, "keys must be MaxText config names"
        ):
          resolve_config.initialize([
              *self._MAXTEXT_TRAINER,
              f"{self._EXTRA}={{{json.dumps(key)}: 1}}",
          ])

  def test_maxtext_extra_flags_is_removed(self):
    with self.assertRaisesRegex(
        ValueError,
        "Unknown configuration key 'dist_training_config.maxtext_extra_flags'",
    ):
      resolve_config.initialize([
          *self._MAXTEXT_TRAINER,
          "dist_training_config.maxtext_extra_flags=remat_policy=custom",
      ])

  def test_maxtext_extra_config_rejects_keys_built_from_the_config(self):
    # Appended last, these would silently override the config values.
    with self.assertRaisesRegex(
        ValueError,
        r"maxtext_extra_config must not set MaxText keys that the trainer"
        r" builds from the config: learning_rate \(set"
        r" rl_training_config\.actor_optimizer_config\.learning_rate instead\),"
        r" ici_fsdp_parallelism \(set the 'fsdp' axis of"
        r" actor_model_config\.mesh instead\)\.",
    ):
      resolve_config.initialize([
          *self._MAXTEXT_TRAINER,
          f"{self._EXTRA}.remat_policy=custom",
          f"{self._EXTRA}.learning_rate=1e-6",
          f"{self._EXTRA}.ici_fsdp_parallelism=4",
      ])
    # Fixed builder settings, which have no config key, stay overridable.
    hp = resolve_config.initialize([
        *self._MAXTEXT_TRAINER,
        f"{self._EXTRA}.use_tokamax_gmm=true",
        f"{self._EXTRA}.grad_dtype=float32",
        f"{self._EXTRA}.context_parallel_load_balance=false",
    ])
    self.assertEqual(
        hp.config_mapping("dist_training_config")["maxtext_extra_config"],
        {
            "use_tokamax_gmm": True,
            "grad_dtype": "float32",
            "context_parallel_load_balance": False,
        },
    )

  def test_freeze_router_validation(self):
    cfg = lambda *a: resolve_config.initialize(list(a)).config_mapping(
        "actor_model_config"
    )
    self.assertIs(cfg()["freeze_router"], False)
    self.assertIs(
        cfg("actor_model_config.freeze_router=true")["freeze_router"], True
    )
    with self.assertRaisesRegex(ValueError, "Unknown configuration key"):
      cfg("model_config.freeze_router=true")
    with self.assertRaisesRegex(TypeError, "freeze_router must be a bool"):
      cfg("actor_model_config.freeze_router=1")
    with self.assertRaisesRegex(ValueError, "not supported by the MaxText"):
      cfg(
          "model_config.model_source=maxtext",
          "model_config.model_path=/ckpt",
          "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
          "weight_sync_config.mode=raiden",
          "actor_model_config.freeze_router=true",
      )

  def test_library_settings(self):
    hp = resolve_config.initialize([])
    vllm = hp.config_mapping("vllm_config")
    self.assertIs(vllm["skip_jax_precompile"], True)
    dist_training = hp.config_mapping("dist_training_config")
    # null keeps the backend's own D2H cap (MaxText's 8 GB, none for Tunix).
    self.assertIsNone(dist_training["ckpt_d2h_concurrent_gb"])
    ws = hp.config_mapping("weight_sync_config")
    self.assertIs(ws["verify_weights"], False)
    # As the earlier launchers did.
    self.assertEqual(ws["raiden"]["data_nics"], "eth0")

    hp = resolve_config.initialize([
        "vllm_config.skip_jax_precompile=false",
        "actor_model_config.model_source=maxtext",
        "actor_model_config.model_path=/ckpt",
        "actor_model_config.kwargs.maxtext_model_name=qwen3-1.7b",
        "dist_training_config.ckpt_d2h_concurrent_gb=16",
        "weight_sync_config.raiden.data_nics=eth0,eth1",
    ])
    vllm = hp.config_mapping("vllm_config")
    self.assertIs(vllm["skip_jax_precompile"], False)
    dist_training = hp.config_mapping("dist_training_config")
    self.assertEqual(dist_training["ckpt_d2h_concurrent_gb"], 16)
    self.assertEqual(
        hp.config_mapping("weight_sync_config")["raiden"]["data_nics"],
        "eth0,eth1",
    )

  def test_invalid_library_settings_fail_loud(self):
    # The nodes turn them into environment variables; a wrong type would be
    # misread there, after the jobs started.
    d2h = "dist_training_config.ckpt_d2h_concurrent_gb"
    for override, error in (
        ("vllm_config.skip_jax_precompile=1", "skip_jax_precompile must be a"),
        ("weight_sync_config.verify_weights=1", "verify_weights must be a"),
        ("weight_sync_config.raiden.data_nics=1", "must be a string"),
        (f"{d2h}=8.5", "must be an int or null"),
        (f"{d2h}=true", "must be an int or null"),
    ):
      with self.subTest(override=override):
        with self.assertRaisesRegex(TypeError, error):
          resolve_config.initialize([override])
    with self.assertRaisesRegex(ValueError, "must be positive"):
      resolve_config.initialize([f"{d2h}=0"])
    # The Tunix trainer has no D2H cap yet.
    with self.assertRaisesRegex(
        ValueError, "only applies to the MaxText trainer"
    ):
      resolve_config.initialize([f"{d2h}=8"])

  def test_moved_library_settings_fail_as_unknown_keys(self):
    for override in (
        "cluster_config.k8s.jax_cache.rollout_skip_jax_precompile=0",
        "cluster_config.k8s.jax_cache.skip_jax_precompile=0",
    ):
      with self.subTest(override=override):
        with self.assertRaisesRegex(
            ValueError, f"Unknown configuration key '{override.split('=')[0]}'"
        ):
          resolve_config.initialize([override])

  def test_library_settings_are_not_exported_to_the_drivers(self):
    # The nodes take them from the config (`node_utils.apply_library_env`,
    # `maxtext_utils.build_maxtext_config`); the drivers pass config only.
    exports = resolve_config.render_bash_exports(resolve_config.initialize([]))
    for name in (
        "WEIGHT_SYNC_VERIFY_WEIGHTS",
        "CKPT_D2H_CONCURRENT_GB",
        "MAXTEXT_EXTRA_FLAGS",
        "K8S_ROLLOUT_SKIP_JAX_PRECOMPILE",
    ):
      with self.subTest(name=name):
        self.assertNotIn(f"export {name}=", exports)

  def test_plain_config_and_fingerprint(self):
    overrides = ["cluster_config.run_id=run_1", "batch_size=8"]
    hp = resolve_config.initialize(overrides)
    plain = resolve_config.plain_config(hp)
    self.assertEqual(plain["batch_size"], 8)
    # Top-level lists are OmegaConf `ListConfig`s in `hp.config`.
    self.assertIs(type(plain["reward_functions"]), list)
    cluster = plain["cluster_config"]
    assert isinstance(cluster, dict)  # Narrows the type for pytype.
    self.assertEqual(cluster["run_id"], "run_1")
    # A detached copy that serializes to JSON.
    cluster["run_id"] = "changed"
    self.assertEqual(hp.config_mapping("cluster_config")["run_id"], "run_1")
    json.dumps(plain)

    fingerprint = resolve_config.config_fingerprint(hp)
    self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
    # The same config resolved again (e.g. by another node) matches, ...
    self.assertEqual(
        resolve_config.config_fingerprint(
            resolve_config.initialize(list(reversed(overrides)))
        ),
        fingerprint,
    )
    # ... and any difference changes it, including the run id.
    for changed in (
        ["cluster_config.run_id=run_1", "batch_size=4"],
        ["cluster_config.run_id=run_2", "batch_size=8"],
    ):
      with self.subTest(changed=changed):
        self.assertNotEqual(
            resolve_config.config_fingerprint(
                resolve_config.initialize(changed)
            ),
            fingerprint,
        )

  def test_plain_config_normalizes_omegaconf_containers(self):
    hp = resolve_config.initialize(["cluster_config.run_id=run_1"])
    # An open section can hold an OmegaConf container with non-string keys.
    hp.config["env_kwargs"] = omegaconf.OmegaConf.create({1: "a", "b": [2]})
    self.assertEqual(
        resolve_config.plain_config(hp)["env_kwargs"], {"1": "a", "b": [2]}
    )
    # Mixed key types would otherwise break the sorted canonical JSON.
    resolve_config.config_fingerprint(hp)
    # A mandatory value that was never set fails instead of logging "???".
    hp.config["env_kwargs"] = omegaconf.OmegaConf.create({"m": "???"})
    with self.assertRaises(omegaconf.MissingMandatoryValue):
      resolve_config.plain_config(hp)

  def test_rollout_cache_params_follow_config(self):
    hp = resolve_config.initialize([
        "agentic_grpo_config.beta=0.0",
        "rollout_engine=vllm",
        "rollout_model_config.mesh.shape=(2,4,2)",
        "rollout_model_config.mesh.axis_names=('fsdp','tp','expert')",
        "rollout_model_config.lora_config={'rank': 16, 'alpha': 32.0}",
        "rollout_config.max_prompt_length=256",
        "rollout_config.total_generation_steps=128",
        "vllm_config.max_model_len=512",
        "vllm_config.enable_prefix_caching=true",
        "vllm_config.async_scheduling=true",
        "cluster_config.k8s.rollout.tpu_slice=tpuv5:2x2x2",
    ])
    self.assertEqual(
        resolve_config.rollout_cache_params(hp),
        gcs_cache.RolloutCacheKeyParams(
            tpu_slice="tpuv5:2x2x2",
            model_name="Qwen3-1.7B",
            sampler="vllm",
            max_prompt_length=256,
            max_response_length=128,
            vllm_max_model_len=512,
            mesh_tp=4,
            mesh_fsdp=2,
            mesh_expert=2,
            use_lora=True,
            lora_rank=16,
            enable_prefix_caching=True,
            vllm_async_scheduling=True,
        ),
    )
    # Without a model name, the model id names it; a MaxText rollout goes by
    # its MaxText model name.
    for overrides, model_name in (
        (("rollout_model_config.model_name=",), "Qwen/Qwen3-1.7B"),
        (
            (
                "model_config.model_source=maxtext",
                "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
                "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
                "weight_sync_config.mode=raiden",
            ),
            "qwen3-1.7b",
        ),
    ):
      with self.subTest(model_name=model_name):
        hp = resolve_config.initialize(
            ["agentic_grpo_config.beta=0.0", *overrides]
        )
        self.assertEqual(
            resolve_config.rollout_cache_params(hp).model_name, model_name
        )
    # In-flight weight updates turn on vLLM's prefix caching, and so the key's.
    hp = resolve_config.initialize([
        "agentic_grpo_config.beta=0.0",
        "vllm_config.enable_prefix_caching=false",
        "weight_sync_config.in_flight_weight_updates=true",
    ])
    self.assertTrue(
        resolve_config.rollout_cache_params(hp).enable_prefix_caching
    )

  def test_rollout_cache_settings_fail_loud(self):
    # `k8s.sh` and the URI derivation must read `disable` the same way.
    with self.assertRaisesRegex(TypeError, "jax_cache.disable must be a bool"):
      resolve_config.initialize([
          "agentic_grpo_config.beta=0.0",
          "cluster_config.k8s.jax_cache.disable=y",
      ])
    # `k8s.sh` would save with `save: null`.
    for save in ("null", "'false'"):
      with self.subTest(save=save):
        with self.assertRaisesRegex(TypeError, "jax_cache.save must be a bool"):
          resolve_config.initialize([
              "agentic_grpo_config.beta=0.0",
              f"cluster_config.k8s.jax_cache.save={save}",
          ])
    for local_dir in ("''", "jax_cache", "null"):
      with self.subTest(local_dir=local_dir):
        with self.assertRaisesRegex(
            ValueError, "local_dir must be a non-empty absolute path"
        ):
          resolve_config.initialize([
              "agentic_grpo_config.beta=0.0",
              f"cluster_config.k8s.jax_cache.local_dir={local_dir}",
          ])
    for key in ("rollout_gcs_dir", "gcs_dir", "bucket"):
      for bad_type in ("123", "true", "[a]"):
        with self.subTest(key=key, bad_type=bad_type):
          with self.assertRaisesRegex(
              TypeError, f"jax_cache.{key} must be a str"
          ):
            resolve_config.initialize([
                "agentic_grpo_config.beta=0.0",
                f"cluster_config.k8s.jax_cache.{key}={bad_type}",
            ])
      with self.subTest(key=key, bad_type=None):
        hp = resolve_config.initialize(["agentic_grpo_config.beta=0.0"])
        hp.config["cluster_config"]["k8s"]["jax_cache"][key] = None
        with self.assertRaisesRegex(
            TypeError, f"jax_cache.{key} must be a str"
        ):
          resolve_config.finalize_dist_config(hp)
    for key in ("rollout_gcs_dir", "gcs_dir"):
      for bad_uri in (
          "my-bucket/cache",
          "s3://my-bucket/cache",
          "-bad",
          "gs://",
          "gs:///p",
          "' gs://r '",
      ):
        with self.subTest(key=key, bad_uri=bad_uri):
          with self.assertRaisesRegex(
              ValueError, f"jax_cache.{key} must start with 'gs://<bucket>'"
          ):
            resolve_config.initialize([
                "agentic_grpo_config.beta=0.0",
                f"cluster_config.k8s.jax_cache.{key}={bad_uri}",
            ])
    for bad_bucket in (
        "gs://",
        "gs:///p",
        "gs:/my-bucket",
        "s3://my-bucket",
        "/bad",
        "-bad",
        "' my-bucket '",
    ):
      with self.subTest(bad_bucket=bad_bucket):
        with self.assertRaisesRegex(
            ValueError,
            "jax_cache.bucket must be a GCS bucket name or 'gs://<bucket>' URI",
        ):
          resolve_config.initialize([
              "agentic_grpo_config.beta=0.0",
              f"cluster_config.k8s.jax_cache.bucket={bad_bucket}",
          ])
    with self.assertRaisesRegex(ValueError, "lora_config is missing 'rank'"):
      resolve_config.initialize([
          "agentic_grpo_config.beta=0.0",
          "rollout_model_config.lora_config={'alpha': 32.0}",
      ])

  def test_k8s_rollout_cache_uri_comes_from_the_config_only(self):
    base = ["agentic_grpo_config.beta=0.0"]
    digest = resolve_config.rollout_cache_params(
        resolve_config.initialize(base)
    ).cache_fingerprint()
    # Hardware, model, rollout slice topology and mesh, and the key's digest.
    derived = f"v5e/qwen3-1.7b/rollout_4x4_ep1_tp2_{digest}"
    bucket = "cluster_config.k8s.jax_cache.bucket=my-bucket"
    gcs_dir = "cluster_config.k8s.jax_cache.gcs_dir=gs://my-cache"
    rollout_gcs_dir = "cluster_config.k8s.jax_cache.rollout_gcs_dir=gs://r/"
    disable = "cluster_config.k8s.jax_cache.disable=true"
    cases = (
        ((), "''"),
        ((bucket,), f"gs://my-bucket/jax_cache/{derived}"),
        (
            ("cluster_config.k8s.jax_cache.bucket=gs://my-bucket",),
            f"gs://my-bucket/jax_cache/{derived}",
        ),
        ((bucket, "cluster_config.k8s.jax_cache.bucket=null"), "''"),
        ((bucket, gcs_dir), f"gs://my-cache/{derived}"),
        ((gcs_dir, rollout_gcs_dir), "gs://r"),
        ((bucket, disable), "''"),
        ((rollout_gcs_dir, disable), "''"),
    )
    # The settings the rollout node reads from its environment, set in the
    # launcher's: none of them may change the URI.
    stray_env = {
        "BUCKET": "stray",
        "DISABLE_JAX_CACHE": "1",
        "JAX_CACHE_BUCKET": "stray",
        "JAX_CACHE_GCS_DIR": "gs://stray",
        "ROLLOUT_JAX_CACHE_GCS_DIR": "gs://stray/rollout",
    }
    with mock.patch.dict(os.environ, stray_env):
      for overrides, uri in cases:
        with self.subTest(overrides=overrides):
          self.assertIn(
              f"\nexport K8S_JAX_CACHE_ROLLOUT_GCS_DIR={uri}\n",
              resolve_config.render_bash_exports(
                  resolve_config.initialize([*base, *overrides])
              ),
          )

  def test_role_env(self):
    hp = resolve_config.initialize([
        "cluster_config.roles.orchestrator.env.ORCH_FLAG=x",
        "cluster_config.roles.trainer.env.XLA_FLAGS='--xla_a --xla_b'",
        "cluster_config.roles.trainer.env.TPU_STDERR_LOG_LEVEL='0'",
        "cluster_config.roles.rollout.env.VLLM_LOGGING_LEVEL=DEBUG",
        "agentic_grpo_config.beta=0.0",
    ])
    self.assertEqual(
        resolve_config.role_env(hp),
        {
            "orchestrator": {"ORCH_FLAG": "x"},
            "trainer": {
                "XLA_FLAGS": "--xla_a --xla_b",
                "TPU_STDERR_LOG_LEVEL": "0",
            },
            "rollout": {"VLLM_LOGGING_LEVEL": "DEBUG"},
            "inference": {},
        },
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            resolve_config.render_bash_exports(hp)
            + 'eval "w=(${ROLE_ENV_TRAINER})"; printf "%s\\n" "${w[@]}"',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    self.assertEqual(
        proc.stdout.splitlines(),
        ["XLA_FLAGS=--xla_a --xla_b", "TPU_STDERR_LOG_LEVEL=0"],
    )

    trainer = "cluster_config.roles.trainer.env"
    with self.assertRaisesRegex(TypeError, "keys and values must be strings"):
      resolve_config.initialize([f"{trainer}.FOO=1"])
    with self.assertRaisesRegex(TypeError, f"{trainer} must be a mapping"):
      resolve_config.initialize([f"{trainer}=null"])
    for name, error in (
        ("T_BATCH_SIZE", "Tunix config override"),
        ("M_REMAT_POLICY", "MaxText config override"),
        ("VERIFY_WEIGHTS", "weight_sync_config.verify_weights"),
        ("DIST_RL_WORKER_PORT", "launcher-managed"),
    ):
      with self.assertRaisesRegex(ValueError, f"{trainer}.*{error}"):
        resolve_config.initialize([f"{trainer}.{name}=x"])
    with self.assertRaisesRegex(
        ValueError, "run_inference_node resolved to false"
    ):
      resolve_config.initialize([
          "agentic_grpo_config.beta=0.0",
          "cluster_config.roles.inference.env.FOO=x",
      ])
    with self.assertRaisesRegex(
        ValueError, "target=external runs the processes"
    ):
      resolve_config.initialize([
          "cluster_config.target=external",
          "cluster_config.artifact_root=/artifacts/run",
          f"{trainer}.FOO=x",
      ])

  def test_render_bash_exports_evaluates_in_bash(self):
    hp = resolve_config.initialize([
        "model_config.model_name=Qwen3-0.6B",
        "model_config.model_source=maxtext",
        "model_config.model_path=/ckpt/qwen3-0.6b/0/items",
        "model_config.kwargs.maxtext_model_name=qwen3-0.6b",
        "weight_sync_config.mode=raiden",
        "kubernetes_config.namespace=swe-ns",
    ])
    exports_str = resolve_config.render_bash_exports(hp)
    proc = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            exports_str
            + '\necho "$CLUSTER_TARGET|$ACTOR_MAXTEXT_MODEL_NAME|'
            '$WEIGHT_SYNC_MODE|$SANDBOX_NAMESPACE|$TRAINER_MAIN|$DRY_RUN"',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    self.assertEqual(
        proc.stdout.strip(),
        "local|qwen3-0.6b|raiden|swe-ns|"
        "tunix.dist_rl.launch.nodes.trainer.main|false",
    )

  def test_execution_targets_default_and_per_role_overrides(self):
    hp = resolve_config.initialize([])
    self.assertEqual(
        hp.cluster_targets,
        execution_target.ClusterTargets(
            trainer=execution_target.JAX_TPU,
            rollout=execution_target.JAX_TPU,
            inference=execution_target.JAX_TPU,
            placement=execution_target.Placement.DISAGGREGATED,
        ),
    )

    hp_override = resolve_config.initialize([
        "cluster_config.target=external",
        "cluster_config.artifact_root=/artifacts/run",
        "execution_targets.target=jax:gpu",
        "execution_targets.trainer=torch:tpu",
        "execution_targets.placement=disaggregated",
    ])
    self.assertEqual(
        hp_override.cluster_targets.trainer, execution_target.TORCH_TPU
    )
    self.assertEqual(
        hp_override.cluster_targets.rollout, execution_target.JAX_GPU
    )
    self.assertEqual(
        hp_override.cluster_targets.inference, execution_target.JAX_GPU
    )
    self.assertEqual(
        hp_override.config["cluster_config"]["local"]["trainer_devices"], "0,1"
    )
    self.assertEqual(
        hp_override.config["cluster_config"]["local"]["rollout_devices"], "2,3"
    )
    self.assertEqual(
        hp_override.config["cluster_config"]["local"]["inference_devices"], ""
    )

    hp_colocated = resolve_config.initialize([
        "execution_targets.target=jax:gpu",
        "execution_targets.placement=colocated",
        "cluster_config.local.trainer_devices=0,1,2,3",
    ])
    self.assertEqual(
        hp_colocated.config["cluster_config"]["local"]["rollout_devices"],
        "0,1,2,3",
    )
    self.assertEqual(
        hp_colocated.config["cluster_config"]["local"]["inference_devices"],
        "0,1,2,3",
    )

    hp_single_zero = resolve_config.initialize([
        "execution_targets.target=jax:gpu",
        "execution_targets.placement=colocated",
        "cluster_config.local.trainer_devices=0",
    ])
    self.assertEqual(
        hp_single_zero.config["cluster_config"]["local"]["trainer_devices"], "0"
    )
    self.assertEqual(
        hp_single_zero.config["cluster_config"]["local"]["rollout_devices"], "0"
    )
    self.assertEqual(
        hp_single_zero.config["cluster_config"]["local"]["inference_devices"],
        "0",
    )

    hp_disagg_ints = resolve_config.initialize([
        "execution_targets.target=jax:gpu",
        "cluster_config.local.trainer_devices=0",
        "cluster_config.local.rollout_devices=1",
        "cluster_config.local.inference_devices=0",
    ])
    self.assertEqual(
        hp_disagg_ints.config["cluster_config"]["local"]["trainer_devices"], "0"
    )
    self.assertEqual(
        hp_disagg_ints.config["cluster_config"]["local"]["rollout_devices"], "1"
    )
    self.assertEqual(
        hp_disagg_ints.config["cluster_config"]["local"]["inference_devices"],
        "0",
    )

  def test_local_auto_detect_devices_and_device_validation(self):
    hp_default = resolve_config.initialize([])
    self.assertFalse(
        hp_default.config["cluster_config"]["local"]["auto_detect_devices"]
    )
    self.assertIn(
        "export LOCAL_AUTO_DETECT_DEVICES=false\n",
        resolve_config.render_bash_exports(hp_default),
    )

    hp = resolve_config.initialize([
        "cluster_config.local.auto_detect_devices=true",
        "cluster_config.local.trainer_devices= 0,1 ",
        "cluster_config.local.inference_devices=   ",
    ])
    local = hp.config["cluster_config"]["local"]
    self.assertTrue(local["auto_detect_devices"])
    self.assertEqual(local["trainer_devices"], "0,1")
    self.assertEqual(local["inference_devices"], "")
    exports = resolve_config.render_bash_exports(hp)
    self.assertIn("export LOCAL_AUTO_DETECT_DEVICES=true\n", exports)
    self.assertIn("export LOCAL_TRAINER_DEVICES=0,1\n", exports)
    self.assertIn("export LOCAL_ROLLOUT_DEVICES=2,3\n", exports)
    self.assertIn("export LOCAL_INFERENCE_DEVICES=''\n", exports)

    with self.assertRaisesRegex(
        TypeError, r"cluster_config\.local\.auto_detect_devices must be a bool"
    ):
      resolve_config.initialize(
          ["cluster_config.local.auto_detect_devices=maybe"]
      )
    with self.assertRaisesRegex(
        TypeError,
        r"cluster_config\.local\.trainer_devices must be a str or non-negative"
        r" int",
    ):
      resolve_config.initialize(["cluster_config.local.trainer_devices=true"])
    with self.assertRaisesRegex(
        ValueError,
        r"cluster_config\.local\.trainer_devices must be non-negative",
    ):
      resolve_config.initialize(["cluster_config.local.trainer_devices=-1"])

  def test_execution_targets_validation_errors(self):
    for overrides, exc_type, pattern in (
        (
            ["execution_targets.target=123"],
            TypeError,
            "execution_targets.target must be a str",
        ),
        (
            ["execution_targets.placement=invalid_mode"],
            ValueError,
            "execution_targets.placement must be one of",
        ),
        (
            ["execution_targets.trainer=torch:gpu"],
            ValueError,
            "Unsupported execution target 'torch:gpu'",
        ),
        (
            [
                "cluster_config.target=external",
                "cluster_config.artifact_root=/artifacts/run",
                "execution_targets.trainer=jax:tpu",
                "execution_targets.rollout=jax:gpu",
                "execution_targets.placement=colocated",
            ],
            ValueError,
            "Placement 'colocated' requires identical execution targets",
        ),
        (
            [
                "execution_targets.trainer=torch:tpu",
                "execution_targets.rollout=jax:tpu",
                "execution_targets.placement=colocated",
            ],
            ValueError,
            "Placement 'colocated' requires identical execution targets",
        ),
        (
            ["weight_sync_config.mode=nccl"],
            ValueError,
            "weight_sync_mode='nccl' requires GPU accelerators on both trainer"
            " and rollout",
        ),
    ):
      with self.subTest(overrides=overrides):
        with self.assertRaisesRegex(exc_type, pattern):
          resolve_config.initialize(overrides)


def _has_key(config: object, dotted_key: str) -> bool:
  """Returns whether the nested mapping `config` has `dotted_key`."""
  node = config
  for part in dotted_key.split("."):
    if not isinstance(node, dict) or part not in node:
      return False
    node = node[part]
  return True


class RetiredKeysTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="key_no_node_reads",
          override="model_config.model_display=true",
          message=(
              "model_config.model_display is no longer a config key: the"
              " distributed RL nodes never read it, so remove it"
          ),
      ),
      dict(
          testcase_name="key_of_a_role_mesh",
          override="actor_model_config.mesh.allocation_policy=COMPACT",
          message=(
              "actor_model_config.mesh.allocation_policy is no longer a"
              " config key"
          ),
      ),
      dict(
          testcase_name="key_of_a_removed_section",
          override="training_config.max_steps=5",
          message=(
              "training_config is no longer a config key: .* set"
              " rl_training_config instead"
          ),
      ),
      dict(
          testcase_name="key_with_one_legal_value",
          override="vllm_config.server_mode=true",
          message=(
              "vllm_config.server_mode is no longer a config key: the vLLM"
              " rollout engines always run in server mode"
          ),
      ),
      dict(
          testcase_name="key_derived_from_use_rollout_logps",
          override="rollout_config.return_logprobs=true",
          message=(
              "rollout_config.return_logprobs is no longer a config key:"
              " derived from agentic_grpo_config.use_rollout_logps"
          ),
      ),
      dict(
          testcase_name="key_derived_from_the_mesh",
          override="vllm_config.tensor_parallel_size=2",
          message="it is the 'tp' axis of rollout_model_config.mesh",
      ),
      dict(
          testcase_name="moved_key",
          override="cluster_config.ports.trainer=20050",
          message=(
              "cluster_config.ports.trainer is no longer a config key: moved"
              " to cluster_config.roles.trainer.port"
          ),
      ),
      dict(
          testcase_name="key_of_a_moved_section",
          override="cluster_config.role_env.trainer.HF_HOME=/dev/shm/hf",
          message=(
              "cluster_config.role_env is no longer a config key: moved to"
              " cluster_config.roles.<role>.env"
          ),
      ),
      dict(
          testcase_name="key_split_per_role",
          override="cluster_config.k8s.user_container_memory=56G",
          message=(
              "moved to cluster_config.roles.trainer.memory.request and"
              " cluster_config.roles.rollout.memory.request"
          ),
      ),
      dict(
          testcase_name="limit_split_per_role",
          override="cluster_config.k8s.user_container_memory_limit=80G",
          message=(
              "moved to cluster_config.roles.trainer.memory.limit and"
              " cluster_config.roles.rollout.memory.limit"
          ),
      ),
      dict(
          testcase_name="renamed_pathways_memory",
          override="cluster_config.k8s.pathways_worker_memory=120G",
          message=(
              "cluster_config.k8s.pathways_worker_memory is no longer a config"
              " key: moved to cluster_config.pathways.server_memory.request"
          ),
      ),
  )
  def test_override_fails_with_what_to_do_instead(
      self, override: str, message: str
  ):
    with self.assertRaisesRegex(ValueError, message):
      resolve_config.initialize([override])

  def test_override_file_fails_the_same_way(self):
    task_yaml = self.create_tempfile(
        "task.yaml", "rl_training_config:\n  max_inflight_computations: 2\n"
    )
    with self.assertRaisesRegex(
        ValueError,
        "rl_training_config.max_inflight_computations is no longer a config"
        " key: the distributed RL nodes never read it",
    ):
      resolve_config.initialize([f"override_config_file={task_yaml.full_path}"])

  def test_every_retired_key_fails_with_its_hint(self):
    for key, hint in resolve_config._RETIRED_KEYS.items():
      with self.subTest(key=key):
        with self.assertRaises(ValueError) as raised:
          resolve_config.initialize([f"{key}=1"])
        self.assertEqual(
            str(raised.exception), f"{key} is no longer a config key: {hint}."
        )

  def test_other_unknown_keys_keep_the_unknown_key_error(self):
    with self.assertRaisesRegex(
        cli_config.UnknownConfigKeyError,
        "Unknown configuration key 'rollout_config.not_a_key'",
    ):
      resolve_config.initialize(["rollout_config.not_a_key=1"])

  def test_no_retired_key_is_in_the_base_config(self):
    base = omegaconf.OmegaConf.to_container(
        omegaconf.OmegaConf.load(resolve_config.DEFAULT_BASE_CONFIG_PATH)
    )
    still_in_base = [
        key for key in resolve_config._RETIRED_KEYS if _has_key(base, key)
    ]
    self.assertEmpty(still_in_base)


class UnimplementedKeysTest(parameterized.TestCase):
  """Keys kept with a TODO; any value but the accepted one fails."""

  def test_base_config_accepts_the_defaults(self):
    hp = resolve_config.initialize([])
    for key, (accepted, _) in resolve_config._UNIMPLEMENTED_KEYS.items():
      section, _, name = key.rpartition(".")
      value = hp.config_mapping(section)[name] if section else hp.config[name]
      with self.subTest(key=key):
        self.assertEqual(value, accepted)

  def test_every_key_rejects_another_value(self):
    for key, (accepted, _) in resolve_config._UNIMPLEMENTED_KEYS.items():
      value = 2 if accepted is not None else 3
      with self.subTest(key=key):
        with self.assertRaisesRegex(
            NotImplementedError,
            rf"{re.escape(key)}={value} is not supported yet .*; leave it at"
            rf" {accepted!r}\.",
        ):
          resolve_config.initialize([f"{key}={value}"])

  @parameterized.named_parameters(
      dict(
          testcase_name="rng_seed",
          override="actor_model_config.rng_seed=0",
          message="LoRA always initializes with seed 0",
      ),
      dict(
          testcase_name="epochs",
          override="num_train_epochs=1",
          message="rl_training_config.max_steps sets the training length",
      ),
      dict(
          testcase_name="evaluation",
          override="rl_training_config.eval_every_n_steps=10",
          message="runs no evaluation yet",
      ),
      dict(
          testcase_name="bool_is_not_one",
          override="agentic_grpo_config.num_iterations=true",
          message="each rollout batch gets one optimizer pass",
      ),
      dict(
          testcase_name="float_is_not_one",
          override="agentic_grpo_config.num_iterations=1.0",
          message="each rollout batch gets one optimizer pass",
      ),
  )
  def test_message_says_why(self, override: str, message: str):
    with self.assertRaisesRegex(NotImplementedError, message):
      resolve_config.initialize([override])

  def test_no_unimplemented_key_is_retired(self):
    self.assertEmpty(
        set(resolve_config._UNIMPLEMENTED_KEYS)
        & set(resolve_config._RETIRED_KEYS)
    )


class RolesTest(absltest.TestCase):

  def test_each_role_has_its_own_memory(self):
    exports = resolve_config.render_bash_exports(
        resolve_config.initialize([
            "cluster_config.roles.orchestrator.memory.request=64G",
            "cluster_config.roles.orchestrator.memory.limit=96G",
            "cluster_config.roles.trainer.memory.request=260G",
            "cluster_config.roles.trainer.memory.limit=340G",
        ])
    )
    for line in (
        "export K8S_CPU_MEMORY=64G",
        "export K8S_CPU_MEMORY_LIMIT=96G",
        "export K8S_TRAINER_MEMORY=260G",
        "export K8S_TRAINER_MEMORY_LIMIT=340G",
        "export K8S_ROLLOUT_MEMORY=48G",
        "export K8S_ROLLOUT_MEMORY_LIMIT=70G",
    ):
      self.assertIn(f"{line}\n", exports)

  def test_empty_memory_exports_empty(self):
    exports = resolve_config.render_bash_exports(
        resolve_config.initialize([
            "cluster_config.roles.trainer.memory.request=''",
            "cluster_config.roles.trainer.memory.limit=''",
        ])
    )
    self.assertIn("export K8S_TRAINER_MEMORY=''\n", exports)
    self.assertIn("export K8S_TRAINER_MEMORY_LIMIT=''\n", exports)

  def test_memory_fields_must_be_strings(self):
    for role in ("orchestrator", "trainer", "rollout"):
      for field in ("request", "limit"):
        with self.subTest(role=role, field=field):
          with self.assertRaisesRegex(
              TypeError,
              rf"cluster_config\.roles\.{role}\.memory\.{field} must be a"
              r" str",
          ):
            resolve_config.initialize(
                [f"cluster_config.roles.{role}.memory.{field}=64"]
            )

  def test_memory_must_be_a_request_and_limit_mapping(self):
    with self.assertRaisesRegex(
        TypeError,
        r"cluster_config\.roles\.trainer\.memory must be a mapping of"
        r" `request` and `limit`",
    ):
      resolve_config.initialize(["cluster_config.roles.trainer.memory=64G"])

  def test_roles_are_the_four_processes(self):
    roles = resolve_config.initialize([]).config_mapping("cluster_config")[
        "roles"
    ]
    self.assertEqual(tuple(roles), resolve_config.PROCESS_ROLES)
    self.assertNotIn("memory", roles["inference"])


class LauncherSettingsTest(absltest.TestCase):
  """The Pathways, Raiden and reservation settings `k8s.sh` passes on."""

  def _exports(self, *overrides: str) -> str:
    return resolve_config.render_bash_exports(
        resolve_config.initialize(list(overrides))
    )

  def test_defaults_match_the_former_environment_defaults(self):
    exports = self._exports()
    for line in (
        "export WEIGHT_SYNC_RAIDEN_USE_FFI=''",
        "export WEIGHT_SYNC_RAIDEN_DATA_NICS=eth0",
        "export WEIGHT_SYNC_RAIDEN_BROADCAST_K=64",
        "export WEIGHT_SYNC_RAIDEN_MULTI_NUMA=false",
        "export K8S_TPU_RESERVATION=''",
        "export K8S_PATHWAYS_IFRT_THRESHOLD_BYTES=''",
        "export K8S_PATHWAYS_WORKER_ENV_JSON='{}'",
        "export K8S_PATHWAYS_PROXY_EXTRA_ARGS=''",
        "export K8S_PATHWAYS_SIDECAR_IMAGE=''",
        "export K8S_PATHWAYS_SIDECAR_MEMORY=16Gi",
        "export K8S_PATHWAYS_SIDECAR_MEMORY_LIMIT=''",
        "export K8S_PATHWAYS_SIDECAR_SHM=true",
        "export K8S_PATHWAYS_CKPT_GCSFUSE_BUCKET=''",
        "export K8S_PATHWAYS_CKPT_GCSFUSE_MOUNT_PATH=/gcs/ckpt",
        (
            "export K8S_PATHWAYS_CKPT_GCSFUSE_MOUNT_OPTIONS="
            "implicit-dirs,metadata-cache:negative-ttl-secs:0"
        ),
        "export K8S_ENABLE_PATHWAYS_PERSISTENCE=false",
        "export K8S_PATHWAYS_PROXY_MEMORY=16G",
        "export K8S_PATHWAYS_PROXY_MEMORY_LIMIT=190G",
        "export K8S_PATHWAYS_RM_MEMORY=4G",
        "export K8S_PATHWAYS_RM_MEMORY_LIMIT=32G",
        "export K8S_PATHWAYS_SERVER_MEMORY=100G",
        "export K8S_PATHWAYS_SERVER_MEMORY_LIMIT=''",
    ):
      self.assertIn(f"{line}\n", exports)

  def test_overrides_reach_the_exports(self):
    exports = self._exports(
        "weight_sync_config.raiden.use_ffi=false",
        "weight_sync_config.raiden.data_nics=",
        "weight_sync_config.raiden.multi_numa=true",
        "cluster_config.k8s.tpu_reservation=my-res",
        (
            "cluster_config.pathways.trainer.ifrt_proxy_large_transfer_threshold"
            "_bytes=16777216"
        ),
        "cluster_config.pathways.worker_env.LIBTPU_INIT_ARGS=--a=1 --b=2",
        # As in a role's `env`, a value that parses as a bool or a number
        # needs quotes.
        "cluster_config.pathways.worker_env.SKIP_X='true'",
        "cluster_config.pathways.proxy_extra_args=--xla_a=1 --xla_b=2",
        "cluster_config.pathways.sidecar.image=sidecar:v1",
        "cluster_config.pathways.sidecar.shm=false",
        "cluster_config.pathways.sidecar.memory.limit=20Gi",
        (
            "cluster_config.pathways.trainer.checkpoint_gcsfuse.bucket="
            "gs://my-bucket/"
        ),
        (
            "cluster_config.pathways.trainer.checkpoint_gcsfuse.mount_path="
            "/mnt/ckpt"
        ),
        (
            "cluster_config.pathways.trainer.checkpoint_gcsfuse.mount_options="
            "only-dir=a"
        ),
        "cluster_config.pathways.trainer.persistence=true",
        "cluster_config.pathways.rm_memory.limit=''",
        "cluster_config.pathways.server_memory.limit=120G",
    )
    for line in (
        "export WEIGHT_SYNC_RAIDEN_USE_FFI=false",
        "export WEIGHT_SYNC_RAIDEN_DATA_NICS=''",
        "export WEIGHT_SYNC_RAIDEN_MULTI_NUMA=true",
        "export K8S_TPU_RESERVATION=my-res",
        "export K8S_PATHWAYS_IFRT_THRESHOLD_BYTES=16777216",
        (
            "export K8S_PATHWAYS_WORKER_ENV_JSON="
            """'{"LIBTPU_INIT_ARGS": "--a=1 --b=2", "SKIP_X": "true"}'"""
        ),
        "export K8S_PATHWAYS_PROXY_EXTRA_ARGS='--xla_a=1 --xla_b=2'",
        "export K8S_PATHWAYS_SIDECAR_IMAGE=sidecar:v1",
        "export K8S_PATHWAYS_SIDECAR_SHM=false",
        "export K8S_PATHWAYS_SIDECAR_MEMORY_LIMIT=20Gi",
        "export K8S_PATHWAYS_CKPT_GCSFUSE_BUCKET=gs://my-bucket/",
        "export K8S_PATHWAYS_CKPT_GCSFUSE_MOUNT_PATH=/mnt/ckpt",
        "export K8S_PATHWAYS_CKPT_GCSFUSE_MOUNT_OPTIONS=only-dir=a",
        "export K8S_ENABLE_PATHWAYS_PERSISTENCE=true",
        "export K8S_PATHWAYS_RM_MEMORY_LIMIT=''",
        "export K8S_PATHWAYS_SERVER_MEMORY_LIMIT=120G",
    ):
      self.assertIn(f"{line}\n", exports)

  def test_malformed_settings_fail(self):
    for override, error, message in (
        (
            "weight_sync_config.raiden.use_ffi=1",
            TypeError,
            "raiden.use_ffi must be a bool or null",
        ),
        (
            "weight_sync_config.raiden.multi_numa=1",
            TypeError,
            "raiden.multi_numa must be a bool",
        ),
        (
            "weight_sync_config.raiden.broadcast_k=0",
            ValueError,
            "raiden.broadcast_k must be a positive int",
        ),
        (
            "weight_sync_config.raiden.broadcast_k=true",
            ValueError,
            "raiden.broadcast_k must be a positive int",
        ),
        (
            (
                "cluster_config.pathways.trainer.ifrt_proxy_large_transfer_threshold"
                "_bytes=0"
            ),
            ValueError,
            r"must be null or an int in \[1, 2\*\*63 - 1\]",
        ),
        (
            "cluster_config.pathways.trainer.persistence=1",
            TypeError,
            "pathways.trainer.persistence must be a bool",
        ),
        (
            "cluster_config.pathways.sidecar.shm=1",
            TypeError,
            "pathways.sidecar.shm must be a bool",
        ),
        (
            "cluster_config.pathways.trainer.checkpoint_gcsfuse.bucket=7",
            TypeError,
            r"trainer\.checkpoint_gcsfuse\.bucket must be a str",
        ),
        (
            (
                "cluster_config.pathways.trainer.checkpoint_gcsfuse.bucket="
                "gs://b/prefix"
            ),
            ValueError,
            r"trainer\.checkpoint_gcsfuse: invalid checkpoint GCSFuse bucket",
        ),
        (
            # Checked with no bucket set too.
            (
                "cluster_config.pathways.trainer.checkpoint_gcsfuse.mount_path="
                "gcs/ckpt"
            ),
            ValueError,
            r"trainer\.checkpoint_gcsfuse: .* must be an absolute path",
        ),
        (
            "cluster_config.pathways.proxy_extra_args=7",
            TypeError,
            "pathways.proxy_extra_args must be a str",
        ),
        (
            "cluster_config.pathways.rm_memory.limit=7",
            TypeError,
            r"pathways\.rm_memory\.limit must be a str",
        ),
        (
            "cluster_config.pathways.sidecar.memory=16Gi",
            TypeError,
            (
                r"pathways\.sidecar\.memory must be a mapping of `request`"
                r" and `limit`"
            ),
        ),
        (
            "cluster_config.k8s.tpu_reservation=7",
            TypeError,
            "k8s.tpu_reservation must be a str",
        ),
        (
            "cluster_config.pathways.worker_env.SOME_VAR=7",
            TypeError,
            "worker_env.SOME_VAR must be a string",
        ),
        (
            # The Pathways template sets it; it has its own key.
            "cluster_config.pathways.worker_env.RAIDEN_BROADCAST_K='128'",
            ValueError,
            (
                "worker_env: Pathways worker environment variable"
                " 'RAIDEN_BROADCAST_K' is set by the template"
            ),
        ),
    ):
      with self.subTest(override=override):
        with self.assertRaisesRegex(error, message):
          resolve_config.initialize([override])

  def test_worker_env_values_pass_through_unchanged(self):
    # `k8s.sh` passes the variables to the manifest generator as JSON, so any
    # string value survives.
    env = {"A": "x\ry=z", "B": "a\nb", "C": " padded", "D": '"quoted"'}
    overlay = self.create_tempfile(
        content=(
            "cluster_config:\n  pathways:\n    worker_env: "
            + json.dumps(env)
            + "\n"
        )
    )
    hp = resolve_config.initialize(
        [f"override_config_file={overlay.full_path}"]
    )
    self.assertEqual(
        json.loads(resolve_config.pathways_worker_env_json(hp)), env
    )


_TASK_OVERRIDES = (
    "data_module=my_task.data",
    "env_class_path=my_env",
    "agent_class_path=my_agent",
    "registry_module=my_task.registry",
)


class TaskConfigTest(absltest.TestCase):

  def test_base_config_names_no_task(self):
    hp = resolve_config.initialize([])
    with self.assertRaisesRegex(
        ValueError,
        "The config names no task: data_module, env_class_path,"
        " agent_class_path, registry_module must be set",
    ):
      resolve_config.validate_task(hp)

  def test_error_names_only_the_empty_keys(self):
    hp = resolve_config.initialize(list(_TASK_OVERRIDES[:2]))
    self.assertStartsWith(
        resolve_config.task_config_error(hp),
        "The config names no task: agent_class_path,"
        " registry_module must be set.",
    )

  def test_config_with_every_task_key_passes(self):
    hp = resolve_config.initialize(list(_TASK_OVERRIDES))
    self.assertEqual(resolve_config.task_config_error(hp), "")
    resolve_config.validate_task(hp)

  def test_bash_exports_carry_the_error_for_the_launcher(self):
    self.assertIn(
        "export TASK_CONFIG_ERROR='The config names no task:",
        resolve_config.render_bash_exports(resolve_config.initialize([])),
    )
    self.assertIn(
        "export TASK_CONFIG_ERROR=''\n",
        resolve_config.render_bash_exports(
            resolve_config.initialize(list(_TASK_OVERRIDES))
        ),
    )


if __name__ == "__main__":
  absltest.main()
