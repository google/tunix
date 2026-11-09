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
import subprocess
from unittest import mock

from absl.testing import absltest
import omegaconf
from tunix.cli import config as cli_config
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
    self.assertEqual(hp.config["rl_training_config"]["eval_every_n_steps"], 50)
    self.assertEqual(hp.config["rollout_config"]["max_prompt_length"], 1024)
    self.assertEqual(
        hp.config["rollout_config"]["total_generation_steps"], 1024
    )
    self.assertEqual(hp.config["agentic_grpo_config"]["num_generations"], 8)
    self.assertEqual(hp.config["agentic_grpo_config"]["beta"], 0.04)
    self.assertEqual(hp.config["agentic_grpo_config"]["epsilon"], 0.2)
    self.assertEqual(hp.config["weight_sync_config"]["mode"], "none")
    self.assertEqual(
        hp.config["cluster_config"]["local"]["trainer_tpu_chips"], "0,1"
    )
    self.assertEqual(
        hp.config["cluster_config"]["local"]["rollout_tpu_chips"], "2,3"
    )
    # Since default beta=0.04 > 0, inference node is automatically enabled.
    self.assertTrue(hp.config["cluster_config"]["run_inference_node"])

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
    self.assertEqual(hp.config["rl_training_config"]["eval_every_n_steps"], 50)
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
    self.assertTrue(cluster["run_inference_node"])
    cluster = resolve_config.initialize(
        ["agentic_grpo_config.beta=0.0"]
    ).config["cluster_config"]
    self.assertFalse(cluster["run_inference_node"])
    # A reference worker started separately registers through discovery; there
    # is no address to configure.
    cluster = resolve_config.initialize(
        ["cluster_config.run_inference_node=false"]
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

    # The program runs one optimizer pass per rollout batch.
    with self.assertRaisesRegex(
        ValueError, "num_iterations must be 1, got 2: StandardRLProgram"
    ):
      resolve_config.initialize(["agentic_grpo_config.num_iterations=2"])

  def test_rollout_random_init_defaults_to_raiden_weight_sync(self):
    hp = resolve_config.initialize([])
    self.assertIsNone(hp.config["vllm_config"]["init_with_random_weights"])
    self.assertIs(resolve_config.rollout_init_with_random_weights(hp), False)
    for mode, init, expected in (
        # Null: random weights only when Raiden's bootstrap sync replaces them.
        ("raiden", "null", True),
        ("none", "null", False),
        ("fallback", "null", False),
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
    for mode in ("none", "fallback"):
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
      for mode in ("none", "fallback"):
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
    # nodes, so `stop` and `otel` resolve the default beta (0.04) on k8s too.
    hp = resolve_config.initialize(["cluster_config.target=k8s"])
    self.assertTrue(hp.config["cluster_config"]["run_inference_node"])
    self.assertIn(
        "export RUN_INFERENCE_NODE=true\n",
        resolve_config.render_bash_exports(hp),
    )
    # A reference worker started separately joins through discovery.
    hp = resolve_config.initialize([
        "cluster_config.target=k8s",
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
        "cpu_memory=64",
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
    # `k8s.sh` omits the flag, so each template keeps its own default.
    self.assertIn("export K8S_TERMINATION_GRACE_SECONDS=''\n", exports)

  def test_orchestrator_choices_fail_loud(self):
    hp = resolve_config.initialize([
        "agentic_grpo_config.trajectory_group_order=prompt_arrival",
        "weight_sync_config.mode=fallback",
    ])
    self.assertEqual(
        hp.config["agentic_grpo_config"]["trajectory_group_order"],
        "prompt_arrival",
    )
    self.assertEqual(hp.config["weight_sync_config"]["mode"], "fallback")
    with self.assertRaisesRegex(
        ValueError, "agentic_grpo_config.trajectory_group_order must be one of"
    ):
      resolve_config.initialize(
          ["agentic_grpo_config.trajectory_group_order=random"]
      )
    with self.assertRaisesRegex(
        ValueError, "weight_sync_config.mode must be one of"
    ):
      resolve_config.initialize(["weight_sync_config.mode=nccl"])
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
          "weight_sync_config.mode=fallback",
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
    self.assertEqual(vllm["kwargs"], {})

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
    # The typed settings stay out of the free-form vLLM engine arguments.
    self.assertEqual(vllm["kwargs"], {})

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
    with self.assertRaisesRegex(TypeError, "vllm_config.server_mode"):
      resolve_config.initialize(["vllm_config.server_mode=0"])
    for bad_ft in ("yes_please", "0"):
      with self.assertRaisesRegex(
          TypeError, r"cluster_config\.rollout_fault_tolerance"
      ):
        resolve_config.initialize(
            [f"cluster_config.rollout_fault_tolerance={bad_ft}"]
        )

  def test_typed_vllm_keys_rejected_in_free_form_kwargs(self):
    with self.assertRaisesRegex(
        ValueError, r"\['detokenize', 'return_routed_experts'\].*typed keys"
    ):
      resolve_config.initialize([
          "vllm_config.kwargs.return_routed_experts=true",
          "vllm_config.kwargs.detokenize=true",
      ])
    # Other free-form engine arguments are still accepted.
    vllm = resolve_config.initialize(
        ["vllm_config.kwargs.attention=flash"]
    ).config["vllm_config"]
    self.assertEqual(vllm["kwargs"], {"attention": "flash"})

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

  def test_maxtext_extra_flags_need_a_maxtext_trainer(self):
    maxtext = [
        "model_config.model_source=maxtext",
        "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
        "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
        "weight_sync_config.mode=raiden",
    ]
    flags = (
        "dist_training_config.maxtext_extra_flags="
        "remat_policy=custom opt_type=sgd"
    )
    # A Tunix trainer would silently ignore them.
    with self.assertRaisesRegex(
        ValueError, "maxtext_extra_flags configures a MaxText trainer"
    ):
      resolve_config.initialize([flags])
    with self.assertRaisesRegex(TypeError, "must be a string"):
      resolve_config.initialize(
          [*maxtext, "dist_training_config.maxtext_extra_flags=3"]
      )
    # A malformed entry fails here, before any job is submitted.
    for malformed in ("remat_policy custom", "=custom"):
      with self.subTest(malformed=malformed):
        with self.assertRaisesRegex(ValueError, "must be key=value pairs"):
          resolve_config.initialize([
              *maxtext,
              f"dist_training_config.maxtext_extra_flags={malformed}",
          ])
    hp = resolve_config.initialize([*maxtext, flags])
    self.assertEqual(
        hp.config_mapping("dist_training_config")["maxtext_extra_flags"],
        "remat_policy=custom opt_type=sgd",
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
    self.assertEqual(ws["raiden_data_nics"], "eth0")

    hp = resolve_config.initialize([
        "vllm_config.skip_jax_precompile=false",
        "actor_model_config.model_source=maxtext",
        "actor_model_config.model_path=/ckpt",
        "actor_model_config.kwargs.maxtext_model_name=qwen3-1.7b",
        "dist_training_config.ckpt_d2h_concurrent_gb=16",
        "weight_sync_config.raiden_data_nics=eth0,eth1",
    ])
    vllm = hp.config_mapping("vllm_config")
    self.assertIs(vllm["skip_jax_precompile"], False)
    dist_training = hp.config_mapping("dist_training_config")
    self.assertEqual(dist_training["ckpt_d2h_concurrent_gb"], 16)
    self.assertEqual(
        hp.config_mapping("weight_sync_config")["raiden_data_nics"],
        "eth0,eth1",
    )

  def test_invalid_library_settings_fail_loud(self):
    # The nodes turn them into environment variables; a wrong type would be
    # misread there, after the jobs started.
    d2h = "dist_training_config.ckpt_d2h_concurrent_gb"
    for override, error in (
        ("vllm_config.skip_jax_precompile=1", "skip_jax_precompile must be a"),
        ("weight_sync_config.verify_weights=1", "verify_weights must be a"),
        ("weight_sync_config.raiden_data_nics=1", "must be a string"),
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


if __name__ == "__main__":
  absltest.main()
