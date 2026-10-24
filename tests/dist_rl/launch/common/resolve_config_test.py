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

import os
import subprocess
from unittest import mock

from absl.testing import absltest
from tunix.cli import config as cli_config
from tunix.dist_rl.launch.common import resolve_config


class ResolveConfigTest(absltest.TestCase):

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
    # A Hugging Face hub id needs HF_TOKEN; a local path does not.
    env_without_token = {
        k: v for k, v in os.environ.items() if k != "HF_TOKEN"
    }
    with mock.patch.dict(os.environ, env_without_token, clear=True):
      with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
        resolve_config.initialize([])
      hp = resolve_config.initialize(
          ["tokenizer_config.tokenizer_path=/tmp/models/qwen"]
      )
    self.assertEqual(
        hp.config["tokenizer_config"]["tokenizer_path"], "/tmp/models/qwen"
    )
    with self.assertRaisesRegex(ValueError, "tokenizer_type"):
      resolve_config.initialize(["tokenizer_config.tokenizer_type=bpe"])

  def test_unknown_key_fails_loud(self):
    with self.assertRaisesRegex(
        ValueError, "rl_training_config.nonexistent_flag"
    ):
      resolve_config.initialize(["rl_training_config.nonexistent_flag=123"])

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

  def test_render_bash_exports_evaluates_in_bash(self):
    hp = resolve_config.initialize([
        "model_config.model_name=Qwen3-0.6B",
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
            + '\necho "$CLUSTER_TARGET|$ACTOR_MODEL_NAME|$WEIGHT_SYNC_MODE|'
            '$SANDBOX_NAMESPACE|$TRAINER_MAIN|$DRY_RUN"',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    self.assertEqual(
        proc.stdout.strip(),
        "local|Qwen3-0.6B|raiden|swe-ns|"
        "tunix.dist_rl.launch.nodes.trainer.main|false",
    )


if __name__ == "__main__":
  absltest.main()
