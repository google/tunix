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

"""Dry-runs `openmath_dist/launcher.sh` and `openmath_dist/k8s_launcher.sh`."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.launch.common import launcher_testing
import yaml

_K8S_IMAGE = "cluster_config.k8s.tunix_image=test-image"
_CKPT = "actor_model_config.model_path=gs://test-bucket/qwen3-0.6b/0/items"

_LOCAL = launcher_testing.Expected(
    name="local",
    script="openmath_dist/launcher.sh",
    example="openmath_dist",
    module="openmath",
    target="local",
    batch_size=256,
    mini_batch_size=256,
    num_generations=8,
    max_steps=20,
    actor_mesh=(1, 2),
    rollout_mesh=(1, 2),
    model_id="Qwen/Qwen3-0.6B",
)

_K8S = launcher_testing.Expected(
    name="k8s",
    script="openmath_dist/k8s_launcher.sh",
    example="openmath_dist",
    module="openmath",
    target="k8s",
    batch_size=256,
    mini_batch_size=256,
    num_generations=8,
    max_steps=20,
    actor_mesh=(16, 1, 1),
    rollout_mesh=(4, 1),
    model_id="Qwen/Qwen3-0.6B",
    extra_args=(_K8S_IMAGE, _CKPT),
)


class OpenMathLaunchersTest(launcher_testing.LauncherTestCase):

  @parameterized.named_parameters(
      (expected.name, expected) for expected in (_LOCAL, _K8S)
  )
  def test_launcher_contract(self, expected: launcher_testing.Expected):
    self.assert_launcher(expected)

  def test_local_derives_default_artifact_and_checkpoint_paths(self):
    _, hp = self.assert_launcher(_LOCAL)
    cfg = hp.config
    self.assertTrue(
        cfg["actor_model_config"]["model_path"].endswith(
            "artifacts/openmath_qwen3_0p6b/maxtext_models/qwen3-0.6b/0/items"
        )
    )
    self.assertTrue(
        cfg["actor_model_config"]["model_download_path"].endswith(
            "artifacts/openmath_qwen3_0p6b/models/Qwen3-0.6B"
        )
    )

  def test_k8s_renders_trainer_2x2x4_and_12_rollout_2x2x1_replicas(self):
    run, hp = self.assert_launcher(_K8S)
    cfg = hp.config
    self.assertEqual(
        cfg["actor_model_config"]["model_download_path"],
        "/dev/shm/models/Qwen3-0.6B",
    )
    self.assertEqual(cfg["agentic_grpo_config"]["off_policy_steps"], 0)
    self.assertFalse(cfg["agentic_grpo_config"]["use_rollout_logps"])
    self.assertEqual(cfg["rl_training_config"]["max_seq_token_per_tpu"], 12288)
    self.assertEqual(cfg["cluster_config"]["rollout_replicas"], 12)
    self.assertEqual(
        cfg["cluster_config"]["k8s"]["trainer"]["tpu_slice"], "tpuv5:2x2x4"
    )
    self.assertEqual(
        cfg["cluster_config"]["k8s"]["rollout"]["tpu_slice"], "tpuv5:2x2x1"
    )

    jobsets = []
    for section in run.stdout.split("\n---\n")[1:]:
      for doc in yaml.safe_load_all(section):
        if doc and doc.get("kind") == "JobSet":
          jobsets.append(doc["metadata"]["name"])
    self.assertLen(jobsets, 14)
    self.assertContainsSubset(
        {
            "testuser-orch",
            "testuser-train",
            "testuser-roll-0",
            "testuser-roll-11",
        },
        jobsets,
    )


if __name__ == "__main__":
  absltest.main()
