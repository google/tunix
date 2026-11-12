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

"""Dry-run tests for data_plane_benchmark local and GKE TPU v5p launchers."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.launch.common import launcher_testing

_MODEL_ID = "tunix/dist_rl/benchmarks/data_plane/tokenizer"
_K8S_IMAGE = "cluster_config.k8s.tunix_image=test-image"

_LOCAL = launcher_testing.Expected(
    name="local",
    script="data_plane/launcher.sh",
    example="data_plane",
    module="data_plane_benchmark",
    target="local",
    batch_size=8,
    mini_batch_size=8,
    num_generations=8,
    max_steps=4,
    actor_mesh=(4, 1),
    rollout_mesh=(4, 1),
    model_id=_MODEL_ID,
)

_K8S = launcher_testing.Expected(
    name="k8s",
    script="data_plane/k8s_launcher.sh",
    example="data_plane",
    module="data_plane_benchmark",
    target="k8s",
    batch_size=8,
    mini_batch_size=8,
    num_generations=8,
    max_steps=4,
    actor_mesh=(64, 1, 1),
    rollout_mesh=(1, 4),
    model_id=_MODEL_ID,
    extra_args=(_K8S_IMAGE,),
)


class LaunchersTest(launcher_testing.LauncherTestCase):
  root = "benchmarks"

  @parameterized.named_parameters(
      (expected.name, expected) for expected in (_LOCAL, _K8S)
  )
  def test_launcher_contract(self, expected: launcher_testing.Expected):
    self.assert_launcher(expected)

  def test_k8s_v5p_topology_and_routed_experts_settings(self):
    run, hp = self.assert_launcher(_K8S)
    config = hp.config
    self.assertEqual(config["weight_sync_config"]["mode"], "noop")
    self.assertEqual(config["agentic_grpo_config"]["beta"], 0.0)
    self.assertTrue(config["vllm_config"]["return_routed_experts"])
    self.assertEqual(
        config["vllm_config"]["routed_experts_encode_mode"], "uint10"
    )
    self.assertEqual(config["vllm_config"]["routed_experts_num_experts"], 384)
    self.assertFalse(config["dist_training_config"]["precompile_trainer"])
    ports = config["cluster_config"]["ports"]
    self.assertEqual(len(set(ports.values())), len(ports))
    k8s = config["cluster_config"]["k8s"]
    self.assertEqual(k8s["trainer"]["jobset_yaml"], "jobset.tpu.yaml")
    self.assertEqual(k8s["trainer"]["tpu_slice"], "tpuv5:2x2x1")
    self.assertEqual(k8s["rollout"]["jobset_yaml"], "jobset.tpu.yaml")
    self.assertEqual(k8s["rollout"]["tpu_slice"], "tpuv5:2x2x1")
    self.assertEqual(config["cluster_config"]["rollout_replicas"], 4)
    self.assertIn("testuser-roll-3", run.stdout)

  def test_cli_overrides_scale_replicas_and_staleness(self):
    run = self.dry_run(
        "data_plane/k8s_launcher.sh",
        _K8S_IMAGE,
        "cluster_config.rollout_replicas=8",
        "agentic_grpo_config.off_policy_steps=5",
    )
    config = run.resolve().config
    self.assertEqual(config["cluster_config"]["rollout_replicas"], 8)
    self.assertEqual(config["agentic_grpo_config"]["off_policy_steps"], 5)
    self.assertIn("testuser-roll-7", run.stdout)


if __name__ == "__main__":
  absltest.main()
