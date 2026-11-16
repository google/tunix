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

"""Dry-runs the GSM8K `launcher.sh` (local) and `k8s_launcher.sh` (GKE)."""

from __future__ import annotations

import os
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.launch.common import launcher_testing
import yaml

_LOCAL = launcher_testing.Expected(
    name="local",
    script="math_gsm8k_dist/launcher.sh",
    example="math_gsm8k_dist",
    module="gsm8k",
    target="local",
    batch_size=4,
    mini_batch_size=2,
    num_generations=8,
    max_steps=1,
    actor_mesh=(1, 2),
    rollout_mesh=(1, 2),
    model_id="Qwen/Qwen3-1.7B",
)
# Copybara replaces `k8s_launcher.sh`'s internal default image with
# `$TUNIX_IMAGE` (`copy.bara.sky`), so the GKE dry runs set one.
_K8S_IMAGE = "cluster_config.k8s.tunix_image=test-image"
_K8S = launcher_testing.Expected(
    name="k8s",
    script="math_gsm8k_dist/k8s_launcher.sh",
    example="math_gsm8k_dist",
    module="gsm8k",
    target="k8s",
    batch_size=4,
    mini_batch_size=2,
    num_generations=8,
    max_steps=1,
    actor_mesh=(16, 1, 1),
    rollout_mesh=(1, 16),
    model_id="Qwen/Qwen3-1.7B",
    extra_args=(_K8S_IMAGE,),
)


class LaunchersTest(launcher_testing.LauncherTestCase):

  @parameterized.named_parameters(
      (expected.name, expected) for expected in (_LOCAL, _K8S)
  )
  def test_launcher_contract(self, expected: launcher_testing.Expected):
    self.assert_launcher(expected)

  def test_local_positive_beta_requires_reference_inference_chips(self):
    run = self.run_launcher(
        "math_gsm8k_dist/launcher.sh", "agentic_grpo_config.beta=0.04"
    )
    self.assertNotEqual(run.returncode, 0)
    self.assertIn("cluster_config.local.inference_devices", run.stderr)
    with_chips = self.dry_run(
        "math_gsm8k_dist/launcher.sh",
        "agentic_grpo_config.beta=0.04",
        "cluster_config.local.inference_devices=4,5",
    )
    self.assertIn("Inference command:", with_chips.stdout)

  def test_local(self):
    run, hp = self.assert_launcher(_LOCAL)
    self.assertNotIn("Inference command:", run.stdout)
    config = hp.config
    self.assertEqual(config["agentic_grpo_config"]["beta"], 0.0)
    self.assertFalse(config["cluster_config"]["run_inference_node"])
    # TFDS uses its default data directory.
    self.assertNotIn("data_dir", config["data_config"])

  def test_k8s(self):
    _, hp = self.assert_launcher(_K8S)
    config = hp.config
    self.assertEqual(config["agentic_grpo_config"]["beta"], 0.0)
    self.assertFalse(config["cluster_config"]["run_inference_node"])
    self.assertNotIn("data_dir", config["data_config"])
    self.assertEqual(config["cluster_config"]["ports"]["orchestrator"], 20000)
    self.assertEqual(config["cluster_config"]["ports"]["rollout"], 20001)
    self.assertEqual(config["cluster_config"]["ports"]["trainer"], 20002)
    self.assertEqual(config["rollout_config"]["max_prompt_length"], 1024)
    self.assertEqual(config["rollout_config"]["total_generation_steps"], 1024)
    self.assertEqual(config["rl_training_config"]["eval_every_n_steps"], 50)
    self.assertEqual(config["model_config"]["kwargs"], {})
    self.assertEqual(hp.mesh_axes("actor_model_config")["fsdp"], 16)
    k8s = config["cluster_config"]["k8s"]
    self.assertEqual(k8s["trainer"]["jobset_yaml"], "jobset.pathways.yaml")
    self.assertEqual(k8s["trainer"]["tpu_slice"], "tpuv5e:4x4")
    self.assertEqual(
        k8s["rollout"]["jobset_yaml"], "leaderworkerset.mcjax.ray.yaml"
    )
    self.assertEqual(k8s["rollout"]["tpu_slice"], "tpuv5e:4x4")
    self.assertEqual(config["cluster_config"]["rollout_replicas"], 1)

  def test_user_args_override_launcher_values(self):
    run = self.dry_run(
        "math_gsm8k_dist/k8s_launcher.sh",
        _K8S_IMAGE,
        "batch_size=8",
        "cluster_config.rollout_replicas=2",
    )
    config = run.resolve().config
    self.assertEqual(config["batch_size"], 8)
    self.assertEqual(config["cluster_config"]["rollout_replicas"], 2)
    self.assertIn("testuser-roll-1", run.stdout)

  def test_k8s_passes_otel_settings_to_every_pod(self):
    otel_env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://testuser-otel:4317",
        "OTEL_METRIC_EXPORT_INTERVAL": "5000",
    }
    with mock.patch.dict(os.environ, otel_env):
      run = self.dry_run("math_gsm8k_dist/k8s_launcher.sh", _K8S_IMAGE)
    exporting = set(
        re.findall(
            r"OTEL_EXPORTER_OTLP_ENDPOINT=http://testuser-otel:4317 "
            r"OTEL_METRIC_EXPORT_INTERVAL=5000 .*--process_main=(\S+)",
            run.stdout,
        )
    )
    self.assertEqual(
        exporting,
        {
            "tunix.dist_rl.launch.nodes.orchestrator.main",
            "tunix.dist_rl.launch.nodes.trainer.main",
            "tunix.dist_rl.launch.nodes.rollout.main",
        },
    )

  def test_k8s_otel_commands_deploy_and_delete_the_collector(self):
    run = self.dry_run("math_gsm8k_dist/k8s_launcher.sh", "otel")
    self.assertIn(
        "kubectl delete deployment testuser-otel -n default", run.stdout
    )
    self.assertIn(
        "kubectl create configmap testuser-otel-config -n default"
        " --from-file=otel-collector.yaml=",
        run.stdout,
    )
    manifests = {}
    for section in run.stdout.split("\n---\n")[1:]:
      for doc in yaml.safe_load_all(section):
        if doc:
          manifests[doc["kind"]] = doc
    self.assertEqual(set(manifests), {"Deployment", "Service"})
    self.assertEqual(manifests["Service"]["spec"]["ports"][0]["port"], 4317)
    run = self.dry_run("math_gsm8k_dist/k8s_launcher.sh", "stop-otel")
    self.assertIn(
        "kubectl delete service testuser-otel -n default", run.stdout
    )
    self.assertNotIn("apiVersion:", run.stdout)


if __name__ == "__main__":
  absltest.main()
