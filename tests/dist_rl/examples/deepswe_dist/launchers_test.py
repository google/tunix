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

"""Dry-runs the DeepSWE `launcher_v2.sh` (local) and `k8s_launcher_v2.sh` (GKE)."""

from __future__ import annotations

import os
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.cli import config as cli_config
from tunix.dist_rl.launch.common import launcher_testing
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import orchestrator
import yaml

_LOCAL = launcher_testing.Expected(
    name="local",
    script="deepswe_dist/launcher_v2.sh",
    example="deepswe_dist",
    module="deepswe",
    target="local",
    batch_size=8,
    mini_batch_size=8,
    num_generations=8,
    max_steps=201,
    actor_mesh=(1, 2),
    rollout_mesh=(1, 2),
    model_id="Qwen/Qwen3-32B",
)
# Copybara replaces `k8s_launcher_v2.sh`'s internal default image with
# `$TUNIX_IMAGE` (`copy.bara.sky`), so the GKE dry runs set one.
_K8S_IMAGE = "cluster_config.k8s.tunix_image=test-image"
_K8S = launcher_testing.Expected(
    name="k8s",
    script="deepswe_dist/k8s_launcher_v2.sh",
    example="deepswe_dist",
    module="deepswe",
    target="k8s",
    batch_size=8,
    mini_batch_size=8,
    num_generations=8,
    max_steps=201,
    actor_mesh=(8, 8, 1),
    rollout_mesh=(1, 4),
    model_id="Qwen/Qwen3-32B",
    extra_args=(_K8S_IMAGE,),
)


class LaunchersTest(launcher_testing.LauncherTestCase):

  @parameterized.named_parameters(
      (expected.name, expected) for expected in (_LOCAL, _K8S)
  )
  def test_launcher_contract(self, expected: launcher_testing.Expected):
    _, hp = self.assert_launcher(expected)
    config = hp.config
    self.assertEqual(config["agentic_grpo_config"]["beta"], 0.0)
    self.assertFalse(config["cluster_config"]["run_inference_node"])
    self.assertEqual(
        config["chat_parser_config"],
        {"type": "auto", "enable_thinking": False},
    )
    self.assertEqual(config["rollout_config"]["eos_tokens"], "151645")
    self.assertEqual(config["env_class_path"], "deepswe_env")
    self.assertEqual(config["agent_class_path"], "deepswe_agent")
    self.assertEqual(
        config["env_kwargs"]["max_steps"],
        config["agentic_grpo_config"]["max_turns"],
    )
    optimizer = config["rl_training_config"]["actor_optimizer_config"]
    self.assertEqual(cli_config.create_learning_rate(optimizer), 1e-6)
    _, derived_rpc_timeout = orchestrator.resolve_weight_sync_timeouts(hp)
    self.assertEqual(derived_rpc_timeout, 5100.0)

  def test_local_positive_beta_requires_reference_inference_chips(self):
    run = self.run_launcher(
        "deepswe_dist/launcher_v2.sh", "agentic_grpo_config.beta=0.04"
    )
    self.assertNotEqual(run.returncode, 0)
    self.assertIn("cluster_config.local.inference_tpu_chips", run.stderr)
    with_chips = self.dry_run(
        "deepswe_dist/launcher_v2.sh",
        "agentic_grpo_config.beta=0.04",
        "cluster_config.local.inference_tpu_chips=4,5",
    )
    self.assertIn("Inference command:", with_chips.stdout)

  def test_local(self):
    run, hp = self.assert_launcher(_LOCAL)
    self.assertNotIn("Inference command:", run.stdout)
    config = hp.config
    rl_train = config["rl_training_config"]
    self.assertTrue(
        rl_train["checkpoint_root_directory"].endswith(
            "/artifacts/deepswe/checkpoints"
        )
    )
    self.assertTrue(
        rl_train["metrics_logging_options"]["log_dir"].endswith(
            "/artifacts/deepswe/metrics"
        )
    )
    self.assertTrue(
        config["data_config"]["cache_dir"].endswith(
            "/artifacts/deepswe/dataset_cache"
        )
    )
    self.assertEqual(
        config["kubernetes_config"],
        {
            "namespace": "rl-tunix-swebench",
            "node_selector_key": None,
            "node_selector_val": None,
        },
    )
    self.assertEqual(
        config["cluster_config"]["local"]["worker_shutdown_grace_secs"], 60
    )
    self.assertTrue(resolve_config.rollout_init_with_random_weights(hp))

  def test_k8s(self):
    run, hp = self.assert_launcher(_K8S)
    config = hp.config
    self.assertEqual(
        config["data_config"]["cache_dir"], "artifacts/deepswe/dataset_cache"
    )
    # The sandbox namespace follows `cluster_config.k8s.namespace`.
    self.assertEqual(config["kubernetes_config"]["namespace"], "default")
    self.assertIn("NAMESPACE=default", run.stdout)
    self.assertEqual(config["cluster_config"]["ports"]["orchestrator"], 20000)
    self.assertEqual(config["cluster_config"]["ports"]["rollout"], 20001)
    self.assertEqual(config["cluster_config"]["ports"]["trainer"], 20002)
    self.assertEqual(hp.mesh_axes("actor_model_config")["fsdp"], 8)
    self.assertEqual(hp.mesh_axes("actor_model_config")["tp"], 8)
    self.assertEqual(hp.mesh_axes("rollout_model_config")["fsdp"], 1)
    self.assertEqual(hp.mesh_axes("rollout_model_config")["tp"], 4)
    k8s = config["cluster_config"]["k8s"]
    self.assertEqual(k8s["cpu_memory"], "64G")
    self.assertEqual(k8s["user_container_memory"], "260G")
    self.assertEqual(k8s["user_container_memory_limit"], "260G")
    self.assertEqual(k8s["trainer"]["jobset_yaml"], "jobset.pathways.yaml")
    self.assertEqual(k8s["trainer"]["tpu_slice"], "tpuv5:4x4x4")
    self.assertIn(
        "name: IFRT_PROXY_GRPC_LARGE_TRANSFER_OPTIMIZATION_THRESHOLD_BYTES\n"
        '                value: "16777216"',
        run.stdout,
    )
    self.assertEqual(k8s["rollout"]["jobset_yaml"], "jobset.tpu.yaml")
    self.assertEqual(k8s["rollout"]["tpu_slice"], "tpuv5:2x2x1")
    self.assertEqual(config["cluster_config"]["rollout_replicas"], 16)
    self.assertIn("testuser-roll-15", run.stdout)

  def test_k8s_ifrt_proxy_threshold_env_override_takes_precedence(self):
    with mock.patch.dict(
        os.environ,
        {"TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES": "33554432"},
    ):
      run = self.dry_run("deepswe_dist/k8s_launcher_v2.sh", _K8S_IMAGE)
    self.assertIn(
        "name: IFRT_PROXY_GRPC_LARGE_TRANSFER_OPTIMIZATION_THRESHOLD_BYTES\n"
        '                value: "33554432"',
        run.stdout,
    )

  def test_k8s_sandbox_namespace_follows_the_gke_namespace(self):
    run = self.dry_run(
        "deepswe_dist/k8s_launcher_v2.sh",
        _K8S_IMAGE,
        "cluster_config.k8s.namespace=team-ns",
    )
    hp = run.resolve()
    self.assertEqual(hp.config["kubernetes_config"]["namespace"], "team-ns")
    self.assertIn("NAMESPACE=team-ns", run.stdout)

  def test_k8s_sandbox_node_selectors_propagate_to_rollout_pod(self):
    run = self.dry_run(
        "deepswe_dist/k8s_launcher_v2.sh",
        _K8S_IMAGE,
        "env_kwargs.use_agent_sandbox=true",
        "kubernetes_config.namespace=rl-tunix-swebench",
        "kubernetes_config.node_selector_key=cloud.google.com/gke-nodepool",
        "kubernetes_config.node_selector_val=swe-sandbox-pool",
    )
    hp = run.resolve()
    self.assertTrue(hp.config["env_kwargs"]["use_agent_sandbox"])
    self.assertIn("NAMESPACE=rl-tunix-swebench", run.stdout)
    self.assertIn(
        "NODE_SELECTOR_KEY=cloud.google.com/gke-nodepool"
        " NODE_SELECTOR_VAL=swe-sandbox-pool",
        run.stdout,
    )

  def test_user_args_override_launcher_values(self):
    run = self.dry_run(
        "deepswe_dist/k8s_launcher_v2.sh",
        _K8S_IMAGE,
        "batch_size=16",
        "cluster_config.rollout_replicas=2",
    )
    hp = run.resolve()
    config = hp.config
    self.assertEqual(config["batch_size"], 16)
    self.assertTrue(resolve_config.rollout_init_with_random_weights(hp))
    self.assertEqual(config["cluster_config"]["rollout_replicas"], 2)
    self.assertIn("testuser-roll-1", run.stdout)

  def test_k8s_passes_otel_settings_to_every_pod(self):
    otel_env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://testuser-otel:4317",
        "OTEL_METRIC_EXPORT_INTERVAL": "5000",
    }
    with mock.patch.dict(os.environ, otel_env):
      run = self.dry_run("deepswe_dist/k8s_launcher_v2.sh", _K8S_IMAGE)
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
    run = self.dry_run("deepswe_dist/k8s_launcher_v2.sh", "otel")
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
    run = self.dry_run("deepswe_dist/k8s_launcher_v2.sh", "stop-otel")
    self.assertIn("kubectl delete service testuser-otel -n default", run.stdout)
    self.assertNotIn("apiVersion:", run.stdout)


if __name__ == "__main__":
  absltest.main()
