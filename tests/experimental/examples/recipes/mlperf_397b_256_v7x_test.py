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

"""Dry-run tests for `experimental/examples/recipes/mlperf_397b_256_v7x.sh`."""

from __future__ import annotations

import os
import pathlib
from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.launch.common import launcher_testing
from tunix.dist_rl.launch.common import resolve_config
import yaml

_RECIPE = (
    pathlib.Path(resolve_config.__file__).parents[3]
    / "experimental/examples/recipes/mlperf_397b_256_v7x.sh"
)


class Mlperf397bRecipeTest(launcher_testing.LauncherTestCase):

  def test_resolves_and_renders_34_jobsets(self):
    run = self.dry_run(
        str(_RECIPE),
        "cluster_config.k8s.tunix_image=test-image",
        "cluster_config.k8s.tpu_reservation=test-v7x-reservation",
        "actor_model_config.model_path=gs://test-bucket/ckpt",
        "data_config.dataset_path=gs://test-bucket/data",
    )
    cfg = run.resolve().config

    self.assertEqual(cfg["model_config"]["model_name"], "Qwen3.5-397B-A17B")
    self.assertEqual(cfg["model_config"]["model_source"], "maxtext")
    self.assertEqual(
        cfg["actor_model_config"]["kwargs"]["maxtext_model_name"],
        "qwen3.5-397b-a17b",
    )
    self.assertEqual(cfg["actor_model_config"]["mesh"]["shape"], "(32,1,2)")
    self.assertEqual(cfg["actor_model_config"]["kwargs"]["mesh_context"], 4)
    self.assertEqual(cfg["rollout_model_config"]["mesh"]["shape"], "(1,1,8)")
    self.assertEqual(
        cfg["data_module"], "tunix.experimental.examples.deepswe_dist.prewarm"
    )
    self.assertEqual(
        cfg["registry_module"],
        "tunix.experimental.examples.deepswe_dist.deepswe",
    )

    roles_cfg = cfg["cluster_config"]["roles"]
    self.assertEqual(roles_cfg["trainer"]["memory"]["request"], "48G")
    self.assertEqual(roles_cfg["trainer"]["memory"]["limit"], "260G")
    trainer_libtpu = roles_cfg["trainer"]["env"]["LIBTPU_INIT_ARGS"]
    self.assertIn("--xla_tpu_scoped_vmem_limit_kib=65472", trainer_libtpu)
    self.assertNotIn("--xla_tpu_scoped_vmem_limit_kib=65536", trainer_libtpu)
    self.assertIn(
        "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=true",
        trainer_libtpu,
    )
    self.assertIn("--xla_tpu_max_hbm_size_mib=84992", trainer_libtpu)

    vllm_cfg = cfg["vllm_config"]
    self.assertTrue(vllm_cfg["async_scheduling"])
    self.assertEqual(vllm_cfg["routed_experts_num_experts"], 512)
    self.assertEqual(vllm_cfg["max_num_seqs"], 16)
    self.assertNotIn("mamba_cache_mode", vllm_cfg["engine_kwargs"])
    additional_cfg = vllm_cfg["engine_kwargs"]["additional_config"]
    self.assertNotIn("enable_continue_decode", additional_cfg)
    self.assertEqual(additional_cfg["custom_mamba_cache_multiplier"], 16)
    sharding_strategy = additional_cfg["sharding"]["sharding_strategy"]
    self.assertEqual(sharding_strategy["expert_parallelism"], 8)
    self.assertFalse(sharding_strategy["enable_dp_attention"])
    self.assertTrue(additional_cfg["maxtext_config"]["fp8_moe"])

    rollout_env = roles_cfg["rollout"]["env"]
    self.assertEqual(rollout_env["ONEHOT_MOE_PERMUTE_THRESHOLD"], "2048")
    self.assertEqual(rollout_env["RAGGED_GATHER_REDUCE_VERSION"], "v3")
    self.assertEqual(rollout_env["TPU_SMALL_ALL_REDUCE"], "1")
    self.assertEqual(rollout_env["TPU_MOE_SINGLE_SORT"], "1")
    self.assertEqual(rollout_env["TPU_MOE_ROUTER_TOPK"], "rowmax")
    self.assertEqual(rollout_env["TPU_MOE_LOWTOK"], "1")

    maxtext_extra = cfg["dist_training_config"]["maxtext_extra_config"]
    self.assertEqual(maxtext_extra["custom_mesh_and_rule"], "cp-as-ep")
    self.assertTrue(maxtext_extra["use_ring_of_experts"])
    self.assertTrue(maxtext_extra["moe_tc_ragged_sort"])
    self.assertFalse(maxtext_extra["use_splash_scheduler"])
    self.assertEqual(maxtext_extra["sa_block_q_dkv"], 1024)
    self.assertEqual(maxtext_extra["sa_block_kv_dkv"], 1024)
    self.assertEqual(maxtext_extra["decoder_layer_input"], "device")
    self.assertEqual(maxtext_extra["kv_tp_size"], 8)
    self.assertTrue(maxtext_extra["rollout_fp8_moe"])
    self.assertTrue(maxtext_extra["fp8_moe_fake_quant"])
    self.assertEqual(
        maxtext_extra["trainable_parameters_mask"],
        "^(?!.*routed_experts/gate/kernel).*",
    )

    k8s_cfg = cfg["cluster_config"]["k8s"]
    self.assertTrue(k8s_cfg["use_dynamic_slicing"])
    self.assertEqual(k8s_cfg["head_nodepool"], "cpu-np")
    self.assertEqual(k8s_cfg["tpu_reservation"], "test-v7x-reservation")
    self.assertEqual(k8s_cfg["rollout"]["jobset_yaml"], "jobset.tpu.yaml")
    self.assertEqual(k8s_cfg["rollout"]["tpu_slice"], "tpu7x:2x2x1")

    self.assertIn("ORCHESTRATOR_ID=testuser-orch", run.stdout)
    self.assertIn(
        "cloud.google.com/reservation-name: test-v7x-reservation", run.stdout
    )
    self.assertIn("SANDBOX_TOLERATIONS=", run.stdout)

    jobsets = {}
    for section in run.stdout.split("\n---\n")[1:]:
      for doc in yaml.safe_load_all(section):
        if doc and doc.get("kind") == "JobSet":
          jobsets[doc["metadata"]["name"]] = doc
    self.assertLen(jobsets, 34)
    self.assertContainsSubset(
        {
            "testuser-orch",
            "testuser-train",
            "testuser-roll-0",
            "testuser-roll-31",
        },
        jobsets.keys(),
    )

    trainer = {
        job["name"]: job["template"]["spec"]["template"]["spec"]
        for job in jobsets["testuser-train"]["spec"]["replicatedJobs"]
    }
    (proxy,) = [
        c for c in trainer["proc"]["initContainers"]
        if c["name"] == "pathways-proxy"
    ]
    self.assertIn("--xla_tpu_scoped_vmem_limit_kib=65472", proxy["args"])
    self.assertNotIn(
        "--DANGEROUS_tpu_runtime_abi_verification_disabled=true", proxy["args"]
    )

    (main,) = [c for c in trainer["proc"]["containers"] if c["name"] == "main"]
    threshold = "IFRT_PROXY_GRPC_LARGE_TRANSFER_OPTIMIZATION_THRESHOLD_BYTES"
    self.assertIn(
        {"name": threshold, "value": "16777216"},
        main["env"],
    )

    (pw_worker,) = [
        c for c in trainer["pw-node"]["containers"]
        if c["name"] == "pathways-worker"
    ]
    pw_env = {e["name"]: e["value"] for e in pw_worker["env"] if "value" in e}
    self.assertEqual(pw_env["SKIP_MEGASCALE_PJRT_CLIENT"], "true")
    self.assertEqual(pw_env["RAIDEN_PARALLELISM"], "16")
    self.assertEqual(pw_env["TPU_RAIDEN_CONTROL_PLANE_BACKEND"], "tcp")
    self.assertEqual(pw_env["ENABLE_MULTI_NUMA"], "0")
    self.assertEqual(pw_env["TPU_RAIDEN_DATA_NICS"], "eth0")
    self.assertIn(
        "--xla_tpu_scoped_vmem_limit_kib=65472", pw_env["LIBTPU_INIT_ARGS"]
    )
    self.assertEndsWith(
        pw_env["LIBTPU_INIT_ARGS"],
        " --megascale_port=-1 --xprof_compress_jftrace=true",
    )

  def test_scales_batch_size_and_overrides_seq_tokens_and_hbm_cap(self):
    with mock.patch.dict(os.environ, {"TRAINER_MAX_HBM_SIZE_MIB": "88000"}):
      run = self.dry_run(
          str(_RECIPE),
          "cluster_config.k8s.tunix_image=test-image",
          "actor_model_config.model_path=gs://test-bucket/ckpt",
          "data_config.dataset_path=gs://test-bucket/data",
          "batch_size=64",
          "cluster_config.rollout_replicas=128",
          "cluster_config.roles.trainer.memory.request=160G",
          "rl_training_config.max_seq_tokens_per_device=32768",
      )
    cfg = run.resolve().config
    self.assertEqual(cfg["batch_size"], 64)
    self.assertEqual(cfg["rl_training_config"]["mini_batch_size"], 64)
    self.assertEqual(
        cfg["rl_training_config"]["max_seq_tokens_per_device"], 32768
    )
    self.assertEqual(cfg["cluster_config"]["rollout_replicas"], 128)
    self.assertEqual(
        cfg["cluster_config"]["roles"]["trainer"]["memory"]["request"], "160G"
    )
    self.assertIn(
        "--xla_tpu_max_hbm_size_mib=88000",
        cfg["cluster_config"]["roles"]["trainer"]["env"]["LIBTPU_INIT_ARGS"],
    )
    self.assertIn(
        "--xla_tpu_max_hbm_size_mib=88000",
        cfg["cluster_config"]["pathways"]["worker_env"]["LIBTPU_INIT_ARGS"],
    )
    self.assertIn("--xla_tpu_max_hbm_size_mib=88000", run.stdout)

  def test_stop(self):
    run = self.dry_run(
        str(_RECIPE),
        "stop",
        "actor_model_config.model_path=gs://test-bucket/ckpt",
    )
    self.assertIn("kubectl delete jobset testuser-orch", run.stdout)
    self.assertIn("testuser-roll-31", run.stdout)


if __name__ == "__main__":
  absltest.main()
