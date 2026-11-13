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

"""Unit tests for `execution_target`."""

from absl.testing import absltest
from absl.testing import parameterized
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.common import execution_target


class ExecutionTargetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("jax_tpu", "jax:tpu", execution_target.JAX_TPU, "jax:tpu"),
      ("jax_gpu", " JAX:GPU ", execution_target.JAX_GPU, "jax:gpu"),
      ("torch_tpu", "torch:tpu", execution_target.TORCH_TPU, "torch:tpu"),
      ("torch_gpu", "torch:gpu", execution_target.TORCH_GPU, "torch:gpu"),
  )
  def test_parse_and_str_roundtrip(
      self,
      spec: str,
      expected: execution_target.ExecutionTarget,
      expected_str: str,
  ):
    parsed = execution_target.ExecutionTarget.parse(spec)
    self.assertEqual(parsed, expected)
    self.assertEqual(str(parsed), expected_str)

  @parameterized.named_parameters(
      ("empty", "", "Invalid execution target"),
      ("missing_colon", "jax_tpu", "Invalid execution target"),
      ("extra_colon", "jax:tpu:v5p", "Invalid execution target"),
      ("unknown_framework", "tensorflow:tpu", "Unsupported framework"),
      ("unknown_accelerator", "jax:cpu", "Unsupported accelerator"),
  )
  def test_parse_rejects_invalid_specs(self, bad_spec: str, expected_msg: str):
    with self.assertRaisesRegex(ValueError, expected_msg):
      execution_target.ExecutionTarget.parse(bad_spec)

  @parameterized.named_parameters(
      (
          "default_jax_tpu",
          execution_target.JAX_TPU,
          execution_target.JAX_TPU,
          execution_target.JAX_TPU,
          execution_target.Placement.DISAGGREGATED,
          common_datatypes.WeightSyncMode.NONE,
      ),
      (
          "disaggregated_torch_and_jax_on_tpu",
          execution_target.TORCH_TPU,
          execution_target.JAX_TPU,
          execution_target.TORCH_TPU,
          execution_target.Placement.DISAGGREGATED,
          common_datatypes.WeightSyncMode.NONE,
      ),
      (
          "disaggregated_tpu_and_gpu",
          execution_target.JAX_TPU,
          execution_target.JAX_GPU,
          execution_target.JAX_TPU,
          execution_target.Placement.DISAGGREGATED,
          common_datatypes.WeightSyncMode.FS,
      ),
      (
          "in_memory_homogeneous_gpu",
          execution_target.JAX_GPU,
          execution_target.JAX_GPU,
          execution_target.JAX_GPU,
          execution_target.Placement.COLOCATED,
          common_datatypes.WeightSyncMode.IN_MEMORY,
      ),
      (
          "nccl_jax_gpu",
          execution_target.JAX_GPU,
          execution_target.JAX_GPU,
          execution_target.JAX_TPU,
          execution_target.Placement.DISAGGREGATED,
          common_datatypes.WeightSyncMode.NCCL,
      ),
      (
          "raiden_jax_tpu",
          execution_target.JAX_TPU,
          execution_target.JAX_TPU,
          execution_target.JAX_GPU,
          execution_target.Placement.DISAGGREGATED,
          common_datatypes.WeightSyncMode.RAIDEN,
      ),
  )
  def test_cluster_targets_accepts_valid_topologies(
      self,
      trainer: execution_target.ExecutionTarget,
      rollout: execution_target.ExecutionTarget,
      inference: execution_target.ExecutionTarget,
      placement: execution_target.Placement,
      weight_sync_mode: common_datatypes.WeightSyncMode,
  ):
    cluster = execution_target.ClusterTargets(
        trainer=trainer,
        rollout=rollout,
        inference=inference,
        placement=placement,
        weight_sync_mode=weight_sync_mode,
    )
    self.assertEqual(cluster.trainer, trainer)
    self.assertEqual(cluster.rollout, rollout)
    self.assertEqual(cluster.inference, inference)
    self.assertEqual(cluster.placement, placement)
    self.assertEqual(cluster.weight_sync_mode, weight_sync_mode)

  @parameterized.named_parameters(
      (
          "unsupported_torch_gpu_trainer",
          dict(trainer=execution_target.TORCH_GPU),
          "Unsupported execution target 'torch:gpu' for role 'trainer'",
      ),
      (
          "unsupported_torch_gpu_rollout",
          dict(rollout=execution_target.TORCH_GPU),
          "Unsupported execution target 'torch:gpu' for role 'rollout'",
      ),
      (
          "unsupported_torch_gpu_inference",
          dict(inference=execution_target.TORCH_GPU),
          "Unsupported execution target 'torch:gpu' for role 'inference'",
      ),
      (
          "colocated_mixed_accelerators",
          dict(
              trainer=execution_target.JAX_TPU,
              rollout=execution_target.JAX_GPU,
              placement=execution_target.Placement.COLOCATED,
          ),
          "Placement 'colocated' requires identical execution targets",
      ),
      (
          "colocated_mixed_frameworks",
          dict(
              trainer=execution_target.TORCH_TPU,
              rollout=execution_target.JAX_TPU,
              placement=execution_target.Placement.COLOCATED,
          ),
          "Placement 'colocated' requires identical execution targets",
      ),
      (
          "colocated_mismatched_inference_target",
          dict(
              trainer=execution_target.JAX_GPU,
              rollout=execution_target.JAX_GPU,
              inference=execution_target.TORCH_TPU,
              placement=execution_target.Placement.COLOCATED,
          ),
          "Placement 'colocated' requires identical execution targets",
      ),
      (
          "in_memory_heterogeneous_frameworks",
          dict(
              trainer=execution_target.TORCH_TPU,
              rollout=execution_target.JAX_TPU,
              weight_sync_mode=common_datatypes.WeightSyncMode.IN_MEMORY,
          ),
          "weight_sync_mode='in_memory' requires homogeneous trainer and"
          " rollout execution targets",
      ),
      (
          "raiden_non_jax_tpu_trainer",
          dict(
              trainer=execution_target.TORCH_TPU,
              rollout=execution_target.JAX_TPU,
              weight_sync_mode=common_datatypes.WeightSyncMode.RAIDEN,
          ),
          "weight_sync_mode='raiden' requires both trainer and rollout"
          " execution targets to be 'jax:tpu'",
      ),
      (
          "raiden_non_jax_tpu_rollout",
          dict(
              trainer=execution_target.JAX_TPU,
              rollout=execution_target.JAX_GPU,
              weight_sync_mode=common_datatypes.WeightSyncMode.RAIDEN,
          ),
          "weight_sync_mode='raiden' requires both trainer and rollout"
          " execution targets to be 'jax:tpu'",
      ),
      (
          "nccl_non_gpu_trainer",
          dict(
              trainer=execution_target.JAX_TPU,
              rollout=execution_target.JAX_GPU,
              weight_sync_mode=common_datatypes.WeightSyncMode.NCCL,
          ),
          "weight_sync_mode='nccl' requires GPU accelerators on both trainer"
          " and rollout",
      ),
  )
  def test_cluster_targets_rejects_invalid_topologies(
      self, kwargs: dict[str, object], expected_msg: str
  ):
    with self.assertRaisesRegex(ValueError, expected_msg):
      execution_target.ClusterTargets(**kwargs)  # pytype: disable=wrong-arg-types


if __name__ == "__main__":
  absltest.main()
