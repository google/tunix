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

"""Tests for fixed-placement replicated data-parallel contracts."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tunix.rl import dp_training


class DPTrainingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.contract = dp_training.DPTrainingContract(
        dp_size=2,
        tp_size=2,
        global_prompts=4,
        num_generations=8,
        local_trajectories=16,
    )

  def test_contract_partition_is_prompt_major(self):
    self.contract.validate()
    np.testing.assert_array_equal(
        self.contract.trajectory_ranks(),
        np.asarray([0] * 16 + [1] * 16, dtype=np.int32),
    )
    self.assertLen(self.contract.rank_indices()[0], 16)
    self.assertLen(self.contract.rank_indices()[1], 16)

  def test_contract_rejects_partial_prompt_group(self):
    with self.assertRaisesRegex(ValueError, "complete prompt groups"):
      dp_training.DPTrainingContract(
          dp_size=2,
          tp_size=2,
          global_prompts=4,
          num_generations=8,
          local_trajectories=15,
      ).validate()

  def test_group_validation_rejects_split_group(self):
    valid = np.repeat(np.arange(4), 8)
    self.contract.validate_prompt_groups(valid)
    split = valid.copy()
    split[15], split[16] = split[16], split[15]
    with self.assertRaisesRegex(ValueError, "split across DP ranks"):
      self.contract.validate_prompt_groups(split)

  def test_partition_inventory_rejects_dp_parameter_shard(self):
    summary = dp_training.validate_dp_replicated_partition_specs(
        {"left": P(None, "tp"), "right": P("tp", None)}, label="params"
    )
    self.assertEqual(summary, {"leaves": 2, "dp_partitioned_leaves": 0})
    with self.assertRaisesRegex(ValueError, "not replicated"):
      dp_training.validate_dp_replicated_partition_specs(
          {"bad": P("dp", "tp")}, label="params"
      )

  def test_fixed_sum_uses_rank_zero_then_rank_one(self):
    rank0 = {"g": jnp.asarray([1.0e8, 1.0], jnp.float32)}
    rank1 = {"g": jnp.asarray([-1.0e8, 2.0], jnp.float32)}
    result = dp_training.fixed_dp2_sum(rank0, rank1)
    np.testing.assert_array_equal(np.asarray(result["g"]), [0.0, 3.0])

  def test_rank_major_reverse_groups_pair_local_ordinals(self):
    groups = self.contract.rank_major_reverse_groups()
    self.assertLen(groups, 16)
    self.assertEqual(groups[0], (0, 16))
    self.assertEqual(groups[-1], (15, 31))
    self.assertEqual(
        sorted(index for group in groups for index in group), list(range(32))
    )

  def test_rank_isolation_reconstructs_group_cotangent(self):
    cotangent = jnp.asarray(
        [[1.0, -2.0, 4.0], [8.0, 16.0, -32.0]], jnp.float32
    )
    rank0 = dp_training.isolate_dp_rank_cotangent(cotangent, rank=0)
    rank1 = dp_training.isolate_dp_rank_cotangent(cotangent, rank=1)
    np.testing.assert_array_equal(rank0 + rank1, cotangent)
    np.testing.assert_array_equal(rank0[1], jnp.zeros((3,), jnp.float32))
    np.testing.assert_array_equal(rank1[0], jnp.zeros((3,), jnp.float32))

  def test_rank_isolation_rejects_wrong_leading_axis(self):
    with self.assertRaisesRegex(ValueError, "leading rank axis"):
      dp_training.isolate_dp_rank_cotangent(
          jnp.ones((3, 4), jnp.float32), rank=0
      )

  def test_fixed_collective_replicates_rank_ordered_sum(self):
    if len(jax.devices()) < 4:
      self.skipTest("requires four visible CPU or accelerator devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=P("dp", "tp"),
        out_specs=P("tp"),
        check_vma=False,
    )
    def reduce_rows(value):
      return dp_training.fixed_dp2_collective(value[0])

    values = jnp.asarray(
        [[1.0e8, 1.0, 3.0, 5.0], [-1.0e8, 2.0, 4.0, 6.0]],
        jnp.float32,
    )
    result = reduce_rows(values)
    np.testing.assert_array_equal(
        np.asarray(result), np.asarray([0.0, 3.0, 7.0, 11.0])
    )
    self.assertTrue(result.sharding.is_fully_replicated is False)

  def test_cleanup_finalizer_supports_server_and_non_server_engines(self):
    class FakeFinalizer:

      def __init__(self):
        self.alive = True

      def detach(self):
        self.alive = False
        return object()

    class JaxModel:
      pass

    class Engine:

      def __init__(self):
        self._finalizer = FakeFinalizer()
        self._model = JaxModel()

      def _get_driver_model_for_cleanup(self):
        return self._model

    for sampler in (
        type("Sampler", (), {"llm": type("LLM", (), {"llm_engine": Engine()})(),
                             "_driver": None})(),
        type("Sampler", (), {"llm": None,
                             "_driver": type("Driver", (), {
                                 "llm_engine": Engine()})()})(),
    ):
      rollout = type("Rollout", (), {"_sampler": sampler})()
      self.assertEqual(
          dp_training.detach_jax_vllm_cleanup_finalizer(rollout),
          {"jax_vllm_finalizer_detached": True},
      )

  def test_cleanup_finalizer_rejects_torch_or_missing_contract(self):
    class TorchModel:

      def modules(self):
        return ()

    class Finalizer:
      alive = True

      def detach(self):
        return object()

    class Engine:
      _finalizer = Finalizer()

      def _get_driver_model_for_cleanup(self):
        return TorchModel()

    torch_sampler = type(
        "Sampler", (), {"llm": type("LLM", (), {"llm_engine": Engine()})(),
                         "_driver": None}
    )()
    with self.assertRaisesRegex(ValueError, "known JAX vLLM cleanup"):
      dp_training.detach_jax_vllm_cleanup_finalizer(
          type("Rollout", (), {"_sampler": torch_sampler})()
      )
    with self.assertRaisesRegex(ValueError, "known JAX vLLM cleanup"):
      dp_training.detach_jax_vllm_cleanup_finalizer(object())


if __name__ == "__main__":
  absltest.main()
