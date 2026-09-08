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

"""Capacity and numerical gates for fixed-leaf DP reduction buckets."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=64"
  ).strip()

from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tunix.models.qwen3 import model as qwen3_model
from tunix.rl import dp_training


def _mesh():
  return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("data", "model"))


def _template(mesh):
  sharding = NamedSharding(mesh, P())
  return {
      "a": jax.device_put(jnp.zeros((4,), jnp.float32), sharding),
      "b": jax.device_put(jnp.zeros((4,), jnp.float32), sharding),
      "c": jax.device_put(jnp.zeros((4,), jnp.float32), sharding),
  }


def _spec_tree(tree):
  return jax.tree.map(
      lambda leaf: jax.ShapeDtypeStruct(
          leaf.shape, leaf.dtype, sharding=leaf.sharding
      ),
      tree,
  )


def _staged(mesh):
  sharding = NamedSharding(mesh, P("data"))
  rank0 = {
      "a": np.asarray([1.0e8, -0.0, 1.25, -3.5], np.float32),
      "b": np.asarray([3.0, 5.0, -7.0, -0.0], np.float32),
      "c": np.asarray([0.5, -2.0, 8.0, 9.0], np.float32),
  }
  rank1 = {
      "a": np.asarray([-1.0e8, 0.0, 2.5, 7.0], np.float32),
      "b": np.asarray([4.0, -5.0, 11.0, -0.0], np.float32),
      "c": np.asarray([1.5, 2.0, -8.0, 12.0], np.float32),
  }
  return {
      key: jax.device_put(
          jnp.asarray(np.stack((rank0[key], rank1[key]))), sharding
      )
      for key in rank0
  }


def _programs(spec, max_local_bytes):
  return dp_training._build_reducer_programs(  # pylint: disable=protected-access
      spec,
      dp_size=2,
      dp_axis="data",
      reduce_mode="",
      check_vma=True,
      compare_mode="full",
      distinct_schedule="every-group",
      max_local_bytes=max_local_bytes,
  )


def _tree_bits(tree):
  return tuple(
      np.asarray(leaf).view(np.uint32).tobytes()
      for leaf in jax.tree.leaves(tree)
  )


def test_shape_only_bucket_schedule_is_stable_and_bounded():
  mesh = _mesh()
  one_gib_float_count = (1024**3) // np.dtype(np.float32).itemsize
  sharding = NamedSharding(mesh, P())
  spec = tuple(
      jax.ShapeDtypeStruct(
          (one_gib_float_count,), jnp.float32, sharding=sharding
      )
      for _ in range(5)
  )
  indices, local_bytes = dp_training._reducer_leaf_buckets(spec)  # pylint: disable=protected-access
  assert indices == ((0, 1), (2, 3), (4,))
  assert local_bytes == (2 * 1024**3, 2 * 1024**3, 1024**3)
  assert sum(local_bytes) == 5 * 1024**3
  assert max(local_bytes) <= (
      dp_training._REDUCER_MAX_LOCAL_BYTES_PER_PROGRAM  # pylint: disable=protected-access
  )


def test_qwen3_8b_dp2_tp2_abstract_gradient_has_eight_bounded_buckets():
  mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))
  config = qwen3_model.ModelConfig.qwen3_8b()
  config.dtype = jnp.bfloat16
  config.param_dtype = jnp.float32
  config.shd_config = qwen3_model.ShardingConfig.get_data_parallel_sharding()
  abstract_model = nnx.eval_shape(
      lambda: qwen3_model.Qwen3(config, rngs=nnx.Rngs(params=0))
  )
  model_state = nnx.state(abstract_model, nnx.Param)
  named_shardings = nnx.get_named_sharding(model_state, mesh)
  gradient_spec = jax.tree.map(
      lambda value, sharding: jax.ShapeDtypeStruct(
          value.shape, jnp.float32, sharding=sharding
      ),
      model_state,
      named_shardings,
  )
  indices, local_bytes = dp_training._reducer_leaf_buckets(  # pylint: disable=protected-access
      gradient_spec
  )
  assert len(jax.tree.leaves(gradient_spec)) == 399
  assert len(indices) == 8
  assert local_bytes == (
      2_100_399_104,
      2_130_875_392,
      2_114_131_968,
      2_130_875_392,
      2_114_131_968,
      2_130_875_392,
      2_114_131_968,
      1_546_665_984,
  )
  assert sum(local_bytes) == 16_382_087_168


def test_invalid_or_single_oversized_leaf_fails_closed():
  mesh = _mesh()
  sharding = NamedSharding(mesh, P())
  small = jax.ShapeDtypeStruct((4,), jnp.float32, sharding=sharding)
  with pytest.raises(ValueError, match="must be positive"):
    dp_training._reducer_leaf_buckets((small,), max_local_bytes=0)  # pylint: disable=protected-access
  oversized = jax.ShapeDtypeStruct((6,), jnp.float32, sharding=sharding)
  with pytest.raises(ValueError, match="leaf=0 bytes=24 cap=20"):
    dp_training._reducer_leaf_buckets(  # pylint: disable=protected-access
        (oversized,), max_local_bytes=20
    )


def test_bucketed_checked_vma_reduction_is_monolithic_bitwise_without_d2h():
  mesh = _mesh()
  spec = _spec_tree(_template(mesh))
  monolithic = _programs(spec, max_local_bytes=1024)
  bucketed = _programs(spec, max_local_bytes=20)
  assert monolithic.reduction_bucket_local_bytes == (48,)
  assert bucketed.reduction_bucket_local_bytes == (16, 16, 16)

  expected = monolithic.reduce(_staged(mesh))
  expected = jax.block_until_ready(expected)
  bucketed_input = _staged(mesh)
  with jax.transfer_guard("disallow"):
    actual = bucketed.reduce(bucketed_input)
    actual = jax.block_until_ready(actual)
    replica_flags = bucketed.compare(actual)
    replica_flags.block_until_ready()

  assert _tree_bits(actual) == _tree_bits(expected)
  np.testing.assert_array_equal(np.asarray(replica_flags), [True, True])


def test_bucketed_replica_compare_negative_fires():
  mesh = _mesh()
  template = _template(mesh)
  programs = _programs(_spec_tree(template), max_local_bytes=20)
  buffers = []
  for flat_index, device in enumerate(mesh.devices.flat):
    value = np.arange(4, dtype=np.float32)
    if flat_index // 2 == 1:
      value[1] += np.float32(1.0)
    buffers.append(jax.device_put(jnp.asarray(value), device))
  divergent = dict(template)
  divergent["b"] = jax.make_array_from_single_device_arrays(
      (4,), NamedSharding(mesh, P()), buffers
  )
  flags = np.asarray(programs.compare(divergent), dtype=np.bool_)
  assert not np.all(flags)


def test_reducer_report_receipts_bucket_schedule_and_fixed_collectives():
  mesh = _mesh()
  template = _template(mesh)
  monolithic = _programs(_spec_tree(template), max_local_bytes=1024)
  expected = jax.block_until_ready(monolithic.reduce(_staged(mesh)))

  def build_bucketed(source_template, **kwargs):
    kwargs.pop("finite_fetch")
    kwargs.pop("require_distinct_fingerprints")
    return dp_training._build_reducer_programs(  # pylint: disable=protected-access
        _spec_tree(source_template), **kwargs, max_local_bytes=20
    )

  with mock.patch.object(
      dp_training, "_reducer_programs_for", side_effect=build_bucketed
  ):
    reducer = dp_training.FixedDPRankGradientReducer(
        template,
        dp_size=2,
        dp_axis="data",
        require_distinct_fingerprints=False,
        check_vma=True,
    )
  reduced, report = reducer.finalize_staged(_staged(mesh))
  assert reducer.reduction_bucket_count == 3
  assert reducer.reduction_bucket_local_bytes == (16, 16, 16)
  assert report["reduction_bucket_count"] == 3
  assert report["reduction_bucket_total_local_bytes"] == 48
  assert report["reduction_bucket_peak_local_bytes"] == 16
  assert report["shard_map_check_vma"] is True
  assert report["reduction_collectives"] == 2
  assert report["post_reduction_replicas_exact"] is True
  assert _tree_bits(reduced) == _tree_bits(expected)
