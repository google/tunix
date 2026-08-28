"""Regression gates for the P74 checked-VMA dispatch layout tax."""

import hashlib
import inspect
import os
from unittest import mock

import jax
import jax.numpy as jnp
from jax._src import array as jax_array
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter as adapter


P = jax.sharding.PartitionSpec
NamedSharding = jax.sharding.NamedSharding
_CHECKED_HASHES = (
    "3bbc2e2052ee668f77601513394d44b4a0dd934136f1fd2ea243750a1ce37936",
    "64df93759cb7459126106cca6143417401a67f25697c4510a35407c20e054d22",
)
_FLAG_OFF_HASHES = (
    "3bbc2e2052ee668f77601513394d44b4a0dd934136f1fd2ea243750a1ce37936",
    "5bd6b6e2342fd12928dc05ba0df39da0c4df0ae31a8c55848e2a06a3fb1a608e",
)
_FLAG_OFF_PARTITION_SOURCE_HASH = (
    "d8c4fdaa37d8cd685e3b565c80e8e1bd27070c13274402d8ba3f491936e8f535"
)


def _inputs():
  if len(jax.devices()) < 4:
    pytest.skip("requires four forced CPU or accelerator devices")
  devices = np.asarray(jax.devices()[:4])
  trainer_mesh = jax.sharding.Mesh(devices.reshape(2, 2), ("dp", "tp"))
  engine_mesh = jax.sharding.Mesh(
      devices.reshape(2, 1, 1, 1, 2, 1),
      ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp"),
  )
  weight = jax.device_put(
      (jnp.arange(8 * 16, dtype=jnp.float32).reshape(8, 16) / 64).astype(
          jnp.bfloat16
      ),
      NamedSharding(trainer_mesh, P(None, "tp")),
  )
  hidden = jax.device_put(
      (jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8) / 32).astype(
          jnp.bfloat16
      ),
      NamedSharding(trainer_mesh, P("dp", None)),
  )
  cotangent = jax.device_put(
      jnp.ones((16, 16), jnp.bfloat16),
      NamedSharding(trainer_mesh, P("dp", "tp")),
  )
  return engine_mesh, weight, hidden, cotangent


def _build_parallel(checked):
  engine_mesh, weight, hidden, cotangent = _inputs()
  segmented = object.__new__(adapter._P28SegmentedEngineForward)
  segmented._engine_mesh = engine_mesh

  def local_pullback(local_weight, local_hidden, local_cotangent):
    _, pullback = jax.vjp(lambda w, h: h @ w, local_weight, local_hidden)
    dweight, dhidden = pullback(local_cotangent)
    return jnp.expand_dims(dweight, 0), dhidden

  with mock.patch.dict(
      os.environ,
      {"CANON_P66_P59_CHECK_VMA": "1" if checked else "0"},
      clear=False,
  ):
    parallel = segmented._p59_parallel_map(
        local_pullback,
        (weight, hidden, cotangent),
        lambda data_axis, axis_size, aligned, manual_axes: (
            adapter._rank_staged_specs(
                aligned[0], data_axis, manual_axes
            ),
            adapter._rank_local_leading_specs(
                aligned[1],
                data_axis,
                axis_size,
                "P74 head hidden output",
                manual_axes,
            ),
        ),
        rank_local_arg_indices=(1, 2),
        module_name=f"zt_tr_p74_test_{'checked' if checked else 'off'}",
        scope_name=f"zt/tr/p74/test/{'checked' if checked else 'off'}",
    )
  return parallel, weight, hidden, cotangent


def _hashes(tree):
  return tuple(
      hashlib.sha256(np.asarray(leaf).tobytes()).hexdigest()
      for leaf in jax.tree.leaves(tree)
  )


def _ordinary_reference(weight, hidden, cotangent):
  _, pullback = jax.vjp(lambda w, h: h @ w, weight, hidden)
  expected_rows = []
  for rank in range(2):
    row_mask = jnp.arange(cotangent.shape[0], dtype=jnp.int32) // 8 == rank
    isolated = jnp.where(
        row_mask[:, None], cotangent, jnp.zeros_like(cotangent)
    )
    expected_rows.append(pullback(isolated)[0])
  _, expected_dhidden = pullback(cotangent)
  return jnp.stack(expected_rows), expected_dhidden


def _reject_jax_array_asarray(original):
  def guarded(value, *args, **kwargs):
    if isinstance(value, jax.Array):
      raise AssertionError("host materialization of jax.Array is forbidden")
    return original(value, *args, **kwargs)

  return guarded


def test_checked_vma_real_function_is_bitwise_and_has_no_host_roundtrip(capsys):
  parallel, weight, hidden, cotangent = _build_parallel(True)
  original_asarray = np.asarray
  with mock.patch.object(
      np, "asarray", side_effect=_reject_jax_array_asarray(original_asarray)
  ), jax.transfer_guard("disallow"):
    actual = parallel(weight, hidden, cotangent)
    jax.block_until_ready(actual)

  expected = _ordinary_reference(weight, hidden, cotangent)
  for actual_leaf, expected_leaf in zip(
      jax.tree.leaves(actual), jax.tree.leaves(expected)
  ):
    np.testing.assert_array_equal(
        np.asarray(actual_leaf), np.asarray(expected_leaf)
    )
  # Frozen from the real pre-P74 function at source HEAD 6b10dc9d in the
  # pinned image. This makes the test a before/after bitwise parity gate.
  assert _hashes(actual) == _CHECKED_HASHES
  assert capsys.readouterr().out.count("[P66.VMA] outer_check_enabled") == 1


def test_flag_off_real_function_and_builder_are_frozen(capsys):
  parallel, weight, hidden, cotangent = _build_parallel(False)
  actual = parallel(weight, hidden, cotangent)
  jax.block_until_ready(actual)
  assert _hashes(actual) == _FLAG_OFF_HASHES
  assert capsys.readouterr().out.count("[P66.VMA] outer_check_enabled") == 0


def _dp_replicated_cotangent(mesh):
  return jax.device_put(
      jnp.ones((16, 16), jnp.bfloat16),
      NamedSharding(mesh, P(None, "tp")),
  )


def test_checked_head_cotangent_partitions_on_device_and_caches_executable():
  engine_mesh, weight, _, _ = _inputs()
  del engine_mesh
  trainer_mesh = weight.sharding.mesh
  cotangent = _dp_replicated_cotangent(trainer_mesh)
  target = NamedSharding(trainer_mesh, P("dp", "tp"))
  adapter._p74_head_cotangent_partitioner.cache_clear()
  original_asarray = np.asarray
  with mock.patch.dict(
      os.environ, {"CANON_P66_P59_CHECK_VMA": "1"}, clear=False
  ), mock.patch.object(
      np, "asarray", side_effect=_reject_jax_array_asarray(original_asarray)
  ), mock.patch.object(
      jax_array,
      "shard_sharded_device_array_slow_path",
      side_effect=AssertionError("sharded-array host fallback is forbidden"),
  ), jax.transfer_guard("disallow"):
    partitioned = adapter._p74_partition_head_cotangent(
        cotangent, trainer_mesh, "P74 checked cotangent"
    )
    again = adapter._p74_partition_head_cotangent(
        cotangent, trainer_mesh, "P74 checked cotangent"
    )
    jax.block_until_ready((partitioned, again))

  assert partitioned.sharding == target
  assert again.sharding == target
  assert partitioned.format.sharding == target
  np.testing.assert_array_equal(
      np.asarray(partitioned), np.asarray(cotangent)
  )
  cache = adapter._p74_head_cotangent_partitioner.cache_info()
  assert cache.misses == 1
  assert cache.hits == 1
  partition = adapter._p74_head_cotangent_partitioner(
      cotangent.format, target
  )
  executable = partition.lower(cotangent).compile()
  input_formats, input_kwargs = executable.input_formats
  assert input_kwargs == {}
  assert input_formats == (cotangent.format,)
  assert executable.output_formats.sharding == target


def test_flag_off_head_partition_and_builder_are_frozen():
  engine_mesh, weight, _, _ = _inputs()
  del engine_mesh
  trainer_mesh = weight.sharding.mesh
  cotangent = _dp_replicated_cotangent(trainer_mesh)
  with mock.patch.dict(
      os.environ, {"CANON_P66_P59_CHECK_VMA": "0"}, clear=False
  ), mock.patch.object(
      adapter,
      "_p74_partition_head_cotangent",
      side_effect=AssertionError("flag-off route entered P74 bridge"),
  ):
    partitioned = adapter._p59_partition_head_cotangent(
        cotangent, trainer_mesh, "P74 flag-off cotangent"
    )
    jax.block_until_ready(partitioned)
  assert partitioned.sharding == NamedSharding(
      trainer_mesh, P("dp", "tp")
  )
  source = inspect.getsource(adapter._p59_partition_head_cotangent)
  assert hashlib.sha256(source.encode()).hexdigest() == (
      _FLAG_OFF_PARTITION_SOURCE_HASH
  )

  captured = []
  original_xprof_jit = adapter._xprof_jit

  def capture_xprof_jit(fun, *, module_name, scope_name, **kwargs):
    captured.append(kwargs)
    return original_xprof_jit(
        fun, module_name=module_name, scope_name=scope_name, **kwargs
    )

  with mock.patch.object(adapter, "_xprof_jit", side_effect=capture_xprof_jit):
    _build_parallel(False)
  assert captured == [{}]


def test_direct_head_partition_negative_control_enters_slow_path():
  engine_mesh, weight, _, _ = _inputs()
  del engine_mesh
  trainer_mesh = weight.sharding.mesh
  cotangent = _dp_replicated_cotangent(trainer_mesh)
  target = NamedSharding(trainer_mesh, P("dp", "tp"))
  original_slow_path = jax_array.shard_sharded_device_array_slow_path
  with mock.patch.object(
      jax_array,
      "shard_sharded_device_array_slow_path",
      wraps=original_slow_path,
  ) as slow_path:
    old_route = jax.device_put(cotangent, target)
    jax.block_until_ready(old_route)
  assert slow_path.call_count == 1


def test_host_materialization_guard_negative_control_fires():
  original_asarray = np.asarray
  victim = jax.jit(lambda: jnp.arange(8, dtype=jnp.float32))()
  with mock.patch.object(
      np, "asarray", side_effect=_reject_jax_array_asarray(original_asarray)
  ), pytest.raises(
      AssertionError, match="host materialization of jax.Array is forbidden"
  ):
    np.asarray(victim)
