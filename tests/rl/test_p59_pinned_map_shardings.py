"""Phase 9 K7: the mapped pullback programs are pinned to their operands'
spellings instead of relabeling every operand and result per call.

The TP1 fixture never leaves the trainer mesh, so the pinning path is
exercised here on a two-mesh setup built directly on the map builder: a
trainer mesh named ('dp', 'tp') and an engine mesh named ('data', 'model')
over the same sixteen devices with a model axis of two.  Results must be
byte-identical to the relabel path and no device_put may run per call.
"""

import os
import types
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=16"
  ).strip()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter as adapter_module

P = jax.sharding.PartitionSpec


def _meshes():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  devices = np.asarray(jax.devices()[:16]).reshape(8, 2)
  return (
      jax.sharding.Mesh(devices, ("dp", "tp")),
      jax.sharding.Mesh(devices, ("data", "model")),
  )


def _build(trainer, engine):
  data_size = int(trainer.shape["dp"])
  leaves = (
      jax.device_put(jnp.full((3,), 0.5, jnp.float32), jax.sharding.NamedSharding(trainer, P())),
  )
  rows = jax.device_put(
      jnp.arange(2.0 * data_size, dtype=jnp.float32).reshape(data_size, 2),
      jax.sharding.NamedSharding(trainer, P("dp", None)),
  )
  engine_rows = jax.device_put(
      jnp.arange(2.0 * data_size, dtype=jnp.float32).reshape(data_size, 2) * 0.25,
      jax.sharding.NamedSharding(engine, P("data", None)),
  )

  def local_fn(local_leaves, local_rows, local_engine_rows):
    grad = jax.tree.map(lambda leaf: leaf[None] * 2.0, local_leaves)
    out = local_rows * local_leaves[0][0] + local_engine_rows
    return grad, out

  fake_self = types.SimpleNamespace(_engine_mesh=engine)
  build = adapter_module._P28SegmentedEngineForward._p59_parallel_map  # pylint: disable=protected-access
  invoke = build(
      fake_self,
      local_fn,
      (leaves, rows, engine_rows),
      lambda data_axis, axis_size, aligned, manual_axes: (
          adapter_module._rank_staged_specs(aligned[0], data_axis, manual_axes),  # pylint: disable=protected-access
          adapter_module._rank_local_leading_specs(  # pylint: disable=protected-access
              aligned[1], data_axis, axis_size, "test output", manual_axes
          ),
      ),
      rank_local_arg_indices=(1, 2),
      module_name="zt_tr_test_pinned_map",
      scope_name="zt/tr/test/pinned",
  )
  return invoke, (leaves, rows, engine_rows)


def test_two_mesh_map_is_pinned_bitwise_and_copies_nothing_per_call():
  trainer, engine = _meshes()
  with mock.patch.dict(os.environ, {"CANON_P66_P59_CHECK_VMA": "0"}, clear=False):
    invoke, args = _build(trainer, engine)
    assert invoke._p59_pinned is True  # pylint: disable=protected-access
    # Reference: the relabel path -- operands onto the engine mesh, the
    # bare program, results back onto the trainer mesh.
    relabel = adapter_module._p59_align_to_mesh  # pylint: disable=protected-access
    calls = []

    def counting(tree, mesh, label):
      calls.append(label)
      return relabel(tree, mesh, label)

    with mock.patch.object(adapter_module, "_p59_align_to_mesh", counting):
      grad, out = invoke(*args)
      jax.block_until_ready((grad, out))
    assert calls == [], calls
    assert out.sharding.mesh == trainer and out.sharding.spec == P("dp", None)
    assert grad[0].sharding.mesh == trainer and out.committed
    leaves, rows, engine_rows = args
    want_out = np.asarray(rows) * 0.5 + np.asarray(engine_rows)
    want_grad = np.full((int(trainer.shape["dp"]), 3), 1.0, np.float32)
    assert np.asarray(out).tobytes() == want_out.astype(np.float32).tobytes()
    assert np.asarray(grad[0]).tobytes() == want_grad.tobytes()
    # A caller that hands an operand in the other spelling is still served.
    engine_spelled_rows = jax.device_put(rows, jax.sharding.NamedSharding(engine, P("data", None)))
    grad2, out2 = invoke(leaves, engine_spelled_rows, engine_rows)
    assert np.asarray(out2).tobytes() == np.asarray(out).tobytes()
    assert out2.sharding.mesh == trainer


def test_single_mesh_map_is_not_pinned():
  """At TP1 the map runs on the trainer mesh itself: nothing to pin."""
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  devices = np.asarray(jax.devices()[:16]).reshape(16, 1)
  trainer = jax.sharding.Mesh(devices, ("dp", "tp"))
  engine = jax.sharding.Mesh(devices, ("data", "model"))
  with mock.patch.dict(os.environ, {"CANON_P66_P59_CHECK_VMA": "0"}, clear=False):
    invoke, args = _build(trainer, engine)
  assert invoke._p59_pinned is False  # pylint: disable=protected-access
  grad, out = invoke(*args)
  assert out.sharding.mesh == trainer
