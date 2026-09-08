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

"""P76 checked device-ticket admission tests."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import ast
import inspect
import textwrap
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter as adapter_lib


def _meshes(dp=2, tp=2):
  devices = np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp)
  return (
      jax.sharding.Mesh(devices, ("dp", "tp")),
      jax.sharding.Mesh(devices, ("data", "model")),
  )


def _put(value, mesh, spec):
  return jax.device_put(
      jnp.asarray(value), jax.sharding.NamedSharding(mesh, spec)
  )


def _completed_tree(mesh, *, offset=0):
  return (
      _put(
          np.arange(2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)
          + offset,
          mesh,
          jax.sharding.PartitionSpec("dp", "tp", None),
      ),
      {
          "bf16": _put(
              np.arange(2 * 8, dtype=np.float32).reshape(2, 8) + offset,
              mesh,
              jax.sharding.PartitionSpec("dp", None),
          ).astype(jnp.bfloat16),
          "f32": _put(
              np.arange(2 * 8, dtype=np.float32).reshape(2, 8) - offset,
              mesh,
              jax.sharding.PartitionSpec("dp", "tp"),
          ),
      },
  )


def _bits(value):
  return np.asarray(value).view(np.uint8).copy()


def _bare_adapter():
  return object.__new__(adapter_lib.Qwen3EngineForwardAdapter)


@pytest.mark.parametrize("starter_dtype", [jnp.int32, jnp.float32, jnp.bfloat16])
def test_ticket_is_bitwise_and_sharding_equal_without_host_transfer(
    starter_dtype,
):
  trainer_mesh, engine_mesh = _meshes()
  adapter = _bare_adapter()
  if starter_dtype == jnp.int32:
    host_starter = np.arange(512, dtype=np.int32)
  else:
    host_starter = np.linspace(-2.0, 2.0, 512, dtype=np.float32)
    host_starter[0] = -0.0
    host_starter[1] = np.nan
  starter_spec = jax.sharding.PartitionSpec("data")

  # Warm the exact starter specialization before guarding transfers.
  warm_tree = _completed_tree(trainer_mesh)
  warm_starter = _put(host_starter, engine_mesh, starter_spec).astype(
      starter_dtype
  )
  jax.block_until_ready(
      adapter._p76_order_next_chunk_start(  # pylint: disable=protected-access
          warm_tree, warm_starter
      )
  )

  completed = _completed_tree(trainer_mesh, offset=7)
  starter = _put(host_starter, engine_mesh, starter_spec).astype(starter_dtype)
  expected = _bits(starter)
  with jax.transfer_guard("disallow"):
    ordered = adapter._p76_order_next_chunk_start(  # pylint: disable=protected-access
        completed, starter
    )
    ordered.block_until_ready()

  assert np.array_equal(_bits(ordered), expected)
  assert ordered.sharding.is_equivalent_to(starter.sharding, ordered.ndim)
  assert not any(value.is_deleted() for value in jax.tree.leaves(completed))


def test_ticket_hlo_has_one_checked_scalar_collective_and_no_reshard_family():
  trainer_mesh, engine_mesh = _meshes()
  adapter = _bare_adapter()
  completed = _completed_tree(trainer_mesh)
  starter = _put(
      np.arange(512, dtype=np.int32),
      engine_mesh,
      jax.sharding.PartitionSpec("data"),
  )
  adapter._p76_order_next_chunk_start(  # pylint: disable=protected-access
      completed, starter
  ).block_until_ready()
  program, = adapter._p76_ticket_programs.values()  # pylint: disable=protected-access
  aligned_starter = adapter_lib._p59_align_to_mesh(  # pylint: disable=protected-access
      starter, trainer_mesh, "P76 test"
  )
  compiled = program.lower(completed, aligned_starter).compile()
  hlo = compiled.as_text()
  stats = compiled.memory_analysis()

  assert hlo.count("all-reduce") == 1
  assert "unreduced_psum" in hlo
  assert "bitcast-convert" in hlo
  assert " xor(" in hlo
  assert " slice(" in hlo
  assert hlo.count("parameter(") >= len(jax.tree.leaves(completed)) + 1
  for forbidden in (
      "all-gather",
      "all-to-all",
      "collective-permute",
      "reduce-scatter",
  ):
    assert forbidden not in hlo
  assert stats.host_argument_size_in_bytes == 0
  assert stats.host_output_size_in_bytes == 0
  assert stats.host_temp_size_in_bytes == 0
  assert stats.temp_size_in_bytes <= 128


def test_checked_vma_rejects_a_varying_ticket_without_tp_reduction():
  trainer_mesh, _ = _meshes()
  tree = _completed_tree(trainer_mesh)
  starter = _put(
      np.arange(512, dtype=np.int32),
      trainer_mesh,
      jax.sharding.PartitionSpec("dp"),
  )

  def bad_local(local_tree, local_starter):
    scalar = jnp.ravel(jax.tree.leaves(local_tree)[0])[0]
    bits = jax.lax.bitcast_convert_type(scalar, jnp.uint32)
    zero = jax.lax.bitwise_xor(
        bits, jax.lax.optimization_barrier(bits)
    )
    return jax.lax.bitwise_xor(
        local_starter, jax.lax.convert_element_type(zero, jnp.int32)
    )

  bad = jax.shard_map(
      bad_local,
      mesh=trainer_mesh,
      in_specs=(
          jax.tree.map(lambda value: value.sharding.spec, tree),
          starter.sharding.spec,
      ),
      out_specs=starter.sharding.spec,
      axis_names=frozenset(trainer_mesh.axis_names),
      check_vma=True,
  )
  with pytest.raises(ValueError, match="require replication"):
    jax.jit(bad).lower(tree, starter)


def test_flag_is_closed_and_defaults_off():
  with mock.patch.dict(os.environ, {}, clear=True):
    assert not adapter_lib._p76_chunk_dependency_ticket_enabled()  # pylint: disable=protected-access
  with mock.patch.dict(
      os.environ,
      {
          "CANON_P76_CHUNK_DEPENDENCY_TICKET": "1",
          "CANON_P32_WORKLOAD": "frozenlake-p45-onehost-dp2-tp2",
      },
      clear=True,
  ):
    assert adapter_lib._p76_chunk_dependency_ticket_enabled()  # pylint: disable=protected-access
  for value in ("yes", "2", "-1"):
    with mock.patch.dict(
        os.environ,
        {"CANON_P76_CHUNK_DEPENDENCY_TICKET": value},
        clear=True,
    ):
      with pytest.raises(adapter_lib.FunctionalMappingError):
        adapter_lib._p76_chunk_dependency_ticket_enabled()  # pylint: disable=protected-access
  with mock.patch.dict(
      os.environ,
      {
          "CANON_P76_CHUNK_DEPENDENCY_TICKET": "1",
          "CANON_P32_WORKLOAD": "frozenlake-m15-onehost-dp2-tp2",
      },
      clear=True,
  ):
    with pytest.raises(adapter_lib.FunctionalMappingError, match="p45"):
      adapter_lib._p76_chunk_dependency_ticket_enabled()  # pylint: disable=protected-access


def test_ticket_call_is_rank_parallel_nonfinal_and_flag_guarded():
  source = textwrap.dedent(inspect.getsource(
      adapter_lib.Qwen3EngineForwardAdapter._p32_reverse_group  # pylint: disable=protected-access
  ))
  tree = ast.parse(source)
  calls = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "_p76_order_next_chunk_start"
  ]
  assert len(calls) == 2
  dependency_guards = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.If)
      and "chunk_dependency_ticket and chunk_index > 0"
      in ast.unparse(node.test)
  ]
  assert len(dependency_guards) == 1
  guarded = set(ast.walk(dependency_guards[0]))
  assert all(call in guarded for call in calls)
  assert "if chunk_dependency_ticket and not rank_parallel" in source
