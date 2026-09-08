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

"""P77 chunk-level device-ready backpressure admission tests."""

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


def test_wait_is_bitwise_exact_and_transfer_guard_clean():
  devices = np.asarray(jax.devices()[:4], dtype=object).reshape(2, 2)
  mesh = jax.sharding.Mesh(devices, ("dp", "tp"))
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("dp", None, "tp")
  )
  left = jax.device_put(
      jnp.arange(256, dtype=jnp.float32).reshape(2, 16, 8), sharding
  )
  right = jax.device_put(
      (jnp.arange(256, dtype=jnp.float32).reshape(2, 16, 8) - 73) / 11,
      sharding,
  )

  @jax.jit
  def add(a, b):
    return {"gradient": a + b, "receipt": a[:, :1] + b[:, :1]}

  expected = add(left, right)
  jax.block_until_ready(expected)
  expected_bits = [
      np.asarray(jax.device_get(value)).view(np.uint8).copy()
      for value in jax.tree.leaves(expected)
  ]
  completed = add(left, right)
  with jax.transfer_guard("disallow"):
    result = adapter_lib._p77_wait_for_chunk_completion(  # pylint: disable=protected-access
        completed
    )

  assert result is None
  for expected_leaf, observed in zip(
      expected_bits, jax.tree.leaves(completed), strict=True
  ):
    assert np.array_equal(
        np.asarray(jax.device_get(observed)).view(np.uint8), expected_leaf
    )


def test_wait_helper_uses_only_device_ready_api():
  source = textwrap.dedent(inspect.getsource(
      adapter_lib._p77_wait_for_chunk_completion  # pylint: disable=protected-access
  ))
  tree = ast.parse(source)
  calls = [
      ast.unparse(node.func)
      for node in ast.walk(tree)
      if isinstance(node, ast.Call)
  ]
  # One device-readiness call on the first leaf of the tree (K14c).
  assert sorted(calls) == ["jax.block_until_ready", "jax.tree.leaves"]
  for forbidden in ("device_get", "np.asarray", "np.array", ".item("):
    assert forbidden not in source


def test_flag_is_closed_p45_only_and_defaults_off():
  with mock.patch.dict(os.environ, {}, clear=True):
    assert not adapter_lib._p77_chunk_backpressure_enabled()  # pylint: disable=protected-access
  with mock.patch.dict(
      os.environ,
      {
          "CANON_P77_CHUNK_BACKPRESSURE": "1",
          "CANON_P32_WORKLOAD": "frozenlake-p45-onehost-dp2-tp2",
      },
      clear=True,
  ):
    assert adapter_lib._p77_chunk_backpressure_enabled()  # pylint: disable=protected-access
  for value in ("yes", "2", "-1"):
    with mock.patch.dict(
        os.environ, {"CANON_P77_CHUNK_BACKPRESSURE": value}, clear=True
    ):
      with pytest.raises(adapter_lib.FunctionalMappingError):
        adapter_lib._p77_chunk_backpressure_enabled()  # pylint: disable=protected-access
  with mock.patch.dict(
      os.environ,
      {
          "CANON_P77_CHUNK_BACKPRESSURE": "1",
          "CANON_P32_WORKLOAD": "frozenlake-m15-onehost-dp2-tp2",
      },
      clear=True,
  ):
    with pytest.raises(adapter_lib.FunctionalMappingError, match="p45"):
      adapter_lib._p77_chunk_backpressure_enabled()  # pylint: disable=protected-access


def test_wait_calls_bracket_accumulation_and_are_flag_guarded():
  source = textwrap.dedent(inspect.getsource(
      adapter_lib.Qwen3EngineForwardAdapter._p32_reverse_group  # pylint: disable=protected-access
  ))
  tree = ast.parse(source)
  calls = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id == "_p77_wait_for_chunk_completion"
  ]
  assert len(calls) == 2
  guards = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.If)
      and ast.unparse(node.test) == "chunk_backpressure"
  ]
  guarded_calls = {
      call
      for guard in guards
      for call in calls
      if call in set(ast.walk(guard))
  }
  assert guarded_calls == set(calls)
  assert "if chunk_backpressure and not rank_parallel" in source
  assert "if chunk_dependency_ticket and chunk_backpressure" in source
  pack_wait = source.index("_p77_wait_for_chunk_completion(chunk_pack)")
  add_dispatch = source.index(
      "self._p70_grad_tree_start(chunk_pack, donate_pack=True)"
  )
  release = source.index("_p70_release_consumed_grad_pack(chunk_pack)")
  accumulation_wait = source.index(
      "_p77_wait_for_chunk_completion(grad_pack)"
  )
  assert pack_wait < add_dispatch < release < accumulation_wait
  assert "pullback_waits={chunk_backpressure_pullback_waits}" in source
  assert (
      "accumulation_waits={chunk_backpressure_accumulation_waits}" in source
  )
