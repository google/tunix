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

"""Phase-0 admission tests for reduce-once checked-VMA reduction."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=4"
  ).strip()

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tunix.rl import dp_training


_COLLECTIVES = frozenset({
    "all_gather",
    "all_to_all",
    "pmax",
    "pmin",
    "ppermute",
    "psum",
    "psum_invariant",
    "psum_scatter",
    "reduce_scatter",
})


def _axis_names(equation):
  axes = equation.params.get(
      "axes", equation.params.get("axis_name", ())
  )
  if not isinstance(axes, tuple):
    axes = (axes,)
  return ",".join(str(axis) for axis in axes)


def _collective_census(function, *arguments):
  counts = {}

  def walk(jaxpr):
    for equation in jaxpr.eqns:
      if equation.primitive.name in _COLLECTIVES:
        key = f"{equation.primitive.name}[{_axis_names(equation)}]"
        counts[key] = counts.get(key, 0) + 1
      for parameter in equation.params.values():
        for nested in (
            parameter if isinstance(parameter, (list, tuple)) else (parameter,)
        ):
          if hasattr(nested, "jaxpr"):
            walk(nested.jaxpr)
          elif hasattr(nested, "eqns"):
            walk(nested)

  walk(jax.make_jaxpr(function)(*arguments).jaxpr)
  return counts


def _mesh(dp_size, tp_size):
  devices = jax.devices()
  if len(devices) < dp_size * tp_size:
    pytest.skip(f"requires {dp_size * tp_size} CPU or accelerator devices")
  return Mesh(
      np.asarray(devices[: dp_size * tp_size]).reshape(dp_size, tp_size),
      ("data", "model"),
  )


def _staged_values(dp_size):
  values = np.asarray([
      [1.0e8, -0.0, 1.25, -3.5, -0.0, 1.0, 2.0, 3.0],
      [-1.0e8, 0.0, 2.5, 7.0, -0.0, 4.0, 5.0, 6.0],
      [2.0, -2.0, -3.75, -1.5, -0.0, -7.0, 8.0, 9.0],
      [3.0, 2.0, 0.0, -2.0, -0.0, 10.0, -11.0, 12.0],
  ], dtype=np.float32)
  return values[:dp_size]


def _mapped_reducer(mesh, dp_size, base_spec, collective, *, check_vma):
  def reduce_local(local_staged):
    return collective(
        jnp.squeeze(local_staged, axis=0),
        dp_size=dp_size,
        axis_name="data",
    )

  return jax.jit(jax.shard_map(
      reduce_local,
      mesh=mesh,
      in_specs=(P("data", *tuple(base_spec)),),
      out_specs=base_spec,
      check_vma=check_vma,
  ))


@pytest.mark.parametrize(
    ("dp_size", "tp_size", "base_spec"),
    ((2, 2, P("model")), (4, 1, P())),
)
def test_vma_collective_is_bitwise_legacy_and_has_no_host_round_trip(
    dp_size, tp_size, base_spec
):
  mesh = _mesh(dp_size, tp_size)
  staged = jax.device_put(
      _staged_values(dp_size),
      NamedSharding(mesh, P("data", *tuple(base_spec))),
  )
  legacy = _mapped_reducer(
      mesh,
      dp_size,
      base_spec,
      dp_training.fixed_dp_collective,
      check_vma=False,
  )
  checked = _mapped_reducer(
      mesh,
      dp_size,
      base_spec,
      dp_training.fixed_dp_vma_collective,
      check_vma=True,
  )
  expected = np.asarray(legacy(staged)).view(np.uint32)
  with jax.transfer_guard("disallow"):
    actual_device = checked(staged)
    actual_device.block_until_ready()
  actual = np.asarray(actual_device).view(np.uint32)
  np.testing.assert_array_equal(actual, expected)
  # The fifth output is the sum of only negative zero inputs.  This catches
  # the tempting float psum publication, which changes its sign bit.
  assert actual[4] == np.uint32(0x80000000)

  stablehlo = str(
      checked.lower(staged).compiler_ir(dialect="stablehlo")
  )
  assert _collective_census(checked, staged) == {
      "ppermute[data]": int(np.log2(dp_size)),
      "pmax[data]": 1,
  }
  assert stablehlo.count("stablehlo.collective_permute") == int(
      np.log2(dp_size)
  )
  assert stablehlo.count("stablehlo.all_reduce") == 1
  assert "stablehlo.maximum" in stablehlo


def test_legacy_collective_is_rejected_by_checked_vma_negative_control():
  mesh = _mesh(2, 2)
  staged = jax.device_put(
      _staged_values(2), NamedSharding(mesh, P("data", "model"))
  )
  legacy_checked = _mapped_reducer(
      mesh,
      2,
      P("model"),
      dp_training.fixed_dp_collective,
      check_vma=True,
  )
  with pytest.raises(ValueError, match="replication"):
    legacy_checked.lower(staged)


def test_reduce_once_selects_checked_bundle_and_off_keeps_legacy_bundle():
  mesh = _mesh(2, 2)
  template = jax.device_put(
      jnp.zeros((8,), jnp.float32), NamedSharding(mesh, P("model"))
  )
  staged = jax.device_put(
      _staged_values(2), NamedSharding(mesh, P("data", "model"))
  )
  with mock.patch.dict(
      os.environ,
      {"CANON_DP_REDUCE_ONCE": "0", "CANON_DP_COLLECTIVE_REDUCE": "0"},
      clear=False,
  ):
    dp_training.reset_reducer_program_cache_for_tests()
    legacy = dp_training.FixedDPRankGradientReducer(
        template, dp_size=2, dp_axis="data"
    )
    legacy_hlo = str(
        legacy._reduce.lower(  # pylint: disable=protected-access
            staged
        ).compiler_ir(dialect="stablehlo")
    )
    expected = np.asarray(
        _mapped_reducer(
            mesh,
            2,
            P("model"),
            dp_training.fixed_dp_collective,
            check_vma=False,
        )(staged)
    ).view(np.uint32)
  # The reducer donates its staged table, so use a fresh physical array for
  # the treatment after materializing the control bits above.
  staged = jax.device_put(
      _staged_values(2), NamedSharding(mesh, P("data", "model"))
  )
  with mock.patch.dict(
      os.environ,
      {"CANON_DP_REDUCE_ONCE": "1", "CANON_DP_COLLECTIVE_REDUCE": "0"},
      clear=False,
  ):
    checked = dp_training.FixedDPRankGradientReducer(
        template, dp_size=2, dp_axis="data"
    )
    actual, report = checked.finalize_staged(staged)
    checked_hlo = str(
        checked._reduce.lower(  # pylint: disable=protected-access
            staged
        ).compiler_ir(dialect="stablehlo")
    )

  assert legacy._check_vma is False  # pylint: disable=protected-access
  assert checked._check_vma is True  # pylint: disable=protected-access
  assert legacy._programs is not checked._programs  # pylint: disable=protected-access
  assert legacy_hlo.count("stablehlo.collective_permute") == 2
  assert "stablehlo.all_reduce" not in legacy_hlo
  legacy_reduce = legacy._reduce  # pylint: disable=protected-access
  assert _collective_census(legacy_reduce, staged) == {
      "ppermute[data]": 2
  }
  checked_staged = jax.device_put(
      _staged_values(2), NamedSharding(mesh, P("data", "model"))
  )
  assert _collective_census(
      checked._reduce, checked_staged  # pylint: disable=protected-access
  ) == {"ppermute[data]": 1, "pmax[data]": 1}
  assert checked_hlo.count("stablehlo.collective_permute") == 1
  assert checked_hlo.count("stablehlo.all_reduce") == 1
  assert report["shard_map_check_vma"] is True
  assert report["reduction_collectives"] == 2
  assert report["rank_local_fingerprint_unique_count"] == 2
  assert report["post_reduction_replicas_exact"] is True
  np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), expected)

  compare_hlo = str(
      checked._compare.lower(  # pylint: disable=protected-access
          actual
      ).compiler_ir(dialect="stablehlo")
  )
  # Replica equality is checked across data, then ANDed across TP so the
  # scalar receipt is honestly replicated on the model axis.
  assert compare_hlo.count("stablehlo.collective_permute") == 1
  assert compare_hlo.count("stablehlo.all_reduce") == 1
  assert "stablehlo.minimum" in compare_hlo


@pytest.mark.parametrize(
    ("mode", "expected_collectives", "required_hlo", "expected_census"),
    (
        (
            "1",
            1,
            "stablehlo.all_reduce",
            {"psum_invariant[data]": 1},
        ),
        (
            "tree",
            2,
            "stablehlo.all_gather",
            {"all_gather[data]": 1, "pmax[data]": 1},
        ),
    ),
)
def test_checked_bundle_covers_existing_collective_selectors(
    mode, expected_collectives, required_hlo, expected_census
):
  mesh = _mesh(4, 1)
  template = jax.device_put(
      jnp.zeros((8,), jnp.float32), NamedSharding(mesh, P())
  )
  staged = jax.device_put(
      _staged_values(4), NamedSharding(mesh, P("data"))
  )
  with mock.patch.dict(
      os.environ,
      {"CANON_DP_REDUCE_ONCE": "1", "CANON_DP_COLLECTIVE_REDUCE": mode},
      clear=False,
  ):
    dp_training.reset_reducer_program_cache_for_tests()
    unchecked = dp_training.FixedDPRankGradientReducer(
        template, dp_size=4, dp_axis="data", check_vma=False
    )
    expected, _ = unchecked.finalize_staged(staged)
    reducer = dp_training.FixedDPRankGradientReducer(
        template, dp_size=4, dp_axis="data"
    )
    checked_staged = jax.device_put(
        _staged_values(4), NamedSharding(mesh, P("data"))
    )
    actual, report = reducer.finalize_staged(checked_staged)
    stablehlo = str(
        reducer._reduce.lower(  # pylint: disable=protected-access
            jax.device_put(
                _staged_values(4), NamedSharding(mesh, P("data"))
            )
        ).compiler_ir(dialect="stablehlo")
    )

  np.testing.assert_array_equal(
      np.asarray(actual).view(np.uint32),
      np.asarray(expected).view(np.uint32),
  )
  assert report["shard_map_check_vma"] is True
  assert report["reduction_collectives"] == expected_collectives
  assert required_hlo in stablehlo
  census_staged = jax.device_put(
      _staged_values(4), NamedSharding(mesh, P("data"))
  )
  assert _collective_census(
      reducer._reduce, census_staged  # pylint: disable=protected-access
  ) == expected_census
