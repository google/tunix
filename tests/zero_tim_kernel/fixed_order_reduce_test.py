# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for `fixed_order_reduce` (4 CPU devices)."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.zero_tim_kernel import fixed_order_reduce
from tunix.zero_tim_kernel import test_utils

_TP = 4
_MODES = fixed_order_reduce.MODES
_ENVS = (
    fixed_order_reduce.GATHER_ENV,
    fixed_order_reduce.SCATTER_ENV,
)


def setUpModule():
  test_utils.configure_cpu(num_devices=_TP)


def _model_mesh():
  return test_utils.cpu_mesh((_TP,), ("model",))


def _sequential_sum(partials) -> np.ndarray:
  """`((p0 + p1) + p2) + p3` with one rounding per addition (numpy)."""
  partials = np.asarray(partials)
  acc = partials[0]
  for part in partials[1:]:
    acc = acc + part
  return acc


def _per_rank_sums(partials, mode, add=None):
  """Every rank's `fixed_order_tp_sum` of `partials[rank]`, stacked."""

  def body(local):
    return fixed_order_reduce.fixed_order_tp_sum(
        local[0], "model", _TP, mode=mode, add=add
    )[None]

  mapped = jax.shard_map(
      body,
      mesh=_model_mesh(),
      in_specs=P("model"),
      out_specs=P("model"),
      check_vma=False,
  )
  return jax.jit(mapped)(partials)


class _EnvTestCase(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    self.addCleanup(patcher.stop)
    for name in _ENVS:
      os.environ.pop(name, None)


class FixedOrderSumTest(_EnvTestCase):

  @parameterized.product(mode=_MODES, rows=(8, 6))
  def test_every_rank_gets_the_rank_ordered_f32_sum(self, mode, rows):
    # rows=6 is not divisible by TP: the scatter form falls back to gather.
    rng = np.random.default_rng(rows)
    magnitude = 10.0 ** rng.integers(-4, 5, size=(_TP, rows, 16))
    partials = (rng.standard_normal((_TP, rows, 16)) * magnitude).astype(
        np.float32
    )
    expected = _sequential_sum(partials)
    sums = np.asarray(_per_rank_sums(jnp.asarray(partials), mode))
    for rank in range(_TP):
      test_utils.assert_bitwise_equal(sums[rank], expected)

  @parameterized.parameters(*_MODES)
  def test_association_order_is_fixed(self, mode):
    # ((1e8 + 1) - 1e8) + 1 == 1 in f32, while (1e8 - 1e8) + (1 + 1) == 2:
    # the result reveals the association order.
    values = np.asarray([1e8, 1.0, -1e8, 1.0], np.float32)
    partials = np.broadcast_to(values[:, None, None], (_TP, 8, 128))
    sums = np.asarray(_per_rank_sums(jnp.asarray(partials), mode))
    np.testing.assert_array_equal(sums, np.ones_like(sums))

  @parameterized.parameters(*_MODES)
  def test_bf16_partials_round_after_every_addition(self, mode):
    rng = np.random.default_rng(3)
    partials = test_utils.random_bf16(rng, (_TP, 8, 64), scale=4.0)
    expected = _sequential_sum(partials)
    sums = np.asarray(_per_rank_sums(partials, mode))
    for rank in range(_TP):
      test_utils.assert_bitwise_equal(sums[rank], expected)

  @parameterized.parameters(*_MODES)
  def test_add_hook_is_applied_once_per_rank_step_in_rank_order(self, mode):
    calls = []

    def add(acc, operand):
      calls.append(None)
      return acc + operand

    values = np.asarray([1e8, 1.0, -1e8, 1.0], np.float32)
    partials = np.broadcast_to(values[:, None, None], (_TP, 8, 128))
    sums = np.asarray(_per_rank_sums(jnp.asarray(partials), mode, add=add))
    self.assertLen(calls, _TP - 1)
    np.testing.assert_array_equal(sums, np.ones_like(sums))

  @parameterized.parameters(*_MODES)
  def test_reduce_precision_add_hook_matches_the_bf16_chain(self, mode):
    def hard_add(acc, operand):
      total = acc.astype(jnp.float32) + operand.astype(jnp.float32)
      rounded = jax.lax.reduce_precision(
          total, exponent_bits=8, mantissa_bits=7
      )
      return rounded.astype(jnp.bfloat16)

    rng = np.random.default_rng(5)
    partials = test_utils.random_bf16(rng, (_TP, 8, 64), scale=4.0)
    expected = np.asarray(_per_rank_sums(partials, mode))
    sums = np.asarray(_per_rank_sums(partials, mode, add=hard_add))
    self.assertEqual(sums.dtype, expected.dtype)
    test_utils.assert_bitwise_equal(sums, expected)

  @parameterized.parameters(*_MODES)
  def test_default_add_is_the_pinned_bf16_chain(self, mode):
    rng = np.random.default_rng(6)
    partials = test_utils.random_bf16(rng, (_TP, 8, 64), scale=4.0)
    default = np.asarray(_per_rank_sums(partials, mode))
    plain = np.asarray(
        _per_rank_sums(partials, mode, add=fixed_order_reduce.plain_add)
    )
    pinned = np.asarray(
        _per_rank_sums(partials, mode, add=fixed_order_reduce.hard_bf16_add)
    )
    self.assertEqual(default.dtype, np.dtype(jnp.bfloat16))
    test_utils.assert_bitwise_equal(default, plain)
    test_utils.assert_bitwise_equal(default, pinned)
    expected = _sequential_sum(partials)
    for rank in range(_TP):
      test_utils.assert_bitwise_equal(default[rank], expected)

  @parameterized.parameters(*_MODES)
  def test_default_add_leaves_f32_chains_plain(self, mode):
    rng = np.random.default_rng(7)
    partials = (rng.standard_normal((_TP, 8, 16)) * 1e3).astype(np.float32)
    default = np.asarray(_per_rank_sums(jnp.asarray(partials), mode))
    plain = np.asarray(
        _per_rank_sums(
            jnp.asarray(partials), mode, add=fixed_order_reduce.plain_add
        )
    )
    self.assertEqual(default.dtype, np.float32)
    test_utils.assert_bitwise_equal(default, plain)
    expected = _sequential_sum(partials)
    for rank in range(_TP):
      test_utils.assert_bitwise_equal(default[rank], expected)

  def test_hard_bf16_add_matches_the_strict_bf16_add(self):
    rng = np.random.default_rng(8)
    a = jnp.asarray(test_utils.random_bf16(rng, (8, 64), scale=4.0))
    b = jnp.asarray(test_utils.random_bf16(rng, (8, 64), scale=4.0))
    pinned = jax.jit(fixed_order_reduce.hard_bf16_add)(a, b)
    self.assertEqual(pinned.dtype, jnp.bfloat16)
    test_utils.assert_bitwise_equal(np.asarray(pinned), np.asarray(a + b))
    a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
    plain32 = jax.jit(fixed_order_reduce.hard_bf16_add)(a32, b32)
    self.assertEqual(plain32.dtype, jnp.float32)
    test_utils.assert_bitwise_equal(np.asarray(plain32), np.asarray(a32 + b32))

  def test_hard_round_and_hard_upcast(self):
    rng = np.random.default_rng(9)
    x_f32 = jnp.asarray(rng.standard_normal((8, 64)), dtype=jnp.float32)
    rounded_bf16 = jax.jit(
        lambda v: fixed_order_reduce.hard_round(v, jnp.bfloat16)
    )(x_f32)
    self.assertEqual(rounded_bf16.dtype, jnp.bfloat16)
    test_utils.assert_bitwise_equal(
        np.asarray(rounded_bf16), np.asarray(x_f32.astype(jnp.bfloat16))
    )
    rounded_f32 = jax.jit(
        lambda v: fixed_order_reduce.hard_round(v, jnp.float32)
    )(x_f32)
    self.assertEqual(rounded_f32.dtype, jnp.float32)
    test_utils.assert_bitwise_equal(np.asarray(rounded_f32), np.asarray(x_f32))
    upcast_from_bf16 = jax.jit(fixed_order_reduce.hard_upcast)(rounded_bf16)
    self.assertEqual(upcast_from_bf16.dtype, jnp.float32)
    test_utils.assert_bitwise_equal(
        np.asarray(upcast_from_bf16),
        np.asarray(rounded_bf16.astype(jnp.float32)),
    )
    upcast_from_f32 = jax.jit(fixed_order_reduce.hard_upcast)(x_f32)
    self.assertEqual(upcast_from_f32.dtype, jnp.float32)
    test_utils.assert_bitwise_equal(
        np.asarray(upcast_from_f32), np.asarray(x_f32)
    )

  def test_unknown_mode_raises(self):
    with self.assertRaisesRegex(ValueError, "unknown fixed-order"):
      fixed_order_reduce.fixed_order_tp_sum(
          jnp.zeros((8, 8)), "model", _TP, mode="tree"
      )

  @parameterized.parameters(*_MODES)
  def test_count_validation_and_singleton_fast_path(self, mode):
    x = jnp.arange(16, dtype=jnp.bfloat16).reshape(4, 4)
    out = fixed_order_reduce.fixed_order_tp_sum(x, "model", 1, mode=mode)
    test_utils.assert_bitwise_equal(np.asarray(out), np.asarray(x))
    for bad_count in (0, -2):
      with self.assertRaisesRegex(ValueError, "count must be positive"):
        fixed_order_reduce.fixed_order_tp_sum(x, "model", bad_count, mode=mode)
    with self.assertRaises(TypeError):
      fixed_order_reduce.fixed_order_tp_sum(  # pyrefly: ignore[missing-argument]
          x, "model", mode=mode
      )


class EnvironmentTest(_EnvTestCase):

  @parameterized.named_parameters(
      ("default", {}, fixed_order_reduce.RING),
      ("gather_zero", {"CANON_FIXED_AR_GATHER": "0"}, fixed_order_reduce.RING),
      ("gather", {"CANON_FIXED_AR_GATHER": "1"}, fixed_order_reduce.GATHER),
      (
          "scatter",
          {"CANON_FIXED_AR_GATHER": "1", "CANON_FIXED_AR_SCATTER": "1"},
          fixed_order_reduce.SCATTER,
      ),
      (
          "scatter_zero",
          {"CANON_FIXED_AR_GATHER": "1", "CANON_FIXED_AR_SCATTER": "0"},
          fixed_order_reduce.GATHER,
      ),
  )
  def test_mode_from_env(self, env, expected):
    os.environ.update(env)
    self.assertEqual(fixed_order_reduce.fixed_ar_mode_from_env(), expected)

  @parameterized.named_parameters(
      ("scatter_without_gather", {"CANON_FIXED_AR_SCATTER": "1"}),
      ("bad_gather", {"CANON_FIXED_AR_GATHER": "2"}),
      (
          "bad_scatter",
          {"CANON_FIXED_AR_GATHER": "1", "CANON_FIXED_AR_SCATTER": "yes"},
      ),
  )
  def test_mode_from_env_rejects(self, env):
    os.environ.update(env)
    with self.assertRaises(RuntimeError):
      fixed_order_reduce.fixed_ar_mode_from_env()


if __name__ == "__main__":
  absltest.main()
