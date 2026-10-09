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
"""Tests for fixed_order_reduce (4 CPU devices)."""

import functools
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.experimental.zero_tim_kernel import fixed_order_reduce
from tunix.experimental.zero_tim_kernel import pallas_matmul
from tunix.experimental.zero_tim_kernel import test_utils

_TP = 4
_MODES = fixed_order_reduce.MODES
_ENVS = (
    fixed_order_reduce.FIXED_AR_ENV,
    fixed_order_reduce.FIXED_AR_EMBED_ENV,
    fixed_order_reduce.FIXED_AR_GATHER_ENV,
    fixed_order_reduce.FIXED_AR_SCATTER_ENV,
    fixed_order_reduce.P59_RANK_PARALLEL_BACKWARD_ENV,
    fixed_order_reduce.P66_CHECK_VMA_ENV,
    fixed_order_reduce.P67_SCOPE_ENV,
)


def setUpModule():
  test_utils.configure_cpu(num_devices=_TP)


def _model_mesh():
  return test_utils.cpu_mesh((_TP,), ("model",))


def _sequential_sum(partials) -> np.ndarray:
  """``((p0 + p1) + p2) + p3`` with one rounding per addition (numpy)."""
  partials = np.asarray(partials)
  acc = partials[0]
  for part in partials[1:]:
    acc = acc + part
  return acc


def _per_rank_sums(partials, mode, add=None):
  """Every rank's fixed_order_tp_sum of ``partials[rank]``, stacked."""

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


def _trace_in_manual_map(mesh, probe, axis_names=None):
  """Returns ``probe()`` evaluated while tracing a shard_map body."""
  observed = []

  def body(x):
    observed.append(probe())
    return x

  kwargs = {} if axis_names is None else {"axis_names": axis_names}
  mapped = jax.shard_map(
      body, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False, **kwargs
  )
  jax.jit(mapped)(jnp.zeros((4,), jnp.float32))
  return observed[0]


def _dot_local_matmul(a, w):
  return jnp.dot(a, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)


_LOCAL_MATMULS = {
    "dot": _dot_local_matmul,
    # Explicit non-contract tiles are honored as written by the kernel.
    "pallas": functools.partial(
        pallas_matmul.matmul,
        interpret=True,
        block_m=8,
        block_n=32,
        block_k=16,
    ),
}


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
    # Needs --xla_allow_excess_precision=false (root cause R3): with
    # excess precision XLA may keep the chained bf16 sum in f32.
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
    # One trace of the SPMD body: `count - 1` additions, left to right.
    self.assertLen(calls, _TP - 1)
    np.testing.assert_array_equal(sums, np.ones_like(sums))

  @parameterized.parameters(*_MODES)
  def test_reduce_precision_add_hook_matches_the_bf16_chain(self, mode):
    # The caller-side form that pins every step's rounding regardless of
    # --xla_allow_excess_precision: upcast both operands, add in f32, round
    # through `reduce_precision(8, 7)`. Under the strict flag it must be the
    # plain bf16 chain, bit for bit.
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
    # `add=None` is `hard_bf16_add`: under the strict flag (the CPU test
    # regime) it must be the plain bf16 chain bit for bit, and it must equal
    # the caller-side hook the bench passes explicitly.
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

  def test_unknown_mode_raises(self):
    with self.assertRaisesRegex(ValueError, "unknown fixed-order"):
      fixed_order_reduce.fixed_order_tp_sum(
          jnp.zeros((8, 8)), "model", _TP, mode="tree"
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

  def test_fixed_ar_is_scoped_to_contract_parallel_sites(self):
    prefixes = (
        "model.layers.3.self_attn.o_proj",
        "model.layers.3.mlp.down_proj",
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.mlp.gate_proj",
    )
    self.assertFalse(
        any(fixed_order_reduce.fixed_ar_enabled_for(p) for p in prefixes)
    )
    os.environ[fixed_order_reduce.FIXED_AR_ENV] = "1"
    self.assertEqual(
        [fixed_order_reduce.fixed_ar_enabled_for(p) for p in prefixes],
        [True, True, False, False],
    )

  def test_fixed_specs(self):
    self.assertEqual(
        fixed_order_reduce.fixed_specs("mn,np->mp"),
        (P(None, "model"), P("model", None)),
    )
    self.assertEqual(
        fixed_order_reduce.fixed_specs("TNH,NHD->TD", "tp"),
        (P(None, "tp", None), P("tp", None, None)),
    )
    with self.assertRaises(ValueError):
      fixed_order_reduce.fixed_specs("TD,DNH->TNH")


class ContractParallelProjectionTest(_EnvTestCase):

  @parameterized.product(mode=_MODES, local=tuple(_LOCAL_MATMULS))
  def test_mlp_equation_on_exact_operands(self, mode, local):
    # Exact operands: every partial and every partial sum is representable in
    # bf16, so the output must be the exactly rounded product.
    rng = np.random.default_rng(0)
    x = test_utils.exact_bf16(rng, (8, 64))
    w = test_utils.exact_bf16(rng, (64, 32))
    out = fixed_order_reduce.contract_parallel_projection(
        x,
        w,
        equation="mn,np->mp",
        mesh=_model_mesh(),
        tp_axis="model",
        local_matmul=_LOCAL_MATMULS[local],
        mode=mode,
    )
    test_utils.assert_bitwise_equal(
        out, test_utils.exact_matmul_reference(x, w)
    )

  @parameterized.product(mode=_MODES, local=tuple(_LOCAL_MATMULS))
  def test_attention_output_equation_on_exact_operands(self, mode, local):
    rng = np.random.default_rng(1)
    x = test_utils.exact_bf16(rng, (8, _TP, 16))
    w = test_utils.exact_bf16(rng, (_TP, 16, 32))
    out = fixed_order_reduce.contract_parallel_projection(
        x,
        w,
        equation="TNH,NHD->TD",
        mesh=_model_mesh(),
        tp_axis="model",
        local_matmul=_LOCAL_MATMULS[local],
        mode=mode,
    )
    expected = test_utils.exact_matmul_reference(
        np.asarray(x, np.float64).reshape(8, -1),
        np.asarray(w, np.float64).reshape(-1, 32),
    )
    test_utils.assert_bitwise_equal(out, expected)

  def test_modes_are_bitwise_equal_on_random_operands(self):
    rng = np.random.default_rng(2)
    x = test_utils.random_bf16(rng, (8, 256))
    w = test_utils.random_bf16(rng, (256, 64), scale=0.1)
    outputs = [
        fixed_order_reduce.contract_parallel_projection(
            x,
            w,
            equation="mn,np->mp",
            mesh=_model_mesh(),
            tp_axis="model",
            local_matmul=_dot_local_matmul,
            mode=mode,
        )
        for mode in _MODES
    ]
    for out in outputs[1:]:
      test_utils.assert_bitwise_equal(out, outputs[0])
    np.testing.assert_allclose(
        test_utils.as_f32(outputs[0]),
        np.asarray(x, np.float64) @ np.asarray(w, np.float64),
        rtol=2e-2,
        atol=5e-2,
    )

  def test_mode_defaults_to_env_and_rejects_unknown(self):
    rng = np.random.default_rng(4)
    x = test_utils.exact_bf16(rng, (8, 64))
    w = test_utils.exact_bf16(rng, (64, 32))
    os.environ[fixed_order_reduce.FIXED_AR_GATHER_ENV] = "1"
    os.environ[fixed_order_reduce.FIXED_AR_SCATTER_ENV] = "1"
    kwargs = dict(
        equation="mn,np->mp",
        mesh=_model_mesh(),
        tp_axis="model",
        local_matmul=_dot_local_matmul,
    )
    out = fixed_order_reduce.contract_parallel_projection(x, w, **kwargs)
    test_utils.assert_bitwise_equal(
        out, test_utils.exact_matmul_reference(x, w)
    )
    with self.assertRaisesRegex(ValueError, "unknown fixed-order"):
      fixed_order_reduce.contract_parallel_projection(
          x, w, mode="tree", **kwargs
      )

  @parameterized.parameters(*_MODES)
  def test_add_hook_threads_through_the_projection(self, mode):
    calls = []

    def add(acc, operand):
      calls.append(None)
      return fixed_order_reduce.hard_bf16_add(acc, operand)

    rng = np.random.default_rng(9)
    x = test_utils.exact_bf16(rng, (8, 64))
    w = test_utils.exact_bf16(rng, (64, 32))
    kwargs = dict(
        equation="mn,np->mp",
        mesh=_model_mesh(),
        tp_axis="model",
        local_matmul=_dot_local_matmul,
        mode=mode,
    )
    out = fixed_order_reduce.contract_parallel_projection(
        x, w, add=add, **kwargs
    )
    self.assertLen(calls, _TP - 1)
    default = fixed_order_reduce.contract_parallel_projection(x, w, **kwargs)
    test_utils.assert_bitwise_equal(out, default)
    test_utils.assert_bitwise_equal(
        out, test_utils.exact_matmul_reference(x, w)
    )


class P59ContextTest(_EnvTestCase):

  def setUp(self):
    super().setUp()
    self.mesh = test_utils.cpu_mesh((2, 2), ("data", "model"))

  def _probe(self, mesh=None, tp_axis="model"):
    engine_mesh = self.mesh if mesh is None else mesh
    return lambda: fixed_order_reduce.p59_local_tp_context(
        mesh=engine_mesh, tp_axis=tp_axis
    )

  def test_requires_the_flag(self):
    self.assertFalse(_trace_in_manual_map(self.mesh, self._probe()))

  def test_true_inside_the_manual_data_model_map(self):
    os.environ[fixed_order_reduce.P59_RANK_PARALLEL_BACKWARD_ENV] = "1"
    self.assertFalse(self._probe()())
    self.assertTrue(_trace_in_manual_map(self.mesh, self._probe()))

  def test_false_for_other_manual_maps(self):
    os.environ[fixed_order_reduce.P59_RANK_PARALLEL_BACKWARD_ENV] = "1"
    self.assertFalse(
        _trace_in_manual_map(_model_mesh(), self._probe(mesh=_model_mesh()))
    )
    self.assertFalse(
        _trace_in_manual_map(self.mesh, self._probe(), axis_names={"model"})
    )

  @parameterized.named_parameters(
      ("topology_differs", (1, 4), "model"),
      ("tp_axis_not_model", (2, 2), "tp"),
  )
  def test_rejects_inconsistent_engine(self, engine_shape, tp_axis):
    os.environ[fixed_order_reduce.P59_RANK_PARALLEL_BACKWARD_ENV] = "1"
    engine_mesh = test_utils.cpu_mesh(engine_shape, ("data", "model"))
    with self.assertRaises(RuntimeError):
      _trace_in_manual_map(
          self.mesh, self._probe(mesh=engine_mesh, tp_axis=tp_axis)
      )

  def test_p66_replicated_value_is_identity_by_default(self):
    value = jnp.arange(4.0)
    self.assertIs(
        fixed_order_reduce.p66_replicated_tp_value(
            value, mesh=self.mesh, tp_axis="model"
        ),
        value,
    )

  def test_p66_replicated_value_keeps_identical_values(self):
    os.environ[fixed_order_reduce.P59_RANK_PARALLEL_BACKWARD_ENV] = "1"
    os.environ[fixed_order_reduce.P66_CHECK_VMA_ENV] = "1"
    value = jnp.linspace(-3.0, 5.0, 32, dtype=jnp.float32)

    def body(x):
      return fixed_order_reduce.p66_replicated_tp_value(
          x, mesh=self.mesh, tp_axis="model"
      )

    out = jax.jit(
        jax.shard_map(
            body,
            mesh=self.mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
    )(value)
    test_utils.assert_bitwise_equal(out, value)


class P59BackwardTest(_EnvTestCase):

  def test_f32_sum_env_validation_and_vma_identity(self):
    value = jnp.ones((4,), jnp.bfloat16)
    os.environ[fixed_order_reduce.P66_CHECK_VMA_ENV] = "2"
    with self.assertRaises(RuntimeError):
      fixed_order_reduce.p59_fixed_order_tp_sum_f32(
          value, axis_name="model", count=_TP
      )
    os.environ[fixed_order_reduce.P66_CHECK_VMA_ENV] = "1"
    self.assertIs(
        fixed_order_reduce.p59_fixed_order_tp_sum_f32(
            value, axis_name="model", count=_TP
        ),
        value,
    )

  def test_f32_sum_is_rank_ordered_and_cast_once(self):
    rng = np.random.default_rng(5)
    values = test_utils.random_bf16(rng, (_TP, 8, 32), scale=3.0)

    def body(local):
      return fixed_order_reduce.p59_fixed_order_tp_sum_f32(
          local[0], axis_name="model", count=_TP
      )[None]

    sums = np.asarray(
        jax.jit(
            jax.shard_map(
                body,
                mesh=_model_mesh(),
                in_specs=P("model"),
                out_specs=P("model"),
                check_vma=False,
            )
        )(values)
    )
    expected = _sequential_sum(np.asarray(values, np.float32)).astype(
        jnp.bfloat16
    )
    for rank in range(_TP):
      test_utils.assert_bitwise_equal(sums[rank], expected)

  def test_column_parallel_rejects_flag_count_mismatch(self):
    with self.assertRaisesRegex(ValueError, "flags"):
      fixed_order_reduce.p59_column_parallel(
          jnp.dot,
          jnp.ones((2, 2)),
          jnp.ones((2, 2)),
          replicated=(True,),
          axis_name="model",
          count=_TP,
      )

  def test_column_parallel_completes_replicated_cotangents(self):
    rng = np.random.default_rng(6)
    x = jnp.asarray(rng.standard_normal((8, 16)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((16, 32)), jnp.float32)
    cot = jnp.asarray(rng.standard_normal((8, 32)), jnp.float32)

    def body(x_replicated, w_local, cot_local):
      def loss(a, b):
        out = fixed_order_reduce.p59_column_parallel(
            jnp.dot,
            a,
            b,
            replicated=(True, False),
            axis_name="model",
            count=_TP,
        )
        return jnp.sum(out * cot_local)

      # Gradients are taken inside the manual map, as P59 does; the outer
      # shard_map transpose would otherwise psum the replicated cotangent.
      dx, dw = jax.grad(loss, argnums=(0, 1))(x_replicated, w_local)
      return dx[None], dw

    dx, dw = jax.jit(
        jax.shard_map(
            body,
            mesh=_model_mesh(),
            in_specs=(P(), P(None, "model"), P(None, "model")),
            out_specs=(P("model"), P(None, "model")),
            check_vma=False,
        )
    )(x, w, cot)
    dx = np.asarray(dx)
    for rank in range(1, _TP):
      test_utils.assert_bitwise_equal(dx[rank], dx[0])
    x64, w64, cot64 = (np.asarray(v, np.float64) for v in (x, w, cot))
    np.testing.assert_allclose(dx[0], cot64 @ w64.T, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(dw), x64.T @ cot64, rtol=1e-5, atol=1e-5
    )


class FixedOrderEmbedTest(_EnvTestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(7)
    self.table = test_utils.random_bf16(rng, (64, 16))

  def _lookup(self, ids):
    return fixed_order_reduce.fixed_order_embed_lookup(
        jnp.asarray(ids, jnp.int32), self.table, mesh=_model_mesh()
    )

  def test_lookup_equals_the_table_rows(self):
    # One id from each end of every rank's 16-row vocabulary slice.
    ids = [0, 15, 16, 31, 32, 47, 48, 63, 5, 5]
    test_utils.assert_bitwise_equal(
        self._lookup(ids), np.asarray(self.table)[ids]
    )

  def test_ids_outside_every_slice_are_zero(self):
    out = np.asarray(self._lookup([64, -1]), np.float32)
    np.testing.assert_array_equal(out, np.zeros_like(out))

  def test_p67_scope_keeps_serving_lookup_unchanged(self):
    # P67 restricts the P66 pmean to P59's outer map; the serving lookup is
    # outside it and must keep the plain ring result.
    os.environ[fixed_order_reduce.P66_CHECK_VMA_ENV] = "1"
    os.environ[fixed_order_reduce.P67_SCOPE_ENV] = "1"
    ids = [3, 17, 40, 63]
    test_utils.assert_bitwise_equal(
        self._lookup(ids), np.asarray(self.table)[ids]
    )

  def test_rejects_malformed_p67_scope(self):
    os.environ[fixed_order_reduce.P67_SCOPE_ENV] = "2"
    with self.assertRaises(RuntimeError):
      self._lookup([1, 2])


if __name__ == "__main__":
  absltest.main()
