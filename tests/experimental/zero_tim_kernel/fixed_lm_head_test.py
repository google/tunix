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
"""Tests for fixed_lm_head (4 CPU devices, Pallas interpret mode).

``RegisteredGeometryTest`` verifies the static contract checks and
traces the real Qwen3-8B TP4 head abstractly.  The numerical tests run a tiny
registered geometry (K=512, local vocab 1000 padded to 1024) so that interpret
mode stays fast; the code path is the production one.
"""

import dataclasses
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.experimental.zero_tim_kernel import fixed_lm_head
from tunix.experimental.zero_tim_kernel import fixed_order_reduce
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import padded_matmul
from tunix.experimental.zero_tim_kernel import test_utils

_ENVS = (
    "CANON_QWEN3_HIDDEN_SIZE",
    "CANON_QWEN3_TP_SIZE",
    "CANON_MATMUL_VJP_PLAIN",
    fixed_lm_head.P66_BACKWARD_ARM_ENV,
    fixed_order_reduce.P59_RANK_PARALLEL_BACKWARD_ENV,
    fixed_order_reduce.P66_CHECK_VMA_ENV,
)
_BF16 = "bfloat16"

# The tiny registered geometry of the numerical tests.
_HIDDEN = 512
_LOCAL_VOCAB = 1000
_PADDED_LOCAL_VOCAB = 1024
_LEARNER_M = 512


def setUpModule():
  test_utils.configure_cpu(num_devices=4)


class _CleanEnvTestCase(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    self.addCleanup(patcher.stop)
    for name in _ENVS:
      os.environ.pop(name, None)


class RegisteredGeometryTest(_CleanEnvTestCase):

  def test_registered_production_shapes(self):
    self.assertEqual(fixed_lm_head.REQUEST_M, (8, 16, 32, 64, 128, 256))
    self.assertEqual(fixed_lm_head.LEARNER_M, (4096,))
    self.assertEqual(fixed_lm_head.QWEN4B_TP8_LEARNER_M, (2048, 4096))
    self.assertEqual(fixed_lm_head.QWEN8B_TP8_LEARNER_M, (1024, 2048, 4096))
    self.assertEqual(
        (fixed_lm_head.BM, fixed_lm_head.BN, fixed_lm_head.BK), (128, 256, 256)
    )
    expected = {
        (2048, 4): ("qwen3-1p7b", "tied_embed", 37984, 38144),
        (4096, 4): ("qwen3-8b", "untied_lm_head", 37984, 38144),
        (4096, 8): ("qwen3-8b-tp8", "untied_lm_head", 18992, 19200),
        (2560, 4): ("qwen3-4b-tp4", "tied_embed", 37984, 38144),
        (2560, 8): ("qwen3-4b", "tied_embed", 18992, 19200),
        (5120, 8): ("qwen3-32b", "untied_lm_head", 18992, 19200),
    }
    self.assertEqual(set(fixed_lm_head.GEOMETRIES), set(expected))
    for (hidden, tp_size), values in expected.items():
      model, endpoint, local_vocab, padded_local_vocab = values
      geometry = fixed_lm_head.resolve_geometry(
          hidden, tp_size, endpoint=endpoint
      )
      self.assertEqual(
          (geometry.model, geometry.local_vocab, geometry.padded_local_vocab),
          (model, local_vocab, padded_local_vocab),
      )
      learner_m = fixed_lm_head.learner_m_for_geometry(geometry)
      for m in fixed_lm_head.REQUEST_M + learner_m:
        with self.subTest(hidden=hidden, tp=tp_size, m=m):
          self.assertEqual(
              fixed_lm_head.validate_global_contract(
                  (m, hidden),
                  (hidden, model_contracts.VOCAB_SIZE),
                  _BF16,
                  _BF16,
                  tp_size=tp_size,
              ),
              m,
          )
          self.assertEqual(
              fixed_lm_head.validate_local_contract(
                  (m, hidden), (hidden, local_vocab), tp_size=tp_size
              ),
              m,
          )
      if (hidden, tp_size) not in ((2560, 8), (4096, 8)):
        with self.subTest(hidden=hidden, tp=tp_size, m=2048):
          with self.assertRaisesRegex(ValueError, "requires semantic M"):
            fixed_lm_head.validate_global_contract(
                (2048, hidden),
                (hidden, model_contracts.VOCAB_SIZE),
                _BF16,
                _BF16,
                tp_size=tp_size,
            )
          with self.assertRaisesRegex(ValueError, "local M invalid"):
            fixed_lm_head.validate_local_contract(
                (2048, hidden), (hidden, local_vocab), tp_size=tp_size
            )

  def test_geometries_agree_with_the_model_contracts(self):
    # The head's padded local vocabulary is exactly the contract-admitted N
    # padding at the head's fixed tiles, so the padded matmul accepts it.
    for (hidden, tp_size), geometry in fixed_lm_head.GEOMETRIES.items():
      with self.subTest(model=geometry.model):
        contract = model_contracts.find_contract(
            hidden_size=hidden, tp_size=tp_size
        )
        self.assertEqual(contract.local_vocab, geometry.local_vocab)
        self.assertEqual(
            padded_matmul.padded_matmul_extents(
                hidden,
                geometry.local_vocab,
                contract=contract,
                block_k=fixed_lm_head.BK,
                block_n=fixed_lm_head.BN,
            ),
            (hidden, geometry.padded_local_vocab),
        )

  def test_shape_dtype_and_topology_negatives(self):
    base = ((16, 4096), (4096, 151936), _BF16, _BF16)
    cases = (
        ((1, 4096), base[1], base[2], base[3], 4),
        ((7, 4096), base[1], base[2], base[3], 4),
        ((24, 4096), base[1], base[2], base[3], 4),
        ((257, 4096), base[1], base[2], base[3], 4),
        ((512, 4096), base[1], base[2], base[3], 4),
        ((2048, 4096), base[1], base[2], base[3], 4),
        ((8192, 4096), base[1], base[2], base[3], 4),
        ((16, 2048), base[1], base[2], base[3], 4),
        ((16, 3072), (3072, 151936), base[2], base[3], 4),
        (base[0], (4096, 152064), base[2], base[3], 4),
        (base[0], base[1], "float32", base[3], 4),
        (base[0], base[1], base[2], "float32", 4),
        (base[0], base[1], base[2], base[3], 2),
        ((16, 5120), (5120, 151936), base[2], base[3], 4),
        ((16, 4096, 1), base[1], base[2], base[3], 4),
    )
    for xshape, wshape, xdtype, wdtype, tp in cases:
      with self.subTest(case=(xshape, wshape, xdtype, wdtype, tp)):
        with self.assertRaises(ValueError):
          fixed_lm_head.validate_global_contract(
              xshape, wshape, xdtype, wdtype, tp_size=tp
          )

  def test_registered_endpoint_is_model_specific(self):
    with self.assertRaisesRegex(ValueError, "endpoint disagrees"):
      fixed_lm_head.resolve_geometry(2560, 8, endpoint="untied_lm_head")
    with self.assertRaisesRegex(ValueError, "endpoint disagrees"):
      fixed_lm_head.resolve_geometry(5120, 8, endpoint="tied_embed")
    self.assertEqual(
        fixed_lm_head.resolve_geometry(2560, 8, endpoint="direct_probe").model,
        "qwen3-4b",
    )

  @parameterized.named_parameters(
      ("hidden_disagrees", {"CANON_QWEN3_HIDDEN_SIZE": "2048"}),
      ("hidden_not_int", {"CANON_QWEN3_HIDDEN_SIZE": "4k"}),
      ("tp_disagrees", {"CANON_QWEN3_TP_SIZE": "8"}),
      ("tp_not_int", {"CANON_QWEN3_TP_SIZE": "four"}),
  )
  def test_profile_environment_must_match_the_shape(self, env):
    os.environ.update(env)
    with self.assertRaises(ValueError):
      fixed_lm_head.resolve_geometry(4096, 4)

  def test_matching_profile_environment_is_accepted(self):
    os.environ.update(model_contracts.get_contract("qwen8b").model_env())
    self.assertEqual(fixed_lm_head.resolve_geometry(4096, 4).model, "qwen3-8b")

  @parameterized.parameters(8, 256, 4096)
  def test_real_qwen8b_head_traces_at_every_admitted_family(self, m):
    mesh = test_utils.cpu_mesh((4,), ("model",))
    local_matmul = fixed_lm_head.canonical_local_matmul(
        model_contracts.find_contract(hidden_size=4096, tp_size=4),
        interpret=True,
    )

    def head(x, w):
      return fixed_lm_head.fixed_lm_head(
          x,
          w,
          mesh=mesh,
          tp_axis="model",
          local_matmul=local_matmul,
          endpoint="untied_lm_head",
      )

    x = jax.ShapeDtypeStruct((m, 4096), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((4096, model_contracts.VOCAB_SIZE), jnp.bfloat16)
    out = jax.eval_shape(head, x, w)
    self.assertEqual(out.shape, (m, model_contracts.VOCAB_SIZE))
    self.assertEqual(out.dtype, jnp.bfloat16)
    if m == 4096:
      # The learner shape's custom VJP (lax.scan over 16 chunks) traces too.
      def loss(a, b):
        return jnp.sum(head(a, b).astype(jnp.float32))

      dx, dw = jax.eval_shape(jax.grad(loss, argnums=(0, 1)), x, w)
      self.assertEqual((dx.shape, dw.shape), (x.shape, w.shape))


class _TinyGeometryTestCase(_CleanEnvTestCase):
  """Registers one tiny geometry (K=512, TP=``TP``) for numerical tests."""

  TP = 4

  def setUp(self):
    super().setUp()
    geometry = fixed_lm_head.Geometry(
        model="tiny",
        hidden=_HIDDEN,
        tp_size=self.TP,
        endpoint="untied_lm_head",
        local_vocab=_LOCAL_VOCAB,
        padded_local_vocab=_PADDED_LOCAL_VOCAB,
    )
    for name, value in (
        ("GEOMETRIES", {(_HIDDEN, self.TP): geometry}),
        ("SUPPORTED_HIDDEN", (_HIDDEN,)),
        ("VOCAB", self.TP * _LOCAL_VOCAB),
        ("LEARNER_M", (_LEARNER_M,)),
    ):
      patcher = mock.patch.object(fixed_lm_head, name, value)
      patcher.start()
      self.addCleanup(patcher.stop)
    # The contract admits the tiny local vocabulary's padding, as the real
    # contracts admit 37984 -> 38144 (the padded width takes a 512 tile).
    contract = dataclasses.replace(
        model_contracts.get_contract("qwen8b"),
        matmul_n_padding={_LOCAL_VOCAB: _PADDED_LOCAL_VOCAB},
    )
    self.local_matmul = fixed_lm_head.canonical_local_matmul(
        contract, interpret=True
    )
    self.vocab = self.TP * _LOCAL_VOCAB
    self.mesh = test_utils.cpu_mesh((self.TP,), ("model",))

  def _head(self, x, w, *, mesh=None, endpoint="untied_lm_head"):
    mesh = self.mesh if mesh is None else mesh

    def run(a, b):
      return fixed_lm_head.fixed_lm_head(
          a,
          b,
          mesh=mesh,
          tp_axis="model",
          local_matmul=self.local_matmul,
          endpoint=endpoint,
      )

    return jax.jit(run)(x, w)


class FixedLmHeadTest(_TinyGeometryTestCase):

  @parameterized.parameters(8, 16, 128)
  def test_request_bucket_rows_equal_the_fixed_program_rows(self, m):
    rng = np.random.default_rng(m)
    x = test_utils.random_bf16(rng, (256, _HIDDEN))
    w = test_utils.random_bf16(rng, (_HIDDEN, self.vocab), scale=0.05)
    full = self._head(x, w)
    self.assertEqual(full.shape, (256, self.vocab))
    test_utils.assert_bitwise_equal(self._head(x[:m], w), full[:m])

  @parameterized.parameters(8, 256, _LEARNER_M)
  def test_exact_operands_give_the_rounded_product(self, m):
    rng = np.random.default_rng(m + 1)
    x = test_utils.exact_bf16(rng, (m, _HIDDEN))
    w = test_utils.exact_bf16(rng, (_HIDDEN, self.vocab))
    test_utils.assert_bitwise_equal(
        self._head(x, w), test_utils.exact_matmul_reference(x, w)
    )

  def test_learner_chunks_run_the_fixed_program(self):
    rng = np.random.default_rng(9)
    x = test_utils.random_bf16(rng, (_LEARNER_M, _HIDDEN))
    w = test_utils.random_bf16(rng, (_HIDDEN, self.vocab), scale=0.05)
    out = self._head(x, w)
    test_utils.assert_bitwise_equal(out[:256], self._head(x[:256], w))
    test_utils.assert_bitwise_equal(out[256:], self._head(x[256:], w))

  def test_learner_vjp_accumulates_chunks_in_ascending_order(self):
    rng = np.random.default_rng(10)
    x = test_utils.random_bf16(rng, (_LEARNER_M, _HIDDEN))
    w = test_utils.random_bf16(rng, (_HIDDEN, self.vocab), scale=0.05)
    cot = test_utils.random_bf16(rng, (_LEARNER_M, self.vocab))

    def loss(a, b, c):
      out = fixed_lm_head.fixed_lm_head(
          a,
          b,
          mesh=self.mesh,
          tp_axis="model",
          local_matmul=self.local_matmul,
          endpoint="untied_lm_head",
      )
      return jnp.sum(out.astype(jnp.float32) * c.astype(jnp.float32))

    grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
    dx, dw = grad(x, w, cot)
    # Each chunk's weight cotangent is the 256-row program's; the loop-carried
    # bf16 accumulator adds them as (0 + dW_0) + dW_1.
    _, dw0 = grad(x[:256], w, cot[:256])
    _, dw1 = grad(x[256:], w, cot[256:])
    dw0, dw1 = np.asarray(dw0), np.asarray(dw1)
    test_utils.assert_bitwise_equal(dw, (np.zeros_like(dw0) + dw0) + dw1)
    x64, w64, cot64 = (np.asarray(v, np.float64) for v in (x, w, cot))
    np.testing.assert_allclose(
        test_utils.as_f32(dx), cot64 @ w64.T, rtol=2e-2, atol=5e-2
    )
    np.testing.assert_allclose(
        test_utils.as_f32(dw), x64.T @ cot64, rtol=2e-2, atol=0.25
    )

  def test_direct_probe_endpoint_is_admitted(self):
    rng = np.random.default_rng(12)
    x = test_utils.random_bf16(rng, (8, _HIDDEN))
    w = test_utils.random_bf16(rng, (_HIDDEN, self.vocab), scale=0.05)
    test_utils.assert_bitwise_equal(
        self._head(x, w, endpoint="direct_probe"), self._head(x, w)
    )

  @parameterized.named_parameters(
      ("m_not_admitted", (24, _HIDDEN), jnp.bfloat16, "untied_lm_head", {}),
      ("m_above_fixed", (384, _HIDDEN), jnp.bfloat16, "untied_lm_head", {}),
      ("f32_input", (8, _HIDDEN), jnp.float32, "untied_lm_head", {}),
      ("endpoint_unknown", (8, _HIDDEN), jnp.bfloat16, "lm_head", {}),
      ("endpoint_mismatch", (8, _HIDDEN), jnp.bfloat16, "tied_embed", {}),
      (
          "hidden_env",
          (8, _HIDDEN),
          jnp.bfloat16,
          "untied_lm_head",
          {"CANON_QWEN3_HIDDEN_SIZE": "4096"},
      ),
      (
          "tp_env",
          (8, _HIDDEN),
          jnp.bfloat16,
          "untied_lm_head",
          {"CANON_QWEN3_TP_SIZE": "8"},
      ),
  )
  def test_rejects_value_errors(self, x_shape, x_dtype, endpoint, env):
    os.environ.update(env)
    x = jnp.zeros(x_shape, x_dtype)
    w = jnp.zeros((_HIDDEN, self.vocab), jnp.bfloat16)
    with self.assertRaises(ValueError):
      self._head(x, w, endpoint=endpoint)

  def test_rejects_missing_mesh_or_axis(self):
    x = jnp.zeros((8, _HIDDEN), jnp.bfloat16)
    w = jnp.zeros((_HIDDEN, self.vocab), jnp.bfloat16)
    kwargs = dict(
        tp_axis="model", local_matmul=self.local_matmul, endpoint="tied_embed"
    )
    with self.assertRaisesRegex(RuntimeError, "live model mesh"):
      fixed_lm_head.fixed_lm_head(x, w, mesh=None, **kwargs)
    with self.assertRaisesRegex(RuntimeError, "lacks axis"):
      fixed_lm_head.fixed_lm_head(
          x, w, mesh=test_utils.cpu_mesh((4,), ("tp",)), **kwargs
      )


class P59FixedLmHeadTest(_TinyGeometryTestCase):
  """The head traced inside P59's outer manual ("data", "model") map."""

  TP = 2

  def setUp(self):
    super().setUp()
    os.environ[fixed_order_reduce.P59_RANK_PARALLEL_BACKWARD_ENV] = "1"
    self.p59_mesh = test_utils.cpu_mesh((2, self.TP), ("data", "model"))

  def _p59_head(self, a, b):
    return fixed_lm_head.fixed_lm_head(
        a,
        b,
        mesh=self.p59_mesh,
        tp_axis="model",
        local_matmul=self.local_matmul,
        endpoint="untied_lm_head",
    )

  def test_local_head_matches_global_head_and_completes_input_grad(self):
    rng = np.random.default_rng(13)
    x = test_utils.random_bf16(rng, (_LEARNER_M, _HIDDEN))
    w = test_utils.random_bf16(rng, (_HIDDEN, self.vocab), scale=0.05)
    cot = test_utils.random_bf16(rng, (_LEARNER_M, self.vocab))

    def body(x_local, w_local, cot_local):
      def loss(a, b):
        out = self._p59_head(a, b)
        value = jnp.sum(out.astype(jnp.float32) * cot_local.astype(jnp.float32))
        return value, out

      (_, out), (dx, dw) = jax.value_and_grad(
          loss, argnums=(0, 1), has_aux=True
      )(x_local, w_local)
      return out, dx[None], dw[None]

    out, dx, dw = jax.jit(
        jax.shard_map(
            body,
            mesh=self.p59_mesh,
            in_specs=(P("data", None), P(None, "model"), P("data", "model")),
            out_specs=(
                P("data", "model"),
                P("model", "data", None),
                P("data", None, "model"),
            ),
            check_vma=False,
        )
    )(x, w, cot)
    # Forward: every DP shard's local logits are the global head's rows (the
    # P59 flag is inert outside the manual map, so this is the global path).
    for shard in range(2):
      rows = slice(256 * shard, 256 * (shard + 1))
      test_utils.assert_bitwise_equal(out[rows], self._head(x[rows], w))
    # Backward: the replicated input's cotangent is completed over TP in rank
    # order, identically on both TP ranks; the weight cotangent stays local.
    dx = np.asarray(dx)
    test_utils.assert_bitwise_equal(dx[1], dx[0])
    x64, w64, cot64 = (np.asarray(v, np.float64) for v in (x, w, cot))
    np.testing.assert_allclose(
        test_utils.as_f32(dx[0]), cot64 @ w64.T, rtol=2e-2, atol=5e-2
    )
    np.testing.assert_allclose(
        test_utils.as_f32(dw).sum(axis=0), x64.T @ cot64, rtol=2e-2, atol=0.25
    )

  def _trace_local(self, mesh, local_m, env=None):
    os.environ.update(env or {})
    x = jnp.zeros((local_m * mesh.shape["data"], _HIDDEN), jnp.bfloat16)
    w = jnp.zeros((_HIDDEN, self.vocab), jnp.bfloat16)

    def body(x_local, w_local):
      return fixed_lm_head.fixed_lm_head(
          x_local,
          w_local,
          mesh=mesh,
          tp_axis="model",
          local_matmul=self.local_matmul,
          endpoint="untied_lm_head",
      )

    return jax.eval_shape(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("data", None), P(None, "model")),
            out_specs=P("data", "model"),
            check_vma=False,
        ),
        x,
        w,
    )

  def test_local_rows_must_reconstruct_one_learner_m(self):
    with self.assertRaisesRegex(ValueError, "reconstruct"):
      self._trace_local(self.p59_mesh, 128)
    self.assertEqual(
        self._trace_local(self.p59_mesh, 256).shape, (_LEARNER_M, self.vocab)
    )

  def test_unit_data_axis_requires_a_p66_arm(self):
    mesh = test_utils.cpu_mesh((1, self.TP), ("data", "model"))
    with self.assertRaisesRegex(RuntimeError, "non-unit DP"):
      self._trace_local(mesh, 256)


class P66UnitDataTest(_TinyGeometryTestCase):
  """The diagnostic P66 DP1xTP4 arms admit a unit data axis at M=256."""

  TP = 4

  def test_p66_arm_admits_unit_data_axis(self):
    os.environ[fixed_order_reduce.P59_RANK_PARALLEL_BACKWARD_ENV] = "1"
    os.environ[fixed_lm_head.P66_BACKWARD_ARM_ENV] = "tp4-p59"
    mesh = test_utils.cpu_mesh((1, self.TP), ("data", "model"))

    def body(x_local, w_local):
      return fixed_lm_head.fixed_lm_head(
          x_local,
          w_local,
          mesh=mesh,
          tp_axis="model",
          local_matmul=self.local_matmul,
          endpoint="untied_lm_head",
      )

    out = jax.eval_shape(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("data", None), P(None, "model")),
            out_specs=P("data", "model"),
            check_vma=False,
        ),
        jax.ShapeDtypeStruct((256, _HIDDEN), jnp.bfloat16),
        jax.ShapeDtypeStruct((_HIDDEN, self.vocab), jnp.bfloat16),
    )
    self.assertEqual(out.shape, (256, self.vocab))


if __name__ == "__main__":
  absltest.main()
