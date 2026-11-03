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
"""Tests for fixed_lm_head (CPU, Pallas interpret mode)."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.zero_tim_kernel import canonical_vjp
from tunix.zero_tim_kernel import fixed_lm_head
from tunix.zero_tim_kernel import model_contracts
from tunix.zero_tim_kernel import test_utils

_HIDDEN = 512
_LOCAL_VOCAB = 1000
_LEARNER_M = 512


def setUpModule():
  test_utils.configure_cpu(num_devices=4)


class FixedLmHeadTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    self.addCleanup(patcher.stop)
    os.environ.pop(canonical_vjp.PLAIN_VJP_ENV, None)
    self.contract = model_contracts.get_contract("qwen8b")
    self.local_matmul = fixed_lm_head.canonical_local_matmul(
        self.contract, interpret=True
    )

  def _head(self, x, w, *, use_fast_plain_vjp: bool = True):
    def run(a, b):
      return fixed_lm_head.fixed_m_lm_head_local(
          a,
          b,
          local_matmul=self.local_matmul,
          use_fast_plain_vjp=use_fast_plain_vjp,
      )

    return jax.jit(run)(x, w)

  @parameterized.parameters(1, 8, 16, 128)
  def test_request_bucket_rows_equal_the_fixed_program_rows(self, m):
    rng = np.random.default_rng(m)
    x = test_utils.random_bf16(rng, (256, _HIDDEN))
    w = test_utils.random_bf16(rng, (_HIDDEN, _LOCAL_VOCAB), scale=0.05)
    full = self._head(x, w)
    self.assertEqual(full.shape, (256, _LOCAL_VOCAB))
    test_utils.assert_bitwise_equal(self._head(x[:m], w), full[:m])

  @parameterized.parameters(8, 256, _LEARNER_M)
  def test_exact_operands_give_the_rounded_product(self, m):
    rng = np.random.default_rng(m + 1)
    x = test_utils.exact_bf16(rng, (m, _HIDDEN))
    w = test_utils.exact_bf16(rng, (_HIDDEN, _LOCAL_VOCAB))
    test_utils.assert_bitwise_equal(
        self._head(x, w), test_utils.exact_matmul_reference(x, w)
    )

  @parameterized.parameters(_LEARNER_M, 300)
  def test_learner_chunks_run_the_fixed_program(self, m):
    rng = np.random.default_rng(m + 9)
    x = test_utils.random_bf16(rng, (m, _HIDDEN))
    w = test_utils.random_bf16(rng, (_HIDDEN, _LOCAL_VOCAB), scale=0.05)
    out = self._head(x, w)
    self.assertEqual(out.shape, (m, _LOCAL_VOCAB))
    test_utils.assert_bitwise_equal(out[:256], self._head(x[:256], w))
    test_utils.assert_bitwise_equal(out[256:], self._head(x[256:], w))

  @parameterized.parameters(True, False)
  def test_learner_vjp_accumulates_chunks_in_ascending_order(
      self, use_fast_plain_vjp
  ):
    rng = np.random.default_rng(10)
    x = test_utils.random_bf16(rng, (_LEARNER_M, _HIDDEN))
    w = test_utils.random_bf16(rng, (_HIDDEN, _LOCAL_VOCAB), scale=0.05)
    cot = test_utils.random_bf16(rng, (_LEARNER_M, _LOCAL_VOCAB))

    def loss(a, b, c):
      out = fixed_lm_head.fixed_m_lm_head_local(
          a,
          b,
          local_matmul=self.local_matmul,
          use_fast_plain_vjp=use_fast_plain_vjp,
      )
      return jnp.sum(out.astype(jnp.float32) * c.astype(jnp.float32))

    grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
    dx, dw = grad(x, w, cot)
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

  @parameterized.parameters(8, 256, 4096)
  def test_real_qwen8b_tp4_head_traces_at_every_family(self, m):
    mesh = test_utils.cpu_mesh((4,), ("tp",))
    contract = model_contracts.find_contract(hidden_size=4096, tp_size=4)
    local_matmul = fixed_lm_head.canonical_local_matmul(
        contract, interpret=True
    )

    def head(x, w):
      return jax.shard_map(
          lambda a, b: fixed_lm_head.fixed_m_lm_head_local(
              a, b, local_matmul=local_matmul
          ),
          mesh=mesh,
          in_specs=(P(None, None), P(None, "tp")),
          out_specs=P(None, "tp"),
          check_vma=False,
      )(x, w)

    x = jax.ShapeDtypeStruct((m, 4096), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((4096, model_contracts.VOCAB_SIZE), jnp.bfloat16)
    out = jax.eval_shape(head, x, w)
    self.assertEqual(out.shape, (m, model_contracts.VOCAB_SIZE))
    self.assertEqual(out.dtype, jnp.bfloat16)
    if m == 4096:

      def loss(a, b):
        return jnp.sum(head(a, b).astype(jnp.float32))

      dx, dw = jax.eval_shape(jax.grad(loss, argnums=(0, 1)), x, w)
      self.assertEqual((dx.shape, dw.shape), (x.shape, w.shape))

  @parameterized.named_parameters(
      ("rank_3_input", (8, _HIDDEN, 1), (_HIDDEN, _LOCAL_VOCAB), jnp.bfloat16),
      ("k_mismatch", (8, 256), (_HIDDEN, _LOCAL_VOCAB), jnp.bfloat16),
      ("f32_input", (8, _HIDDEN), (_HIDDEN, _LOCAL_VOCAB), jnp.float32),
      ("empty_m", (0, _HIDDEN), (_HIDDEN, _LOCAL_VOCAB), jnp.bfloat16),
      ("empty_vocab", (8, _HIDDEN), (_HIDDEN, 0), jnp.bfloat16),
  )
  def test_rejects_invalid_operands(self, x_shape, w_shape, dtype):
    x = jnp.zeros(x_shape, dtype)
    w = jnp.zeros(w_shape, jnp.bfloat16)
    with self.assertRaises(ValueError):
      fixed_lm_head.fixed_m_lm_head_local(x, w, local_matmul=self.local_matmul)

  def test_custom_local_matmul_is_differentiated_when_m_exceeds_fixed_m(self):
    rng = np.random.default_rng(21)
    x = test_utils.random_bf16(rng, (_LEARNER_M, _HIDDEN))
    w = test_utils.random_bf16(rng, (_HIDDEN, _LOCAL_VOCAB), scale=0.05)

    def scaled_matmul(a, b, **kwargs):
      return (
          jnp.float32(2.0)
          * self.local_matmul(a, b, **kwargs).astype(jnp.float32)
      ).astype(jnp.bfloat16)

    def loss_custom(a, b):
      out = fixed_lm_head.fixed_m_lm_head_local(
          a, b, local_matmul=scaled_matmul
      )
      return jnp.sum(out.astype(jnp.float32))

    def loss_unscaled(a, b):
      out = fixed_lm_head.fixed_m_lm_head_local(
          a,
          b,
          local_matmul=self.local_matmul,
          use_fast_plain_vjp=False,
      )
      return jnp.sum(out.astype(jnp.float32))

    dx_custom, dw_custom = jax.jit(jax.grad(loss_custom, argnums=(0, 1)))(x, w)
    dx_base, dw_base = jax.jit(jax.grad(loss_unscaled, argnums=(0, 1)))(x, w)
    np.testing.assert_allclose(
        test_utils.as_f32(dx_custom),
        2.0 * test_utils.as_f32(dx_base),
        rtol=2e-2,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        test_utils.as_f32(dw_custom),
        2.0 * test_utils.as_f32(dw_base),
        rtol=2e-2,
        atol=5e-2,
    )


if __name__ == "__main__":
  absltest.main()
