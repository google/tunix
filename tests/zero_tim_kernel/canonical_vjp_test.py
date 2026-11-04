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
"""Tests for canonical_vjp (CPU, Pallas interpret mode)."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.zero_tim_kernel import canonical_vjp
from tunix.zero_tim_kernel import model_contracts
from tunix.zero_tim_kernel import padded_matmul
from tunix.zero_tim_kernel import pallas_norm_matmul
from tunix.zero_tim_kernel import pallas_rmsnorm
from tunix.zero_tim_kernel import pallas_swiglu
from tunix.zero_tim_kernel import test_utils

_EPS = 1e-6


def setUpModule():
  test_utils.configure_cpu()


def _grads(fn, cotangent, *args):
  """Gradients of <fn(*args), cotangent> with respect to every argument."""

  def loss(*operands):
    return jnp.sum(fn(*operands).astype(jnp.float32) * cotangent)

  return jax.grad(loss, argnums=tuple(range(len(args))))(*args)


class CanonicalReplicaTest(absltest.TestCase):

  def test_canonical_matmul_known_answer_with_k_padding(self):
    contract = model_contracts.get_contract("qwen4b")  # K 1216 -> 1280.
    rng = np.random.default_rng(0)
    x = test_utils.exact_bf16(rng, (16, 1216))
    y = test_utils.exact_bf16(rng, (1216, 256))
    test_utils.assert_bitwise_equal(
        canonical_vjp.canonical_matmul(x, y, contract=contract),
        test_utils.exact_matmul_reference(x, y),
    )

  def test_canonical_matmul_matches_the_pallas_forward_on_exact_data(self):
    contract = model_contracts.get_contract("qwen8b")
    rng = np.random.default_rng(1)
    x = test_utils.exact_bf16(rng, (8, 512))
    y = test_utils.exact_bf16(rng, (512, 512))
    test_utils.assert_bitwise_equal(
        canonical_vjp.canonical_matmul(x, y, contract=contract),
        padded_matmul.matmul(x, y, contract=contract, interpret=pltpu.InterpretParams()),
    )

  def test_replicas_reject_bad_operands(self):
    contract = model_contracts.get_contract("qwen8b")
    with self.assertRaises(ValueError):
      canonical_vjp.canonical_matmul(
          jnp.zeros((8, 256), jnp.float32),
          jnp.zeros((256, 256), jnp.float32),
          contract=contract,
      )
    with self.assertRaises(ValueError):
      canonical_vjp.canonical_swiglu(
          jnp.zeros((8, 256), jnp.bfloat16), jnp.zeros((8, 128), jnp.bfloat16)
      )


class CoatTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    self.addCleanup(patcher.stop)
    os.environ.pop(canonical_vjp.PLAIN_VJP_ENV, None)
    self.contract = model_contracts.get_contract("qwen8b")
    rng = np.random.default_rng(2)
    self.x = test_utils.random_bf16(rng, (16, 512))
    self.gamma = test_utils.random_bf16(rng, (512,))
    self.y = test_utils.random_bf16(rng, (512, 256), scale=0.05)
    self.cotangent = jnp.asarray(rng.standard_normal((16, 256)), jnp.float32)

  def _pallas_forward(self, a, b):
    return padded_matmul.matmul(a, b, contract=self.contract, interpret=pltpu.InterpretParams())

  def test_plain_env_parsing(self):
    self.assertTrue(canonical_vjp.plain_matmul_vjp_enabled())
    os.environ[canonical_vjp.PLAIN_VJP_ENV] = "0"
    self.assertFalse(canonical_vjp.plain_matmul_vjp_enabled())
    os.environ[canonical_vjp.PLAIN_VJP_ENV] = "yes"
    with self.assertRaises(ValueError):
      canonical_vjp.plain_matmul_vjp_enabled()

  def test_matmul_primal_is_the_forward_verbatim(self):
    coat = canonical_vjp.matmul(
        self.x, self.y, forward=self._pallas_forward, contract=self.contract
    )
    test_utils.assert_bitwise_equal(coat, self._pallas_forward(self.x, self.y))

  def test_matmul_default_pullback_is_the_plain_dot_transpose(self):
    grads = _grads(
        lambda a, b: canonical_vjp.matmul(a, b, forward=self._pallas_forward),
        self.cotangent,
        self.x,
        self.y,
    )
    expected = _grads(
        canonical_vjp.plain_matmul, self.cotangent, self.x, self.y
    )
    for grad, want in zip(grads, expected):
      test_utils.assert_bitwise_equal(grad, want)

  def test_matmul_replica_pullback(self):
    os.environ[canonical_vjp.PLAIN_VJP_ENV] = "0"
    grads = _grads(
        lambda a, b: canonical_vjp.matmul(
            a, b, forward=self._pallas_forward, contract=self.contract
        ),
        self.cotangent,
        self.x,
        self.y,
    )
    expected = _grads(
        lambda a, b: canonical_vjp.canonical_matmul(
            a, b, contract=self.contract
        ),
        self.cotangent,
        self.x,
        self.y,
    )
    for grad, want in zip(grads, expected):
      test_utils.assert_bitwise_equal(grad, want)
    # Both pullbacks compute the same gradient up to summation order.
    os.environ[canonical_vjp.PLAIN_VJP_ENV] = "1"
    plain = _grads(
        lambda a, b: canonical_vjp.matmul(a, b, forward=self._pallas_forward),
        self.cotangent,
        self.x,
        self.y,
    )
    for grad, want in zip(plain, expected):
      np.testing.assert_allclose(
          test_utils.as_f32(grad),
          test_utils.as_f32(want),
          rtol=2e-2,
          atol=2e-2,
      )

  def test_replica_pullback_requires_a_contract(self):
    os.environ[canonical_vjp.PLAIN_VJP_ENV] = "0"
    with self.assertRaisesRegex(ValueError, "contract"):
      _grads(
          lambda a, b: canonical_vjp.matmul(a, b, forward=self._pallas_forward),
          self.cotangent,
          self.x,
          self.y,
      )

  def test_swiglu_pullback_is_the_canonical_replica(self):
    rng = np.random.default_rng(3)
    gate = test_utils.random_bf16(rng, (16, 512), scale=3.0)
    up = test_utils.random_bf16(rng, (16, 512))
    cotangent = jnp.asarray(rng.standard_normal((16, 512)), jnp.float32)

    def forward(g, u):
      return pallas_swiglu.swiglu(g, u, interpret=pltpu.InterpretParams())

    coat = canonical_vjp.swiglu(gate, up, forward=forward)
    test_utils.assert_bitwise_equal(coat, forward(gate, up))
    grads = _grads(
        lambda g, u: canonical_vjp.swiglu(g, u, forward=forward),
        cotangent,
        gate,
        up,
    )
    expected = _grads(canonical_vjp.canonical_swiglu, cotangent, gate, up)
    for grad, want in zip(grads, expected):
      test_utils.assert_bitwise_equal(grad, want)

  def test_rmsnorm_pullback_is_the_canonical_replica(self):
    cotangent = jnp.asarray(
        np.random.default_rng(4).standard_normal((16, 512)), jnp.float32
    )

    def forward(a, w):
      return pallas_rmsnorm.rmsnorm(a, w, epsilon=_EPS, interpret=pltpu.InterpretParams())

    def replica(a, w):
      return pallas_rmsnorm.canonical_rmsnorm(a, w, epsilon=_EPS)

    grads = _grads(
        lambda a, w: canonical_vjp.rmsnorm(a, w, epsilon=_EPS, forward=forward),
        cotangent,
        self.x,
        self.gamma,
    )
    expected = _grads(replica, cotangent, self.x, self.gamma)
    for grad, want in zip(grads, expected):
      test_utils.assert_bitwise_equal(grad, want)

  @parameterized.parameters("1", "0")
  def test_norm_matmul_pullback(self, plain):
    os.environ[canonical_vjp.PLAIN_VJP_ENV] = plain
    rng = np.random.default_rng(5)
    x = test_utils.random_bf16(rng, (128, 512))
    cotangent = jnp.asarray(rng.standard_normal((128, 256)), jnp.float32)

    def forward(a, g, b):
      return pallas_norm_matmul.norm_matmul(
          a, g, b, epsilon=_EPS, interpret=pltpu.InterpretParams()
      )

    def coat(a, g, b):
      return canonical_vjp.norm_matmul(
          a, g, b, epsilon=_EPS, forward=forward, contract=self.contract
      )

    test_utils.assert_bitwise_equal(
        coat(x, self.gamma, self.y), forward(x, self.gamma, self.y)
    )

    def reference(a, g, b):
      normed = pallas_rmsnorm.canonical_rmsnorm(a, g, epsilon=_EPS)
      if plain == "1":
        return canonical_vjp.plain_matmul(normed, b)
      return canonical_vjp.canonical_matmul(normed, b, contract=self.contract)

    grads = _grads(coat, cotangent, x, self.gamma, self.y)
    expected = _grads(reference, cotangent, x, self.gamma, self.y)
    for grad, want in zip(grads, expected):
      np.testing.assert_allclose(
          test_utils.as_f32(grad),
          test_utils.as_f32(want),
          rtol=1e-2,
          atol=1e-2,
      )

  @parameterized.parameters("1", "0")
  def test_fused_matmul_and_fused_norm_matmul(self, plain):
    os.environ[canonical_vjp.PLAIN_VJP_ENV] = plain
    rng = np.random.default_rng(6)
    x = test_utils.random_bf16(rng, (12, 512))
    w1 = test_utils.random_bf16(rng, (512, 256), scale=0.05)
    w2 = test_utils.random_bf16(rng, (512, 128), scale=0.05)
    cot = jnp.asarray(rng.standard_normal((12, 384)), jnp.float32)

    out_fused = canonical_vjp.fused_matmul(
        x, (w1, w2), contract=self.contract, interpret=pltpu.InterpretParams()
    )
    out_concat = padded_matmul.matmul(
        x,
        jnp.concatenate((w1, w2), axis=1),
        contract=self.contract,
        interpret=pltpu.InterpretParams(),
    )
    test_utils.assert_bitwise_equal(out_fused, out_concat)

    def fused_mm_ref(a, b1, b2):
      w_cat = jnp.concatenate((b1, b2), axis=1)
      if plain == "1":
        return canonical_vjp.plain_matmul(a, w_cat)
      return canonical_vjp.canonical_matmul(a, w_cat, contract=self.contract)

    dx_mm, dw1_mm, dw2_mm = _grads(
        lambda a, b1, b2: canonical_vjp.fused_matmul(
            a, (b1, b2), contract=self.contract, interpret=pltpu.InterpretParams()
        ),
        cot,
        x,
        w1,
        w2,
    )
    dx_mm_ref, dw1_mm_ref, dw2_mm_ref = _grads(fused_mm_ref, cot, x, w1, w2)
    test_utils.assert_bitwise_equal(dx_mm, dx_mm_ref)
    test_utils.assert_bitwise_equal(dw1_mm, dw1_mm_ref)
    test_utils.assert_bitwise_equal(dw2_mm, dw2_mm_ref)

    out_norm_fused = canonical_vjp.fused_norm_matmul(
        x,
        self.gamma,
        (w1, w2),
        norm_eps=_EPS,
        contract=self.contract,
        interpret=pltpu.InterpretParams(),
    )
    normed = pallas_rmsnorm.rmsnorm(x, self.gamma, epsilon=_EPS, interpret=pltpu.InterpretParams())
    out_two_step = canonical_vjp.fused_matmul(
        normed, (w1, w2), contract=self.contract, interpret=pltpu.InterpretParams()
    )
    test_utils.assert_bitwise_equal(out_norm_fused, out_two_step)

    def fused_norm_mm_ref(a, g, b1, b2):
      h = pallas_rmsnorm.canonical_rmsnorm(a, g, epsilon=_EPS)
      return fused_mm_ref(h, b1, b2)

    dx, dg, dw1, dw2 = _grads(
        lambda a, g, b1, b2: canonical_vjp.fused_norm_matmul(
            a,
            g,
            (b1, b2),
            norm_eps=_EPS,
            contract=self.contract,
            interpret=pltpu.InterpretParams(),
        ),
        cot,
        x,
        self.gamma,
        w1,
        w2,
    )
    dx_ref, dg_ref, dw1_ref, dw2_ref = _grads(
        fused_norm_mm_ref, cot, x, self.gamma, w1, w2
    )
    test_utils.assert_bitwise_equal(dx, dx_ref)
    test_utils.assert_bitwise_equal(dg, dg_ref)
    test_utils.assert_bitwise_equal(dw1, dw1_ref)
    test_utils.assert_bitwise_equal(dw2, dw2_ref)

  def test_fused_ops_validate_inputs_and_wide_n_padding(self):
    qwen4b = model_contracts.get_contract("qwen4b")
    rng = np.random.default_rng(7)
    x = test_utils.random_bf16(rng, (16, 256))
    gamma = test_utils.random_bf16(rng, (256,))
    # 1216 + 1216 = 2432 triggers wide_n_tiles -> (2560, block_n=1280).
    w_gate = test_utils.random_bf16(rng, (256, 1216), scale=0.05)
    w_up = test_utils.random_bf16(rng, (256, 1216), scale=0.05)
    out_fused = canonical_vjp.fused_norm_matmul(
        x,
        gamma,
        (w_gate, w_up),
        norm_eps=_EPS,
        contract=qwen4b,
        interpret=pltpu.InterpretParams(),
    )
    normed = pallas_rmsnorm.rmsnorm(x, gamma, epsilon=_EPS, interpret=pltpu.InterpretParams())
    out_two_step = canonical_vjp.fused_matmul(
        normed, (w_gate, w_up), contract=qwen4b, interpret=pltpu.InterpretParams()
    )
    test_utils.assert_bitwise_equal(out_fused, out_two_step)

    with self.assertRaisesRegex(ValueError, "w_parts"):
      canonical_vjp.fused_matmul(x, (), contract=qwen4b, interpret=pltpu.InterpretParams())
    with self.assertRaisesRegex(ValueError, "w_parts"):
      canonical_vjp.fused_norm_matmul(
          x, gamma, (), norm_eps=_EPS, contract=qwen4b, interpret=pltpu.InterpretParams()
      )
    with self.assertRaisesRegex(ValueError, "w_parts"):
      canonical_vjp.fused_matmul(
          x,
          (jnp.zeros((128, 128), jnp.bfloat16),),
          contract=qwen4b,
          interpret=pltpu.InterpretParams(),
      )


if __name__ == "__main__":
  absltest.main()
