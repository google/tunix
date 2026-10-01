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
from tunix.experimental.zero_tim_kernel import canonical_vjp
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import padded_matmul
from tunix.experimental.zero_tim_kernel import padded_swiglu
from tunix.experimental.zero_tim_kernel import pallas_norm_matmul
from tunix.experimental.zero_tim_kernel import pallas_rmsnorm
from tunix.experimental.zero_tim_kernel import test_utils

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
        padded_matmul.matmul(x, y, contract=contract, interpret=True),
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
    return padded_matmul.matmul(a, b, contract=self.contract, interpret=True)

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
      return padded_swiglu.swiglu(g, u, contract=self.contract, interpret=True)

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
      return pallas_rmsnorm.rmsnorm(a, w, epsilon=_EPS, interpret=True)

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
          a, g, b, epsilon=_EPS, interpret=True
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


if __name__ == "__main__":
  absltest.main()
