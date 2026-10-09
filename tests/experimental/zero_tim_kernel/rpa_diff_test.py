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
"""Tests for rpa_diff (prefill-only differentiable RPA wrapper)."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import rpa_diff

_T, _NQ, _NKV, _HD = 16, 4, 2, 8
_SEQ_LENS = (5, 7)  # Two prefill sequences; rows 12..15 are padding.
_SM_SCALE = 1.0 / np.sqrt(_HD)


def _metadata(kv_lens=None):
  cu_q_lens = np.zeros((5,), np.int32)
  cu_q_lens[1 : len(_SEQ_LENS) + 1] = np.cumsum(_SEQ_LENS)
  cu_q_lens[len(_SEQ_LENS) + 1 :] = cu_q_lens[len(_SEQ_LENS)]
  if kv_lens is None:
    kv_lens = list(_SEQ_LENS) + [0, 0]
  return (
      jnp.asarray(kv_lens, jnp.int32),
      jnp.zeros((4, 1), jnp.int32),  # page_indices (unused by the replica).
      jnp.asarray(cu_q_lens),
      jnp.asarray([0, 0, len(_SEQ_LENS)], jnp.int32),  # distribution.
  )


def _qkv(seed=0):
  rng = np.random.default_rng(seed)
  return tuple(
      jnp.asarray(rng.standard_normal(shape), jnp.float32)
      for shape in ((_T, _NQ, _HD), (_T, _NKV, _HD), (_T, _NKV, _HD))
  )


def _float64_reference(q, k, v):
  """Per-sequence causal GQA attention; padding rows are zero."""
  q, k, v = (np.asarray(a, np.float64) for a in (q, k, v))
  out = np.zeros_like(q)
  group = _NQ // _NKV
  start = 0
  for length in _SEQ_LENS:
    rows = slice(start, start + length)
    for head in range(_NQ):
      scores = q[rows, head] @ k[rows, head // group].T * _SM_SCALE
      scores = np.where(
          np.tril(np.ones((length, length), bool)), scores, -np.inf
      )
      weights = np.exp(scores - scores.max(-1, keepdims=True))
      weights /= weights.sum(-1, keepdims=True)
      out[rows, head] = weights @ v[rows, head // group]
    start += length
  return out


def _replica(q, k, v, kv_lens, page_indices, cu_q_lens, distribution):
  return rpa_diff.replica(
      q,
      k,
      v,
      None,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      sm_scale=_SM_SCALE,
      num_seqs=distribution[2],
  )


class ReplicaTest(absltest.TestCase):

  def test_replica_matches_float64_reference(self):
    q, k, v = _qkv()
    out = _replica(q, k, v, *_metadata())
    np.testing.assert_allclose(
        np.asarray(out), _float64_reference(q, k, v), rtol=1e-5, atol=1e-5
    )

  def test_replica_accepts_traced_num_seqs(self):
    q, k, v = _qkv()
    out = jax.jit(_replica)(q, k, v, *_metadata())
    np.testing.assert_allclose(
        np.asarray(out), _float64_reference(q, k, v), rtol=1e-5, atol=1e-5
    )

  def test_decode_batch_is_refused_loudly(self):
    # kv_len > q_len would need the paged cache: the replica returns NaN.
    q, k, v = _qkv()
    out = _replica(q, k, v, *_metadata(kv_lens=[9, 7, 0, 0]))
    self.assertTrue(np.all(np.isnan(np.asarray(out))))

  def test_num_seqs_is_required(self):
    q, k, v = _qkv()
    kv_lens, page_indices, cu_q_lens, _ = _metadata()
    with self.assertRaises(ValueError):
      rpa_diff.replica(
          q,
          k,
          v,
          None,
          kv_lens,
          page_indices,
          cu_q_lens,
          None,
          sm_scale=_SM_SCALE,
      )


class MakeDiffRpaTest(absltest.TestCase):

  def test_forward_is_the_kernel_and_backward_is_the_replica(self):
    # The stand-in kernel returns 2x the replica, so a replica-based VJP is
    # distinguishable from differentiating the kernel.
    def kernel(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, dist):
      out = _replica(q, k, v, kv_lens, page_indices, cu_q_lens, dist)
      return 2.0 * out, kv_cache + 1.0

    diff = rpa_diff.make_diff_rpa(kernel, sm_scale=_SM_SCALE)
    q, k, v = _qkv(1)
    cache = jnp.zeros((4, 2, 2, 2, _HD), jnp.float32)
    meta = _metadata()
    weight = jnp.asarray(
        np.random.default_rng(2).standard_normal((_T, _NQ, _HD)), jnp.float32
    )

    out, new_cache = jax.jit(diff)(q, k, v, cache, *meta)
    expected_out, expected_cache = kernel(q, k, v, cache, *meta)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected_out))
    np.testing.assert_array_equal(
        np.asarray(new_cache), np.asarray(expected_cache)
    )

    def loss(q_, k_, v_):
      return jnp.sum(diff(q_, k_, v_, cache, *meta)[0] * weight)

    def replica_loss(q_, k_, v_):
      return jnp.sum(_replica(q_, k_, v_, *meta) * weight)

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)
    expected = jax.jit(jax.grad(replica_loss, argnums=(0, 1, 2)))(q, k, v)
    for grad, want in zip(grads, expected):
      np.testing.assert_allclose(
          np.asarray(grad), np.asarray(want), rtol=1e-5, atol=1e-6
      )
    # Padding rows receive no gradient.
    self.assertTrue(np.all(np.asarray(grads[0])[sum(_SEQ_LENS) :] == 0))

  def test_scratch_kv_cache_shape_and_allocation_match_rpa_v3_contract(self):
    # bf16: kv_packing = 2, head_dim aligned to 128
    k_bf16 = jnp.zeros((48, 3, 64), dtype=jnp.bfloat16)
    kv_lens = jnp.asarray([20, 28, 0, 0], dtype=jnp.int32)
    shape_bf16 = rpa_diff.scratch_kv_cache_shape(k_bf16, page_size=16)
    # 48 tokens / 16 = 3 pages; ceil(3 * 2 / 2) = 3; packing = 2;
    # align(64, 128) = 128
    self.assertEqual(shape_bf16, (3, 16, 3, 2, 128))
    scratch_cache, scratch_pages = rpa_diff.allocate_scratch_kv_cache(
        k_bf16, kv_lens, page_size=16
    )
    self.assertEqual(scratch_cache.shape, (3, 16, 3, 2, 128))
    self.assertEqual(scratch_cache.dtype, jnp.bfloat16)
    self.assertEqual(scratch_pages.shape, (4 * 3,))
    self.assertEqual(scratch_pages.dtype, jnp.int32)
    self.assertTrue(
        np.all(
            (np.asarray(scratch_pages) >= 0) & (np.asarray(scratch_pages) < 3)
        )
    )

    # f32: kv_packing = 1
    k_f32 = jnp.zeros((16, 2, 8), dtype=jnp.float32)
    shape_f32 = rpa_diff.scratch_kv_cache_shape(k_f32, page_size=16)
    self.assertEqual(shape_f32, (2, 16, 4, 1, 128))

  def test_make_diff_rpa_and_bind_scratch_kv_cache_without_explicit_cache(self):
    observed_cache_shapes = []

    def strict_rpa_v3_kernel(
        q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, dist
    ):
      # Enforce RPA v3 contract: kv_cache and page_indices must not be None
      # and must match the 5D TPU layout.
      self.assertIsNotNone(kv_cache)
      self.assertIsNotNone(page_indices)
      expected_shape = rpa_diff.scratch_kv_cache_shape(k, page_size=16)
      self.assertEqual(kv_cache.shape, expected_shape)
      self.assertEqual(page_indices.ndim, 1)
      self.assertEqual(page_indices.shape[0] % kv_lens.shape[0], 0)
      observed_cache_shapes.append(kv_cache.shape)
      out = _replica(q, k, v, kv_lens, page_indices, cu_q_lens, dist)
      return 2.0 * out, kv_cache + 1.0

    q, k, v = _qkv(3)
    kv_lens, _, cu_q_lens, dist = _metadata()
    weight = jnp.asarray(
        np.random.default_rng(4).standard_normal((_T, _NQ, _HD)), jnp.float32
    )

    # 1. Forward-only bind_scratch_kv_cache (6-arg cacheless & 8-arg None)
    bound_fwd = rpa_diff.bind_scratch_kv_cache(
        strict_rpa_v3_kernel, page_size=16
    )
    out_6 = jax.jit(bound_fwd)(q, k, v, kv_lens, cu_q_lens, dist)
    out_8, new_scratch = jax.jit(bound_fwd)(
        q, k, v, None, kv_lens, None, cu_q_lens, dist
    )
    np.testing.assert_array_equal(np.asarray(out_6), np.asarray(out_8))
    self.assertEqual(
        new_scratch.shape, rpa_diff.scratch_kv_cache_shape(k, page_size=16)
    )

    # 2. Differentiable make_diff_rpa with 6-arg cacheless signature across
    #    2 layers
    diff = rpa_diff.make_diff_rpa(
        strict_rpa_v3_kernel, sm_scale=_SM_SCALE, page_size=16
    )

    def two_layer_cacheless_loss(q_, k_, v_):
      h1 = diff(q_, k_, v_, kv_lens, cu_q_lens, dist)
      h2 = diff(h1, k_, v_, kv_lens, cu_q_lens, dist)
      return jnp.sum(h2 * weight)

    def two_layer_replica_loss(q_, k_, v_):
      # Forward of strict_rpa_v3_kernel scales output by 2.0, while backward
      # differentiates replica(q, k, v) at each layer.
      h1 = jax.lax.stop_gradient(
          2.0 * _replica(q_, k_, v_, kv_lens, None, cu_q_lens, dist)
          - _replica(q_, k_, v_, kv_lens, None, cu_q_lens, dist)
      ) + _replica(q_, k_, v_, kv_lens, None, cu_q_lens, dist)
      h2 = _replica(h1, k_, v_, kv_lens, None, cu_q_lens, dist)
      return jnp.sum(h2 * weight)

    grads = jax.jit(jax.grad(two_layer_cacheless_loss, argnums=(0, 1, 2)))(
        q, k, v
    )
    expected = jax.jit(jax.grad(two_layer_replica_loss, argnums=(0, 1, 2)))(
        q, k, v
    )
    for grad, want in zip(grads, expected):
      np.testing.assert_allclose(
          np.asarray(grad), np.asarray(want), rtol=1e-5, atol=1e-6
      )


if __name__ == "__main__":
  absltest.main()
