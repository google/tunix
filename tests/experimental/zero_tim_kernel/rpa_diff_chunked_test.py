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
"""Tests for rpa_diff_chunked (fp64 gradient checks of the cache-reading VJP).

In the chain test, a 96-token sequence is
prefilled in three 32-token chunks through the paged cache and compared with
a single full-causal attention oracle (value, gradient and finite difference).
"""

import os
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import rpa_diff_chunked

_NQ, _NKV, _HD, _PAGE, _NP = 8, 2, 32, 16, 12
_T, _CHUNK, _D = 96, 32, 64
_SM = 1.0 / np.sqrt(_HD)


class _X64TestCase(absltest.TestCase):
  """Runs each test with float64 enabled, scoped to the test."""

  def setUp(self):
    super().setUp()
    self.enter_context(jax.enable_x64(True))


def _write_and_attend(q, k, v, cache, kv_len, q_len, page_indices):
  """RPA v3 contract on this layout: write the chunk's kv, attend causally."""
  ctx = kv_len - q_len
  pos = ctx + jnp.arange(q.shape[0])
  pg = page_indices[pos // _PAGE]
  off = pos % _PAGE
  ok = jnp.arange(q.shape[0]) < q_len
  new_cache = cache.at[pg, off, :, 0].set(
      jnp.where(ok[:, None, None], k, cache[pg, off, :, 0])
  )
  new_cache = new_cache.at[pg, off, :, 1].set(
      jnp.where(ok[:, None, None], v, new_cache[pg, off, :, 1])
  )
  replica = rpa_diff_chunked.make_chunked_replica(
      sm_scale=_SM,
      page_size=_PAGE,
      out_dtype=jnp.float64,
      compute_dtype=jnp.float64,
  )
  return replica(q, k, v, cache, kv_len, q_len, page_indices), new_cache


def _causal_oracle(q, k, v):
  kg = jnp.repeat(k, _NQ // _NKV, axis=1)
  vg = jnp.repeat(v, _NQ // _NKV, axis=1)
  s = jnp.einsum("qhd,shd->hqs", q, kg) * _SM
  length = q.shape[0]
  mask = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
  s = jnp.where(mask[None], s, -1e30)
  return jnp.einsum("hqs,shd->qhd", jax.nn.softmax(s, -1), vg)


class ChunkedChainTest(_X64TestCase):

  def test_chunked_chain_matches_full_causal_oracle(self):
    rng = np.random.default_rng(5)
    mk = lambda *shape: jnp.asarray(rng.normal(size=shape) * 0.07)
    wq, wk, wv = mk(_D, _NQ * _HD), mk(_D, _NKV * _HD), mk(_D, _NKV * _HD)
    x = mk(_T, _D)
    table = jnp.asarray(rng.permutation(_NP)[: _T // _PAGE].astype(np.int32))
    self.assertEqual(x.dtype, jnp.float64)

    op = rpa_diff_chunked.make_diff_rpa_chunked(
        _write_and_attend,
        sm_scale=_SM,
        page_size=_PAGE,
        num_q_heads=_NQ,
        num_kv_heads=_NKV,
        out_dtype=jnp.float64,
        compute_dtype=jnp.float64,
    )

    def chain(x_):
      cache = jnp.zeros((_NP, _PAGE, _NKV, 2, _HD))
      outs = []
      for c0 in range(0, _T, _CHUNK):
        xc = x_[c0 : c0 + _CHUNK]
        q = (xc @ wq).reshape(-1, _NQ, _HD)
        k = (xc @ wk).reshape(-1, _NKV, _HD)
        v = (xc @ wv).reshape(-1, _NKV, _HD)
        o, cache = op(
            q, k, v, cache, jnp.int32(c0 + _CHUNK), jnp.int32(_CHUNK), table
        )
        outs.append(o.reshape(-1, _NQ * _HD))
      return jnp.concatenate(outs)

    def oracle(x_):
      q = (x_ @ wq).reshape(-1, _NQ, _HD)
      k = (x_ @ wk).reshape(-1, _NKV, _HD)
      v = (x_ @ wv).reshape(-1, _NKV, _HD)
      return _causal_oracle(q, k, v).reshape(-1, _NQ * _HD)

    target = mk(_T, _NQ * _HD)
    chain_loss = lambda x_: jnp.sum(chain(x_) * target)
    oracle_loss = lambda x_: jnp.sum(oracle(x_) * target)

    self.assertLess(abs(float(chain_loss(x)) - float(oracle_loss(x))), 1e-12)
    chain_grad = np.asarray(jax.grad(chain_loss)(x))
    oracle_grad = np.asarray(jax.grad(oracle_loss)(x))
    relative = np.linalg.norm(chain_grad - oracle_grad) / np.linalg.norm(
        oracle_grad
    )
    self.assertLess(relative, 1e-12)

    i, j = 40, 17
    best = np.inf
    for eps in (1e-5, 1e-6, 1e-7):
      xp = np.asarray(x).copy()
      xp[i, j] += eps
      xm = np.asarray(x).copy()
      xm[i, j] -= eps
      fd = (
          float(chain_loss(jnp.asarray(xp)))
          - float(chain_loss(jnp.asarray(xm)))
      ) / (2 * eps)
      best = min(
          best, abs(fd - chain_grad[i, j]) / (abs(chain_grad[i, j]) + 1e-300)
      )
    self.assertLess(best, 1e-6)


class RaggedAdapterTest(_X64TestCase):
  """Two prefill sequences through the full RPA v3 signature."""

  _SEQS = ((0, 24), (24, 64))  # Static query row ranges (q_len 24 and 40).
  _PAGES_PER_SEQ = 3

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    self.addCleanup(patcher.stop)
    os.environ.pop(rpa_diff_chunked.MAX_SEQS_ENV, None)
    rng = np.random.default_rng(7)
    total = self._SEQS[-1][1]
    self.q = jnp.asarray(rng.normal(size=(total, _NQ, _HD)) * 0.3)
    self.k = jnp.asarray(rng.normal(size=(total, _NKV, _HD)) * 0.3)
    self.v = jnp.asarray(rng.normal(size=(total, _NKV, _HD)) * 0.3)
    self.target = jnp.asarray(rng.normal(size=(total, _NQ, _HD)))
    self.cache = jnp.zeros((_NP, _PAGE, _NKV, 2, _HD))
    self.kv_lens = jnp.asarray([hi - lo for lo, hi in self._SEQS], jnp.int32)
    self.cu_q_lens = jnp.asarray([0] + [hi for _, hi in self._SEQS], jnp.int32)
    self.page_indices = jnp.asarray(
        rng.permutation(_NP)[: 2 * self._PAGES_PER_SEQ].astype(np.int32)
    )
    self.distribution = jnp.asarray([0, 0, 2], jnp.int32)

  def _kernel(self, q, k, v, cache, kv_lens, page_indices, cu_q_lens, dist):
    """Stand-in ragged RPA v3 kernel for two static prefill sequences."""
    del kv_lens, cu_q_lens, dist  # The row ranges are static here.
    out = jnp.zeros_like(q)
    for i, (lo, hi) in enumerate(self._SEQS):
      table = page_indices[
          i * self._PAGES_PER_SEQ : (i + 1) * self._PAGES_PER_SEQ
      ]
      o_i, cache = _write_and_attend(
          q[lo:hi], k[lo:hi], v[lo:hi], cache, hi - lo, hi - lo, table
      )
      out = out.at[lo:hi].set(o_i)
    return out, cache

  def _grads(self):
    op = rpa_diff_chunked.make_diff_rpa_chunked_ragged(
        self._kernel,
        sm_scale=_SM,
        page_size=_PAGE,
        num_q_heads=_NQ,
        num_kv_heads=_NKV,
        out_dtype=jnp.float64,
        compute_dtype=jnp.float64,
    )

    def loss(q, k, v):
      out, _ = op(
          q,
          k,
          v,
          self.cache,
          self.kv_lens,
          self.page_indices,
          self.cu_q_lens,
          self.distribution,
      )
      return jnp.sum(out * self.target)

    return jax.grad(loss, argnums=(0, 1, 2))(self.q, self.k, self.v)

  def _oracle_grads(self, lo, hi):
    def loss(q, k, v):
      return jnp.sum(_causal_oracle(q, k, v) * self.target[lo:hi])

    return jax.grad(loss, argnums=(0, 1, 2))(
        self.q[lo:hi], self.k[lo:hi], self.v[lo:hi]
    )

  def _assert_close(self, actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    self.assertLess(
        np.linalg.norm(actual - expected) / np.linalg.norm(expected), 1e-12
    )

  def test_all_sequences_with_max_seqs_2(self):
    os.environ[rpa_diff_chunked.MAX_SEQS_ENV] = "2"
    grads = self._grads()
    for lo, hi in self._SEQS:
      for grad, want in zip(grads, self._oracle_grads(lo, hi)):
        self._assert_close(grad[lo:hi], want)

  def test_default_differentiates_only_the_first_sequence(self):
    # Event 21: the default of one unrolled sequence keeps compile time
    # bounded; later sequences silently receive zero gradient.
    grads = self._grads()
    (lo0, hi0), (lo1, hi1) = self._SEQS
    for grad, want in zip(grads, self._oracle_grads(lo0, hi0)):
      self._assert_close(grad[lo0:hi0], want)
    for grad in grads:
      self.assertTrue(np.all(np.asarray(grad)[lo1:hi1] == 0))


if __name__ == "__main__":
  absltest.main()
