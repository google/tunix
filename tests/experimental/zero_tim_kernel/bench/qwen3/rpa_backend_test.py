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

"""Tests for `rpa_backend` (CPU: the kernel's contract replica is injected).

The real RPA v3 kernel needs a TPU. Every test here injects
`rpa_backend.reference_kernel`, the jittable pure-JAX replica of the kernel's
*contract*, and therefore proves the adapter's bookkeeping -- layouts, page
tables, positions, metadata, the TP map, the cache threading, the VJP
wrapper, the model glue -- with tolerance-based assertions against dense f32
attention. Bitwise claims about the kernel itself are only made by
`rpa_backend_tpu_test` with the real kernel.
"""

import functools
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import rpa_canonical
from tunix.experimental.zero_tim_kernel import rpa_diff_chunked
from tunix.experimental.zero_tim_kernel import test_utils
from tunix.experimental.zero_tim_kernel.bench.qwen3 import canon_attention
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_canon_benchmark
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_model_canon as qwen3_canon
from tunix.experimental.zero_tim_kernel.bench.qwen3 import rpa_backend
from tunix.models.qwen3 import model as qwen3_stock


def setUpModule():
  test_utils.configure_cpu(num_devices=4)


_D = 128
_REF = rpa_backend.reference_kernel


def _cacheless(*args, **kwargs):
  """`attention_cacheless` on the contract replica, forward-only by default."""
  kwargs.setdefault('kernel', _REF)
  kwargs.setdefault('differentiable', False)
  return rpa_backend.attention_cacheless(*args, **kwargs)


def _cached(*args, **kwargs):
  """`attention_cached` on the contract replica, forward-only by default."""
  kwargs.setdefault('kernel', _REF)
  kwargs.setdefault('differentiable', False)
  return rpa_backend.attention_cached(*args, **kwargs)


def _rng(seed: int) -> np.random.Generator:
  return np.random.default_rng(seed)


def _operands(rng, b, t, qh, kh, d=_D, dtype=jnp.float32):
  q = jnp.asarray(rng.standard_normal((b, t, qh, d)), dtype)
  k = jnp.asarray(rng.standard_normal((b, t, kh, d)), dtype)
  v = jnp.asarray(rng.standard_normal((b, t, kh, d)), dtype)
  return q, k, v


def _dense_reference(q, k_all, v_all, q_pos, *, scale):
  """f32 attention of `q[B, T]` at positions `q_pos[B, T]` over `k_all[B, S]`.

  Slot `s` of `k_all` / `v_all` is position `s`; a query sees the keys at
  positions `<= q_pos` (prefix causal on absolute positions).

  Args:
    q: Queries `[B, T, QH, D]`.
    k_all: Keys `[B, S, KH, D]`.
    v_all: Values `[B, S, KH, D]`.
    q_pos: int32 `[B, T]` absolute query positions.
    scale: Softmax scale.

  Returns:
    `out[B, T, QH, D]` in the query dtype.
  """
  b, t, qh, d = q.shape
  kh = k_all.shape[2]
  qf = q.astype(jnp.float32).reshape(b, t, kh, qh // kh, d)
  scores = (
      jnp.einsum('bthgd,bshd->bhgts', qf, k_all.astype(jnp.float32)) * scale
  )
  slots = jnp.arange(k_all.shape[1], dtype=jnp.int32)
  keep = slots[None, None, :] <= q_pos[:, :, None]  # [B, T, S]
  scores = jnp.where(keep[:, None, None], scores, jnp.finfo(jnp.float32).min)
  probs = jax.nn.softmax(scores, axis=-1)
  out = jnp.einsum('bhgts,bshd->bthgd', probs, v_all.astype(jnp.float32))
  return out.reshape(b, t, qh, d).astype(q.dtype)


def _causal_reference(q, k, v, *, scale):
  b, t = q.shape[:2]
  pos = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[None], (b, t))
  return _dense_reference(q, k, v, pos, scale=scale)


def _dense_cache(b, s, kh, d, dtype):
  """One layer of the bench's `init_cache`."""
  return {
      'k': jnp.zeros((b, s, kh, d), dtype),
      'v': jnp.zeros((b, s, kh, d), dtype),
      'end_index': jnp.zeros((b,), jnp.int32),
  }


def _tp_mesh(tp_size: int) -> jax.sharding.Mesh:
  return test_utils.cpu_mesh((1, tp_size), ('fsdp', 'tp'))


# The project's gradient gate (gradients have no bitwise contract): the same
# thresholds as `qwen3_model_canon_test.test_param_grads_match_stock_tp4`.
_GRAD_RATIO_TOL = 0.02
_GRAD_COSINE_MIN = 0.999


def _grad_stats(got, want) -> tuple[float, float]:
  """`(norm ratio got / want, cosine)` of two gradient arrays."""
  got = np.asarray(got, np.float32).ravel()
  want = np.asarray(want, np.float32).ravel()
  norm_got, norm_want = float(np.linalg.norm(got)), float(np.linalg.norm(want))
  ratio = norm_got / norm_want
  cosine = float(np.dot(got, want) / (norm_got * norm_want + 1e-30))
  return ratio, cosine


class _RecordingKernel:
  """Delegates to `reference_kernel`; records the call (eager calls only)."""

  def __init__(self):
    self.calls = []

  def __call__(self, *args, **kwargs):
    shapes = tuple((tuple(a.shape), str(a.dtype)) for a in args)
    metadata = tuple(np.asarray(a) for a in args[4:])
    self.calls.append((shapes, dict(kwargs), metadata))
    return _REF(*args, **kwargs)


class LayoutTest(parameterized.TestCase):

  def test_packing_and_paged_cache_shape(self):
    self.assertEqual(rpa_backend.kv_packing(jnp.bfloat16), 2)
    self.assertEqual(rpa_backend.kv_packing(jnp.float16), 2)
    self.assertEqual(rpa_backend.kv_packing(jnp.float32), 1)
    self.assertEqual(
        rpa_backend.paged_cache_shape(8, 16, 2, 128, jnp.bfloat16),
        (8, 16, 2, 2, 128),
    )
    self.assertEqual(
        rpa_backend.paged_cache_shape(8, 16, 2, 128, jnp.float32),
        (8, 16, 4, 1, 128),
    )
    # Odd head counts and a head dim below the 128 alignment.
    self.assertEqual(
        rpa_backend.paged_cache_shape(3, 32, 3, 64, jnp.bfloat16),
        (3, 32, 3, 2, 128),
    )

  @parameterized.named_parameters(
      ('bf16_kh2_d128', jnp.bfloat16, 2, 128),
      ('f32_kh3_d128', jnp.float32, 3, 128),
      ('bf16_kh3_d64_padded', jnp.bfloat16, 3, 64),
  )
  def test_merge_kv_layout_and_split_round_trip(self, dtype, kh, d):
    rng = _rng(0)
    k = jnp.asarray(rng.standard_normal((5, kh, d)), dtype)
    v = jnp.asarray(rng.standard_normal((5, kh, d)), dtype)
    kv = rpa_backend.merge_kv(k, v)
    packing = rpa_backend.kv_packing(dtype)
    n2p = rpa_backend.round_up(2 * kh, packing) // packing
    self.assertEqual(kv.shape, (5, n2p, packing, 128))
    flat = kv.reshape(5, n2p * packing, 128)
    for h in range(kh):  # combined head 2h is K_h, 2h + 1 is V_h
      test_utils.assert_bitwise_equal(flat[:, 2 * h, :d], k[:, h])
      test_utils.assert_bitwise_equal(flat[:, 2 * h + 1, :d], v[:, h])
    np.testing.assert_array_equal(np.asarray(flat[:, :, d:], np.float32), 0.0)
    k2, v2 = rpa_backend.split_kv(kv, kh, d)
    test_utils.assert_bitwise_equal(k2, k)
    test_utils.assert_bitwise_equal(v2, v)

  def test_merge_kv_rejects_mismatched_operands(self):
    k = jnp.zeros((2, 2, 128), jnp.bfloat16)
    with self.assertRaises(ValueError):
      rpa_backend.merge_kv(k, jnp.zeros((2, 2, 128), jnp.float32))
    with self.assertRaises(ValueError):
      rpa_backend.merge_kv(k, jnp.zeros((2, 1, 128), jnp.bfloat16))

  def test_dense_to_paged_slot_mapping_and_round_trip(self):
    b, s, kh, ps = 2, 32, 2, 16
    rng = _rng(1)
    k = test_utils.random_bf16(rng, (b, s, kh, _D))
    v = test_utils.random_bf16(rng, (b, s, kh, _D))
    paged = rpa_backend.dense_to_paged(k, v, ps)
    self.assertEqual(
        paged.shape,
        rpa_backend.paged_cache_shape(b * s // ps, ps, kh, _D, k.dtype),
    )
    for bb in range(b):
      for slot in (0, 7, 16, 31):
        page = bb * (s // ps) + slot // ps
        test_utils.assert_bitwise_equal(
            paged[page, slot % ps, :, 0, :], k[bb, slot]
        )
        test_utils.assert_bitwise_equal(
            paged[page, slot % ps, :, 1, :], v[bb, slot]
        )
    k2, v2 = rpa_backend.paged_to_dense(paged, b, kh, _D)
    test_utils.assert_bitwise_equal(k2, k)
    test_utils.assert_bitwise_equal(v2, v)
    with self.assertRaisesRegex(ValueError, 'multiple of page_size'):
      rpa_backend.dense_to_paged(k, v, 24)
    with self.assertRaisesRegex(ValueError, 'equal tables'):
      rpa_backend.paged_to_dense(paged, 3, kh, _D)


class MetadataTest(parameterized.TestCase):

  def test_build_metadata(self):
    kv_lens = jnp.array([5, 9], jnp.int32)
    md = rpa_backend.build_metadata(
        batch_size=2,
        seq_len=4,
        pages_per_seq=3,
        kv_lens=kv_lens,
        force_mixed=False,
    )
    np.testing.assert_array_equal(md.kv_lens, [5, 9])
    np.testing.assert_array_equal(md.cu_q_lens, [0, 4, 8])
    np.testing.assert_array_equal(md.page_indices, np.arange(6))
    # Natural split: `T != 1` sequences take the mixed pass ...
    np.testing.assert_array_equal(md.distribution, [0, 0, 2])
    decode = functools.partial(
        rpa_backend.build_metadata,
        batch_size=2,
        seq_len=1,
        pages_per_seq=3,
        kv_lens=kv_lens,
    )
    # ... `T == 1` sequences the decode pass, unless forced to mixed.
    np.testing.assert_array_equal(
        decode(force_mixed=False).distribution, [2, 2, 2]
    )
    np.testing.assert_array_equal(
        decode(force_mixed=True).distribution, [0, 0, 2]
    )
    for a in (md.kv_lens, md.page_indices, md.cu_q_lens, md.distribution):
      self.assertEqual(a.dtype, jnp.int32)
    with self.assertRaisesRegex(ValueError, r'kv_lens'):
      rpa_backend.build_metadata(
          batch_size=3,
          seq_len=1,
          pages_per_seq=1,
          kv_lens=kv_lens,
          force_mixed=True,
      )

  def test_token_rows_and_flatten(self):
    self.assertEqual(rpa_backend.token_rows(1, 1, 256), 256)
    self.assertEqual(rpa_backend.token_rows(2, 4, 256), 256)
    self.assertEqual(rpa_backend.token_rows(4, 64, 256), 256)
    self.assertEqual(rpa_backend.token_rows(4, 65, 256), 512)
    x = jnp.arange(2 * 3 * 4 * 2, dtype=jnp.float32).reshape(2, 3, 4, 2)
    flat = rpa_backend.flatten_tokens(x, 8)
    self.assertEqual(flat.shape, (8, 4, 2))
    np.testing.assert_array_equal(flat[:6], x.reshape(6, 4, 2))
    np.testing.assert_array_equal(flat[6:], 0.0)
    np.testing.assert_array_equal(rpa_backend.unflatten_tokens(flat, 2, 3), x)
    with self.assertRaises(ValueError):
      rpa_backend.flatten_tokens(x, 5)

  def test_kernel_kwargs_and_describe(self):
    pinned = rpa_canonical.canonical_block_size_kwargs()
    self.assertEqual(
        rpa_backend.kernel_kwargs(
            scale=0.5, pin_block_sizes=True, out_dtype=None
        ),
        {'sm_scale': 0.5, **pinned},
    )
    self.assertEqual(
        rpa_backend.kernel_kwargs(
            scale=0.5, pin_block_sizes=False, out_dtype=None
        ),
        {'sm_scale': 0.5},
    )
    self.assertEqual(
        rpa_backend.kernel_kwargs(
            scale=0.5, pin_block_sizes=False, out_dtype=jnp.float32
        ),
        {'sm_scale': 0.5, 'out_dtype': jnp.dtype(jnp.float32)},
    )
    self.assertEqual(
        rpa_backend.describe(),
        {
            'page_size': 16,
            'token_bucket': 256,
            'block_sizes': (128, 512, 128, 512),
            'distribution': '[0, 0, B] (forced mixed)',
            'out_dtype': 'kernel default',
        },
    )
    self.assertEqual(
        rpa_backend.describe(
            page_size=128,
            pin_block_sizes=False,
            force_mixed=False,
            out_dtype=jnp.float32,
        ),
        {
            'page_size': 128,
            'token_bucket': 256,
            'block_sizes': 'kernel heuristic',
            'distribution': 'natural',
            'out_dtype': 'float32',
        },
    )

  def test_resolve_kernel_precedence_and_import_error(self):
    def sentinel(*args, **kwargs):
      del args, kwargs

    self.assertIs(rpa_backend.resolve_kernel(sentinel), sentinel)
    with mock.patch.object(rpa_backend, 'KERNEL', _REF):
      self.assertIs(rpa_backend.resolve_kernel(None), _REF)
      self.assertIs(rpa_backend.resolve_kernel(sentinel), sentinel)
    with (
        mock.patch.object(rpa_backend, 'KERNEL', None),
        mock.patch.object(rpa_backend, '_loaded_kernel', None),
        mock.patch.object(
            rpa_backend, 'KERNEL_MODULE', 'tunix_no_such_module.kernel'
        ),
    ):
      with self.assertRaisesRegex(ImportError, 'not importable'):
        rpa_backend.resolve_kernel(None)


class ReferenceKernelTest(absltest.TestCase):

  def test_reference_kernel_contract(self):
    """Hand-built ragged metadata: context rows, two lengths, an idle slot."""
    qh, kh, ps, pps = 4, 2, 16, 2
    max_num_seqs, rows, scale = 3, 32, 0.2
    # Slot 0: 4 context rows already in the cache + 5 new rows (kv_len 9);
    # slot 1: 3 new rows, no context; slot 2: idle (`distribution[2] == 2`).
    cu_q_lens = jnp.array([0, 5, 8, 8], jnp.int32)
    kv_lens = jnp.array([9, 3, 0], jnp.int32)
    page_indices = jnp.arange(max_num_seqs * pps, dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], jnp.int32)
    rng = _rng(2)
    q = jnp.asarray(rng.standard_normal((rows, qh, _D)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((rows, kh, _D)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((rows, kh, _D)), jnp.float32)
    cache_shape = rpa_backend.paged_cache_shape(
        max_num_seqs * pps, ps, kh, _D, jnp.float32
    )
    cache0 = jnp.asarray(rng.standard_normal(cache_shape), jnp.float32)
    out, cache1 = jax.jit(functools.partial(_REF, sm_scale=scale))(
        q, k, v, cache0, kv_lens, page_indices, cu_q_lens, distribution
    )

    def dense(cache, slot):
      pages = cache[slot * pps : (slot + 1) * pps].reshape(
          pps * ps, *cache.shape[2:]
      )
      return rpa_backend.split_kv(pages, kh, _D)  # [S, KH, D] each

    # Slot 0: rows 0..4 sit at positions 4..8 and see the 4 context rows.
    k0, v0 = dense(cache0, 0)
    k_all = k0.at[4:9].set(k[0:5])
    v_all = v0.at[4:9].set(v[0:5])
    pos = jnp.arange(4, 9, dtype=jnp.int32)[None]
    want0 = _dense_reference(
        q[None, 0:5], k_all[None], v_all[None], pos, scale=scale
    )
    np.testing.assert_allclose(out[0:5], want0[0], rtol=1e-5, atol=1e-5)
    # Slot 1: rows 5..7 at positions 0..2 (its context slots are not visible).
    k1, v1 = dense(cache0, 1)
    k_all = k1.at[0:3].set(k[5:8])
    v_all = v1.at[0:3].set(v[5:8])
    pos = jnp.arange(3, dtype=jnp.int32)[None]
    want1 = _dense_reference(
        q[None, 5:8], k_all[None], v_all[None], pos, scale=scale
    )
    np.testing.assert_allclose(out[5:8], want1[0], rtol=1e-5, atol=1e-5)
    # Rows that belong to no sequence come back as zeros.
    np.testing.assert_array_equal(out[8:], 0.0)
    # The cache: new rows written at their positions, everything else kept.
    k0n, v0n = dense(cache1, 0)
    test_utils.assert_bitwise_equal(k0n[4:9], k[0:5])
    test_utils.assert_bitwise_equal(v0n[4:9], v[0:5])
    test_utils.assert_bitwise_equal(k0n[:4], k0[:4])
    test_utils.assert_bitwise_equal(k0n[9:], k0[9:])
    k1n, _ = dense(cache1, 1)
    test_utils.assert_bitwise_equal(k1n[0:3], k[5:8])
    test_utils.assert_bitwise_equal(k1n[3:], k1[3:])
    test_utils.assert_bitwise_equal(cache1[2 * pps :], cache0[2 * pps :])


class AdapterTest(parameterized.TestCase):
  """The adapter around the contract replica, f32 and forward-only."""

  @parameterized.named_parameters(
      ('b1_t5_ps16', 1, 5, 16),
      ('b2_t20_ps16', 2, 20, 16),
      ('b3_t7_ps128', 3, 7, 128),
      ('b2_t130_ps16_two_buckets', 2, 130, 16),
  )
  def test_cacheless_matches_dense_causal_reference(self, b, t, page_size):
    q, k, v = _operands(_rng(3), b, t, 4, 2)
    scale = _D**-0.5
    out = rpa_backend.attention_cacheless(
        q,
        k,
        v,
        scale=scale,
        page_size=page_size,
        kernel=_REF,
        differentiable=False,
    )
    self.assertEqual(out.shape, q.shape)
    self.assertEqual(out.dtype, q.dtype)
    np.testing.assert_allclose(
        out, _causal_reference(q, k, v, scale=scale), rtol=1e-5, atol=1e-5
    )

  @parameterized.named_parameters(
      ('ps16_cache32', 16, 32), ('ps128_cache128', 128, 128)
  )
  def test_cached_prefill_then_decode_matches_cacheless(
      self, page_size, cache_size
  ):
    b, l_prompt, t_total, qh, kh = 2, 5, 9, 4, 2
    q, k, v = _operands(_rng(4), b, t_total, qh, kh)
    scale = _D**-0.5
    cache = _dense_cache(b, cache_size, kh, _D, q.dtype)
    cache, out_prefill = _cached(
        cache,
        q[:, :l_prompt],
        k[:, :l_prompt],
        v[:, :l_prompt],
        scale=scale,
        page_size=page_size,
    )
    pages = b * cache_size // page_size
    self.assertEqual(
        cache['k'].shape,
        rpa_backend.paged_cache_shape(pages, page_size, kh, _D, q.dtype),
    )
    self.assertEqual(cache['v'].shape, ())
    np.testing.assert_array_equal(cache['end_index'], [l_prompt] * b)
    outs = [out_prefill]
    for pos in range(l_prompt, t_total):
      step = slice(pos, pos + 1)
      cache, out_step = _cached(
          cache,
          q[:, step],
          k[:, step],
          v[:, step],
          scale=scale,
          page_size=page_size,
      )
      outs.append(out_step)
    np.testing.assert_array_equal(cache['end_index'], [t_total] * b)
    got = jnp.concatenate(outs, axis=1)
    want = _cacheless(q, k, v, scale=scale, page_size=page_size)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        got, _causal_reference(q, k, v, scale=scale), rtol=1e-5, atol=1e-5
    )
    # The kernel copies the new rows into the cache: bitwise the inputs, the
    # slots past the sequence still zero.
    k_stored, v_stored = rpa_backend.paged_to_dense(cache['k'], b, kh, _D)
    test_utils.assert_bitwise_equal(k_stored[:, :t_total], k)
    test_utils.assert_bitwise_equal(v_stored[:, :t_total], v)
    np.testing.assert_array_equal(k_stored[:, t_total:], 0.0)
    np.testing.assert_array_equal(v_stored[:, t_total:], 0.0)

  def test_cached_positions_follow_each_sequence_end_index(self):
    """`kv_len = end_index + T` per sequence, not one length for the batch."""
    b, t_fill, qh, kh = 2, 6, 4, 2
    q, k, v = _operands(_rng(5), b, t_fill + 1, qh, kh)
    scale = _D**-0.5
    cache = _dense_cache(b, 32, kh, _D, q.dtype)
    cache, _ = _cached(
        cache, q[:, :t_fill], k[:, :t_fill], v[:, :t_fill], scale=scale
    )
    # Sequence 0 is rewound to position 3: its next row overwrites slot 3 and
    # sees `[0, 4)`; sequence 1 continues at position 6 and sees `[0, 7)`.
    cache['end_index'] = jnp.array([3, 6], jnp.int32)
    new_cache, out = _cached(
        cache, q[:, -1:], k[:, -1:], v[:, -1:], scale=scale
    )
    np.testing.assert_array_equal(new_cache['end_index'], [4, 7])
    k_all = jnp.concatenate([k[:, :t_fill], jnp.zeros_like(k[:, :1])], axis=1)
    v_all = jnp.concatenate([v[:, :t_fill], jnp.zeros_like(v[:, :1])], axis=1)
    k_all = k_all.at[0, 3].set(k[0, -1]).at[1, 6].set(k[1, -1])
    v_all = v_all.at[0, 3].set(v[0, -1]).at[1, 6].set(v[1, -1])
    pos = jnp.array([[3], [6]], jnp.int32)
    want = _dense_reference(q[:, -1:], k_all, v_all, pos, scale=scale)
    np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-5)
    k_stored, _ = rpa_backend.paged_to_dense(new_cache['k'], b, kh, _D)
    test_utils.assert_bitwise_equal(k_stored[:, :7], k_all)

  def test_batch_subset_matches_full_batch(self):
    q, k, v = _operands(_rng(6), 3, 6, 4, 2)
    scale = _D**-0.5
    full = _cacheless(q, k, v, scale=scale)
    one = _cacheless(q[1:2], k[1:2], v[1:2], scale=scale)
    np.testing.assert_allclose(one[0], full[1], rtol=1e-6, atol=1e-6)

  def test_tp4_head_sharded_call_matches_single_device(self):
    b, t, qh, kh = 2, 6, 8, 4
    q, k, v = _operands(_rng(7), b, t, qh, kh)
    scale = _D**-0.5
    want = _cacheless(q, k, v, scale=scale)
    mesh = _tp_mesh(4)

    @jax.jit
    def cacheless(q, k, v):
      return _cacheless(q, k, v, scale=scale, mesh=mesh)

    @jax.jit
    def cached(cache, q, k, v):
      return _cached(cache, q, k, v, scale=scale, mesh=mesh)

    with mesh:
      got = cacheless(q, k, v)
      new_cache, out_prefill = cached(
          _dense_cache(b, 32, kh, _D, q.dtype), q, k, v
      )
      with self.assertRaisesRegex(ValueError, 'TP degree'):
        # KH=2 cannot be split over 4 ranks (no replicated fallback).
        _cacheless(
            q[:, :, :4], k[:, :, :2], v[:, :, :2], scale=scale, mesh=mesh
        )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_prefill, want, rtol=1e-5, atol=1e-5)
    self.assertEqual(
        new_cache['k'].shape,
        rpa_backend.paged_cache_shape(b * 32 // 16, 16, kh, _D, q.dtype),
    )
    k_stored, _ = rpa_backend.paged_to_dense(new_cache['k'], b, kh, _D)
    test_utils.assert_bitwise_equal(k_stored[:, :t], k)

  def test_engine_form_reaches_the_kernel(self):
    """Pinned blocks, forced mixed, the 256-row bucket, `kv_len`, `out_dtype`."""
    rec = _RecordingKernel()
    b, t, qh, kh = 2, 4, 4, 2
    q, k, v = _operands(_rng(8), b, t, qh, kh)
    rpa_backend.attention_cacheless(
        q, k, v, scale=0.1, kernel=rec, differentiable=False
    )
    self.assertLen(rec.calls, 1)
    shapes, kwargs, metadata = rec.calls[0]
    self.assertEqual(shapes[0], ((256, qh, _D), 'float32'))
    self.assertEqual(shapes[1], ((256, kh, _D), 'float32'))
    self.assertEqual(shapes[3], ((b * 1, 16, 2 * kh, 1, _D), 'float32'))
    self.assertEqual(
        kwargs, {'sm_scale': 0.1, **rpa_canonical.canonical_block_size_kwargs()}
    )
    kv_lens, page_indices, cu_q_lens, distribution = metadata
    np.testing.assert_array_equal(kv_lens, [t, t])
    np.testing.assert_array_equal(page_indices, [0, 1])
    np.testing.assert_array_equal(cu_q_lens, [0, t, 2 * t])
    np.testing.assert_array_equal(distribution, [0, 0, b])

    # A decode step with the Leave-One-Out knobs: natural split (decode
    # pass), the kernel's own block sizes, f32 accumulators.
    rec.calls.clear()
    cache = _dense_cache(b, 32, kh, _D, q.dtype)
    cache['end_index'] = jnp.array([5, 5], jnp.int32)
    rpa_backend.attention_cached(
        cache,
        q[:, :1],
        k[:, :1],
        v[:, :1],
        scale=0.1,
        kernel=rec,
        differentiable=False,
        pin_block_sizes=False,
        force_mixed=False,
        out_dtype=jnp.float32,
    )
    self.assertLen(rec.calls, 1)
    shapes, kwargs, metadata = rec.calls[0]
    self.assertEqual(shapes[0], ((256, qh, _D), 'float32'))
    self.assertEqual(shapes[3], ((b * 2, 16, 2 * kh, 1, _D), 'float32'))
    self.assertEqual(
        kwargs, {'sm_scale': 0.1, 'out_dtype': jnp.dtype(jnp.float32)}
    )
    kv_lens, page_indices, cu_q_lens, distribution = metadata
    np.testing.assert_array_equal(kv_lens, [6, 6])
    np.testing.assert_array_equal(page_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(cu_q_lens, [0, 1, 2])
    np.testing.assert_array_equal(distribution, [b, b, b])

  def test_fail_closed_static_checks(self):
    q, k, v = _operands(_rng(9), 2, 4, 4, 2)
    with self.assertRaisesRegex(NotImplementedError, 'packing-2'):
      _cacheless(q, k, v, scale=0.1, differentiable=True)  # f32 VJP
    with self.assertRaisesRegex(ValueError, 'head_dim'):
      _cacheless(q[..., :64], k[..., :64], v[..., :64], scale=0.1)
    with self.assertRaisesRegex(ValueError, 'not a multiple'):
      _cacheless(q[:, :, :3], k, v, scale=0.1)
    with self.assertRaisesRegex(ValueError, 'pinned bkv'):
      _cacheless(q, k, v, scale=0.1, page_size=24)
    with self.assertRaisesRegex(ValueError, r'expected \[B, T, H, D\]'):
      _cacheless(q[0], k[0], v[0], scale=0.1)
    bq, bk, bv = (x.astype(jnp.bfloat16) for x in (q, k, v))
    with self.assertRaisesRegex(ValueError, 'multiple of 2'):
      _cacheless(
          bq,
          bk,
          bv,
          scale=0.1,
          page_size=3,
          pin_block_sizes=False,
          differentiable=True,
      )
    with self.assertRaisesRegex(ValueError, 'dtypes differ'):
      _cacheless(bq, k, v, scale=0.1)
    with self.assertRaisesRegex(ValueError, 'width of the query dtype'):
      _cacheless(bq, bk, bv, scale=0.1, out_dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, 'multiple of page_size'):
      _cached(_dense_cache(2, 20, 2, _D, q.dtype), q, k, v, scale=0.1)
    wrong_batch = _dense_cache(3, 32, 2, _D, q.dtype)
    wrong_batch['end_index'] = jnp.zeros((2,), jnp.int32)
    with self.assertRaisesRegex(ValueError, 'cache batch'):
      _cached(wrong_batch, q, k, v, scale=0.1)
    cache = _dense_cache(2, 32, 2, _D, q.dtype)
    cache['end_index'] = jnp.zeros((3,), jnp.int32)
    with self.assertRaisesRegex(ValueError, 'end_index'):
      _cached(cache, q, k, v, scale=0.1)
    paged, _ = _cached(_dense_cache(2, 32, 2, _D, q.dtype), q, k, v, scale=0.1)
    with self.assertRaisesRegex(ValueError, 'page_size'):
      _cached(paged, q, k, v, scale=0.1, page_size=32)
    paged['k'] = paged['k'][0, 0]
    with self.assertRaisesRegex(ValueError, 'cache rank'):
      _cached(paged, q, k, v, scale=0.1)


class GlueTest(absltest.TestCase):

  def test_attention_core_rejects_rpa_fwd(self):
    self.assertEqual(
        canon_attention.ATTENTION_FWD_MODES,
        (canon_attention.XLA_FWD, canon_attention.PALLAS_FWD),
    )
    self.assertEqual(
        canon_attention.ATTENTION_FWD_CHOICES,
        canon_attention.ATTENTION_FWD_MODES + (canon_attention.RPA_FWD,),
    )
    q = jnp.zeros((1, 2, 2, 4, _D), jnp.bfloat16)
    k = jnp.zeros((1, 2, 4, _D), jnp.bfloat16)
    mask = jnp.ones((1, 4, 4), bool)
    with self.assertRaisesRegex(ValueError, 'engine kernel'):
      canon_attention.attention_core(
          q, k, k, mask, scale=0.1, fwd=canon_attention.RPA_FWD
      )
    self.assertFalse(qwen3_canon.rpa_attention_selected())
    with mock.patch.object(
        canon_attention, 'ATTENTION_FWD', canon_attention.RPA_FWD
    ):
      self.assertTrue(qwen3_canon.rpa_attention_selected())

  def test_model_glue_contract_and_dispatch(self):
    b, t, qh, kh = 1, 4, 4, 2
    q, k, v = _operands(_rng(10), b, t, qh, kh, dtype=jnp.bfloat16)
    idx = jnp.arange(t)
    mask = jnp.broadcast_to((idx[:, None] >= idx[None, :])[None], (b, t, t))
    with mock.patch.object(rpa_backend, 'KERNEL', _REF):
      with self.assertRaisesRegex(ValueError, 'attn_mask=None'):
        qwen3_canon.rpa_attention_cacheless(q, k, v, scale=0.1, attn_mask=None)
      with self.assertRaisesRegex(ValueError, 'segment_ids'):
        qwen3_canon.rpa_attention_cacheless(
            q,
            k,
            v,
            scale=0.1,
            attn_mask=mask,
            segment_ids=jnp.zeros((b, t), jnp.int32),
        )
      cache = _dense_cache(b, 16, kh, _D, q.dtype)
      with self.assertRaisesRegex(ValueError, 'attn_mask=None'):
        qwen3_canon.rpa_attention_cached(
            cache, q, k, v, scale=0.1, attn_mask=None
        )
      # The happy paths (no mesh active) are the adapter calls.
      out = qwen3_canon.rpa_attention_cacheless(
          q, k, v, scale=0.1, attn_mask=mask
      )
      test_utils.assert_bitwise_equal(
          out, rpa_backend.attention_cacheless(q, k, v, scale=0.1)
      )
      new_cache, out_cached = qwen3_canon.rpa_attention_cached(
          cache, q, k, v, scale=0.1, attn_mask=mask[:, :, :1].repeat(16, 2)
      )
      np.testing.assert_array_equal(new_cache['end_index'], [t])
      np.testing.assert_allclose(
          test_utils.as_f32(out_cached),
          test_utils.as_f32(out),
          rtol=2e-2,
          atol=2e-2,
      )


class VjpTest(absltest.TestCase):
  """The chunked replica VJP around the adapter (bf16 operands)."""

  def test_bf16_grads_match_dense_autodiff(self):
    b, t, qh, kh = 2, 8, 4, 2
    rng = _rng(11)
    q, k, v = _operands(rng, b, t, qh, kh, dtype=jnp.bfloat16)
    scale = _D**-0.5
    w = jnp.asarray(rng.standard_normal((b, t, qh, _D)), jnp.float32)

    def loss_rpa(q, k, v):
      out = rpa_backend.attention_cacheless(q, k, v, scale=scale, kernel=_REF)
      return jnp.sum(out.astype(jnp.float32) * w)

    def loss_dense(q, k, v):
      out = _causal_reference(q, k, v, scale=scale)
      return jnp.sum(out.astype(jnp.float32) * w)

    with mock.patch.dict(os.environ, {rpa_diff_chunked.MAX_SEQS_ENV: str(b)}):
      g_rpa = jax.jit(jax.grad(loss_rpa, argnums=(0, 1, 2)))(q, k, v)
    g_dense = jax.jit(jax.grad(loss_dense, argnums=(0, 1, 2)))(q, k, v)
    for name, got, want in zip('qkv', g_rpa, g_dense):
      self.assertEqual(got.shape, want.shape)
      self.assertEqual(got.dtype, want.dtype)
      ratio, cosine = _grad_stats(got, want)
      self.assertAlmostEqual(
          ratio, 1.0, delta=_GRAD_RATIO_TOL, msg=f'd{name}: {ratio}'
      )
      self.assertGreaterEqual(
          cosine, _GRAD_COSINE_MIN, msg=f'd{name}: {cosine}'
      )

  def test_unroll_limit_drops_the_other_sequences_and_warns(self):
    """`CANON_VJP2_MAX_SEQS` unset: only sequence 0 gets cotangents."""
    b, t, qh, kh = 2, 4, 4, 2
    q, k, v = _operands(_rng(12), b, t, qh, kh, dtype=jnp.bfloat16)

    def loss(q, k, v):
      out = rpa_backend.attention_cacheless(q, k, v, scale=0.1, kernel=_REF)
      return jnp.sum(out.astype(jnp.float32))

    env = {
        name: value
        for name, value in os.environ.items()
        if name != rpa_diff_chunked.MAX_SEQS_ENV
    }
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(rpa_backend, '_warned', set()),
        mock.patch.object(rpa_backend.logging, 'warning') as warning,
    ):
      grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
      self.assertEqual(warning.call_count, 1)
      jax.grad(loss, argnums=(0, 1, 2))(q, k, v)  # same B: warned already
      self.assertEqual(warning.call_count, 1)
      # Forward-only programs and batches inside the limit do not warn.
      rpa_backend.attention_cacheless(
          q, k, v, scale=0.1, kernel=_REF, differentiable=False
      )
      jax.grad(loss, argnums=(0,))(q[:1], k[:1], v[:1])
      self.assertEqual(warning.call_count, 1)
    for g in grads:
      g = np.asarray(g, np.float32)
      self.assertGreater(np.abs(g[0]).max(), 0.0)
      np.testing.assert_array_equal(g[1], 0.0)


def _model_config(tp_size: int) -> qwen3_stock.ModelConfig:
  """`qwen3_model_canon_test._make_test_config(use_tied_embedding=False)`."""
  return qwen3_stock.ModelConfig(
      num_layers=2,
      vocab_size=1024 * tp_size,
      embed_dim=256,
      hidden_dim=512,
      num_heads=max(4, 2 * tp_size),
      head_dim=128,
      num_kv_heads=max(2, tp_size),
      norm_eps=1e-6,
      rope_theta=1_000_000,
      use_tied_embedding=False,
      shd_config=qwen3_canon.get_tp_sharding_config('tp'),
      dtype=jnp.dtype(jnp.bfloat16),
      param_dtype=jnp.dtype(jnp.bfloat16),
  )


def _param_grads(model, tokens: jax.Array, *, use_canon_logsoftmax: bool):
  """`d(-mean token logprob) / d params` of a cacheless causal prefill."""
  graphdef, params, other_state = nnx.split(model, nnx.Param, ...)
  b, l_total = tokens.shape
  positions = jnp.broadcast_to(
      jnp.arange(l_total, dtype=jnp.int32)[None, :], (b, l_total)
  )
  idx = jnp.arange(l_total, dtype=jnp.int32)
  causal_mask = jnp.broadcast_to(
      (idx[:, None] >= idx[None, :])[None, :, :], (b, l_total, l_total)
  )

  def loss_fn(param_state):
    m = nnx.merge(graphdef, param_state, other_state)
    logits, _ = m(tokens, positions, None, causal_mask)
    lps, _ = qwen3_canon.next_token_logprobs(
        logits,
        tokens,
        use_canon_logsoftmax=use_canon_logsoftmax,
        exact_residual_dtype=model.config.dtype,
    )
    return -jnp.mean(lps)

  return jax.jit(jax.grad(loss_fn))(params)


class ModelTest(absltest.TestCase):
  """`Qwen3Canon` with `ATTENTION_FWD = RPA_FWD` on the contract replica.

  Model-level plumbing: the paged `LayerCache` through the sampler's prefill
  + `lax.scan` decode, the cacheless scoring and learner programs, the TP=4
  `shard_map`s and the VJP wrapper under `jax.grad`. The replica is f32
  arithmetic with the kernel's bookkeeping, so the Zero-TIM columns are
  checked to a tolerance here (the shapes of its reductions differ between
  the programs); the bitwise question is `rpa_backend_tpu_test`'s.
  """

  def setUp(self):
    super().setUp()
    # The jit caches hold traces of the default attention; the knobs are read
    # at trace time.
    jax.clear_caches()
    self.enter_context(
        mock.patch.object(
            canon_attention, 'ATTENTION_FWD', canon_attention.RPA_FWD
        )
    )
    self.enter_context(mock.patch.object(rpa_backend, 'KERNEL', _REF))
    self.addCleanup(jax.clear_caches)

  def test_model_suite_rows_rpa_fwd(self):
    tp_size = 4 if len(jax.devices()) >= 4 else 1
    rows = qwen3_canon_benchmark.run_benchmark_suite(
        preset='mini_qwen3',
        batch_size=2,
        prompt_len=4,
        max_new_tokens=3,
        cache_size=32,
        num_iters=1,
        run_ablations=False,
        include_fwdbwd=True,
        tp_size=tp_size,
    )
    print(
        f'[rpa_fwd suite, tp={tp_size}]\n'
        + qwen3_canon_benchmark.format_markdown_report(rows),
        flush=True,
    )
    by_name = {r.name: r for r in rows}
    self.assertEqual(
        list(by_name),
        [
            'stock_qwen3 (model.py)',
            'canon_all_disabled',
            'canon_all_enabled (Zero-TIM)',
        ],
    )
    # `canon_all_disabled` does not route through the RPA backend
    # (`use_canon_attention=False`): still the stock program, bitwise.
    disabled = by_name['canon_all_disabled']
    self.assertEqual(disabled.vs_stock_max_abs_diff, 0.0)
    self.assertEqual(disabled.decode_vs_prefill_max_diff, 0.0)
    self.assertEqual(disabled.batch1_vs_batchn_max_diff, 0.0)
    self.assertEqual(disabled.fwd_vs_fwdbwd_max_diff, 0.0)
    enabled = by_name['canon_all_enabled (Zero-TIM)']
    for field in (
        'decode_vs_prefill_max_diff',
        'batch1_vs_batchn_max_diff',
        'fwd_vs_fwdbwd_max_diff',
    ):
      self.assertLessEqual(getattr(enabled, field), 0.1, msg=field)
    self.assertLess(enabled.vs_stock_max_abs_diff, 1.0)
    self.assertTrue(np.isfinite(enabled.vs_canon_max_abs_diff))
    for row in rows:
      for field in (
          'sample_latency_ms',
          'prefill_latency_ms',
          'fwdbwd_latency_ms',
      ):
        self.assertGreater(getattr(row, field), 0.0, msg=f'{row.name}.{field}')

  def test_model_param_grads_rpa_fwd_tp4(self):
    """TP=4 parameter gradients vs stock `Qwen3` with the replica VJP.

    The bugs this guards against are integer factors of TP in the
    `check_vma=False` `shard_map` of the backend (cotangent scaling); the
    thresholds leave room for the replica's rounding (the stock model's
    attention is bf16 XLA, the replica f32).
    """
    tp_size = 4
    cfg = _model_config(tp_size)
    env = {rpa_diff_chunked.MAX_SEQS_ENV: '2'}
    with mock.patch.dict(os.environ, env), _tp_mesh(tp_size):
      stock = qwen3_stock.Qwen3(cfg, rngs=nnx.Rngs(params=0))
      qwen3_canon.init_non_zero_mlp_weights(stock, seed=7)
      canon = qwen3_canon.Qwen3Canon(
          cfg,
          rngs=nnx.Rngs(params=0),
          canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
      )
      qwen3_canon.copy_weights(stock, canon)
      tokens = jax.random.randint(
          jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size, dtype=jnp.int32
      )
      g_stock = _param_grads(stock, tokens, use_canon_logsoftmax=False)
      g_canon = _param_grads(canon, tokens, use_canon_logsoftmax=True)
    ref = jax.tree_util.tree_leaves_with_path(g_stock)
    test = jax.tree_util.tree_leaves_with_path(g_canon)
    self.assertLen(test, len(ref))
    worst_ratio_dev, min_cosine = 0.0, 1.0
    for (p_ref, r), (p_test, t) in zip(ref, test):
      path = jax.tree_util.keystr(p_ref)
      self.assertEqual(path, jax.tree_util.keystr(p_test))
      ratio, cosine = _grad_stats(t, r)
      worst_ratio_dev = max(worst_ratio_dev, abs(ratio - 1.0))
      min_cosine = min(min_cosine, cosine)
      self.assertAlmostEqual(
          ratio,
          1.0,
          delta=_GRAD_RATIO_TOL,
          msg=f'{path}: grad-norm ratio {ratio:.4f}',
      )
      self.assertGreaterEqual(
          cosine, _GRAD_COSINE_MIN, msg=f'{path}: grad cosine {cosine:.5f}'
      )
    print(
        f'[rpa_fwd grad-check tp={tp_size}] {len(ref)} leaves, worst'
        f' |ratio-1| = {worst_ratio_dev:.2e}, min cosine = {min_cosine:.6f}',
        flush=True,
    )


if __name__ == '__main__':
  absltest.main()
