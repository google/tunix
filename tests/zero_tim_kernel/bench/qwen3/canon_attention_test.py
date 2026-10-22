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

"""Tests for `canon_attention` (Pallas forward and flash backward)."""

from __future__ import annotations

from collections.abc import Callable
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.zero_tim_kernel import test_utils
from tunix.zero_tim_kernel.bench.qwen3 import canon_attention


def setUpModule():
  test_utils.configure_cpu()


_D = 64


def _interpret() -> bool:
  """Pallas interpret mode on CPU/v6e; the real Mosaic kernel on TPU v4/v5p."""
  return test_utils.is_interpret_default()


def _causal_mask(b: int, t: int, s: int) -> jax.Array:
  """Causal mask for the last `t` of `s` positions (prefill / learner)."""
  q_pos = np.arange(t)[:, None] + (s - t)
  k_pos = np.arange(s)[None, :]
  return jnp.asarray(np.broadcast_to(k_pos <= q_pos, (b, t, s)))


def _random_mask(
    rng: np.random.Generator, b: int, t: int, s: int, density: float = 0.6
) -> jax.Array:
  return jnp.asarray(rng.random((b, t, s)) < density)


def _inputs(rng, b, kh, g, t, s, d=_D):
  q = test_utils.random_bf16(rng, (b, kh, g, t, d))
  k = test_utils.random_bf16(rng, (b, kh, s, d))
  v = test_utils.random_bf16(rng, (b, kh, s, d))
  do = test_utils.random_bf16(rng, (b, kh, g, t, d))
  return q, k, v, do


def _dense_ref_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    *,
    scale: float,
) -> jax.Array:
  """Dense f32 reference attention for numeric/gradient verification."""
  q_f32 = q.astype(jnp.float32) * jnp.float32(scale * canon_attention.LOG2_E)
  k_f32 = k.astype(jnp.float32)
  v_f32 = v.astype(jnp.float32)
  qk = jnp.einsum(
      'bhgtd,bhsd->bhgts', q_f32, k_f32, preferred_element_type=jnp.float32
  )
  mask5 = mask[:, None, None, :, :]
  qk_masked = jnp.where(mask5, qk, -jnp.inf)
  valid_any = jnp.any(mask5, axis=-1, keepdims=True)
  m = jnp.max(qk_masked, axis=-1, keepdims=True)
  m_safe = jnp.where(jnp.isneginf(m), 0.0, m)
  p = jnp.where(mask5, jnp.exp2(qk_masked - m_safe), 0.0)
  denom = jnp.where(valid_any, jnp.sum(p, axis=-1, keepdims=True), 1.0)
  out = jnp.einsum(
      'bhgts,bhsd->bhgtd', p / denom, v_f32, preferred_element_type=jnp.float32
  )
  return out.astype(q.dtype)


def _core(
    mask, *, bwd=None, fwd=None, kv_block_size=512, d=_D
) -> Callable[..., jax.Array]:
  scale = 1.0 / np.sqrt(d)
  return lambda q, k, v: canon_attention.attention_core(
      q,
      k,
      v,
      mask,
      scale=scale,
      kv_block_size=kv_block_size,
      interpret=_interpret(),
      bwd=bwd,
      fwd=fwd,
  )


def _forward(
    mask, *, kv_block_size=512, d=_D
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
  scale = 1.0 / np.sqrt(d)
  return jax.jit(
      lambda q, k, v: canon_attention.flash_fwd_pallas(
          q,
          k,
          v,
          mask,
          scale=scale,
          kv_block_size=kv_block_size,
          interpret=_interpret(),
      )
  )


@functools.partial(jax.jit, static_argnums=0)
def _vjp(fn, q, k, v, do):
  out, pullback = jax.vjp(fn, q, k, v)
  return (out, *pullback(do))


def _grads(mask, q, k, v, do, *, bwd=None, fwd=None, kv_block_size=512):
  fn = _core(mask, bwd=bwd, fwd=fwd, kv_block_size=kv_block_size, d=q.shape[-1])
  return _vjp(fn, q, k, v, do)


def _ref_grads(mask, q, k, v, do):
  scale = 1.0 / np.sqrt(q.shape[-1])
  fn = lambda q_in, k_in, v_in: _dense_ref_attention(
      q_in, k_in, v_in, mask, scale=scale
  )
  return _vjp(fn, q, k, v, do)


GEOMETRIES = (
    # name, b, kh, g, t, s, kv_block_size
    ('prefill_t64', 2, 2, 2, 64, 64, 512),
    ('decode_like_t18_s70', 1, 1, 4, 18, 70, 512),
    ('t129_s129_g2', 2, 1, 2, 129, 129, 512),
    ('t256_s256', 1, 2, 2, 256, 256, 512),
    ('t384_three_query_blocks', 1, 1, 2, 384, 384, 512),
    ('s600_two_kv_blocks', 1, 1, 2, 64, 600, 512),
    ('bkv128_s256', 1, 1, 2, 64, 256, 128),
)


class CanonAttentionTest(parameterized.TestCase):

  def _assert_close(
      self,
      actual,
      expected,
      *,
      name: str,
      rtol: float,
      cos_min: float = 0.999,
      ratio_tol: float = 0.02,
  ) -> None:
    a = test_utils.as_f32(actual)
    e = test_utils.as_f32(expected)
    self.assertEqual(a.shape, e.shape, name)
    self.assertTrue(np.all(np.isfinite(a)), f'{name}: non-finite values')
    na = float(np.linalg.norm(a))
    ne = float(np.linalg.norm(e))
    self.assertGreater(ne, 0.0, f'{name}: all-zero reference')
    cos = float(np.dot(a.ravel(), e.ravel()) / (na * ne))
    self.assertGreaterEqual(cos, cos_min, f'{name}: cos={cos}')
    self.assertLessEqual(
        abs(na / ne - 1.0), ratio_tol, f'{name}: norm ratio={na / ne}'
    )
    np.testing.assert_allclose(
        a, e, rtol=rtol, atol=rtol * float(np.max(np.abs(e))), err_msg=name
    )

  @parameterized.named_parameters(*GEOMETRIES)
  def test_forward_is_untouched_by_vjp(self, b, kh, g, t, s, kv_block_size):
    self.assertEqual(canon_attention.ATTENTION_FWD, canon_attention.PALLAS_FWD)
    rng = np.random.default_rng(1)
    q, k, v, _ = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    expected, m_p, l_p = _forward(mask, kv_block_size=kv_block_size)(q, k, v)
    self.assertEqual(expected.shape, (b, kh, g, t, _D))
    tp = canon_attention.round_up(t, canon_attention.FWD_QUERY_BLOCK)
    self.assertEqual(m_p.shape, (b, kh, g, tp))
    self.assertEqual(l_p.shape, (b, kh, g, tp))
    out = jax.jit(_core(mask, kv_block_size=kv_block_size))(q, k, v)
    test_utils.assert_bitwise_equal(out, expected)
    out_vjp, *_ = _grads(mask, q, k, v, q, kv_block_size=kv_block_size)
    test_utils.assert_bitwise_equal(out_vjp, expected)

  @parameterized.named_parameters(('d64', 64), ('d128', 128))
  def test_pallas_fwd_is_invariant_to_query_length_batch_and_cache(self, d):
    """The contract: one row, one key set -> one result, in every program."""
    b, kh, g, t, s = 2, 1, 2, 256, 256
    rng = np.random.default_rng(9)
    q, k, v, _ = _inputs(rng, b, kh, g, t, s, d=d)
    mask = _causal_mask(b, t, s)
    fwd = lambda m, *xs: _forward(m, d=d)(*xs)[0]
    full = fwd(mask, q, k, v)
    for p in (0, 1, 63, 64, 127, 128, 129, 200, 255):
      row = fwd(mask[:, p : p + 1, :], q[:, :, :, p : p + 1], k, v)
      test_utils.assert_bitwise_equal(row[:, :, :, 0], full[:, :, :, p])
    one = fwd(mask[:1], q[:1], k[:1], v[:1])
    test_utils.assert_bitwise_equal(one, full[:1])
    for s_cache in (384, 600):
      k_c = test_utils.random_bf16(rng, (b, kh, s_cache, d))
      v_c = test_utils.random_bf16(rng, (b, kh, s_cache, d))
      k_c = k_c.at[:, :, :s].set(k)
      v_c = v_c.at[:, :, :s].set(v)
      mask_c = jnp.pad(mask, ((0, 0), (0, 0), (0, s_cache - s)))
      test_utils.assert_bitwise_equal(fwd(mask_c, q, k_c, v_c), full)
      p = 200
      row = fwd(mask_c[:, p : p + 1], q[:, :, :, p : p + 1], k_c, v_c)
      test_utils.assert_bitwise_equal(row[:, :, :, 0], full[:, :, :, p])

  def test_pallas_fwd_row_pass_through_and_sub_block_skipping_are_exact(self):
    b, kh, g, t, s = 1, 1, 2, 128, 256
    rng = np.random.default_rng(10)
    q, k, v, _ = _inputs(rng, b, kh, g, t, s)
    fwd = lambda m, *xs: _forward(m)(*xs)[0]
    mask = np.zeros((b, t, s), dtype=bool)
    mask[:, :64, :128] = True
    mask[:, 64:, :] = True
    full = fwd(jnp.asarray(mask), q, k, v)
    short = fwd(
        jnp.asarray(mask[:, :64, :128]),
        q[:, :, :, :64],
        k[:, :, :128],
        v[:, :, :128],
    )
    test_utils.assert_bitwise_equal(full[:, :, :, :64], short)
    mask = np.zeros((b, t, s), dtype=bool)
    mask[:, :, 128:] = True
    skipped = fwd(jnp.asarray(mask), q, k, v)
    shifted = fwd(
        jnp.ones((b, t, 128), dtype=bool), q, k[:, :, 128:], v[:, :, 128:]
    )
    test_utils.assert_bitwise_equal(skipped, shifted)
    mask[:, :8, :] = False
    out = fwd(jnp.asarray(mask), q, k, v)
    np.testing.assert_array_equal(test_utils.as_f32(out)[:, :, :, :8, :], 0.0)
    test_utils.assert_bitwise_equal(out[:, :, :, 8:], skipped[:, :, :, 8:])

  @parameterized.named_parameters(*GEOMETRIES)
  def test_flash_fwd_and_bwd_match_reference(
      self, b, kh, g, t, s, kv_block_size
  ):
    rng = np.random.default_rng(3)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    ref_out, *ref_grads = _ref_grads(mask, q, k, v, do)
    got_out, *got_grads = _grads(mask, q, k, v, do, kv_block_size=kv_block_size)
    self._assert_close(got_out, ref_out, name='out', rtol=3e-2, cos_min=0.9999)
    for name, a, e in zip(('dq', 'dk', 'dv'), got_grads, ref_grads):
      self.assertEqual(a.dtype, e.dtype, name)
      self._assert_close(a, e, name=name, rtol=5e-2)

  def test_random_mask_and_fully_masked_rows(self):
    b, kh, g, t, s = 2, 1, 2, 64, 200
    rng = np.random.default_rng(4)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    mask = np.array(_random_mask(rng, b, t, s))
    mask[1, :8, :] = False  # Rows with no valid key at all.
    mask[0, 40:48, 128:] = False  # Rows whose last key sub-block is masked.
    mask = jnp.asarray(mask)
    ref_out, *ref = _ref_grads(mask, q, k, v, do)
    out_m, *got = _grads(mask, q, k, v, do)
    self._assert_close(out_m, ref_out, name='out', rtol=3e-2, cos_min=0.9999)
    for name, a, e in zip(('dq', 'dk', 'dv'), got, ref):
      self._assert_close(a, e, name=name, rtol=5e-2)
    dq = test_utils.as_f32(got[0])
    np.testing.assert_array_equal(dq[1, :, :, :8, :], 0.0)
    np.testing.assert_array_equal(test_utils.as_f32(out_m)[1, :, :, :8, :], 0.0)

  def test_block_skipping_is_exact(self):
    """Skipped (all-masked) sub-blocks contribute nothing in the backward."""
    b, kh, g, t, s = 1, 1, 2, 256, 512
    rng = np.random.default_rng(5)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    mask = np.zeros((b, t, s), dtype=bool)
    mask[:, :, 128:256] = True
    mask = jnp.asarray(mask)
    _, *ref = _ref_grads(mask, q, k, v, do)
    _, *got = _grads(mask, q, k, v, do)
    for name, a, e in zip(('dq', 'dk', 'dv'), got, ref):
      self._assert_close(a, e, name=name, rtol=2e-2, cos_min=0.9999)
    dk = test_utils.as_f32(got[1])
    dv = test_utils.as_f32(got[2])
    np.testing.assert_array_equal(dk[:, :, :128, :], 0.0)
    np.testing.assert_array_equal(dk[:, :, 256:, :], 0.0)
    np.testing.assert_array_equal(dv[:, :, :128, :], 0.0)
    np.testing.assert_array_equal(dv[:, :, 256:, :], 0.0)

  def test_pallas_bwd_under_checkpoint(self):
    """`jax.checkpoint` (CHECKPOINT_ATTN_CORE / Remat=BLOCK) around the op."""
    b, kh, g, t, s = 1, 2, 2, 64, 64
    rng = np.random.default_rng(6)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    fn = _core(mask)
    _, *ref = _vjp(fn, q, k, v, do)
    _, *got = _vjp(jax.checkpoint(fn), q, k, v, do)
    for name, a, e in zip(('dq', 'dk', 'dv'), got, ref):
      self._assert_close(a, e, name=name, rtol=1e-2, cos_min=0.9999)

  def test_row_lse2(self):
    m = jnp.asarray([[1.0, -jnp.inf, 2.0]], jnp.float32)
    l = jnp.asarray([[2.0, 0.0, 1.0]], jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(canon_attention.row_lse2(m, l)), [[2.0, 0.0, 2.0]]
    )

  def test_rejects_unknown_mode_and_bad_kv_block(self):
    b, kh, g, t, s = 1, 1, 2, 64, 64
    rng = np.random.default_rng(7)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    with self.assertRaisesRegex(ValueError, 'unknown attention backward'):
      canon_attention.attention_core(q, k, v, mask, scale=0.125, bwd='nope')
    with self.assertRaisesRegex(ValueError, 'unknown attention forward'):
      canon_attention.attention_core(q, k, v, mask, scale=0.125, fwd='nope')
    with self.assertRaisesRegex(ValueError, 'multiple of 128'):
      _grads(mask, q, k, v, do, kv_block_size=64)
    with self.assertRaisesRegex(ValueError, 'one dtype'):
      _grads(mask, q, k.astype(jnp.float32), v, do)


if __name__ == '__main__':
  absltest.main()
