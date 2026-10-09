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

"""Tests for `canon_attention` (A4 Pallas forward, B4 flash backward).

What CPU interpret mode can prove:

* the Pallas forward (`PALLAS_FWD`, the Zero-TIM contract) is the same bits
  for a query row whether it is evaluated as a `T=1` decode program, inside a
  `T=L` prefill, at `B=1` or `B=N`, against a short or a longer (masked) key
  cache, and that fully masked sub-blocks / rows without a valid key are strict
  no-ops;
* the Pallas forward matches the XLA scan (`XLA_FWD`) to bf16 rounding on
  every geometry the bench uses (query padding, `sp=512` key padding, several
  key blocks, several query blocks per head, GQA groups, fully masked rows),
  and for a given forward every backward mode leaves the primal untouched
  (bitwise), so B4 cannot move the contract;
* the Pallas flash backward matches the dense `jnp` flash backward (same math,
  different blocking / accumulation order) within bf16 rounding;
* both flash backwards match JAX autodiff of the scan (`AUTODIFF`, the Step 7
  gradient) up to the bf16 rounding of the MXU operands, under the same gate
  as the whole-model TP=4 grad parity test (`cos >= 0.999`,
  `|norm ratio - 1| <= 0.02`).

The TPU variant of this target runs the same tests against the real Mosaic
kernels (CPU interpret mode cannot catch layout / lowering failures and
compiles the kernel bodies with a different backend).
"""

from __future__ import annotations

from collections.abc import Callable
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import test_utils
from tunix.experimental.zero_tim_kernel.bench.qwen3 import canon_attention


def setUpModule():
  test_utils.configure_cpu()


_D = 64


def _interpret() -> bool:
  """Pallas interpret mode on CPU; the real Mosaic kernel on TPU."""
  return jax.default_backend() == 'cpu'


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


def _core(
    mask, *, bwd, fwd=None, kv_block_size=512, d=_D
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
    fwd: str, mask, *, kv_block_size=512, d=_D
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
  """Jitted `(out, m_final, l_final)` of one forward implementation."""
  scale = 1.0 / np.sqrt(d)
  if fwd == canon_attention.PALLAS_FWD:
    fn = lambda q, k, v: canon_attention.flash_fwd_pallas(
        q,
        k,
        v,
        mask,
        scale=scale,
        kv_block_size=kv_block_size,
        interpret=_interpret(),
    )
  else:
    fn = lambda q, k, v: canon_attention.online_softmax_scan(
        q, k, v, mask, scale=scale, kv_block_size=kv_block_size
    )
  return jax.jit(fn)


@functools.partial(jax.jit, static_argnums=0)
def _vjp(fn, q, k, v, do):
  out, pullback = jax.vjp(fn, q, k, v)
  return (out, *pullback(do))


def _grads(mask, q, k, v, do, *, bwd, fwd=None, kv_block_size=512):
  fn = _core(mask, bwd=bwd, fwd=fwd, kv_block_size=kv_block_size, d=q.shape[-1])
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
  def test_forward_is_untouched_by_every_bwd_mode(
      self, b, kh, g, t, s, kv_block_size
  ):
    self.assertEqual(canon_attention.ATTENTION_FWD, canon_attention.PALLAS_FWD)
    rng = np.random.default_rng(1)
    q, k, v, _ = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    for fwd in canon_attention.ATTENTION_FWD_MODES:
      expected, _, _ = _forward(fwd, mask, kv_block_size=kv_block_size)(q, k, v)
      self.assertEqual(expected.shape, (b, kh, g, t, _D))
      for mode in canon_attention.ATTENTION_BWD_MODES:
        if mode == canon_attention.AUTODIFF and fwd != canon_attention.XLA_FWD:
          continue  # No JVP for the Pallas kernel (rejected, see below).
        out = jax.jit(
            _core(mask, bwd=mode, fwd=fwd, kv_block_size=kv_block_size)
        )(q, k, v)
        test_utils.assert_bitwise_equal(out, expected)
        # The primal must also be untouched when the op is differentiated.
        out_vjp, *_ = _grads(
            mask, q, k, v, q, bwd=mode, fwd=fwd, kv_block_size=kv_block_size
        )
        test_utils.assert_bitwise_equal(out_vjp, expected)
    # `fwd=None` is the module default.
    out_default = jax.jit(_core(mask, bwd=None, kv_block_size=kv_block_size))(
        q, k, v
    )
    pallas_out, _, _ = _forward(
        canon_attention.PALLAS_FWD, mask, kv_block_size=kv_block_size
    )(q, k, v)
    test_utils.assert_bitwise_equal(out_default, pallas_out)

  @parameterized.named_parameters(*GEOMETRIES)
  def test_pallas_fwd_matches_xla_scan(self, b, kh, g, t, s, kv_block_size):
    rng = np.random.default_rng(8)
    q, k, v, _ = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    out_p, m_p, l_p = _forward(
        canon_attention.PALLAS_FWD, mask, kv_block_size=kv_block_size
    )(q, k, v)
    out_x, m_x, l_x = _forward(
        canon_attention.XLA_FWD, mask, kv_block_size=kv_block_size
    )(q, k, v)
    self.assertEqual(out_p.dtype, out_x.dtype)
    tp = canon_attention.round_up(t, canon_attention.FWD_QUERY_BLOCK)
    self.assertEqual(m_p.shape, (b, kh, g, tp))
    self.assertEqual(l_p.shape, (b, kh, g, tp))
    self.assertEqual(m_p.dtype, jnp.float32)
    self.assertEqual(l_p.dtype, jnp.float32)
    # Same math, different association order / operand rounding (on CPU the
    # scan's f32 einsums are not rounded to bf16): bf16-level agreement.
    self._assert_close(out_p, out_x, name='out', rtol=3e-2, cos_min=0.9999)
    lse_p = np.asarray(canon_attention.row_lse2(m_p, l_p))[..., :t]
    lse_x = np.asarray(canon_attention.row_lse2(m_x, l_x))[..., :t]
    np.testing.assert_allclose(lse_p, lse_x, atol=5e-2, err_msg='lse2')
    # Every causal row has a valid key; the padded rows have none.
    l_p = np.asarray(l_p)
    self.assertTrue(np.all(l_p[..., :t] > 0.0))
    np.testing.assert_array_equal(l_p[..., t:], 0.0)
    np.testing.assert_array_equal(np.asarray(m_p)[..., t:], -np.inf)

  @parameterized.named_parameters(('d64', 64), ('d128', 128))
  def test_pallas_fwd_is_invariant_to_query_length_batch_and_cache(self, d):
    """The contract: one row, one key set -> one result, in every program."""
    b, kh, g, t, s = 2, 1, 2, 256, 256
    rng = np.random.default_rng(9)
    q, k, v, _ = _inputs(rng, b, kh, g, t, s, d=d)
    mask = _causal_mask(b, t, s)
    fwd = lambda m, *xs: _forward(canon_attention.PALLAS_FWD, m, d=d)(*xs)[0]
    full = fwd(mask, q, k, v)  # The prefill / learner program.
    # (1) Decode: the `T=1` program for a row at position `p` (both query
    # blocks of the prefill, block edges, the first and the last row).
    for p in (0, 1, 63, 64, 127, 128, 129, 200, 255):
      row = fwd(mask[:, p : p + 1, :], q[:, :, :, p : p + 1], k, v)
      test_utils.assert_bitwise_equal(row[:, :, :, 0], full[:, :, :, p])
    # (2) Batch: the `B=1` program vs the `B=2` one.
    one = fwd(mask[:1], q[:1], k[:1], v[:1])
    test_utils.assert_bitwise_equal(one, full[:1])
    # (3) Cache length: the same rows against a longer cache whose tail is
    # masked -- random (not zero) keys, one and two key blocks -- as a decode
    # step sees it.
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
    fwd = lambda m, *xs: _forward(canon_attention.PALLAS_FWD, m)(*xs)[0]
    # Rows 0..63 see keys 0..127 only, rows 64..127 see all 256 keys: the
    # second 128-key sub-block IS processed for the query block (it has valid
    # keys for rows 64..127) and rows 0..63 must carry their state through it
    # unchanged -- the same bits as a program whose keys end at 128 (where that
    # sub-block is key padding and skipped).
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
    # Only keys 128..255 visible: sub-block 0 is skipped by the block map, so
    # the result equals attending to those keys as sub-block 0 of a shorter
    # key set (the first processed sub-block starts from the empty state).
    mask = np.zeros((b, t, s), dtype=bool)
    mask[:, :, 128:] = True
    skipped = fwd(jnp.asarray(mask), q, k, v)
    shifted = fwd(
        jnp.ones((b, t, 128), dtype=bool), q, k[:, :, 128:], v[:, :, 128:]
    )
    test_utils.assert_bitwise_equal(skipped, shifted)
    # A row without any valid key is exactly 0 (`l = 0 -> denom 1`).
    mask[:, :8, :] = False
    out = fwd(jnp.asarray(mask), q, k, v)
    np.testing.assert_array_equal(test_utils.as_f32(out)[:, :, :, :8, :], 0.0)
    test_utils.assert_bitwise_equal(out[:, :, :, 8:], skipped[:, :, :, 8:])

  @parameterized.named_parameters(*GEOMETRIES)
  def test_pallas_bwd_matches_xla_bwd(self, b, kh, g, t, s, kv_block_size):
    rng = np.random.default_rng(2)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    _, *ref = _grads(
        mask,
        q,
        k,
        v,
        do,
        bwd=canon_attention.XLA_BWD,
        kv_block_size=kv_block_size,
    )
    _, *got = _grads(
        mask,
        q,
        k,
        v,
        do,
        bwd=canon_attention.PALLAS_BWD,
        kv_block_size=kv_block_size,
    )
    for name, a, e in zip(('dq', 'dk', 'dv'), got, ref):
      self.assertEqual(a.dtype, e.dtype, name)
      # Same math, different accumulation order: bf16 rounding only.
      self._assert_close(a, e, name=name, rtol=2e-2, cos_min=0.9999)

  @parameterized.named_parameters(*GEOMETRIES)
  def test_flash_bwds_match_autodiff(self, b, kh, g, t, s, kv_block_size):
    rng = np.random.default_rng(3)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    _, *ref = _grads(
        mask,
        q,
        k,
        v,
        do,
        bwd=canon_attention.AUTODIFF,
        fwd=canon_attention.XLA_FWD,
        kv_block_size=kv_block_size,
    )
    # With the Pallas forward the backward recomputes `p` from that forward's
    # own `lse2` (and `rowsum(do * o)` from its `o`); the gradient of the same
    # function is the same up to the forwards' bf16-level disagreement.
    for fwd in canon_attention.ATTENTION_FWD_MODES:
      for mode in (canon_attention.XLA_BWD, canon_attention.PALLAS_BWD):
        _, *got = _grads(
            mask, q, k, v, do, bwd=mode, fwd=fwd, kv_block_size=kv_block_size
        )
        for name, a, e in zip(('dq', 'dk', 'dv'), got, ref):
          self.assertEqual(a.dtype, e.dtype, name)
          self._assert_close(a, e, name=f'{fwd}/{mode}:{name}', rtol=5e-2)

  def test_random_mask_and_fully_masked_rows(self):
    b, kh, g, t, s = 2, 1, 2, 64, 200
    rng = np.random.default_rng(4)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    mask = np.array(_random_mask(rng, b, t, s))
    mask[1, :8, :] = False  # Rows with no valid key at all.
    mask[0, 40:48, 128:] = False  # Rows whose last key sub-block is masked.
    mask = jnp.asarray(mask)
    out, *ref = _grads(
        mask,
        q,
        k,
        v,
        do,
        bwd=canon_attention.AUTODIFF,
        fwd=canon_attention.XLA_FWD,
    )
    out_pallas, _, _ = _forward(canon_attention.PALLAS_FWD, mask)(q, k, v)
    self._assert_close(out_pallas, out, name='out', rtol=3e-2, cos_min=0.9999)
    for mode in (canon_attention.XLA_BWD, canon_attention.PALLAS_BWD):
      out_m, *got = _grads(mask, q, k, v, do, bwd=mode)
      test_utils.assert_bitwise_equal(out_m, out_pallas)
      for name, a, e in zip(('dq', 'dk', 'dv'), got, ref):
        self._assert_close(a, e, name=f'{mode}:{name}', rtol=5e-2)
      dq = test_utils.as_f32(got[0])
      np.testing.assert_array_equal(dq[1, :, :, :8, :], 0.0)
    np.testing.assert_array_equal(test_utils.as_f32(out)[1, :, :, :8, :], 0.0)
    np.testing.assert_array_equal(
        test_utils.as_f32(out_pallas)[1, :, :, :8, :], 0.0
    )

  def test_block_skipping_is_exact(self):
    """Skipped (all-masked) sub-blocks contribute nothing in either backward."""
    b, kh, g, t, s = 1, 1, 2, 256, 512
    rng = np.random.default_rng(5)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    # Only keys [128, 256) are visible: three of four 128-key sub-blocks are
    # skipped by the block map for every query block.
    mask = np.zeros((b, t, s), dtype=bool)
    mask[:, :, 128:256] = True
    mask = jnp.asarray(mask)
    _, *ref = _grads(mask, q, k, v, do, bwd=canon_attention.XLA_BWD)
    _, *got = _grads(mask, q, k, v, do, bwd=canon_attention.PALLAS_BWD)
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
    fn = _core(mask, bwd=canon_attention.PALLAS_BWD)
    _, *ref = _vjp(fn, q, k, v, do)
    _, *got = _vjp(jax.checkpoint(fn), q, k, v, do)
    # The kernel is the same program either way; only the XLA-side row
    # statistic `rowsum(do * o)` is recomputed from the rematerialized forward
    # inside the backward program, where the compiler may reduce it in another
    # order. On TPU the rows whose exact `dq` is 0 (one visible key) then carry
    # a different ~1e-7 rounding residue, so the gradients agree to bf16
    # rounding, not bitwise.
    for name, a, e in zip(('dq', 'dk', 'dv'), got, ref):
      self._assert_close(a, e, name=name, rtol=1e-2, cos_min=0.9999)

  def test_row_lse2(self):
    m = jnp.asarray([[1.0, -jnp.inf, 2.0]], jnp.float32)
    l = jnp.asarray([[2.0, 0.0, 1.0]], jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(canon_attention.row_lse2(m, l)), [[2.0, 0.0, 2.0]]
    )

  def test_hard_round_and_upcast_match_astype(self):
    rng = np.random.default_rng(11)
    # Random magnitudes over most of the normal f32 range (clear of denormals,
    # whose flushing is backend-specific) plus exact round-to-even ties (bf16
    # keeps 8 significant bits: 1 + 2**-8 is a tie, 1 + 3 * 2**-8 too).
    magnitudes = 2.0 ** rng.integers(-100, 100, size=4096)
    f32 = np.finfo(np.float32)
    values = np.concatenate([
        (rng.standard_normal(4096) * magnitudes).astype(np.float32),
        np.asarray(
            [1 + 2.0**-8, 1 + 3 * 2.0**-8, -(1 + 2.0**-8), 2.0**-100, 0.0],
            np.float32,
        ),
        # Overflow (`f32.max` rounds up to `inf` in bf16 under RNE) and the
        # infinities pass through both paths identically; NaN payloads are
        # backend-specific like denormals and stay out.
        np.asarray([f32.max, -f32.max, np.inf, -np.inf], np.float32),
    ])
    x = jnp.asarray(values)
    rounded = jax.jit(lambda a: canon_attention.hard_round(a, jnp.bfloat16))(x)
    self.assertEqual(rounded.dtype, jnp.bfloat16)
    test_utils.assert_bitwise_equal(rounded, x.astype(jnp.bfloat16))
    # A materialized bf16 operand upcasts unchanged.
    upcast = jax.jit(canon_attention.hard_upcast)(rounded)
    self.assertEqual(upcast.dtype, jnp.float32)
    test_utils.assert_bitwise_equal(upcast, rounded.astype(jnp.float32))
    # An unrounded f32 that stands in for a dropped bf16 convert is rounded
    # to the bits the bf16 buffer would have held.
    y = jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)
    test_utils.assert_bitwise_equal(
        y, x.astype(jnp.bfloat16).astype(jnp.float32)
    )
    # Other dtypes are a plain cast.
    test_utils.assert_bitwise_equal(
        canon_attention.hard_round(x, jnp.float32), x
    )
    test_utils.assert_bitwise_equal(canon_attention.hard_upcast(x), x)

  @parameterized.named_parameters(*GEOMETRIES)
  def test_hard_rounding_scan_is_bitwise_the_scan(
      self, b, kh, g, t, s, kv_block_size
  ):
    # Under the strict `--xla_allow_excess_precision=false` of these tests the
    # `reduce_precision` rounding points coincide with the `astype` ones.
    rng = np.random.default_rng(3)
    q, k, v, _ = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    scale = 1.0 / np.sqrt(_D)

    def scan(hard):
      return jax.jit(
          lambda q, k, v: canon_attention.online_softmax_scan(
              q,
              k,
              v,
              mask,
              scale=scale,
              kv_block_size=kv_block_size,
              hard_rounding=hard,
          )
      )(q, k, v)

    for a, e in zip(scan(True), scan(False)):
      test_utils.assert_bitwise_equal(a, e)
    out = jax.jit(
        lambda q, k, v: canon_attention.attention_core(
            q,
            k,
            v,
            mask,
            scale=scale,
            kv_block_size=kv_block_size,
            interpret=_interpret(),
            fwd=canon_attention.XLA_FWD,
            hard_rounding=True,
        )
    )(q, k, v)
    test_utils.assert_bitwise_equal(out, scan(False)[0])

  def test_rejects_unknown_mode_and_bad_kv_block(self):
    b, kh, g, t, s = 1, 1, 2, 64, 64
    rng = np.random.default_rng(7)
    q, k, v, do = _inputs(rng, b, kh, g, t, s)
    mask = _causal_mask(b, t, s)
    with self.assertRaisesRegex(ValueError, 'unknown attention backward'):
      canon_attention.attention_core(q, k, v, mask, scale=0.125, bwd='nope')
    with self.assertRaisesRegex(ValueError, 'unknown attention forward'):
      canon_attention.attention_core(q, k, v, mask, scale=0.125, fwd='nope')
    with self.assertRaisesRegex(ValueError, 'has no JVP'):
      canon_attention.attention_core(
          q, k, v, mask, scale=0.125, bwd=canon_attention.AUTODIFF
      )
    for fwd in canon_attention.ATTENTION_FWD_MODES:
      with self.assertRaisesRegex(ValueError, 'multiple of 128'):
        _grads(
            mask,
            q,
            k,
            v,
            do,
            bwd=canon_attention.PALLAS_BWD,
            fwd=fwd,
            kv_block_size=64,
        )
    with self.assertRaisesRegex(ValueError, 'one dtype'):
      _grads(
          mask,
          q,
          k.astype(jnp.float32),
          v,
          do,
          bwd=canon_attention.PALLAS_BWD,
      )
    with self.assertRaisesRegex(ValueError, 'one dtype'):
      _grads(
          mask,
          q,
          k.astype(jnp.float32),
          v,
          do,
          bwd=canon_attention.XLA_BWD,
          fwd=canon_attention.PALLAS_FWD,
      )


if __name__ == '__main__':
  absltest.main()
