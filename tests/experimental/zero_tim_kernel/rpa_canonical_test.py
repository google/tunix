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
"""Tests for rpa_canonical (4 CPU devices)."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.experimental.zero_tim_kernel import rpa_canonical
from tunix.experimental.zero_tim_kernel import rpa_diff
from tunix.experimental.zero_tim_kernel import rpa_diff_chunked
from tunix.experimental.zero_tim_kernel import test_utils

_CANONICAL = (128, 512, 128, 512)
_P59_ENV = "CANON_P59_RANK_PARALLEL_BACKWARD"
_ENVS = (
    rpa_canonical.FORCE_MIXED_ENV,
    rpa_canonical.RPA_VJP2_ENV,
    rpa_canonical.RPA_VJP_ENV,
    _P59_ENV,
) + tuple(name for _, name in rpa_canonical.BLOCK_SIZE_ENVS)


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


class BlockSizeTest(_CleanEnvTestCase):

  def test_parse_block_sizes(self):
    self.assertIsNone(rpa_canonical.parse_block_sizes(None))
    self.assertIsNone(rpa_canonical.parse_block_sizes(""))
    self.assertEqual(
        rpa_canonical.parse_block_sizes("128,512,128,512"), _CANONICAL
    )
    with self.assertRaises(ValueError):
      rpa_canonical.parse_block_sizes("128,big")

  def test_pinned_kwargs_follow_the_given_environment(self):
    environ = {"CANON_RPA_D": "128,512,128,512", "CANON_RPA_P": "64,256,64,256"}
    self.assertEqual(
        rpa_canonical.pinned_block_size_kwargs(environ=environ),
        {"d_block_sizes": _CANONICAL, "p_block_sizes": (64, 256, 64, 256)},
    )
    self.assertEqual(rpa_canonical.pinned_block_size_kwargs(environ={}), {})
    # The head-dim-64 kernel is never pinned (as on the branch).
    self.assertEqual(
        rpa_canonical.pinned_block_size_kwargs(use_hd64=True, environ=environ),
        {},
    )

  def test_pinned_kwargs_default_to_os_environ(self):
    os.environ["CANON_RPA_M"] = "128,512,128,512"
    self.assertEqual(
        rpa_canonical.pinned_block_size_kwargs(), {"m_block_sizes": _CANONICAL}
    )

  def test_canonical_bundle(self):
    canonical = rpa_canonical.canonical_block_size_kwargs()
    self.assertEqual(
        canonical,
        {
            "d_block_sizes": _CANONICAL,
            "p_block_sizes": _CANONICAL,
            "m_block_sizes": _CANONICAL,
        },
    )
    self.assertEqual(rpa_canonical.CANONICAL_MIN_TOKEN_BUCKET, 256)
    # The canonical engine profile's env strings select the same kwargs.
    environ = {
        name: "128,512,128,512" for _, name in rpa_canonical.BLOCK_SIZE_ENVS
    }
    self.assertEqual(
        rpa_canonical.pinned_block_size_kwargs(environ=environ), canonical
    )


class ForceMixedTest(_CleanEnvTestCase):

  def test_force_mixed_distribution(self):
    distribution = jnp.asarray([3, 5, 9], jnp.int32)
    out = rpa_canonical.force_mixed_distribution(distribution)
    self.assertEqual(out.dtype, jnp.int32)
    np.testing.assert_array_equal(np.asarray(out), [0, 0, 9])
    jitted = jax.jit(rpa_canonical.force_mixed_distribution)(distribution)
    np.testing.assert_array_equal(np.asarray(jitted), [0, 0, 9])

  def test_maybe_force_mixed_is_env_gated(self):
    distribution = jnp.asarray([3, 5, 9], jnp.int32)
    self.assertIs(
        rpa_canonical.maybe_force_mixed_distribution(distribution),
        distribution,
    )
    os.environ[rpa_canonical.FORCE_MIXED_ENV] = "1"
    np.testing.assert_array_equal(
        np.asarray(rpa_canonical.maybe_force_mixed_distribution(distribution)),
        [0, 0, 9],
    )


class DifferentiableRpaTest(_CleanEnvTestCase):

  _KWARGS = dict(sm_scale=0.125, page_size=16, num_q_heads=8, num_kv_heads=2)

  def test_kernel_is_unchanged_without_a_vjp_flag(self):
    kernel = mock.Mock()
    self.assertIs(
        rpa_canonical.differentiable_rpa(kernel, **self._KWARGS), kernel
    )

  def test_vjp2_selects_the_chunked_cache_wrapper(self):
    os.environ[rpa_canonical.RPA_VJP2_ENV] = "1"
    # VJP2 takes precedence over the legacy prefill-only VJP.
    os.environ[rpa_canonical.RPA_VJP_ENV] = "1"
    kernel = mock.Mock()
    with (
        mock.patch.object(
            rpa_diff_chunked, "make_diff_rpa_chunked_ragged", autospec=True
        ) as chunked,
        mock.patch.object(rpa_diff, "make_diff_rpa", autospec=True) as legacy,
    ):
      wrapped = rpa_canonical.differentiable_rpa(kernel, **self._KWARGS)
    chunked.assert_called_once_with(kernel, **self._KWARGS)
    legacy.assert_not_called()
    self.assertIs(wrapped, chunked.return_value)

  def test_vjp_selects_the_prefill_only_wrapper(self):
    os.environ[rpa_canonical.RPA_VJP_ENV] = "1"
    kernel = mock.Mock()
    with mock.patch.object(rpa_diff, "make_diff_rpa", autospec=True) as legacy:
      wrapped = rpa_canonical.differentiable_rpa(
          kernel, use_causal_mask=False, **self._KWARGS
      )
    legacy.assert_called_once_with(
        kernel, sm_scale=0.125, use_causal_mask=False
    )
    self.assertIs(wrapped, legacy.return_value)

  def test_wrappers_are_real_callables(self):
    kernel = mock.Mock()
    for env in (rpa_canonical.RPA_VJP2_ENV, rpa_canonical.RPA_VJP_ENV):
      with self.subTest(env=env), mock.patch.dict(os.environ, {env: "1"}):
        wrapped = rpa_canonical.differentiable_rpa(kernel, **self._KWARGS)
        self.assertIsNot(wrapped, kernel)
        self.assertTrue(callable(wrapped))
    kernel.assert_not_called()

  @parameterized.product(
      env=(rpa_canonical.RPA_VJP2_ENV, rpa_canonical.RPA_VJP_ENV),
      flag=("use_hd64", "has_attention_sink"),
  )
  def test_unsupported_paths_raise(self, env, flag):
    os.environ[env] = "1"
    with self.assertRaises(NotImplementedError):
      rpa_canonical.differentiable_rpa(
          mock.Mock(), **{flag: True}, **self._KWARGS
      )


class P59LocalAttentionTest(_CleanEnvTestCase):

  def setUp(self):
    super().setUp()
    self.mesh = test_utils.cpu_mesh((2, 2), ("data", "model"))

  def _in_manual_map(self, engine_mesh, axis_names=None):
    observed = []

    def body(x):
      observed.append(rpa_canonical.p59_local_attention_context(engine_mesh))
      return x

    kwargs = {} if axis_names is None else {"axis_names": axis_names}
    jax.jit(
        jax.shard_map(
            body,
            mesh=self.mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
            **kwargs,
        )
    )(jnp.zeros((4,), jnp.float32))
    return observed[0]

  def test_context(self):
    self.assertFalse(self._in_manual_map(self.mesh))
    os.environ[_P59_ENV] = "1"
    self.assertFalse(rpa_canonical.p59_local_attention_context(self.mesh))
    self.assertTrue(self._in_manual_map(self.mesh))
    self.assertFalse(self._in_manual_map(self.mesh, axis_names={"model"}))

  @parameterized.named_parameters(
      ("topology_differs", (1, 4), ("data", "model")),
      ("engine_lacks_data_axis", (4,), ("model",)),
  )
  def test_rejects_inconsistent_engine(self, shape, axis_names):
    os.environ[_P59_ENV] = "1"
    with self.assertRaises(RuntimeError):
      self._in_manual_map(test_utils.cpu_mesh(shape, axis_names))

  def _operands(
      self,
      *,
      q=(16, 4, 128),
      k=(16, 2, 128),
      v=None,
      cache=(8, 16, 2, 2, 128),
  ):
    return tuple(
        jax.ShapeDtypeStruct(shape, jnp.bfloat16)
        for shape in (q, k, k if v is None else v, cache)
    )

  def test_validate_accepts_tp_local_operands(self):
    # bf16 packs two values per 32-bit word: K and V interleave into
    # ceil(2 * kv_heads / packing) cache heads.
    rpa_canonical.validate_p59_local_attention(*self._operands(), tp_size=4)
    rpa_canonical.validate_p59_local_attention(
        *self._operands(
            q=(16, 8, 128), k=(16, 1, 128), cache=(8, 16, 1, 2, 128)
        ),
        tp_size=8,
    )

  @parameterized.named_parameters(
      ("q_rank", dict(q=(16, 512))),
      ("cache_rank", dict(cache=(8, 16, 2, 256))),
      ("k_v_differ", dict(v=(16, 1, 128))),
      ("tokens_differ", dict(k=(8, 2, 128))),
      ("head_dim_differs", dict(k=(16, 2, 64))),
      ("gqa_ratio", dict(q=(16, 3, 128))),
      ("no_kv_heads", dict(k=(16, 0, 128))),
      ("zero_packing", dict(cache=(8, 16, 2, 0, 128))),
      ("cache_heads", dict(cache=(8, 16, 4, 2, 128))),
      ("cache_head_dim", dict(cache=(8, 16, 2, 2, 64))),
  )
  def test_validate_rejects(self, overrides):
    with self.assertRaises(ValueError):
      rpa_canonical.validate_p59_local_attention(
          *self._operands(**overrides), tp_size=4
      )


def _fixed_qk_scores(
    q_row: jax.Array, k_block: jax.Array, sm_scale: float
) -> jax.Array:
  """Computes q @ k_block.T * sm_scale with strict left-to-right D reduction."""
  q_f32 = q_row.astype(jnp.float32)
  k_f32 = k_block.astype(jnp.float32)
  head_dim = q_f32.shape[0]
  init = jnp.zeros((k_f32.shape[0],), dtype=jnp.float32)
  scores = jax.lax.fori_loop(
      0, head_dim, lambda d, acc: acc + q_f32[d] * k_f32[:, d], init
  )
  return scores * jnp.float32(sm_scale)


def _fixed_rowmax(s_block: jax.Array) -> jax.Array:
  """Reduces max(s_block) strictly left-to-right across the KV block."""
  return jax.lax.fori_loop(
      0,
      s_block.shape[0],
      lambda t, acc: jnp.maximum(acc, s_block[t]),
      jnp.float32(-jnp.inf),
  )


def _fixed_rowsum(p_block: jax.Array) -> jax.Array:
  """Reduces sum(p_block) strictly left-to-right from 0.0 across the KV block."""
  return jax.lax.fori_loop(
      0,
      p_block.shape[0],
      lambda t, acc: acc + p_block[t],
      jnp.float32(0.0),
  )


def _fixed_pv_matmul(p_block: jax.Array, v_block: jax.Array) -> jax.Array:
  """Computes p_block @ v_block with strict left-to-right bkv_csz reduction."""
  v_f32 = v_block.astype(jnp.float32)
  init = jnp.zeros((v_f32.shape[1],), dtype=jnp.float32)
  return jax.lax.fori_loop(
      0,
      p_block.shape[0],
      lambda t, acc: acc + p_block[t] * v_f32[t],
      init,
  )


def _rpa_v3_kv_loop_minimal(
    q_row: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    bkv_csz: int,
    sm_scale: float = 0.08838834764831845,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Minimal reproduction of RPA v3 step1_qk_softmax + step2_pv KV loop.

  Every sub-operation inside a block (QK^T dot, rowmax, exp, rowsum, PV dot) is
  pinned to a deterministic left-to-right reduction. Only the KV block size
  `bkv_csz` varies (e.g. 16 vs 32, matching the measured Qwen3-32B@tp4 decode
  vs prefill difference).

  Args:
    q_row: Query vector of shape `[head_dim]`.
    k: Key matrix of shape `[kv_len, head_dim]`.
    v: Value matrix of shape `[kv_len, head_dim]`.
    bkv_csz: KV compute block size dividing `kv_len`.
    sm_scale: Softmax scale factor.

  Returns:
    Tuple `(out_bf16, out_f32, l_final, all_p)` where `all_p` is the
    concatenation of per-block `p = exp(s - m_curr)` values.
  """
  kv_len, head_dim = k.shape
  assert kv_len % bkv_csz == 0
  m = jnp.float32(-jnp.inf)
  l = jnp.float32(0.0)
  o = jnp.zeros((head_dim,), dtype=jnp.float32)
  p_blocks = []

  for start in range(0, kv_len, bkv_csz):
    k_blk = k[start : start + bkv_csz]
    v_blk = v[start : start + bkv_csz]

    # flash_attention_step1_qk_softmax (kernel.py:456-513)
    s = _fixed_qk_scores(q_row, k_blk, sm_scale)
    s_rowmax = _fixed_rowmax(s)
    m_prev = m
    m_curr = jnp.maximum(m_prev, s_rowmax)
    m = m_curr
    p = jnp.exp(s - m_curr)
    p_blocks.append(p)

    p_rowsum = _fixed_rowsum(p)
    exp_m_diff = jnp.exp(m_prev - m_curr)
    l = exp_m_diff * l + p_rowsum

    # flash_attention_step2_pv (kernel.py:530-537)
    pv = _fixed_pv_matmul(p, v_blk)
    o = exp_m_diff * o + pv

  out_f32 = o / l
  return out_f32.astype(jnp.bfloat16), out_f32, l, jnp.concatenate(p_blocks)


class BlockSizeNumericsDriftMinimalReproTest(parameterized.TestCase):
  """Minimal reproduction proving why RPA block_size (bkv_csz) changes numerics.

  Even after every individual sub-operator (QK^T, rowmax, exp, rowsum, PV) is
  pinned to a deterministic left-to-right reduction, changing `bkv_csz` from
  16 (decode) to 32 (prefill) alters the attention output via two independent
  mechanisms along the KV sequence reduction axis:
    1. Blockwise contraction grouping: `PV` and `rowsum(P)` reset their local
       accumulator to 0 at every `bkv_csz` boundary, turning a 32-step chain
       `sum(0..31)` into `(sum(0..15)) + (sum(16..31))`.
    2. Online-softmax rescaling: whenever a later KV block has a larger max
       logit `m_curr > m_prev`, `bkv_csz=16` computes
       `exp(m_prev - m_curr) * exp(s - m_prev)` whereas `bkv_csz=32` computes
       `exp(s - m_curr)` directly in one block.
  """

  def test_qk_scores_are_bitwise_independent_of_bkv_csz(self):
    key = jax.random.PRNGKey(0)
    kq, kk = jax.random.split(key)
    q = jax.random.normal(kq, (128,), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (32, 128), dtype=jnp.bfloat16)

    # QK^T contracts over head_dim (128), NOT bkv_csz: splitting K into 16-row
    # vs 32-row blocks produces 100% bit-identical scores s[0:32].
    s_32 = _fixed_qk_scores(q, k, sm_scale=0.125)
    s_16 = jnp.concatenate([
        _fixed_qk_scores(q, k[:16], sm_scale=0.125),
        _fixed_qk_scores(q, k[16:], sm_scale=0.125),
    ])
    np.testing.assert_array_equal(np.asarray(s_16), np.asarray(s_32))

  def test_cause1_kv_contraction_grouping_drifts_even_when_rescale_is_one(self):
    key = jax.random.PRNGKey(42)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (128,), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (32, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (32, 128), dtype=jnp.bfloat16)

    # Put the global maximum at t=0 (block 0) and an equal score at t=16
    # (block 1), so m_curr == m_prev in block 1 and exp_m_diff == exp(0) == 1.0
    # (zero online-softmax rescaling, p_0 == p_16 == 1.0).
    k_max = jnp.sign(q) * jnp.bfloat16(0.5)
    k = k.at[0].set(k_max).at[16].set(k_max)
    # Place +256 at t=0 and -256 at t=16:
    # - In bkv_csz=32, +256 (t=0) and -256 (t=16) cancel at step t=16 BEFORE
    #   t=17..31 are accumulated, so t=17..31 are added to an O(1) running sum.
    # - In bkv_csz=16, block 1 starts a fresh 0.0 accumulator at t=16, so
    #   t=17..31 are accumulated onto -256.0 (swamping ~8 mantissa bits) before
    #   block 0 (+256) and block 1 (-256) are summed.
    v = v.at[0].set(jnp.bfloat16(256.0)).at[16].set(jnp.bfloat16(-256.0))

    out16_bf16, out16_f32, _, p16 = _rpa_v3_kv_loop_minimal(q, k, v, bkv_csz=16)
    out32_bf16, out32_f32, _, p32 = _rpa_v3_kv_loop_minimal(q, k, v, bkv_csz=32)

    # Because block 0 holds the global max, every token's unnormalized
    # probability p_t = exp(s_t - m_global) is 100% bitwise identical!
    np.testing.assert_array_equal(np.asarray(p16), np.asarray(p32))

    # Yet out (PV / l) drifts in both f32 and bf16 between bkv_csz=16 and 32
    # purely because bkv_csz=16 groups the KV contraction as
    # (sum_{t=0..15}) + (sum_{t=16..31}) instead of sum_{t=0..31}.
    self.assertFalse(
        np.array_equal(
            np.asarray(out16_f32.view(jnp.uint32)),
            np.asarray(out32_f32.view(jnp.uint32)),
        )
    )
    self.assertFalse(
        np.array_equal(
            np.asarray(out16_bf16.view(jnp.uint16)),
            np.asarray(out32_bf16.view(jnp.uint16)),
        )
    )

    # Prove this drift is 100% explained by the (0..15) + (16..31) grouping:
    exact_two_block_pv = _fixed_pv_matmul(p32[:16], v[:16]) + _fixed_pv_matmul(
        p32[16:], v[16:]
    )
    exact_two_block_l = _fixed_rowsum(p32[:16]) + _fixed_rowsum(p32[16:])
    np.testing.assert_array_equal(
        np.asarray((exact_two_block_pv / exact_two_block_l).view(jnp.uint32)),
        np.asarray(out16_f32.view(jnp.uint32)),
    )

  def test_cause2_online_softmax_rescaling_drifts_even_with_one_active_v(self):
    # Isolate online-softmax rescaling from PV multi-element summation rounding
    # by making V non-zero ONLY at a single token t=1 in block 0 (with s_1 < m_0
    # in block 0, and a larger global max m_1 > m_0 at t=16 in block 1).
    # Because V is zero everywhere except t=1, PV has only ONE non-zero term
    # (no multi-term PV sum at all!), yet o_final still drifts because:
    #   bkv_csz=16 computes: exp(m_0 - m_1) * (exp(s_1 - m_0) * v_1)
    #   bkv_csz=32 computes: exp(s_1 - m_1) * v_1
    q = jnp.ones((128,), dtype=jnp.bfloat16)
    k = jnp.full((32, 128), jnp.bfloat16(-0.5))
    k = k.at[0].set(jnp.bfloat16(0.25))  # block 0 local max m_0
    k = k.at[1].set(
        jnp.bfloat16(0.125)
    )  # active V token in block 0 (s_1 < m_0)
    k = k.at[16].set(jnp.bfloat16(0.375))  # block 1 global max m_1 > m_0

    v = jnp.zeros((32, 128), dtype=jnp.bfloat16)
    v = v.at[1].set(jnp.linspace(0.5, 1.5, 128, dtype=jnp.bfloat16))

    _, out16_f32, _, p16 = _rpa_v3_kv_loop_minimal(
        q, k, v, bkv_csz=16, sm_scale=0.05
    )
    _, out32_f32, _, p32 = _rpa_v3_kv_loop_minimal(
        q, k, v, bkv_csz=32, sm_scale=0.05
    )

    # For token t=1 (where s_1 < m_0 < m_1), exp(s_1 - m_0) * exp(m_0 - m_1)
    # != exp(s_1 - m_1) in IEEE-754 float32!
    s_all = _fixed_qk_scores(q, k, sm_scale=0.05)
    m0, m1 = s_all[0], s_all[16]
    rescaled_p_blk0 = p16[:16] * jnp.exp(m0 - m1)
    direct_p_blk0 = p32[:16]
    self.assertFalse(
        np.array_equal(
            np.asarray(rescaled_p_blk0.view(jnp.uint32)),
            np.asarray(direct_p_blk0.view(jnp.uint32)),
        )
    )
    self.assertFalse(
        np.array_equal(
            np.asarray(out16_f32.view(jnp.uint32)),
            np.asarray(out32_f32.view(jnp.uint32)),
        )
    )


if __name__ == "__main__":
  absltest.main()
