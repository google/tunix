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
"""Tests for canonical_logsoftmax (CPU, Pallas interpret mode + 8-device SPMD)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.zero_tim_kernel import canonical_logsoftmax
from tunix.zero_tim_kernel import test_utils


def setUpModule():
  test_utils.configure_cpu(8)


def _logits(seed: int, rows: int, vocab: int, scale: float = 3.0):
  rng = np.random.default_rng(seed)
  return jnp.asarray(
      rng.standard_normal((rows, vocab)).astype(np.float32) * scale
  )


def _tokens(seed: int, rows: int, vocab: int):
  rng = np.random.default_rng(seed + 1)
  return jnp.asarray(rng.integers(0, vocab, size=(rows,)).astype(np.int32))


def _log_softmax_f64(logits) -> np.ndarray:
  x = np.asarray(logits, np.float64)
  shifted = x - x.max(axis=-1, keepdims=True)
  return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class TilesPerStepTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, 1),
      (5, 5),
      (8, 8),
      (9, 5),
      (16, 8),
      (19, 5),
      (38, 8),
      (75, 5),
      (149, 6),
      (150, 6),
      (152, 8),
  )
  def test_minimal_padding_widest_on_ties(self, tiles, expected):
    tps = canonical_logsoftmax._tiles_per_step(tiles)
    self.assertEqual(tps, expected)
    # The Stage 2 operand width never changes.
    padded = -(-tiles // tps) * tps
    self.assertEqual(
        canonical_logsoftmax._summary_width(padded),
        canonical_logsoftmax._summary_width(tiles),
    )


class ValidateAndShapeInvarianceTest(parameterized.TestCase):

  def test_validate_accepts_bf16_f32_and_arbitrary_positive_shapes(self):
    for dtype in (jnp.bfloat16, jnp.float32):
      for m, vocab in ((1, 512), (5, 1000), (8, 151936), (1024, 151936)):
        logits = jax.ShapeDtypeStruct((m, vocab), dtype)
        self.assertEqual(canonical_logsoftmax._validate(logits), (m, vocab))
    for bad in (
        jax.ShapeDtypeStruct((0, 512), jnp.float32),
        jax.ShapeDtypeStruct((8, 0), jnp.float32),
        jax.ShapeDtypeStruct((2, 4, 512), jnp.float32),
        jax.ShapeDtypeStruct((8, 512), jnp.int32),
    ):
      with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
        canonical_logsoftmax._validate(bad)

  @parameterized.parameters(1, 5, 8, 260)
  def test_unaligned_and_large_m_match_padded_program_bitwise(self, m):
    vocab = 512
    logits = _logits(m, 300, vocab)
    tokens = _tokens(m, 300, vocab)
    full = canonical_logsoftmax.gathered_logprobs(
        logits, tokens, interpret=True
    )
    sliced = canonical_logsoftmax.gathered_logprobs(
        logits[:m], tokens[:m], interpret=True
    )
    self.assertLen(sliced, len(full))
    for got, want in zip(sliced, full):
      test_utils.assert_bitwise_equal(got, want[:m])


class LogSoftmaxTest(parameterized.TestCase):

  @parameterized.parameters(512, 1000, 9000)
  def test_matches_float64_log_softmax(self, vocab):
    logits = _logits(vocab, 5, vocab)
    out = canonical_logsoftmax.log_softmax(logits, interpret=True)
    self.assertEqual(out.shape, (5, vocab))
    self.assertEqual(out.dtype, jnp.float32)
    np.testing.assert_allclose(
        np.asarray(out), _log_softmax_f64(logits), rtol=1e-6, atol=1e-5
    )

  @parameterized.parameters(512, 1000, 9000)
  def test_bf16_input_matches_upcast_f32_bitwise_forward_and_vjp(self, vocab):
    logits_bf16 = _logits(vocab + 30, 5, vocab).astype(jnp.bfloat16)
    logits_f32 = logits_bf16.astype(jnp.float32)
    tokens = _tokens(vocab + 30, 5, vocab)
    cot_1d = _logits(vocab + 31, 5, 1, scale=1.0)[:, 0]
    cot_2d = _logits(vocab + 32, 5, vocab, scale=1.0)

    ls_bf16, ls_vjp_bf16 = jax.vjp(
        lambda x: canonical_logsoftmax.log_softmax(x, interpret=True),
        logits_bf16,
    )
    ls_f32, ls_vjp_f32 = jax.vjp(
        lambda x: canonical_logsoftmax.log_softmax(x, interpret=True),
        logits_f32,
    )
    test_utils.assert_bitwise_equal(ls_bf16, ls_f32)
    (grad_ls_bf16,) = ls_vjp_bf16(cot_2d)
    (grad_ls_f32,) = ls_vjp_f32(cot_2d)
    self.assertEqual(grad_ls_bf16.dtype, jnp.bfloat16)
    test_utils.assert_bitwise_equal(
        grad_ls_bf16, grad_ls_f32.astype(jnp.bfloat16)
    )

    lp_bf16, lp_vjp_bf16 = jax.vjp(
        lambda x: canonical_logsoftmax.token_logprobs(
            x, tokens, interpret=True
        ),
        logits_bf16,
    )
    lp_f32, lp_vjp_f32 = jax.vjp(
        lambda x: canonical_logsoftmax.token_logprobs(
            x, tokens, interpret=True
        ),
        logits_f32,
    )
    test_utils.assert_bitwise_equal(lp_bf16, lp_f32)
    (grad_lp_bf16,) = lp_vjp_bf16(cot_1d)
    (grad_lp_f32,) = lp_vjp_f32(cot_1d)
    self.assertEqual(grad_lp_bf16.dtype, jnp.bfloat16)
    test_utils.assert_bitwise_equal(
        grad_lp_bf16, grad_lp_f32.astype(jnp.bfloat16)
    )

    gathered_bf16 = canonical_logsoftmax.gathered_logprobs(
        logits_bf16, tokens, interpret=True
    )
    gathered_f32 = canonical_logsoftmax.gathered_logprobs(
        logits_f32, tokens, interpret=True
    )
    for got, want in zip(gathered_bf16, gathered_f32):
      test_utils.assert_bitwise_equal(got, want)

  @parameterized.parameters(1000, 9000)
  def test_rows_are_independent_of_the_batch(self, vocab):
    # `vocab=9000` with 19 rows: 3 row blocks x 2 vocab steps in interpret
    # mode, so the fused kernel's scratch is reused across both grid axes.
    logits = _logits(1, 19, vocab)
    full = canonical_logsoftmax.log_softmax(logits, interpret=True)
    head = canonical_logsoftmax.log_softmax(logits[:5], interpret=True)
    test_utils.assert_bitwise_equal(head, full[:5])
    perm = np.random.default_rng(2).permutation(19)
    permuted = canonical_logsoftmax.log_softmax(logits[perm], interpret=True)
    test_utils.assert_bitwise_equal(permuted, np.asarray(full)[perm])

  def test_vjp_is_the_analytic_log_softmax_vjp(self):
    logits = _logits(3, 8, 1000)
    weights = _logits(4, 8, 1000, scale=1.0)

    def loss(x):
      return jnp.sum(
          weights * canonical_logsoftmax.log_softmax(x, interpret=True)
      )

    grad = jax.grad(loss)(logits)
    probability = np.exp(_log_softmax_f64(logits))
    w = np.asarray(weights, np.float64)
    expected = w - probability * w.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(grad), expected, rtol=1e-5, atol=1e-5)

  def test_rejects_bad_operands(self):
    for bad in (
        jnp.zeros((8, 512), jnp.int32),
        jnp.zeros((2, 4, 512), jnp.float32),
        jnp.zeros((0, 512), jnp.float32),
    ):
      with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
        canonical_logsoftmax.log_softmax(bad, interpret=True)
    with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
      canonical_logsoftmax.token_logprobs(
          jnp.zeros((8, 512), jnp.float32),
          jnp.zeros((7,), jnp.int32),
          interpret=True,
      )
    with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
      canonical_logsoftmax.gathered_logprobs(
          jnp.zeros((8, 512), jnp.float32),
          jnp.zeros((7,), jnp.int32),
          interpret=True,
      )

  @parameterized.parameters(512, 1000, 9000)
  def test_token_logprobs_matches_log_softmax_forward_and_vjp_bitwise(
      self, vocab
  ):
    logits = _logits(vocab + 10, 7, vocab)
    tokens = _tokens(vocab + 10, 7, vocab)
    cotangent = _logits(vocab + 11, 7, 1, scale=1.0)[:, 0]

    def ref_fn(x):
      full = canonical_logsoftmax.log_softmax(x, interpret=True)
      return jnp.take_along_axis(full, tokens[:, None], axis=-1)[:, 0]

    def opt_fn(x):
      return canonical_logsoftmax.token_logprobs(x, tokens, interpret=True)

    ref_out, ref_vjp = jax.vjp(ref_fn, logits)
    opt_out, opt_vjp = jax.vjp(opt_fn, logits)
    test_utils.assert_bitwise_equal(opt_out, ref_out)
    (ref_grad,) = ref_vjp(cotangent)
    (opt_grad,) = opt_vjp(cotangent)
    test_utils.assert_bitwise_equal(opt_grad, ref_grad)


class GatheredLogprobsTest(parameterized.TestCase):

  def _materialize_then_gather(self, logits, tokens):
    full = np.asarray(canonical_logsoftmax.log_softmax(logits, interpret=True))
    rows = np.arange(full.shape[0])
    token_logprob = full[rows, np.asarray(tokens)]
    return (
        token_logprob,
        full.max(axis=-1),
        full.argmax(axis=-1).astype(np.int32),
        (full >= token_logprob[:, None]).sum(axis=-1).astype(np.int32),
    )

  @parameterized.parameters(1000, 9000)
  def test_matches_materialize_then_gather_bitwise(self, vocab):
    logits = _logits(vocab + 1, 7, vocab)
    tokens = _tokens(vocab + 1, 7, vocab)
    got = canonical_logsoftmax.gathered_logprobs(logits, tokens, interpret=True)
    want = self._materialize_then_gather(logits, tokens)
    self.assertLen(got, 4)
    for value, expected in zip(got, want):
      test_utils.assert_bitwise_equal(value, expected)

  def test_ties_take_the_lowest_index(self):
    vocab = 9000
    logits = np.array(_logits(5, 8, vocab))
    logits[0, [5, 700]] = 30.0
    logits[1, [100, 8500]] = 30.0
    logits = jnp.asarray(logits)
    tokens = jnp.asarray([700, 8500, 0, 1, 2, 3, 4, 5], jnp.int32)
    token_logprob, top_value, top_index, rank = (
        canonical_logsoftmax.gathered_logprobs(logits, tokens, interpret=True)
    )
    self.assertEqual(int(top_index[0]), 5)
    self.assertEqual(int(top_index[1]), 100)
    self.assertEqual(int(rank[0]), 2)
    self.assertEqual(int(rank[1]), 2)
    test_utils.assert_bitwise_equal(token_logprob[:2], top_value[:2])
    want = self._materialize_then_gather(logits, tokens)
    for value, expected in zip(
        (token_logprob, top_value, top_index, rank), want
    ):
      test_utils.assert_bitwise_equal(value, expected)


class VocabParallelLogSoftmaxTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("aligned_4096_m5", 4096, 4, 5),
      ("non_tile_aligned_6144_m5", 6144, 4, 5),
      ("partial_tail_5000_m9", 5000, 4, 9),
      ("qwen3_prod_151936_tp2", 151936, 2, 5),
      ("qwen3_prod_151936_tp4", 151936, 4, 5),
      ("qwen3_prod_151936_tp8", 151936, 8, 5),
  )
  def test_vocab_parallel_matches_unsharded_bitwise(
      self, vocab: int, tp_size: int, m: int
  ):
    self.assertTrue(
        canonical_logsoftmax.vocab_parallel_logsoftmax_admitted(vocab, tp_size)
    )
    mesh = test_utils.cpu_mesh((tp_size,), ("tp",))
    logits_bf16 = _logits(vocab + m, m, vocab).astype(jnp.bfloat16)
    tokens = _tokens(vocab + m, m, vocab)
    cot_1d = _logits(vocab + m + 1, m, 1, scale=1.0)[:, 0]

    ref_lp, ref_lp_vjp = jax.vjp(
        lambda x: canonical_logsoftmax.token_logprobs(
            x, tokens, interpret=True
        ),
        logits_bf16,
    )
    (ref_grad_lp,) = ref_lp_vjp(cot_1d)
    ref_ls = canonical_logsoftmax.log_softmax(logits_bf16, interpret=True)

    with mesh:
      vp_lp, vp_lp_vjp = jax.vjp(
          lambda x: canonical_logsoftmax.vocab_parallel_token_logprobs(
              x,
              tokens,
              mesh=mesh,
              tp_axis="tp",
              tp_size=tp_size,
              interpret=True,
          ),
          logits_bf16,
      )
      (vp_grad_lp,) = vp_lp_vjp(cot_1d)
      vp_ls = canonical_logsoftmax.vocab_parallel_log_softmax(
          logits_bf16,
          mesh=mesh,
          tp_axis="tp",
          tp_size=tp_size,
          interpret=True,
      )

    test_utils.assert_bitwise_equal(vp_lp, ref_lp)
    test_utils.assert_bitwise_equal(vp_grad_lp, ref_grad_lp)
    test_utils.assert_bitwise_equal(vp_ls, ref_ls)

  @parameterized.parameters(6, 13)
  def test_vocab_parallel_with_data_axes_matches_unsharded_bitwise(
      self, m: int
  ):
    vocab, fsdp, tp_size = 3000, 2, 2  # v_local = 1500 (non-aligned + tail)
    mesh = test_utils.cpu_mesh((fsdp, tp_size), ("fsdp", "tp"))
    logits_f32 = _logits(700 + m, m, vocab)
    logits_bf16 = logits_f32.astype(jnp.bfloat16)
    tokens = _tokens(700 + m, m, vocab)
    cot_1d = _logits(800 + m, m, 1, scale=1.0)[:, 0]
    cot_2d = _logits(900 + m, m, vocab, scale=1.0)

    ref_lp, ref_lp_vjp = jax.vjp(
        lambda x: canonical_logsoftmax.token_logprobs(
            x, tokens, interpret=True
        ),
        logits_bf16,
    )
    (ref_grad_lp,) = ref_lp_vjp(cot_1d)
    ref_ls, ref_ls_vjp = jax.vjp(
        lambda x: canonical_logsoftmax.log_softmax(x, interpret=True),
        logits_f32,
    )
    (ref_grad_ls,) = ref_ls_vjp(cot_2d)

    with mesh:
      # Omit `num_shards` so `_resolve_row_sharding` derives `fsdp=2` from mesh.
      vp_lp, vp_lp_vjp = jax.vjp(
          lambda x: canonical_logsoftmax.vocab_parallel_token_logprobs(
              x,
              tokens,
              mesh=mesh,
              tp_axis="tp",
              tp_size=tp_size,
              data_axes=("fsdp",),
              interpret=True,
          ),
          logits_bf16,
      )
      (vp_grad_lp,) = vp_lp_vjp(cot_1d)
      vp_ls, vp_ls_vjp = jax.vjp(
          lambda x: canonical_logsoftmax.vocab_parallel_log_softmax(
              x,
              mesh=mesh,
              tp_axis="tp",
              tp_size=tp_size,
              data_axes=("fsdp",),
              interpret=True,
          ),
          logits_f32,
      )
      (vp_grad_ls,) = vp_ls_vjp(cot_2d)

    test_utils.assert_bitwise_equal(vp_lp, ref_lp)
    test_utils.assert_bitwise_equal(vp_grad_lp, ref_grad_lp)
    test_utils.assert_bitwise_equal(vp_ls, ref_ls)
    np.testing.assert_allclose(
        np.asarray(vp_grad_ls), np.asarray(ref_grad_ls), rtol=1e-5, atol=1e-5
    )

  def test_vocab_parallel_rejects_mismatched_tokens_and_num_shards(self):
    mesh = test_utils.cpu_mesh((2, 2), ("fsdp", "tp"))
    logits = _logits(11, 6, 4096).astype(jnp.bfloat16)
    with mesh:
      for bad_tokens in (
          jnp.zeros((1,), jnp.int32),
          jnp.zeros((6, 1), jnp.int32),
      ):
        with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
          canonical_logsoftmax.vocab_parallel_token_logprobs(
              logits,
              bad_tokens,
              mesh=mesh,
              tp_axis="tp",
              tp_size=2,
              interpret=True,
          )
      with self.assertRaisesRegex(ValueError, "num_shards=1 does not match"):
        canonical_logsoftmax.vocab_parallel_log_softmax(
            logits,
            mesh=mesh,
            tp_axis="tp",
            tp_size=2,
            data_axes=("fsdp",),
            num_shards=1,
            interpret=True,
        )


if __name__ == "__main__":
  absltest.main()
