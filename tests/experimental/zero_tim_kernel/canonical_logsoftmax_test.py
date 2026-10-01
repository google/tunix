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
"""Tests for canonical_logsoftmax (CPU, Pallas interpret mode)."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import canonical_logsoftmax
from tunix.experimental.zero_tim_kernel import test_utils

_CONTINUE_DECODE_ENV = "CANON_CONTINUE_DECODE"


def setUpModule():
  test_utils.configure_cpu()


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


class _EnvTestCase(parameterized.TestCase):
  """Runs every test with CANON_PALLAS_LOGSOFTMAX=1 and no bucket overrides."""

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ, {canonical_logsoftmax.ENV: "1"})
    patcher.start()
    self.addCleanup(patcher.stop)
    for name in (canonical_logsoftmax.ROW_BUCKET_ENV, _CONTINUE_DECODE_ENV):
      os.environ.pop(name, None)


class RowBucketTest(_EnvTestCase):

  def test_bucket_admission_and_rounding(self):
    self.assertTrue(canonical_logsoftmax.row_bucket_admitted(8))
    self.assertTrue(canonical_logsoftmax.row_bucket_admitted(256))
    self.assertFalse(canonical_logsoftmax.row_bucket_admitted(4))
    self.assertFalse(canonical_logsoftmax.row_bucket_admitted(12))
    self.assertFalse(canonical_logsoftmax.row_bucket_admitted(264))
    self.assertEqual(canonical_logsoftmax.row_bucket(1), 8)
    self.assertEqual(canonical_logsoftmax.row_bucket(8), 8)
    self.assertEqual(canonical_logsoftmax.row_bucket(9), 16)
    self.assertEqual(canonical_logsoftmax.row_bucket(200), 200)
    self.assertEqual(canonical_logsoftmax.row_bucket(300), 256)

  def test_validate_accepts_buckets_and_pins_vocab(self):
    # Shape-only structs: the TPU admission never reads values.
    vocab = canonical_logsoftmax.PRODUCTION_V
    for m in (8, 32, 128, 256):
      logits = jax.ShapeDtypeStruct((m, vocab), jnp.float32)
      self.assertEqual(
          canonical_logsoftmax._validate(logits, interpret=False)[0], m
      )
    for bad in ((4, vocab), (12, vocab), (8, 1000)):
      with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
        canonical_logsoftmax._validate(
            jax.ShapeDtypeStruct(bad, jnp.float32), interpret=False
        )

  def test_bucket_switch_defaults_on_and_zero_disables(self):
    self.assertTrue(canonical_logsoftmax.row_bucket_enabled())
    os.environ[canonical_logsoftmax.ROW_BUCKET_ENV] = "0"
    self.assertFalse(canonical_logsoftmax.row_bucket_enabled())

  def test_continue_decode_bucket_matches_padded_program_interpret(self):
    # CPU interpret path (block_rows=1): the bucket and the padded M=256
    # program must agree bitwise on every real row.
    vocab = 4 * canonical_logsoftmax.VOCAB_ALIGN
    logits = _logits(0, 8, vocab)
    tokens = _tokens(0, 8, vocab)
    os.environ[_CONTINUE_DECODE_ENV] = "8"
    os.environ[canonical_logsoftmax.ROW_BUCKET_ENV] = "0"
    padded = canonical_logsoftmax.continue_decode_gathered_logprobs(
        logits, tokens, interpret=True
    )
    os.environ[canonical_logsoftmax.ROW_BUCKET_ENV] = "1"
    bucket = canonical_logsoftmax.continue_decode_gathered_logprobs(
        logits, tokens, interpret=True
    )
    self.assertLen(bucket, len(padded))
    for got, want in zip(bucket, padded):
      test_utils.assert_bitwise_equal(got, want)


class LogSoftmaxTest(_EnvTestCase):

  @parameterized.parameters(512, 1000, 9000)
  def test_matches_float64_log_softmax(self, vocab):
    # 1000 and 9000 exercise the finfo.min padding of a partial vocabulary
    # group; 9000 also the left-to-right combine of two groups.
    logits = _logits(vocab, 8, vocab)
    out = canonical_logsoftmax.log_softmax(logits, interpret=True)
    self.assertEqual(out.shape, (8, vocab))
    self.assertEqual(out.dtype, jnp.float32)
    np.testing.assert_allclose(
        np.asarray(out), _log_softmax_f64(logits), rtol=1e-6, atol=1e-5
    )

  def test_rows_are_independent_of_the_batch(self):
    logits = _logits(1, 16, 1000)
    full = canonical_logsoftmax.log_softmax(logits, interpret=True)
    head = canonical_logsoftmax.log_softmax(logits[:8], interpret=True)
    test_utils.assert_bitwise_equal(head, full[:8])
    perm = np.random.default_rng(2).permutation(16)
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

  def test_rejects_missing_env_and_bad_operands(self):
    good = jnp.zeros((8, 512), jnp.float32)
    for bad in (
        jnp.zeros((8, 512), jnp.bfloat16),
        jnp.zeros((2, 4, 512), jnp.float32),
    ):
      with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
        canonical_logsoftmax.log_softmax(bad, interpret=True)
    os.environ.pop(canonical_logsoftmax.ENV)
    with self.assertRaisesRegex(
        canonical_logsoftmax.CanonicalLogSoftmaxError, "is required"
    ):
      canonical_logsoftmax.log_softmax(good, interpret=True)
    with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
      canonical_logsoftmax.gathered_logprobs(
          good, jnp.zeros((8,), jnp.int32), interpret=True
      )
    with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
      canonical_logsoftmax.token_logprobs(
          good, jnp.zeros((8,), jnp.int32), interpret=True
      )

  @parameterized.parameters(512, 1000, 9000)
  def test_token_logprobs_matches_log_softmax_forward_and_vjp_bitwise(
      self, vocab
  ):
    logits = _logits(vocab + 10, 8, vocab)
    tokens = _tokens(vocab + 10, 8, vocab)
    cotangent = _logits(vocab + 11, 8, 1, scale=1.0)[:, 0]

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


class GatheredLogprobsTest(_EnvTestCase):

  def _materialize_then_gather(self, logits, tokens):
    full = np.asarray(canonical_logsoftmax.log_softmax(logits, interpret=True))
    rows = np.arange(full.shape[0])
    token_logprob = full[rows, np.asarray(tokens)]
    return (
        token_logprob,
        full.max(axis=-1),
        # np.argmax returns the lowest index among ties, like lax.top_k.
        full.argmax(axis=-1).astype(np.int32),
        (full >= token_logprob[:, None]).sum(axis=-1).astype(np.int32),
    )

  @parameterized.parameters(1000, 9000)
  def test_matches_materialize_then_gather_bitwise(self, vocab):
    logits = _logits(vocab + 1, 8, vocab)
    tokens = _tokens(vocab + 1, 8, vocab)
    got = canonical_logsoftmax.gathered_logprobs(logits, tokens, interpret=True)
    want = self._materialize_then_gather(logits, tokens)
    self.assertLen(got, 4)
    for value, expected in zip(got, want):
      test_utils.assert_bitwise_equal(value, expected)

  def test_ties_take_the_lowest_index(self):
    vocab = 9000
    logits = np.array(_logits(5, 8, vocab))
    # Row 0: tie inside one vocabulary group; row 1: tie across groups (the
    # strict > update keeps the earlier group's index).
    logits[0, [5, 700]] = 30.0
    logits[1, [100, 8500]] = 30.0
    logits = jnp.asarray(logits)
    tokens = jnp.asarray([700, 8500, 0, 1, 2, 3, 4, 5], jnp.int32)
    token_logprob, top_value, top_index, rank = (
        canonical_logsoftmax.gathered_logprobs(logits, tokens, interpret=True)
    )
    self.assertEqual(int(top_index[0]), 5)
    self.assertEqual(int(top_index[1]), 100)
    # The sampled token ties the top-1 value, so both tied entries count.
    self.assertEqual(int(rank[0]), 2)
    self.assertEqual(int(rank[1]), 2)
    test_utils.assert_bitwise_equal(token_logprob[:2], top_value[:2])
    want = self._materialize_then_gather(logits, tokens)
    for value, expected in zip(
        (token_logprob, top_value, top_index, rank), want
    ):
      test_utils.assert_bitwise_equal(value, expected)

  def test_rows_are_independent_of_the_batch(self):
    logits = _logits(6, 16, 1000)
    tokens = _tokens(6, 16, 1000)
    full = canonical_logsoftmax.gathered_logprobs(
        logits, tokens, interpret=True
    )
    head = canonical_logsoftmax.gathered_logprobs(
        logits[:8], tokens[:8], interpret=True
    )
    for got, want in zip(head, full):
      test_utils.assert_bitwise_equal(got, np.asarray(want)[:8])


class ContinueDecodeTest(_EnvTestCase):

  def test_bucket_runs_the_gathered_program_at_its_own_rows(self):
    os.environ[_CONTINUE_DECODE_ENV] = "16"
    logits = _logits(7, 16, 512)
    tokens = _tokens(7, 16, 512)
    got = canonical_logsoftmax.continue_decode_gathered_logprobs(
        logits, tokens, interpret=True
    )
    want = canonical_logsoftmax.gathered_logprobs(
        logits, tokens, interpret=True
    )
    for value, expected in zip(got, want):
      test_utils.assert_bitwise_equal(value, expected)

  @parameterized.named_parameters(
      ("env_unset", None, (8, 512), (8,)),
      ("env_zero", "0", (8, 512), (8,)),
      ("env_out_of_range", "65", (8, 512), (8,)),
      ("env_not_a_number", "x", (8, 512), (8,)),
      ("logits_rank", "8", (8, 4, 128), (8,)),
      ("tokens_rank", "8", (8, 512), (8, 1)),
      ("rows_differ", "8", (8, 512), (16,)),
      ("rows_not_a_bucket", "8", (12, 512), (12,)),
      ("rows_between_buckets", "8", (64, 512), (64,)),
  )
  def test_rejects(self, env_value, logits_shape, tokens_shape):
    if env_value is not None:
      os.environ[_CONTINUE_DECODE_ENV] = env_value
    with self.assertRaises(canonical_logsoftmax.CanonicalLogSoftmaxError):
      canonical_logsoftmax.continue_decode_gathered_logprobs(
          jnp.zeros(logits_shape, jnp.float32),
          jnp.zeros(tokens_shape, jnp.int32),
          interpret=True,
      )


if __name__ == "__main__":
  absltest.main()
