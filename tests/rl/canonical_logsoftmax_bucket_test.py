"""Row-bucket admission for the canonical log-softmax (tasks/zero_tim_perf2 B2 knife 1)."""
import os
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

from tunix.rl import canonical_logsoftmax as cls


class RowBucketTest(absltest.TestCase):

  def test_bucket_admission_and_rounding(self):
    self.assertTrue(cls.row_bucket_admitted(8))
    self.assertTrue(cls.row_bucket_admitted(256))
    self.assertFalse(cls.row_bucket_admitted(4))
    self.assertFalse(cls.row_bucket_admitted(12))
    self.assertFalse(cls.row_bucket_admitted(264))
    self.assertEqual(cls.row_bucket(1), 8)
    self.assertEqual(cls.row_bucket(8), 8)
    self.assertEqual(cls.row_bucket(9), 16)
    self.assertEqual(cls.row_bucket(200), 200)
    self.assertEqual(cls.row_bucket(300), 256)

  def test_validate_accepts_buckets_and_pins_vocab(self):
    with mock.patch.dict(os.environ, {cls.ENV: "1"}, clear=False):
      for m in (8, 32, 128, 256):
        self.assertEqual(
            cls._validate(jnp.zeros((m, cls.PRODUCTION_V), jnp.float32), interpret=False)[0],
            m,
        )
      for bad in ((4, cls.PRODUCTION_V), (12, cls.PRODUCTION_V), (8, 1000)):
        with self.assertRaises(cls.CanonicalLogSoftmaxError):
          cls._validate(jnp.zeros(bad, jnp.float32), interpret=False)

  def test_bucket_switch_defaults_on_and_zero_disables(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop(cls.ROW_BUCKET_ENV, None)
      self.assertTrue(cls.row_bucket_enabled())
    with mock.patch.dict(os.environ, {cls.ROW_BUCKET_ENV: "0"}, clear=False):
      self.assertFalse(cls.row_bucket_enabled())

  def test_continue_decode_bucket_matches_padded_program_interpret(self):
    # CPU interpret path (block_rows=1): the bucket and the padded program
    # must agree bitwise on every real row; the TPU evidence is the probe.
    rng = np.random.default_rng(0)
    vocab = 4 * cls.VOCAB_ALIGN
    logits = jnp.asarray(rng.standard_normal((8, vocab)).astype(np.float32) * 3)
    tokens = jnp.asarray(rng.integers(0, vocab, size=(8,)).astype(np.int32))
    with mock.patch.dict(
        os.environ, {cls.ENV: "1", "CANON_CONTINUE_DECODE": "8"}, clear=False
    ):
      with mock.patch.dict(os.environ, {cls.ROW_BUCKET_ENV: "0"}, clear=False):
        padded = cls.continue_decode_gathered_logprobs(logits, tokens, interpret=True)
      with mock.patch.dict(os.environ, {cls.ROW_BUCKET_ENV: "1"}, clear=False):
        bucket = cls.continue_decode_gathered_logprobs(logits, tokens, interpret=True)
    self.assertEqual(len(padded), len(bucket))
    for a, b in zip(padded, bucket):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


if __name__ == "__main__":
  absltest.main()
