# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import mock_rollout


class FakeTokenizer:
  """A fake tokenizer for testing purposes.

  This class simulates a tokenizer with configurable vocab size, pad/eos IDs,
  and an option to raise an exception during encoding.
  """

  def __init__(self, vocab_size=100, pad_id=0, eos_id=1, fail_encode=False):
    self.vocab_size = vocab_size
    self._pad_id = pad_id
    self._eos_id = eos_id
    self._fail_encode = fail_encode

  def encode(self, text):
    if self._fail_encode:
      raise ValueError("Encode failed")
    # Return dummy token IDs based on string length to simulate tokenization
    return [min(i, self.vocab_size - 1) for i in range(len(text))]

  def decode(self, tokens):
    return "decoded_" + "_".join(str(t) for t in tokens)

  def pad_id(self):
    return self._pad_id

  def eos_id(self):
    return self._eos_id


class FakeTokenizerProperties:
  # Some tokenizers have these as properties rather than functions
  def __init__(self, pad_id=0, eos_id=1):
    self.pad_id = pad_id
    self.eos_id = eos_id


class MockRolloutTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.base_rc = base_rollout.RolloutConfig(
        max_prompt_length=10,
        max_tokens_to_generate=15,
        return_logprobs=True,
        seed=42,
    )

  def _create_mock_rollout(self, **kwargs):
    kwargs.setdefault("vocab_size", 100)
    kwargs.setdefault("pad_id", 0)
    kwargs.setdefault("eos_id", 1)
    kwargs.setdefault("min_generation_time", 0.01)
    kwargs.setdefault("max_generation_time", 0.02)
    return mock_rollout.MockRollout(**kwargs)

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_basic(self, mock_sleep):
    m = self._create_mock_rollout()
    out = m.generate(["prompt 1", "prompt 2"], rollout_config=self.base_rc)

    self.assertLen(out.text, 2)
    self.assertLen(out.logits, 2)
    self.assertLen(out.tokens, 2)
    self.assertLen(out.logprobs, 2)
    self.assertEqual(out.left_padded_prompt_tokens.shape, (2, 10))
    self.assertTrue(mock_sleep.called)

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_single_prompt(self, mock_sleep):
    m = self._create_mock_rollout()
    out = m.generate("single prompt", rollout_config=self.base_rc)

    self.assertLen(out.text, 1)
    self.assertLen(out.logits, 1)
    self.assertLen(out.tokens, 1)
    self.assertLen(out.logprobs, 1)
    self.assertEqual(out.left_padded_prompt_tokens.shape, (1, 10))

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_no_logprobs(self, mock_sleep):
    m = self._create_mock_rollout()
    rc = dataclasses.replace(self.base_rc, return_logprobs=False)
    out = m.generate(["prompt 1"], rollout_config=rc)

    self.assertIsNone(out.logprobs)

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_with_tokenizer(self, mock_sleep):
    tokenizer = FakeTokenizer(vocab_size=100)
    m = self._create_mock_rollout(tokenizer=tokenizer)
    out = m.generate(["prompt"], rollout_config=self.base_rc)

    self.assertLen(out.text, 1)
    self.assertNotEmpty(out.tokens[0])
    self.assertTrue(out.text[0].startswith("decoded_"))

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_tokenizer_encode_exception(self, mock_sleep):
    tokenizer = FakeTokenizer(vocab_size=100, fail_encode=True)
    m = self._create_mock_rollout(tokenizer=tokenizer)
    out = m.generate(["prompt"], rollout_config=self.base_rc)

    self.assertLen(out.tokens[0], len(out.text[0].split()))

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_prompt_truncation(self, mock_sleep):
    tokenizer = FakeTokenizer(vocab_size=100)
    m = self._create_mock_rollout(tokenizer=tokenizer)
    rc = dataclasses.replace(self.base_rc, max_prompt_length=3)
    # FakeTokenizer encodes string length, so "long_prompt" is 11 tokens.
    out = m.generate(["long_prompt"], rollout_config=rc)

    self.assertEqual(out.left_padded_prompt_tokens.shape, (1, 3))
    # It should be truncated, so no pad_id is present at the start.
    self.assertFalse(
        np.any(out.left_padded_prompt_tokens == tokenizer.pad_id())
    )

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_reproducibility_with_seed(self, mock_sleep):
    rollout_config = dataclasses.replace(self.base_rc, seed=42)
    m1 = self._create_mock_rollout(rollout_config=rollout_config)
    m2 = self._create_mock_rollout(rollout_config=rollout_config)

    out1 = m1.generate(["prompt 1", "prompt 2"], rollout_config=rollout_config)
    out2 = m2.generate(["prompt 1", "prompt 2"], rollout_config=rollout_config)

    self.assertEqual(out1.text, out2.text)
    np.testing.assert_array_equal(out1.tokens[0], out2.tokens[0])
    np.testing.assert_array_equal(out1.tokens[1], out2.tokens[1])
    np.testing.assert_array_equal(out1.logits[0], out2.logits[0])
    np.testing.assert_array_equal(out1.logits[1], out2.logits[1])

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_reproducibility_with_jax_seed(self, mock_sleep):
    rollout_config = dataclasses.replace(self.base_rc, seed=jnp.array(42))
    m1 = self._create_mock_rollout(rollout_config=rollout_config)
    m2 = self._create_mock_rollout(rollout_config=rollout_config)

    out1 = m1.generate(["prompt"], rollout_config=rollout_config)
    out2 = m2.generate(["prompt"], rollout_config=rollout_config)

    self.assertEqual(out1.text, out2.text)
    np.testing.assert_array_equal(out1.tokens[0], out2.tokens[0])
    np.testing.assert_array_equal(out1.logits[0], out2.logits[0])

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_sleep_time_bounds(self, mock_sleep):
    m = self._create_mock_rollout()
    m.generate(["prompt"], rollout_config=self.base_rc)

    mock_sleep.assert_called_once()
    sleep_time = mock_sleep.call_args[0][0]
    self.assertBetween(sleep_time, 0.01, 0.02)

  def test_get_per_token_logps(self):
    m = self._create_mock_rollout()
    prompt_tokens = jnp.zeros((2, 5))
    completion_tokens = jnp.ones((2, 10))

    logps = m.get_per_token_logps(prompt_tokens, completion_tokens)

    self.assertEqual(logps.shape, (2, 10))
    np.testing.assert_array_equal(logps, np.zeros((2, 10), dtype=np.float32))

  def test_update_params(self):
    m = self._create_mock_rollout()
    # update_params is a no-op, just ensure it doesn't raise
    m.update_params({"dummy": "tree"})

  @parameterized.named_parameters(
      {
          "testcase_name": "without_tokenizer",
          "tokenizer": None,
          "expected_pad_id": 5,
          "expected_eos_id": 10,
      },
      {
          "testcase_name": "with_callable_tokenizer_methods",
          "tokenizer": FakeTokenizer(pad_id=7, eos_id=14),
          "expected_pad_id": 7,
          "expected_eos_id": 14,
      },
      {
          "testcase_name": "with_property_tokenizer_attributes",
          "tokenizer": FakeTokenizerProperties(pad_id=8, eos_id=16),
          "expected_pad_id": 8,
          "expected_eos_id": 16,
      },
  )
  def test_pad_id_eos_id(self, tokenizer, expected_pad_id, expected_eos_id):
    if tokenizer is None:
      m = self._create_mock_rollout(
          pad_id=expected_pad_id, eos_id=expected_eos_id
      )
    else:
      m = self._create_mock_rollout(tokenizer=tokenizer)
    self.assertEqual(m.pad_id(), expected_pad_id)
    self.assertEqual(m.eos_id(), expected_eos_id)

  def test_model_property(self):
    dummy_model = {"weights": [1, 2, 3]}
    m = self._create_mock_rollout(rollout_actor=dummy_model)
    self.assertEqual(m.model(), dummy_model)

  @parameterized.named_parameters(
      {
          "testcase_name": "uniform_default",
          "dist": "uniform",
          "mean": None,
          "std": None,
          "expected_min": 1,
          "expected_max": 15,
      },
      {
          "testcase_name": "uniform_custom",
          "dist": "uniform",
          "mean": 10.0,
          "std": 2.0,
          "expected_min": 6,
          "expected_max": 14,
      },
      {
          "testcase_name": "normal_default",
          "dist": "normal",
          "mean": None,
          "std": None,
          "expected_min": 1,
          "expected_max": 15,
      },
      {
          "testcase_name": "skewed_default",
          "dist": "skewed",
          "mean": None,
          "std": None,
          "expected_min": 1,
          "expected_max": 15,
      },
      {
          "testcase_name": "fixed",
          "dist": "fixed",
          "mean": 8.0,
          "std": None,
          "expected_min": 8,
          "expected_max": 8,
      },
  )
  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_distributions(
      self, mock_sleep, dist, mean, std, expected_min, expected_max
  ):
    m = self._create_mock_rollout(
        length_distribution=dist,
        length_mean=mean,
        length_std=std,
    )
    out = m.generate(["prompt"] * 5, rollout_config=self.base_rc)

    self.assertLen(out.text, 5)
    lengths = [len(tokens) for tokens in out.tokens]
    # Verify all lengths are within bounds
    self.assertTrue(
        all(expected_min <= length <= expected_max for length in lengths),
        f"Lengths {lengths} not in bounds [{expected_min}, {expected_max}] for"
        f" dist={dist}",
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "normal_custom",
          "dist": "normal",
          "mean": 200.0,
          "std": 30.0,
          "expected_mean": 200.0,
          "expected_std": 30.0,
      },
      {
          "testcase_name": "skewed_custom",
          "dist": "skewed",
          "mean": 100.0,
          "std": 20.0,
          "expected_mean": 100.0,
          "expected_std": 20.0,
      },
      {
          "testcase_name": "normal_default",
          "dist": "normal",
          "mean": None,
          "std": None,
          "expected_mean": 250.0,
          "expected_std": 500.0 / 6.0,
      },
      {
          "testcase_name": "skewed_default",
          "dist": "skewed",
          "mean": None,
          "std": None,
          "expected_mean": 125.0,
          "expected_std": 500.0 / 8.0,
      },
  )
  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_distributions_statistical(
      self, mock_sleep, dist, mean, std, expected_mean, expected_std
  ):
    rc = dataclasses.replace(
        self.base_rc,
        max_tokens_to_generate=500,
    )
    m = self._create_mock_rollout(
        length_distribution=dist,
        length_mean=mean,
        length_std=std,
    )
    # Generate a large batch to check statistics
    batch_size = 1000
    out = m.generate(["prompt"] * batch_size, rollout_config=rc)

    lengths = [len(tokens) for tokens in out.tokens]
    empirical_mean = np.mean(lengths)
    empirical_std = np.std(lengths)

    # Check that empirical values are close to expected
    self.assertAlmostEqual(empirical_mean, expected_mean, delta=10.0)
    self.assertAlmostEqual(empirical_std, expected_std, delta=5.0)

  @mock.patch.object(mock_rollout.logging, "warning", autospec=True)
  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_unknown_distribution_fallback(
      self, mock_sleep, mock_warning
  ):
    m = self._create_mock_rollout(length_distribution="unknown_dist")
    out = m.generate(["prompt"], rollout_config=self.base_rc)
    self.assertLen(out.text, 1)
    self.assertBetween(
        len(out.tokens[0]), 1, self.base_rc.max_tokens_to_generate
    )
    mock_warning.assert_called_once_with(
        "Unknown length distribution %r, falling back to uniform",
        "unknown_dist",
    )

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_tokenizer_returns_empty(self, mock_sleep):
    class EmptyTokenizer:

      def encode(self, text):
        return []

      def decode(self, tokens):
        return ""

    tokenizer = EmptyTokenizer()
    m = self._create_mock_rollout(tokenizer=tokenizer)
    out = m.generate(["prompt"], rollout_config=self.base_rc)

    self.assertLen(out.text, 1)
    self.assertLen(out.tokens[0], 1)

  @mock.patch.object(mock_rollout.time, "sleep", autospec=True)
  def test_generate_empty_prompts(self, mock_sleep):
    m = self._create_mock_rollout()
    out = m.generate([], rollout_config=self.base_rc)

    self.assertEmpty(out.text)
    self.assertEmpty(out.logits)
    self.assertEmpty(out.tokens)
    self.assertEqual(
        out.left_padded_prompt_tokens.shape, (0, self.base_rc.max_prompt_length)
    )
    self.assertEmpty(out.logprobs)

  def test_get_target_length_max_tokens_one(self):
    m = self._create_mock_rollout(length_distribution="skewed")
    length = m._get_target_length(max_tokens=1)
    self.assertEqual(length, 1)

    length = m._get_target_length(max_tokens=0)
    self.assertEqual(length, 0)


if __name__ == "__main__":
  absltest.main()
