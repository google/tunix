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
      ("without_tokenizer", None, 5, 10),
      (
          "with_callable_tokenizer_methods",
          FakeTokenizer(pad_id=7, eos_id=14),
          7,
          14,
      ),
      (
          "with_property_tokenizer_attributes",
          FakeTokenizerProperties(pad_id=8, eos_id=16),
          8,
          16,
      ),
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


if __name__ == "__main__":
  absltest.main()
