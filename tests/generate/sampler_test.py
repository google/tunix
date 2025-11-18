# Copyright 2025 Google LLC
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

# Forked from flax/examples/gemma/sampler_test.py

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import numpy as np
from tunix.generate import sampler as sampler_lib
from tunix.generate import utils
from tunix.tests import test_common as tc


class SamplerTest(parameterized.TestCase):

  def assertReasonableTensor(self, array, expected_shape=None):
    self.assertIsNotNone(array)
    if expected_shape is not None:
      self.assertEqual(array.shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='case1',
          max_prompt_length=None,
          echo=False,
          return_logits=False,
      ),
      dict(
          testcase_name='case2',
          max_prompt_length=4,
          echo=True,
          return_logits=True,
      ),
      dict(
          testcase_name='case3',
          max_prompt_length=4,
          echo=False,
          return_logits=False,
      ),
      dict(
          testcase_name='case4',
          max_prompt_length=1,
          echo=False,
          return_logits=True,
      ),
  )
  def test_samples_padding_output(self, max_prompt_length, echo, return_logits):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=64,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    max_generation_steps = 10
    result_padded = sampler(
        ['input string', 'hello world'],
        max_generation_steps=max_generation_steps,
        return_logits=return_logits,
        max_prompt_length=max_prompt_length,
        echo=echo,
        pad_output=True,
    )

    result_not_padded = sampler(
        ['input string', 'hello world'],
        max_generation_steps=max_generation_steps,
        return_logits=return_logits,
        max_prompt_length=max_prompt_length,
        echo=echo,
    )

    for i in range(len(result_not_padded.text)):
      self.assertEqual(result_not_padded.text[i], result_padded.text[i])
      if return_logits:
        valid_length = (
            utils.find_last_non_pad_idx(result_padded.tokens[i], vocab.pad_id())
            + 1
        )
        np.testing.assert_allclose(
            result_not_padded.logits[i],
            result_padded.logits[i][:valid_length],
        )
        np.testing.assert_allclose(
            result_not_padded.tokens[i],
            result_padded.tokens[i][:valid_length],
        )
        if not echo:
          np.testing.assert_equal(
              result_padded.tokens[i].shape[0], max_generation_steps
          )

  @parameterized.named_parameters(
      dict(
          testcase_name='case1',
          max_prompt_length=None,
          echo=False,
      ),
      dict(
          testcase_name='case2',
          max_prompt_length=4,
          echo=True,
      ),
      dict(
          testcase_name='case3',
          max_prompt_length=4,
          echo=False,
      ),
      dict(
          testcase_name='case4',
          max_prompt_length=1,
          echo=False,
      ),
  )
  def test_samples(self, max_prompt_length, echo):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=64,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    result = sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        return_logits=True,
        max_prompt_length=max_prompt_length,
        echo=echo,
    )
    self.assertIsNotNone(result)
    self.assertLen(result.logits, 2)
    if echo:
      self.assertEqual(result.logits[0].shape, (13, vocab.GetPieceSize()))
    else:
      self.assertEqual(result.logits[0].shape, (10, vocab.GetPieceSize()))

    # With 1 beam, the beam search result should be the
    # same as the greedy output
    result_beam_search_1 = sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        return_logits=True,
        max_prompt_length=max_prompt_length,
        echo=echo,
        beam_size=1,
    )
    self.assertIsNotNone(result_beam_search_1)
    self.assertEqual(result_beam_search_1.text, result.text)

    # Check with multiple beams, it still works.
    result_beam_search_2 = sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        return_logits=True,
        max_prompt_length=max_prompt_length,
        echo=echo,
        beam_size=2,
    )
    self.assertIsNotNone(result_beam_search_2)

    top_p_result = sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        temperature=9,
        top_p=0.95,
        echo=echo,
    )
    self.assertIsNotNone(top_p_result)
    self.assertNotEqual(result.text, top_p_result.text)

    top_p_result_2 = sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        temperature=9,
        top_p=0.95,
        seed=42,
        echo=echo,
    )
    self.assertIsNotNone(top_p_result_2)
    self.assertNotEqual(top_p_result.text, top_p_result_2.text)

    top_k_result = sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        temperature=9,
        top_p=0.95,
        top_k=3,
        seed=42,
        echo=echo,
    )
    self.assertIsNotNone(top_k_result)
    self.assertNotEqual(top_p_result_2.text, top_k_result.text)

  def test_prompt_padding_bucketization(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=64,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    self.assertEqual(sampler._compiled_prefill_fn._cache_size(), 0)  # pytype: disable=attribute-error
    sampler(
        ['input', 'hello'],
        max_generation_steps=10,
    )
    self.assertEqual(sampler._compiled_prefill_fn._cache_size(), 1)  # pytype: disable=attribute-error

    sampler(
        ['input input input input input', 'hello hello'],
        max_generation_steps=10,
    )

    sampler(
        ['input input input input input input', 'hello hello'],
        max_generation_steps=10,
    )
    self.assertEqual(sampler._compiled_prefill_fn._cache_size(), 2)  # pytype: disable=attribute-error

  def test_state_update(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    input_strings = ['input string', 'hello world']
    original_logits = sampler(
        input_strings, max_generation_steps=10, return_logits=True
    ).logits

    new_transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler.transformer_state = nnx.variables(new_transformer, nnx.Param)
    new_logits = sampler(
        input_strings, max_generation_steps=10, return_logits=True
    ).logits
    with self.assertRaises(AssertionError):
      for orig, new in zip(original_logits, new_logits):
        np.testing.assert_allclose(orig, new, atol=1e-1, rtol=1e-1)

  def test_lora_state_update(self):
    vocab = tc.MockVocab()
    transformer = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
            rngs=nnx.Rngs(0),
        )
    )

    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    input_strings = ['input string', 'hello world']
    original_logits = sampler(
        input_strings, max_generation_steps=10, return_logits=True
    ).logits

    new_transformer = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
            rngs=nnx.Rngs(42),
        )
    )
    # Since LoRA_b is initialized to 0, we need to add a small perturbation to
    # the LoRA params to make sure that the new params are different from the
    # original params.
    new_lora_params = nnx.variables(new_transformer, nnx.LoRAParam)
    new_lora_params = jax.tree.map(lambda x: x + 0.1, new_lora_params)

    sampler.transformer_state = new_lora_params
    new_logits = sampler(
        input_strings, max_generation_steps=10, return_logits=True
    ).logits
    with self.assertRaises(AssertionError):
      for orig, new in zip(original_logits, new_logits):
        np.testing.assert_allclose(orig, new, atol=1e-1, rtol=1e-1)

  def test_invalid_state_update(self):
    vocab = tc.MockVocab()

    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize(), num_layers=4),
        rngs=nnx.Rngs(0),
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )

    new_transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize(), num_layers=6),
        rngs=nnx.Rngs(42),
    )
    with self.assertRaisesRegex(ValueError, '.*must have the same structure.*'):
      sampler.transformer_state = nnx.variables(new_transformer, nnx.Param)

  def test_invalid_lora_state_update(self):
    vocab = tc.MockVocab()

    transformer = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(
                vocab_size=vocab.GetPieceSize(), num_layers=4
            ),
            rngs=nnx.Rngs(0),
        )
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )

    new_transformer = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(
                vocab_size=vocab.GetPieceSize(), num_layers=6
            ),
            rngs=nnx.Rngs(42),
        )
    )
    with self.assertRaisesRegex(ValueError, '.*must have the same structure.*'):
      sampler.transformer_state = nnx.variables(new_transformer, nnx.LoRAParam)

  def test_eos_tokens(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=64,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    result = sampler(
        ['input string training', 'hello world'],
        max_generation_steps=10,
        return_logits=True,
        max_prompt_length=4,
        eos_tokens=[7, 21],
        temperature=0.9,
        top_p=1.0,
        seed=42,
    )
    np.testing.assert_equal(
        result.tokens, [np.array([8, 14, 5]), np.array([14])]
    )


if __name__ == '__main__':
  absltest.main()
