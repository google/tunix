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

import dataclasses
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import sampler as sampler_lib
from tunix.generate import utils
from tunix.models.gemma4 import model as gemma4_model_lib
from tunix.tests import test_common as tc


@dataclasses.dataclass(kw_only=True)
class ModelConfigWithDtype(tc.ModelConfig):
  dtype: jax.numpy.dtype = jax.numpy.bfloat16


class SamplerTest(parameterized.TestCase):

  def assertReasonableTensor(self, array, expected_shape=None):
    self.assertIsNotNone(array)
    if expected_shape is not None:
      self.assertEqual(array.shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='fallback',
          config_class=tc.ModelConfig,
          expected_dtype=jax.numpy.float32,
      ),
      dict(
          testcase_name='from_config',
          config_class=ModelConfigWithDtype,
          expected_dtype=jax.numpy.bfloat16,
      ),
  )
  def test_dtype(self, config_class, expected_dtype):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=config_class(vocab_size=vocab.GetPieceSize()),
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
    self.assertEqual(sampler.dtype, expected_dtype)

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

  def test_multimodal_samples(self):
    vocab = tc.MockVocab(is_multimodal=True)
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(
            vocab_size=vocab.GetPieceSize(), vision_config=tc.VisionConfig()
        ),
        rngs=nnx.Rngs(42),
    )

    class DummyImageProcessor:

      def __call__(self, images):
        # returns dummy processed images
        return np.ones((len(images), 1, 32, 32, 3), dtype=np.float32)

    image_processor = DummyImageProcessor()

    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=64,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
        image_processor=image_processor,  # pytype: disable=wrong-arg-types
    )

    max_generation_steps = 8

    # We pass in 2 strings and 2 corresponding dummy images
    images = [
        np.zeros((32, 32, 3)),
        np.zeros((32, 32, 3)),
    ]

    result = sampler(
        [
            'quantization <soi> <img> <img> Tunix',
            '<soi> <img> <img> Parallax distributed',
        ],
        max_generation_steps=max_generation_steps,
        return_logits=True,
        max_prompt_length=8,
        echo=True,
        images=images,
    )

    self.assertIsNotNone(result)
    self.assertReasonableTensor(result.tokens)
    self.assertReasonableTensor(result.logits)
    np.testing.assert_allclose(
        result.tokens,
        np.array([
            [1, 21, 23, 22, 22, 14, 8, 25, 8, 25, 8, 25, 8, 25],
            [1, 23, 22, 22, 15, 18, 8, 25, 8, 25, 8, 25, 8, 25],
        ]),
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

  def test_logprobs(self):
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
    # Test greedy logprobs
    result = sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        return_logprobs=True,
    )
    self.assertIsNotNone(result.logprobs)
    self.assertLen(result.logprobs, 2)
    for logprobs, tokens in zip(result.logprobs, result.tokens):
      self.assertNotEmpty(logprobs)
      self.assertLen(logprobs, tokens.shape[0])

    # Test top_p logprobs
    top_p_result = sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        return_logprobs=True,
        temperature=1.0,
        top_p=0.9,
    )
    self.assertIsNotNone(top_p_result.logprobs)
    self.assertLen(top_p_result.logprobs, 2)
    for logprobs, tokens in zip(top_p_result.logprobs, top_p_result.tokens):
      self.assertNotEmpty(logprobs)
      self.assertLen(logprobs, tokens.shape[0])

    # Test beam search logprobs
    beam_result = sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        return_logprobs=True,
        beam_size=2,
    )
    self.assertIsNotNone(beam_result.logprobs)
    self.assertLen(beam_result.logprobs, 2)
    for logprobs, tokens in zip(beam_result.logprobs, beam_result.tokens):
      self.assertNotEmpty(logprobs)
      self.assertLen(logprobs, tokens.shape[0])

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

  def test_decode_stops_after_prefill_for_single_generation_step(self):
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
    sampler.eos_ids = jnp.array([vocab.eos_id()])
    max_prompt_length = 4
    max_generation_steps = 1
    prompt_tokens = sampler.tokenize('input string')
    all_input_ids = jnp.array([
        utils.pad_to_length(
            prompt_tokens,
            target_length=max_prompt_length,
            pad_value=vocab.pad_id(),
            left=True,
        )
    ])
    total_sampling_steps = max_prompt_length + max_generation_steps
    sampling_state = sampler.init_sample_state(
        all_input_ids=all_input_ids,
        total_sampling_steps=total_sampling_steps,
        include_logits=False,
        forbidden_token_ids=None,
        temperature=0.0,
        top_p=None,
        top_k=None,
        seed=jax.random.PRNGKey(0),
        beam_size=None,
        include_logprobs=False,
    )

    after_prefill = sampler._prefill_fn(
        sampler._flattened_transformer_state, sampling_state, None, echo=False
    )
    self.assertEqual(after_prefill.decoding_step, total_sampling_steps - 1)

    after_decode = sampler._decode_fn(
        sampler._flattened_transformer_state, after_prefill
    )
    self.assertEqual(after_decode.decoding_step, total_sampling_steps - 1)
    np.testing.assert_array_equal(
        np.asarray(after_decode.token_buffer),
        np.asarray(after_prefill.token_buffer),
    )

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
        seed=0,
    )
    np.testing.assert_equal(
        result.tokens, [np.array([14]), np.array([12, 1, 17])]
    )

  def test_forbidden_token_ids(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=128,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )

    vocab_size = vocab.GetPieceSize()
    num_allowed_tokens = vocab_size // 4
    forbidden_tokens = set(range(num_allowed_tokens, vocab_size))
    # EOS is forbidden so we are sure to get a full length generation.
    forbidden_tokens.add(vocab.eos_id())
    max_generation_steps = 100

    result = sampler(
        ['input string'],
        max_generation_steps=max_generation_steps,
        return_logits=False,
        forbidden_tokens=forbidden_tokens,
        temperature=1.0,  # Ensure some randomness
        seed=123,
    )
    self.assertLen(result.tokens[0], max_generation_steps)
    self.assertNoCommonElements(result.tokens[0], forbidden_tokens)

  def test_gemma4_smoke_test(self):
    """Runs a sampling call with a dummy Gemma4 config.

    Useful to catch JAX compilation and model implementation errors early.
    """
    config = gemma4_model_lib.ModelConfig(
        num_layers=2,
        num_embed=32,
        embed_dim=16,
        hidden_dim=16,
        num_heads=4,
        head_dim=16,
        num_kv_heads=1,
        per_layer_input_dim=16,
        sliding_window_size=4,
        param_dtype=jnp.bfloat16,
        attention_pattern=(
            gemma4_model_lib.AttentionType.LOCAL_SLIDING,
            gemma4_model_lib.AttentionType.GLOBAL,
        ),
        final_logit_softcap=30.0,
        local_rope_proportion=1.0,
        global_rope_proportion=0.25,
        global_key_size=16,
        k_eq_v_global=False,
        local_base_frequency=10000,
        global_base_frequency=1000000,
        local_scale_factor=1.0,
        global_scale_factor=1.0,
    )
    rngs = nnx.Rngs(0)
    model = gemma4_model_lib.Gemma4(config, rngs=rngs)
    cache_config = sampler_lib.CacheConfig(
        cache_size=32,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
    )
    mock_tokenizer = tc.MockVocab()
    mock_tokenizer.DecodeIds = mock.MagicMock()
    mock_tokenizer.DecodeIds.return_value = 'decoded_string'
    sampler = sampler_lib.Sampler(model, mock_tokenizer, cache_config)
    sampler(
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=10,
    )

  def test_gemma4_decode_only_last_token_consistency(self):
    """Verifies that decode_only_last_token yields identical generated tokens and logits."""
    config = gemma4_model_lib.ModelConfig(
        num_layers=2,
        num_embed=32,
        embed_dim=16,
        hidden_dim=16,
        num_heads=4,
        head_dim=16,
        num_kv_heads=1,
        per_layer_input_dim=16,
        sliding_window_size=4,
        param_dtype=jnp.bfloat16,
        attention_pattern=(
            gemma4_model_lib.AttentionType.LOCAL_SLIDING,
            gemma4_model_lib.AttentionType.GLOBAL,
        ),
        final_logit_softcap=30.0,
        local_rope_proportion=1.0,
        global_rope_proportion=0.25,
        global_key_size=16,
        k_eq_v_global=False,
        local_base_frequency=10000,
        global_base_frequency=1000000,
        local_scale_factor=1.0,
        global_scale_factor=1.0,
    )
    rngs = nnx.Rngs(42)
    model = gemma4_model_lib.Gemma4(config, rngs=rngs)
    cache_config = sampler_lib.CacheConfig(
        cache_size=32,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
    )
    mock_tokenizer = tc.MockVocab()
    mock_tokenizer.DecodeIds = mock.MagicMock()
    mock_tokenizer.DecodeIds.return_value = 'decoded_string'

    # Run 1: Optimized (decode_only_last_token = True)
    sampler_opt = sampler_lib.Sampler(model, mock_tokenizer, cache_config)
    self.assertTrue(sampler_opt._supports_decode_only_last_token)
    res_opt = sampler_opt(
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=10,
        return_logits=True,
        echo=False,
    )

    # Run 2: Unoptimized (force decode_only_last_token = False)
    sampler_unopt = sampler_lib.Sampler(model, mock_tokenizer, cache_config)
    sampler_unopt._supports_decode_only_last_token = False
    res_unopt = sampler_unopt(
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=10,
        return_logits=True,
        echo=False,
    )

    # Verify tokens and generated logits are identical
    self.assertEqual(len(res_opt.tokens), len(res_unopt.tokens))
    for t_opt, t_unopt in zip(res_opt.tokens, res_unopt.tokens):
      np.testing.assert_array_equal(t_opt, t_unopt)
    self.assertEqual(len(res_opt.logits), len(res_unopt.logits))
    for l_opt, l_unopt in zip(res_opt.logits, res_unopt.logits):
      self.assertEqual(l_opt.shape, l_unopt.shape)
      np.testing.assert_allclose(l_opt, l_unopt, atol=1e-5, rtol=1e-5)

  def test_sampler_gemma4_multimodal(self):
    vocab = tc.MockVocab(
        mapping_text_to_id={
            '<pad>': 0,
            '<s>': 1,
            '</s>': 2,
            'Describe:': 3,
            '<img>': 258880,
            '<soi>': 255999,
            '<eoi>': 258882,
            '<audio>': 258881,
            '<soa>': 256000,
            '<eoa>': 258883,
        }
    )
    # Since there are a lot of holes (missing ids) in our vocab.
    vocab.DecodeIds = mock.MagicMock()
    vocab.DecodeIds.return_value = 'decoded_string'

    config = gemma4_model_lib.ModelConfig.gemma4_e2b()
    config = dataclasses.replace(
        config,
        num_layers=1,
        num_heads=2,
        head_dim=16,
        embed_dim=32,
        hidden_dim=64,
        num_embed=vocab.GetPieceSize(),
        frac_shared_layers=0.0,
    )
    config.vision_encoder = gemma4_model_lib.vision.VisionEncoderConfig(
        d_model=16,
        num_layers=1,
        num_heads=2,
        ffw_hidden=32,
        patch_size=4,
        output_length=5,
    )
    config.audio_encoder = gemma4_model_lib.audio.ConformerConfig(
        num_layers=1,
        model_dims=16,
        atten_num_heads=2,
        lm_model_dims=32,
    )

    rngs = nnx.Rngs(42)
    transformer = gemma4_model_lib.Gemma4(config, rngs=rngs, text_only=False)

    sampler = sampler_lib.Sampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=128,
            num_layers=1,
            num_kv_heads=1,
            head_dim=16,
        ),
    )

    # Let prompt contain image placeholder tag
    prompt = "Describe: <img> <audio>"
    dummy_image = np.ones((16, 16, 3), dtype=np.uint8)
    dummy_audio = np.zeros(16000, dtype=np.float32)

    result = sampler(
        [prompt],
        images=[dummy_image],
        audios=[dummy_audio],
        max_prompt_length=32,
        max_generation_steps=5,
    )

    self.assertIsNotNone(result)
    self.assertIsNotNone(result.tokens)
    self.assertGreater(len(result.tokens[0]), 0)


if __name__ == '__main__':
  absltest.main()
