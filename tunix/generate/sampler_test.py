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

"""Tests for the Sampler integration, focusing on top-k sampling."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from tunix.generate import sampler as sampler_lib
from tunix.generate import top_k as top_k_lib # For referencing sample_top_k if needed for oracle

# Mock Tokenizer
class MockTokenizer:
  def __init__(self, vocab_size=100, pad_id=0, bos_id=1, eos_id=2):
    self.vocab_size = vocab_size
    self._pad_id = pad_id
    self._bos_id = bos_id
    self._eos_id = eos_id

  def encode(self, s: str) -> list[int]:
    # Simple fixed encoding for testing
    if s == "hello":
      return [10, 20, 30]
    elif s == "world":
      return [40, 50]
    return [len(s)]

  def decode(self, ids: list[int]) -> str:
    return f"decoded_{ids}"

  def pad_id(self) -> int:
    return self._pad_id

  def bos_id(self) -> int:
    return self._bos_id

  def eos_id(self) -> int:
    return self._eos_id

# Mock Transformer
class MockTransformer(nnx.Module):
  def __init__(self, vocab_size: int, *, rngs: nnx.Rngs):
    self.vocab_size = vocab_size
    self.num_embed = vocab_size # Ensure num_embed is present
    self.dummy_param = nnx.Param(jnp.zeros(1)) # Add a dummy parameter

  def __call__(self, tokens, positions, cache, attention_mask, output_hidden_states=False):
    batch_size, seq_len = tokens.shape
    # Return fixed logits for predictability in tests
    # For example, make the logit for token_id 'i' be 'i' itself, cycling if necessary
    logits_output = jnp.arange(self.vocab_size, dtype=jnp.float32)
    logits_output = jnp.tile(logits_output, (batch_size, seq_len, 1))

    # Create a dummy cache structure similar to what Sampler expects
    # This needs to align with CacheConfig used in the Sampler
    new_cache = cache # In a real model, cache would be updated
    if output_hidden_states:
        # Add dummy hidden states if requested
        dummy_hidden_states = jnp.zeros((batch_size, seq_len, 10)) # 10 is arbitrary embed_dim
        self.all_hidden_states = nnx.Variable(dummy_hidden_states, collection='intermediates')


    return logits_output, new_cache


class SamplerTopKIntegrationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab_size = 50
    self.mock_tokenizer = MockTokenizer(vocab_size=self.vocab_size)

    # CacheConfig similar to what might be used
    self.cache_config = sampler_lib.CacheConfig(
        cache_size=128,
        num_layers=1,
        num_kv_heads=1,
        head_dim=1
    )
    self.transformer_key = nnx.Rngs(params=jax.random.key(0))


  def test_top_k_sampling_mode_selection(self):
    mock_transformer = MockTransformer(self.vocab_size, rngs=self.transformer_key)
    sampler = sampler_lib.Sampler(
        transformer=mock_transformer,
        tokenizer=self.mock_tokenizer,
        cache_config=self.cache_config
    )

    input_ids = jnp.array([[10, 20]]) # dummy input
    seed = jax.random.PRNGKey(42)

    # Test that top_k mode is selected when top_k is provided and others are not
    state = sampler.init_sample_state(
        all_input_ids=input_ids,
        total_sampling_steps=10,
        include_logits=False,
        forbidden_token_ids=None,
        temperature=1.0,
        top_p=None,
        top_k=5, # Set top_k
        penalty_alpha=None,
        seed=seed,
        beam_size=None
    )
    self.assertEqual(state.sampling_mode, "top_k")
    self.assertEqual(state.sampling_parameters['top_k'], 5)

  @parameterized.parameters(
      {"top_k_val": 3, "temperature": 1.0, "gen_len": 5, "seed": 0},
      {"top_k_val": 1, "temperature": 0.5, "gen_len": 3, "seed": 1},
      {"top_k_val": 10, "temperature": 1.5, "gen_len": 4, "seed": 2},
  )
  def test_top_k_end_to_end_sampling(self, top_k_val, temperature, gen_len, seed):
    mock_transformer = MockTransformer(self.vocab_size, rngs=self.transformer_key)
    sampler = sampler_lib.Sampler(
        transformer=mock_transformer,
        tokenizer=self.mock_tokenizer,
        cache_config=self.cache_config
    )

    prng_seed = jax.random.PRNGKey(seed)

    # The mock transformer always returns logits where logit[i] = i
    # So, the top_k tokens will be [vocab_size-1, vocab_size-2, ..., vocab_size-k]
    expected_top_k_indices = jnp.arange(self.vocab_size - top_k_val, self.vocab_size)[::-1]

    outputs = sampler(
        input_strings=["hello"],
        total_generation_steps=gen_len,
        temperature=temperature,
        top_k=top_k_val,
        seed=prng_seed
    )

    self.assertLen(outputs.text, 1)
    self.assertLen(outputs.tokens, 1)
    generated_tokens = outputs.tokens[0]

    # The prompt tokens are from mock_tokenizer.encode("hello") -> [1, 10, 20, 30] (bos + encoded)
    # Bos id is 1, "hello" is [10,20,30]
    # padded_prompt_tokens are left padded to max_prompt_length
    # max_prompt_length for a single string "hello" (len 3 + bos) is 4
    prompt_len = len(self.mock_tokenizer.encode("hello")) + (1 if self.mock_tokenizer.bos_id() else 0)

    # Check only generated tokens (after prompt)
    for i in range(gen_len):
      token_idx_in_output = i
      # If echo=False (default), tokens list only contains generated tokens
      # If echo=True, it would be prompt_len + i
      # Based on current Sampler impl, output.tokens are *only* generated ones if echo=False

      # Because our mock model always returns the same logits, every step will sample from the same set
      self.assertIn(generated_tokens[token_idx_in_output], expected_top_k_indices)

  def test_top_k_sampling_with_k_greater_than_vocab(self):
    top_k_val = self.vocab_size + 10 # K > vocab_size
    temperature = 1.0
    gen_len = 3
    prng_seed = jax.random.PRNGKey(3)

    mock_transformer = MockTransformer(self.vocab_size, rngs=self.transformer_key)
    sampler = sampler_lib.Sampler(
        transformer=mock_transformer,
        tokenizer=self.mock_tokenizer,
        cache_config=self.cache_config
    )

    # All tokens are candidates
    expected_top_k_indices = jnp.arange(self.vocab_size)

    outputs = sampler(
        input_strings=["world"],
        total_generation_steps=gen_len,
        temperature=temperature,
        top_k=top_k_val,
        seed=prng_seed
    )
    generated_tokens = outputs.tokens[0]
    for i in range(gen_len):
        self.assertIn(generated_tokens[i], expected_top_k_indices)


if __name__ == "__main__":
  absltest.main()
