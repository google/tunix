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

"""Tests for VanillaSampler in experimental/rollout."""
# Forked from flax/examples/gemma/sampler_test.py

import asyncio
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.common import configs
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.rollout import vanilla_sampler
from tunix.experimental.weight_sync import weight_sync
from tunix.generate import beam_search as beam_search_lib
from tunix.generate import utils
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma4 import model as gemma4_model_lib
from tunix.processors import audio_processor as audio_processor_lib
from tunix.processors import image_processor as image_processor_lib
from tunix.tests import test_common as tc


def _run_prefill_and_decode(
    sampler: vanilla_sampler.VanillaSampler,
    input_strings=None,
    *,
    prompt_token_ids=None,
    max_generation_steps: int = 10,
    max_prompt_length: int | None = None,
    echo: bool = False,
    pad_output: bool = False,
    forbidden_tokens=None,
    images=None,
    audios=None,
    max_audio_length: int | None = None,
    max_audio_clips: int | None = None,
):
  """Helper exercising VanillaSampler's prefill and decode primitives directly."""
  return_logits = sampler.config.return_logits
  return_logprobs = sampler.config.return_logprobs
  beam_size = sampler.config.beam_size
  exact_input = prompt_token_ids is not None
  tokens = utils.resolve_prompt_tokens(
      input_strings,
      prompt_token_ids,
      sampler.tokenize,
      max_generation_steps=max_generation_steps,
      max_total_length=sampler.cache_size,
      max_length_name='cache_size',
      single_output_per_row=beam_size is None,
  )
  forbidden_token_ids = tuple(forbidden_tokens) if forbidden_tokens else None

  is_gemma4 = sampler.transformer.__class__.__name__ == 'Gemma4'
  processed_images = None
  if is_gemma4 and images is not None:
    processed_images, tokens = image_processor_lib.process_gemma4_inputs(
        images,
        tokens,
        sampler.transformer.vision_encoder,
        sampler.tokenizer.pad_id(),
    )
  elif images is not None and sampler.image_processor is not None:
    processed_images = jnp.array(sampler.image_processor(images))

  processed_audios = None
  if audios is not None and is_gemma4:
    processed_audios, tokens = audio_processor_lib.process_gemma4_inputs(
        audios=audios,
        tokens=tokens,
        audio_encoder=sampler.transformer.audio_encoder,
        max_audio_length=max_audio_length,
        max_audio_clips=max_audio_clips,
    )

  all_input_ids, prompt_lengths, max_prompt_length = (
      utils.left_pad_prompt_tokens(
          tokens,
          max_prompt_length,
          sampler.tokenizer.pad_id(),
          max_allowed_length=sampler.cache_size - max_generation_steps,
      )
  )
  total_sampling_steps = max_prompt_length + max_generation_steps
  sampling_state = sampler.init_sample_state(
      jnp.array(all_input_ids),
      total_sampling_steps=total_sampling_steps,
      forbidden_token_ids=forbidden_token_ids,
      prompt_lengths=(
          jnp.asarray(prompt_lengths, dtype=jnp.int32)
          if exact_input
          else None
      ),
  )
  sampling_state = sampler._compiled_prefill_fn(
      sampler._flattened_transformer_state,
      sampling_state,
      images=processed_images,
      audios=processed_audios,
      echo=echo,
  )
  sampling_state = sampler._compiled_decode_fn(
      sampler._flattened_transformer_state, sampling_state
  )
  token_buffers = sampling_state.token_buffer
  logits_buffers = sampling_state.logits_buffer
  final_logprobs_buffer = sampling_state.logprobs_buffer

  if sampling_state.sampling_mode == 'beam_search':
    updated_args = beam_search_lib.finalize_beam_search_state(
        sampling_state.beam_search_sampling_state,
        sampling_state.token_buffer,
        sampling_state.logits_buffer,
        sampling_state.logprobs_buffer,
    )
    token_buffers = updated_args['token_buffer']
    logits_buffers = updated_args['logits_buffer']
    final_logprobs_buffer = updated_args['logprobs_buffer']

  def _output_slice_bounds(token_buffer: np.ndarray, prompt_length: int):
    if not echo:
      start_idx = max_prompt_length
    elif exact_input:
      start_idx = max_prompt_length - int(prompt_length)
    else:
      start_idx = utils.np_find_first_non_pad_idx(
          token_buffer, sampler.tokenizer.pad_id()
      )
    end_idx = (
        utils.np_find_first_eos_idx(
            token_buffer[max_prompt_length:], sampler.eos_ids
        )
        + max_prompt_length
    )
    return start_idx, end_idx

  if pad_output:
    max_len = total_sampling_steps if echo else max_generation_steps
    lengths, out_tokens, out_logits = utils.padded_fill_tokens_and_logits(
        token_buffers,
        logits_buffers,
        return_logits,
        echo,
        sampler.tokenizer.pad_id(),
        sampler.eos_ids,
        max_prompt_length,
        max_len,
        jnp.asarray(prompt_lengths, dtype=jnp.int32) if exact_input else None,
    )
    out_tokens, lengths = jax.device_get(out_tokens), jax.device_get(lengths)
    decoded_outputs = [
        sampler.tokenizer.decode(toks[:length].tolist())
        for toks, length in zip(out_tokens, lengths)
    ]
    out_logprobs = None
    if return_logprobs and final_logprobs_buffer is not None:
      out_logprobs = []
      token_buffers = jax.device_get(token_buffers)
      final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
      for i, token_buffer in enumerate(token_buffers):
        start_idx, end_idx = _output_slice_bounds(
            token_buffer, int(prompt_lengths[i])
        )
        length = end_idx - start_idx
        sliced_logprobs = final_logprobs_buffer[i][start_idx:end_idx]
        padded_logprobs = np.pad(
            sliced_logprobs,
            (0, max_len - length),
            mode='constant',
            constant_values=0.0,
        )
        out_logprobs.append(padded_logprobs.tolist())
  else:
    out_tokens = []
    out_logits = []
    out_logprobs = [] if return_logprobs else None
    token_buffers = jax.device_get(token_buffers)
    if return_logprobs and final_logprobs_buffer is not None:
      final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
    if return_logits and logits_buffers is not None:
      logits_buffers = jax.device_get(logits_buffers)
    for i, token_buffer in enumerate(token_buffers):
      start_idx, end_idx = _output_slice_bounds(
          token_buffer, int(prompt_lengths[i])
      )
      out_tokens.append(token_buffer[start_idx:end_idx])
      if return_logits and logits_buffers is not None:
        out_logits.append(logits_buffers[i][start_idx:end_idx])
      if return_logprobs and final_logprobs_buffer is not None:
        out_logprobs.append(
            final_logprobs_buffer[i][start_idx:end_idx].tolist()
        )
    decoded_outputs = [
        sampler.tokenizer.decode(toks.tolist()) for toks in out_tokens
    ]

  return (
      decoded_outputs,
      out_tokens,
      out_logits if return_logits else [],
      all_input_ids,
      out_logprobs,
      prompt_lengths,
  )


class VanillaSamplerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = tc.MockVocab()
    self.transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    self.config = configs.RolloutConfig(
        kv_cache_size=64,
        temperature=0.0,
        weight_sync_mode=weight_sync.WeightSyncMode.FALLBACK,
    )
    self.vanilla_sampler = vanilla_sampler.VanillaSampler(
        server_id='tpu_slice_01',
        config=self.config,
        transformer=self.transformer,
        tokenizer=self.vocab,
    )
    self.vanilla_sampler.initialize()

  def _make_sampler(
      self,
      transformer=None,
      tokenizer=None,
      image_processor=None,
      raiden_sync_delegate=None,
      **config_kwargs,
  ) -> vanilla_sampler.VanillaSampler:
    cfg_kwargs = {'kv_cache_size': 64, 'temperature': 0.0}
    cfg_kwargs.update(config_kwargs)
    sampler = vanilla_sampler.VanillaSampler(
        server_id='tpu_slice_test',
        config=configs.RolloutConfig(**cfg_kwargs),
        transformer=transformer if transformer is not None else self.transformer,
        tokenizer=tokenizer if tokenizer is not None else self.vocab,
        image_processor=image_processor,
        raiden_sync_delegate=raiden_sync_delegate,
    )
    sampler.initialize()
    return sampler

  def assertReasonableTensor(self, array, expected_shape=None):
    self.assertIsNotNone(array)
    if expected_shape is not None:
      self.assertEqual(array.shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='default_float32',
          model_dtype=jax.numpy.float32,
      ),
      dict(
          testcase_name='bfloat16',
          model_dtype=jax.numpy.bfloat16,
      ),
  )
  def test_dtype(self, model_dtype):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(
            vocab_size=vocab.GetPieceSize(), dtype=model_dtype
        ),
        rngs=nnx.Rngs(42),
    )
    sampler = self._make_sampler(transformer=transformer, tokenizer=vocab)
    self.assertEqual(sampler.dtype, model_dtype)

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
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        return_logits=return_logits,
    )
    max_generation_steps = 10
    text_padded, tokens_padded, logits_padded, _, _, _ = (
        _run_prefill_and_decode(
            sampler,
            ['input string', 'hello world'],
            max_generation_steps=max_generation_steps,
            max_prompt_length=max_prompt_length,
            echo=echo,
            pad_output=True,
        )
    )
    text_unpadded, tokens_unpadded, logits_unpadded, _, _, _ = (
        _run_prefill_and_decode(
            sampler,
            ['input string', 'hello world'],
            max_generation_steps=max_generation_steps,
            max_prompt_length=max_prompt_length,
            echo=echo,
            pad_output=False,
        )
    )

    for i in range(len(text_unpadded)):
      self.assertEqual(text_unpadded[i], text_padded[i])
      if return_logits:
        valid_length = (
            utils.find_last_non_pad_idx(tokens_padded[i], vocab.pad_id()) + 1
        )
        np.testing.assert_allclose(
            logits_unpadded[i],
            logits_padded[i][:valid_length],
        )
        np.testing.assert_allclose(
            tokens_unpadded[i],
            tokens_padded[i][:valid_length],
        )
        if not echo:
          np.testing.assert_equal(
              tokens_padded[i].shape[0], max_generation_steps
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
        return np.ones((len(images), 1, 32, 32, 3), dtype=np.float32)

    image_processor = DummyImageProcessor()
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        image_processor=image_processor,
        return_logits=True,
    )
    max_generation_steps = 8
    images = [
        np.zeros((32, 32, 3)),
        np.zeros((32, 32, 3)),
    ]
    _, tokens, logits, _, _, _ = _run_prefill_and_decode(
        sampler,
        [
            'quantization <soi> <img> <img> Tunix',
            '<soi> <img> <img> Parallax distributed',
        ],
        max_generation_steps=max_generation_steps,
        max_prompt_length=8,
        echo=True,
        images=images,
    )
    self.assertReasonableTensor(tokens)
    self.assertReasonableTensor(logits)
    np.testing.assert_allclose(
        tokens,
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
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        return_logits=True,
    )
    text, _, logits, _, _, _ = _run_prefill_and_decode(
        sampler,
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=max_prompt_length,
        echo=echo,
    )
    self.assertLen(logits, 2)
    if echo:
      self.assertEqual(logits[0].shape, (13, vocab.GetPieceSize()))
    else:
      self.assertEqual(logits[0].shape, (10, vocab.GetPieceSize()))

    # With 1 beam, the beam search result should be the same as the greedy output
    sampler_beam_1 = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        return_logits=True,
        beam_size=1,
    )
    text_beam_1, _, _, _, _, _ = _run_prefill_and_decode(
        sampler_beam_1,
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=max_prompt_length,
        echo=echo,
    )
    self.assertEqual(text_beam_1, text)

    # Check with multiple beams, it still works.
    sampler_beam_2 = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        return_logits=True,
        beam_size=2,
    )
    text_beam_2, _, _, _, _, _ = _run_prefill_and_decode(
        sampler_beam_2,
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=max_prompt_length,
        echo=echo,
    )
    self.assertIsNotNone(text_beam_2)

    sampler_top_p = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        temperature=9.0,
        top_p=0.95,
        seed=0,
    )
    top_p_text, _, _, _, _, _ = _run_prefill_and_decode(
        sampler_top_p,
        ['input string', 'hello world'],
        max_generation_steps=10,
        echo=echo,
    )
    self.assertNotEqual(text, top_p_text)

    sampler_top_p_2 = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        temperature=9.0,
        top_p=0.95,
        seed=42,
    )
    top_p_text_2, _, _, _, _, _ = _run_prefill_and_decode(
        sampler_top_p_2,
        ['input string', 'hello world'],
        max_generation_steps=10,
        echo=echo,
    )
    self.assertNotEqual(top_p_text, top_p_text_2)

    sampler_top_k = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        temperature=9.0,
        top_p=0.95,
        top_k=3,
        seed=42,
    )
    top_k_text, _, _, _, _, _ = _run_prefill_and_decode(
        sampler_top_k,
        ['input string', 'hello world'],
        max_generation_steps=10,
        echo=echo,
    )
    self.assertNotEqual(top_p_text_2, top_k_text)

  def test_logprobs(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    reqs = [
        base_sampler_lib.SamplingRequest(
            request_id='req_0',
            prompt='input string',
            sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
        ),
        base_sampler_lib.SamplingRequest(
            request_id='req_1',
            prompt='hello world',
            sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
        ),
    ]
    # Test greedy logprobs
    greedy_sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        return_logprobs=True,
        temperature=0.0,
    )
    greedy_responses = asyncio.run(greedy_sampler.sample(reqs))
    self.assertLen(greedy_responses, 2)
    for resp in greedy_responses:
      self.assertIsNotNone(resp.logprobs)
      self.assertNotEmpty(resp.logprobs)
      self.assertLen(resp.logprobs, resp.token_ids.shape[0])

    # Test top_p logprobs
    top_p_sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        return_logprobs=True,
        temperature=1.0,
        top_p=0.9,
    )
    top_p_responses = asyncio.run(top_p_sampler.sample(reqs))
    self.assertLen(top_p_responses, 2)
    for resp in top_p_responses:
      self.assertIsNotNone(resp.logprobs)
      self.assertNotEmpty(resp.logprobs)
      self.assertLen(resp.logprobs, resp.token_ids.shape[0])

    # Test beam search logprobs
    beam_sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        return_logprobs=True,
        beam_size=2,
    )
    beam_responses = asyncio.run(beam_sampler.sample(reqs))
    self.assertLen(beam_responses, 2)
    for resp in beam_responses:
      self.assertIsNotNone(resp.logprobs)
      self.assertNotEmpty(resp.logprobs)
      self.assertLen(resp.logprobs, resp.token_ids.shape[0])

  def test_prompt_padding_bucketization(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = self._make_sampler(transformer=transformer, tokenizer=vocab)
    self.assertEqual(sampler._compiled_prefill_fn._cache_size(), 0)
    asyncio.run(
        sampler.sample([
            base_sampler_lib.SamplingRequest(
                prompt='input',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
            ),
            base_sampler_lib.SamplingRequest(
                prompt='hello',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
            ),
        ])
    )
    self.assertEqual(sampler._compiled_prefill_fn._cache_size(), 1)

    asyncio.run(
        sampler.sample([
            base_sampler_lib.SamplingRequest(
                prompt='input input input input input',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
            ),
            base_sampler_lib.SamplingRequest(
                prompt='hello hello',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
            ),
        ])
    )

    asyncio.run(
        sampler.sample([
            base_sampler_lib.SamplingRequest(
                prompt='input input input input input input',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
            ),
            base_sampler_lib.SamplingRequest(
                prompt='hello hello',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
            ),
        ])
    )
    self.assertEqual(sampler._compiled_prefill_fn._cache_size(), 2)

  def test_decode_stops_after_prefill_for_single_generation_step(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        temperature=0.0,
        top_p=None,
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
        forbidden_token_ids=None,
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
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        kv_cache_size=1024,
        return_logits=True,
    )
    input_strings = ['input string', 'hello world']
    _, _, original_logits, _, _, _ = _run_prefill_and_decode(
        sampler, input_strings, max_generation_steps=10
    )

    new_transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler.transformer_state = nnx.variables(new_transformer, nnx.Param)
    _, _, new_logits, _, _, _ = _run_prefill_and_decode(
        sampler, input_strings, max_generation_steps=10
    )
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
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        kv_cache_size=1024,
        return_logits=True,
    )
    input_strings = ['input string', 'hello world']
    _, _, original_logits, _, _, _ = _run_prefill_and_decode(
        sampler, input_strings, max_generation_steps=10
    )

    new_transformer = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
            rngs=nnx.Rngs(42),
        )
    )
    new_lora_params = nnx.variables(new_transformer, nnx.LoRAParam)
    new_lora_params = jax.tree.map(lambda x: x + 0.1, new_lora_params)

    sampler.transformer_state = new_lora_params
    _, _, new_logits, _, _, _ = _run_prefill_and_decode(
        sampler, input_strings, max_generation_steps=10
    )
    with self.assertRaises(AssertionError):
      for orig, new in zip(original_logits, new_logits):
        np.testing.assert_allclose(orig, new, atol=1e-1, rtol=1e-1)

  def test_invalid_state_update(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize(), num_layers=4),
        rngs=nnx.Rngs(0),
    )
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        kv_cache_size=1024,
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
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        kv_cache_size=1024,
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
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        return_logits=True,
        eos_tokens=[7, 21],
        temperature=0.9,
        top_p=1.0,
        seed=0,
    )
    with mock.patch.object(
        jax.random,
        'split',
        return_value=(jax.random.PRNGKey(0), jax.random.PRNGKey(0)),
    ):
      _, tokens, _, _, _, _ = _run_prefill_and_decode(
          sampler,
          ['input string training', 'hello world'],
          max_generation_steps=10,
          max_prompt_length=4,
      )
    np.testing.assert_equal(
        tokens, [np.array([14]), np.array([12, 1, 17])]
    )

  def test_forbidden_token_ids(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        kv_cache_size=128,
        temperature=1.0,
        seed=123,
    )
    vocab_size = vocab.GetPieceSize()
    num_allowed_tokens = vocab_size // 4
    forbidden_tokens = set(range(num_allowed_tokens, vocab_size))
    forbidden_tokens.add(vocab.eos_id())
    max_generation_steps = 100

    _, tokens, _, _, _, _ = _run_prefill_and_decode(
        sampler,
        ['input string'],
        max_generation_steps=max_generation_steps,
        forbidden_tokens=forbidden_tokens,
    )
    self.assertLen(tokens[0], max_generation_steps)
    self.assertNoCommonElements(tokens[0], forbidden_tokens)

  def test_gemma4_smoke_test(self):
    """Runs a sampling call with a dummy Gemma4 config."""
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
        dtype=jnp.bfloat16,
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
    mock_tokenizer = tc.MockVocab()
    mock_tokenizer.DecodeIds = mock.MagicMock()
    mock_tokenizer.DecodeIds.return_value = 'decoded_string'
    sampler = self._make_sampler(
        transformer=model,
        tokenizer=mock_tokenizer,
        kv_cache_size=32,
    )
    asyncio.run(
        sampler.sample([
            base_sampler_lib.SamplingRequest(
                prompt='input string',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
            ),
            base_sampler_lib.SamplingRequest(
                prompt='hello world',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=10),
            ),
        ])
    )

  def test_gemma3_decode_only_last_token_consistency(self):
    """Verifies that decode_only_last_token yields identical generated tokens and logits."""
    config = gemma3_model_lib.ModelConfig(
        num_layers=2,
        num_embed=32,
        embed_dim=16,
        hidden_dim=16,
        num_heads=4,
        head_dim=16,
        num_kv_heads=1,
        sliding_window_size=4,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
    )
    rngs = nnx.Rngs(42)
    model = gemma3_model_lib.Gemma3(config, rngs=rngs)
    mock_tokenizer = tc.MockVocab()
    mock_tokenizer.DecodeIds = mock.MagicMock()
    mock_tokenizer.DecodeIds.return_value = 'decoded_string'

    sampler_opt = self._make_sampler(
        transformer=model,
        tokenizer=mock_tokenizer,
        kv_cache_size=32,
        return_logits=True,
    )
    self.assertTrue(sampler_opt._supports_decode_only_last_token)
    _, tokens_opt, logits_opt, _, _, _ = _run_prefill_and_decode(
        sampler_opt,
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=10,
        echo=False,
    )

    sampler_unopt = self._make_sampler(
        transformer=model,
        tokenizer=mock_tokenizer,
        kv_cache_size=32,
        return_logits=True,
    )
    sampler_unopt._supports_decode_only_last_token = False
    _, tokens_unopt, logits_unopt, _, _, _ = _run_prefill_and_decode(
        sampler_unopt,
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=10,
        echo=False,
    )

    self.assertEqual(len(tokens_opt), len(tokens_unopt))
    for t_opt, t_unopt in zip(tokens_opt, tokens_unopt):
      np.testing.assert_array_equal(t_opt, t_unopt)
    self.assertEqual(len(logits_opt), len(logits_unopt))
    for l_opt, l_unopt in zip(logits_opt, logits_unopt):
      self.assertEqual(l_opt.shape, l_unopt.shape)
      np.testing.assert_allclose(l_opt, l_unopt, atol=1e-5, rtol=1e-5)

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
        dtype=jnp.bfloat16,
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
    mock_tokenizer = tc.MockVocab()
    mock_tokenizer.DecodeIds = mock.MagicMock()
    mock_tokenizer.DecodeIds.return_value = 'decoded_string'

    sampler_opt = self._make_sampler(
        transformer=model,
        tokenizer=mock_tokenizer,
        kv_cache_size=32,
        return_logits=True,
    )
    self.assertTrue(sampler_opt._supports_decode_only_last_token)
    _, tokens_opt, logits_opt, _, _, _ = _run_prefill_and_decode(
        sampler_opt,
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=10,
        echo=False,
    )

    sampler_unopt = self._make_sampler(
        transformer=model,
        tokenizer=mock_tokenizer,
        kv_cache_size=32,
        return_logits=True,
    )
    sampler_unopt._supports_decode_only_last_token = False
    _, tokens_unopt, logits_unopt, _, _, _ = _run_prefill_and_decode(
        sampler_unopt,
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=10,
        echo=False,
    )

    self.assertEqual(len(tokens_opt), len(tokens_unopt))
    for t_opt, t_unopt in zip(tokens_opt, tokens_unopt):
      np.testing.assert_array_equal(t_opt, t_unopt)
    self.assertEqual(len(logits_opt), len(logits_unopt))
    for l_opt, l_unopt in zip(logits_opt, logits_unopt):
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
    sampler = self._make_sampler(
        transformer=transformer,
        tokenizer=vocab,
        kv_cache_size=128,
    )

    prompt = 'Describe: <img> <audio>'
    dummy_image = np.ones((16, 16, 3), dtype=np.uint8)
    dummy_audio = np.zeros(16000, dtype=np.float32)

    _, tokens, _, _, _, _ = _run_prefill_and_decode(
        sampler,
        [prompt],
        images=[dummy_image],
        audios=[dummy_audio],
        max_prompt_length=32,
        max_generation_steps=5,
    )
    self.assertIsNotNone(tokens)
    self.assertGreater(len(tokens[0]), 0)

  def test_update_params(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    sampler = self._make_sampler(transformer=transformer, tokenizer=vocab)

    source_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(1),
    )
    new_embedding = jnp.ones_like(source_model.emb.embedding.value) * 123.0
    source_model.emb.embedding.value = new_embedding
    new_state = nnx.state(source_model)

    sampler.update_params(new_state)

    state_dict = {
        '.'.join(str(p) for p in path): var
        for path, var in sampler.transformer_state.flat_state()
    }
    np.testing.assert_allclose(
        np.array(state_dict['emb.embedding'].value),
        np.array(new_embedding),
    )

  def test_update_params_with_filter_types(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    sampler = self._make_sampler(transformer=transformer, tokenizer=vocab)

    source_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(2),
    )
    new_embedding = jnp.ones_like(source_model.emb.embedding.value) * 77.0
    source_model.emb.embedding.value = new_embedding
    new_state = nnx.state(source_model, nnx.Param)

    sampler.update_params(new_state, filter_types=(nnx.Param,))

    state_dict = {
        '.'.join(str(p) for p in path): var
        for path, var in sampler.transformer_state.flat_state()
    }
    np.testing.assert_allclose(
        np.array(state_dict['emb.embedding'].value),
        np.array(new_embedding),
    )

  def test_update_params_precision_conversion(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(
            vocab_size=vocab.GetPieceSize(), dtype=jnp.float32
        ),
        rngs=nnx.Rngs(0),
    )
    sampler = self._make_sampler(transformer=transformer, tokenizer=vocab)

    source_model = tc.ToyTransformer(
        config=tc.ModelConfig(
            vocab_size=vocab.GetPieceSize(), dtype=jnp.bfloat16
        ),
        rngs=nnx.Rngs(1),
    )
    new_embedding = jnp.ones_like(source_model.emb.embedding.value) * 42.0
    source_model.emb.embedding.value = new_embedding
    new_state = nnx.state(source_model)

    sampler.update_params(new_state)

    state_dict = {
        '.'.join(str(p) for p in path): var
        for path, var in sampler.transformer_state.flat_state()
    }
    self.assertEqual(state_dict['emb.embedding'].value.dtype, jnp.float32)
    np.testing.assert_allclose(
        np.array(state_dict['emb.embedding'].value),
        np.array(jnp.ones_like(state_dict['emb.embedding'].value) * 42.0),
    )

  def test_single_sampling_request(self):
    req = base_sampler_lib.SamplingRequest(
        request_id='req_01',
        prompt='input string',
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=10,
        ),
    )
    response = asyncio.run(self.vanilla_sampler.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, 'req_01')
    self.assertIsNotNone(response.text)
    self.assertGreater(response.prompt_token_ids.size, 0)

  def test_per_request_seed_advances(self):
    stochastic_sampler = vanilla_sampler.VanillaSampler(
        server_id='tpu_slice_stochastic',
        config=configs.RolloutConfig(
            kv_cache_size=64,
            temperature=1.0,
            top_p=1.0,
            seed=42,
            weight_sync_mode=weight_sync.WeightSyncMode.FALLBACK,
        ),
        transformer=self.transformer,
        tokenizer=self.vocab,
    )
    stochastic_sampler.initialize()
    rng_before = np.asarray(stochastic_sampler._rng).copy()
    req1 = base_sampler_lib.SamplingRequest(
        request_id='req_seed_1',
        prompt='input string',
        sampling_params=base_sampler_lib.SamplingParams(max_tokens=8),
    )
    req2 = base_sampler_lib.SamplingRequest(
        request_id='req_seed_2',
        prompt='input string',
        sampling_params=base_sampler_lib.SamplingParams(max_tokens=8),
    )
    asyncio.run(stochastic_sampler.sample(req1))
    rng_after_1 = np.asarray(stochastic_sampler._rng).copy()
    asyncio.run(stochastic_sampler.sample(req2))
    rng_after_2 = np.asarray(stochastic_sampler._rng).copy()
    self.assertFalse(np.array_equal(rng_before, rng_after_1))
    self.assertFalse(np.array_equal(rng_after_1, rng_after_2))

  def test_sampling_request_with_logprobs(self):
    sampler_with_logprobs = vanilla_sampler.VanillaSampler(
        server_id='tpu_slice_logprobs',
        config=configs.RolloutConfig(
            kv_cache_size=64,
            temperature=0.0,
            return_logprobs=True,
            weight_sync_mode=weight_sync.WeightSyncMode.FALLBACK,
        ),
        transformer=self.transformer,
        tokenizer=self.vocab,
    )
    sampler_with_logprobs.initialize()
    req = base_sampler_lib.SamplingRequest(
        request_id='req_logprobs',
        prompt='input string',
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=10,
        ),
    )
    response = asyncio.run(sampler_with_logprobs.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, 'req_logprobs')
    self.assertIsNotNone(response.logprobs)

  def test_batch_sampling_requests(self):
    reqs = [
        base_sampler_lib.SamplingRequest(
            request_id='req_a',
            prompt='input string 1',
            sampling_params=base_sampler_lib.SamplingParams(
                max_tokens=8,
                temperature=0.0,
            ),
        ),
        base_sampler_lib.SamplingRequest(
            request_id='req_b',
            prompt='hello world 2',
            sampling_params=base_sampler_lib.SamplingParams(
                max_tokens=8,
                temperature=0.0,
            ),
        ),
    ]
    responses = asyncio.run(self.vanilla_sampler.sample(reqs))
    self.assertIsInstance(responses, list)
    self.assertLen(responses, 2)
    self.assertEqual(responses[0].request_id, 'req_a')
    self.assertEqual(responses[1].request_id, 'req_b')
    self.assertIsNotNone(responses[0].text)
    self.assertIsNotNone(responses[1].text)
    self.assertGreater(responses[0].prompt_token_ids.size, 0)
    self.assertGreater(responses[1].prompt_token_ids.size, 0)

  def test_weight_sync_without_raiden_delegate(self):
    self.assertIsNone(asyncio.run(self.vanilla_sampler.bind_weight_sync()))
    self.assertTrue(asyncio.run(self.vanilla_sampler.pre_weight_sync()))
    new_state = self.vanilla_sampler.transformer_state
    req = base_sampler_lib.WeightSyncRequest(weights=new_state)
    self.assertTrue(
        asyncio.run(self.vanilla_sampler.weight_sync(sync_request=req))
    )
    self.assertTrue(asyncio.run(self.vanilla_sampler.post_weight_sync()))
    with self.assertRaises(NotImplementedError):
      asyncio.run(self.vanilla_sampler.get_weight_sync_metadata())

  def test_weight_sync_with_raiden_delegate(self):
    mock_delegate = mock.MagicMock()
    mock_delegate.is_bounded.return_value = False
    mock_delegate.bind_weight_sync = mock.AsyncMock(return_value=True)
    mock_delegate.get_weight_sync_metadata = mock.AsyncMock(
        return_value=[{'unit': 'rollout'}]
    )
    mock_delegate.pre_weight_sync = mock.AsyncMock(return_value=True)
    mock_delegate.weight_sync = mock.AsyncMock(return_value=10)
    mock_delegate.post_weight_sync = mock.AsyncMock(return_value=True)
    mock_delegate.abort_weight_sync = mock.AsyncMock(return_value=True)

    sampler_with_raiden = vanilla_sampler.VanillaSampler(
        server_id='tpu_slice_raiden',
        config=configs.RolloutConfig(
            kv_cache_size=64,
            weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
        ),
        transformer=self.transformer,
        tokenizer=self.vocab,
        raiden_sync_delegate=mock_delegate,
    )
    sampler_with_raiden.initialize()

    sync_req = base_sampler_lib.WeightSyncRequest(policy_version=10)

    # 1. bind_weight_sync
    asyncio.run(sampler_with_raiden.bind_weight_sync(sync_req))
    mock_delegate.bind_weight_sync.assert_awaited_once_with(
        sync_request=sync_req,
        state=sampler_with_raiden.transformer_state,
        sampler=sampler_with_raiden,
    )
    mock_delegate.is_bounded.return_value = True

    # 2. get_weight_sync_metadata
    metadata = asyncio.run(sampler_with_raiden.get_weight_sync_metadata())
    self.assertEqual(metadata, [{'unit': 'rollout'}])

    # 3. pre_weight_sync
    self.assertTrue(asyncio.run(sampler_with_raiden.pre_weight_sync(sync_req)))
    mock_delegate.pre_weight_sync.assert_awaited_once_with(
        sync_request=sync_req
    )

    # 4. weight_sync
    version = asyncio.run(sampler_with_raiden.weight_sync(sync_req))
    self.assertEqual(version, 10)
    mock_delegate.weight_sync.assert_awaited_once_with(sync_request=sync_req)

    # 5. post_weight_sync
    self.assertTrue(asyncio.run(sampler_with_raiden.post_weight_sync(sync_req)))
    mock_delegate.post_weight_sync.assert_awaited_once_with(
        sync_request=sync_req
    )

  def test_real_raiden_delegate_instantiation_and_sync(self):
    sampler_with_raiden = vanilla_sampler.VanillaSampler(
        server_id='tpu_slice_raiden',
        config=configs.RolloutConfig(
            kv_cache_size=64,
            weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
        ),
        transformer=self.transformer,
        tokenizer=self.vocab,
    )
    sampler_with_raiden.initialize()
    self.assertTrue(sampler_with_raiden.enable_raiden)
    self.assertIsNotNone(sampler_with_raiden.raiden_sync_delegate)
    self.assertEqual(
        sampler_with_raiden.raiden_sync_delegate._synchronizers[0].job_name,
        'tpu_slice_raiden',
    )
    self.assertFalse(sampler_with_raiden.raiden_sync_delegate.is_bounded())


class SamplerTokenInputTest(absltest.TestCase):

  def _make_sampler(self, cache_size: int = 16, **config_kwargs):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    cfg_kwargs = {
        'kv_cache_size': cache_size,
        'temperature': 0.0,
        'top_p': None,
        'top_k': 1,
    }
    cfg_kwargs.update(config_kwargs)
    sampler = vanilla_sampler.VanillaSampler(
        server_id='tpu_slice_token_input',
        config=configs.RolloutConfig(**cfg_kwargs),
        transformer=transformer,
        tokenizer=vocab,
    )
    sampler.initialize()
    return sampler

  def test_prompt_token_ids_bypasses_tokenize_and_returns_exact_padded_ids(
      self,
  ):
    sampler = self._make_sampler(cache_size=16)
    with mock.patch.object(
        sampler,
        'tokenize',
        side_effect=AssertionError('tokenize must not be called'),
    ):
      responses = asyncio.run(
          sampler.sample([
              base_sampler_lib.SamplingRequest(
                  request_id='req_0',
                  prompt=[0, 3],
                  sampling_params=base_sampler_lib.SamplingParams(max_tokens=2),
              ),
              base_sampler_lib.SamplingRequest(
                  request_id='req_1',
                  prompt=[4, 0, 5],
                  sampling_params=base_sampler_lib.SamplingParams(max_tokens=2),
              ),
          ])
      )

    np.testing.assert_array_equal(
        responses[0].prompt_token_ids,
        np.array([0, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        responses[1].prompt_token_ids,
        np.array([4, 0, 5], dtype=np.int32),
    )

    _, tokens, _, padded_prompt_tokens, _, prompt_lengths = (
        _run_prefill_and_decode(
            sampler,
            prompt_token_ids=[[0, 3], [4, 0, 5]],
            max_generation_steps=2,
            max_prompt_length=4,
            echo=True,
            pad_output=True,
        )
    )
    np.testing.assert_array_equal(
        padded_prompt_tokens,
        np.array([[0, 0, 0, 3], [0, 4, 0, 5]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        prompt_lengths, np.array([2, 3], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        tokens[0][:2], np.array([0, 3], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        tokens[1][:3], np.array([4, 0, 5], dtype=np.int32)
    )

    state = sampler.init_sample_state(
        jnp.array([[0, 0, 0, 3], [0, 4, 0, 5]], dtype=jnp.int32),
        total_sampling_steps=6,
        forbidden_token_ids=None,
        prompt_lengths=jnp.array([2, 3], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(state.input_mask[:, :4]),
        np.array([[False, False, True, True], [False, True, True, True]]),
    )
    np.testing.assert_array_equal(
        np.asarray(state.positions[:, :4]),
        np.array([[0, 0, 0, 1], [0, 0, 1, 2]], dtype=np.int32),
    )

  def test_input_strings_populates_prompt_lengths(self):
    sampler = self._make_sampler(cache_size=16)
    responses = asyncio.run(
        sampler.sample([
            base_sampler_lib.SamplingRequest(
                request_id='req_0',
                prompt='input string',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=2),
            ),
            base_sampler_lib.SamplingRequest(
                request_id='req_1',
                prompt='hello world',
                sampling_params=base_sampler_lib.SamplingParams(max_tokens=2),
            ),
        ])
    )
    expected_lengths = np.array(
        [
            len(sampler.tokenize('input string')),
            len(sampler.tokenize('hello world')),
        ],
        dtype=np.int32,
    )
    actual_lengths = np.array(
        [len(r.prompt_token_ids) for r in responses], dtype=np.int32
    )
    np.testing.assert_array_equal(actual_lengths, expected_lengths)

  def test_rejects_invalid_inputs(self):
    sampler = self._make_sampler(cache_size=5)
    sampler_beam = self._make_sampler(cache_size=5, beam_size=2)
    with self.assertRaisesRegex(ValueError, 'one output per row'):
      asyncio.run(
          sampler_beam.sample([
              base_sampler_lib.SamplingRequest(
                  prompt=[1],
                  sampling_params=base_sampler_lib.SamplingParams(max_tokens=1),
              )
          ])
      )
    with self.assertRaisesRegex(ValueError, 'exceeds cache_size'):
      asyncio.run(
          sampler.sample([
              base_sampler_lib.SamplingRequest(
                  prompt=[1, 2, 3, 4],
                  sampling_params=base_sampler_lib.SamplingParams(max_tokens=2),
              )
          ])
      )
    with self.assertRaisesRegex(ValueError, 'must not be empty'):
      asyncio.run(sampler.sample([]))
    with self.assertRaisesRegex(ValueError, '1-D'):
      asyncio.run(
          sampler.sample([
              base_sampler_lib.SamplingRequest(
                  prompt=np.zeros((1, 2), dtype=np.int32),
                  sampling_params=base_sampler_lib.SamplingParams(max_tokens=1),
              )
          ])
      )

  def test_check_prompt_echo(self):
    expected = [np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)]
    utils.check_prompt_echo(
        expected,
        [
            {
                'prompt_token_ids': [1, 2],
                'meta_info': {'id': '0', 'prompt_tokens': 2},
            },
            {'meta_info': {'id': '1', 'prompt_tokens': 1}},
        ],
    )
    with self.assertRaisesRegex(ValueError, 'Expected 2 outputs, got 1'):
      utils.check_prompt_echo(expected, [{'meta_info': {'id': '0'}}])
    with self.assertRaisesRegex(ValueError, 'Duplicate request_id'):
      utils.check_prompt_echo(
          expected,
          [
              {'meta_info': {'id': 'dup', 'prompt_tokens': 2}},
              {'meta_info': {'id': 'dup', 'prompt_tokens': 1}},
          ],
      )
    with self.assertRaisesRegex(ValueError, 'missing request_id'):
      utils.check_prompt_echo(
          [np.array([1, 2], dtype=np.int32)],
          [{'prompt_token_ids': [1, 2]}],
      )
    with self.assertRaisesRegex(ValueError, 'missing prompt echo'):
      utils.check_prompt_echo(
          [np.array([1, 2], dtype=np.int32)],
          [{'meta_info': {'id': '0'}}],
      )
    with self.assertRaisesRegex(
        ValueError, 'prompt_token_ids differed from input'
    ):
      utils.check_prompt_echo(
          [np.array([1, 2], dtype=np.int32)],
          [{'request_id': '0', 'prompt_token_ids': [1, 999]}],
      )
    with self.assertRaisesRegex(
        ValueError, 'prompt_token_ids differed from input'
    ):
      utils.check_prompt_echo(
          [np.array([1, 2], dtype=np.int32)],
          [{'meta_info': {'id': '0', 'prompt_tokens': 3}}],
      )


if __name__ == '__main__':
  absltest.main()
