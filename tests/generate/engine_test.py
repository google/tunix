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

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import numpy as np

from tunix.generate import engine as engine_lib
from tunix.generate import sampler_v2 as sampler_lib
from tunix.tests import test_common as tc


@dataclasses.dataclass(kw_only=True)
class ModelConfigWithDtype(tc.ModelConfig):
  dtype: jax.numpy.dtype = jax.numpy.bfloat16


def run_engine_complete(
    engine: engine_lib.LLMEngine,
    prompts: list[str],
    max_generation_steps: int = 10,
    max_prompt_length: int | None = None,
    echo: bool = False,
    return_logits: bool = False,
    return_logprobs: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int = 0,
    eos_tokens: list[int] | None = None,
    forbidden_tokens: set[int] | None = None,
):
  reqs = []
  for i, p in enumerate(prompts):
    reqs.append(engine.add_request(str(i), p))

  if max_prompt_length is not None:
    for req in reqs:
      # truncate prompt to max length
      req.token_ids = req.token_ids[:max_prompt_length]
      req.prompt_length = min(req.prompt_length, max_prompt_length)

  # Ensure the engine executes on our set bounds to force exact matching
  engine.cache_config.max_tokens_to_generate = max_generation_steps
  engine.scheduler.max_tokens_to_generate = max_generation_steps

  sampling_config = sampler_lib.SamplingConfig(
      temperature=temperature,
      top_p=top_p,
      top_k=top_k if top_k is not None else -1,
      forbidden_token_ids=tuple(forbidden_tokens) if forbidden_tokens else None,
      eos_token_ids=tuple(eos_tokens) if eos_tokens else None,
      return_logprobs=return_logprobs,
  )
  engine.sampler._seed = seed

  while engine.has_unfinished_requests():
    engine.step(sampling_config=sampling_config, return_logits=return_logits)

  texts = []
  tokens = []
  logits = []
  logprobs = []
  for req in reqs:
    gen_tokens = req.token_ids
    if not echo:
      gen_tokens = req.token_ids[req.prompt_length:]

    texts.append(engine.tokenizer.DecodeIds(gen_tokens))
    tokens.append(np.array(gen_tokens))
    if return_logits:
      logits.append(np.array(req.logits))
    if return_logprobs:
      logprobs.append(np.array(req.logprobs))

  return sampler_lib.SamplerOutput(
      text=texts,
      tokens=tokens,
      logits=logits if return_logits else None,
      logprobs=logprobs if return_logprobs else None
  )


class EngineTest(parameterized.TestCase):

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

    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(),
    )
    self.assertEqual(engine.sampler.dtype, expected_dtype)

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
    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(max_tpu_bytes=1*1024*1024),
    )
    
    result = run_engine_complete(
        engine,
        ['input string', 'hello world'],
        max_generation_steps=10,
        max_prompt_length=max_prompt_length,
        return_logits=True,
        echo=echo,
    )

    self.assertIsNotNone(result)

    top_p_result = run_engine_complete(
        engine,
        ['input string', 'hello world'],
        max_generation_steps=10,
        temperature=9.0,
        top_p=0.95,
        echo=echo,
    )
    self.assertIsNotNone(top_p_result)
    self.assertNotEqual(result.text, top_p_result.text)

    top_p_result_2 = run_engine_complete(
        engine,
        ['input string', 'hello world'],
        max_generation_steps=10,
        temperature=9.0,
        top_p=0.95,
        seed=42,
        echo=echo,
    )
    self.assertIsNotNone(top_p_result_2)
    self.assertNotEqual(top_p_result.text, top_p_result_2.text)

    top_k_result = run_engine_complete(
        engine,
        ['input string', 'hello world'],
        max_generation_steps=10,
        temperature=9.0,
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
    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(max_tpu_bytes=1*1024*1024),
    )
    # Test greedy logprobs
    result = run_engine_complete(
        engine,
        ['input string', 'hello world'],
        max_generation_steps=10,
        return_logprobs=True,
        echo=False
    )
    self.assertIsNotNone(result.logprobs)
    self.assertLen(result.logprobs, 2)
    for logprobs, tokens in zip(result.logprobs, result.tokens):
      self.assertLen(logprobs, tokens.shape[0])

    # Test top_p logprobs
    top_p_result = run_engine_complete(
        engine,
        ['input string', 'hello world'],
        max_generation_steps=10,
        return_logprobs=True,
        temperature=1.0,
        top_p=0.9,
        echo=False
    )
    self.assertIsNotNone(top_p_result.logprobs)
    self.assertLen(top_p_result.logprobs, 2)
    for logprobs, tokens in zip(top_p_result.logprobs, top_p_result.tokens):
      self.assertNotEmpty(logprobs)
      self.assertLen(logprobs, tokens.shape[0])

  def test_decode_stops_after_prefill_for_single_generation_step(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(max_tpu_bytes=1*1024*1024),
    )

    req = engine.add_request("req1", "input string")
    engine.cache_config.max_tokens_to_generate = 1
    engine.scheduler.max_tokens_to_generate = 1

    # 1 Step
    completed_reqs = engine.step(return_logits=False)

    # Check that after one step, the engine correctly stops request 1
    self.assertEqual(len(engine.scheduler._running_requests), 0)
    self.assertEqual(completed_reqs[0], req)

  def test_state_update(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(max_tpu_bytes=1*1024*1024),
    )
    input_strings = ['input string', 'hello world']
    original_logits = run_engine_complete(
        engine, input_strings, max_generation_steps=10, return_logits=True
    ).logits

    new_transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    engine.sampler.transformer_state = nnx.variables(new_transformer, nnx.Param)
    new_logits = run_engine_complete(
        engine, input_strings, max_generation_steps=10, return_logits=True
    ).logits
    with self.assertRaises(AssertionError):
      for orig, new in zip(original_logits, new_logits):  # pyrefly: ignore[bad-argument-type]
        np.testing.assert_allclose(orig, new, atol=1e-1, rtol=1e-1)

  def test_lora_state_update(self):
    vocab = tc.MockVocab()
    transformer = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
            rngs=nnx.Rngs(0),
        )
    )

    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(max_tpu_bytes=1*1024*1024),
    )
    input_strings = ['input string', 'hello world']
    original_logits = run_engine_complete(
        engine, input_strings, max_generation_steps=10, return_logits=True
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

    engine.sampler.transformer_state = new_lora_params
    new_logits = run_engine_complete(
        engine, input_strings, max_generation_steps=10, return_logits=True
    ).logits
    with self.assertRaises(AssertionError):
      for orig, new in zip(original_logits, new_logits):  # pyrefly: ignore[bad-argument-type]
        np.testing.assert_allclose(orig, new, atol=1e-1, rtol=1e-1)

  def test_invalid_state_update(self):
    vocab = tc.MockVocab()

    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize(), num_layers=4),
        rngs=nnx.Rngs(0),
    )
    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(max_tpu_bytes=10*1024*1024),
    )

    new_transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize(), num_layers=6),
        rngs=nnx.Rngs(42),
    )
    with self.assertRaisesRegex(ValueError, '.*must have the same structure.*'):
      engine.sampler.transformer_state = nnx.variables(new_transformer, nnx.Param)

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
    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(max_tpu_bytes=10*1024*1024),
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
      engine.sampler.transformer_state = nnx.variables(new_transformer, nnx.LoRAParam)

  def test_eos_tokens(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(max_tpu_bytes=1*1024*1024),
    )
    result = run_engine_complete(
        engine,
        ['input string training', 'hello world'],
        max_generation_steps=10,
        return_logits=True,
        max_prompt_length=4,
        eos_tokens=[7, 21],
        temperature=0.9,
        top_p=1.0,
        seed=0,
        echo=False
    )

    np.testing.assert_equal(result.tokens, [np.array([]), np.array([12, 1, 17])])

  def test_forbidden_token_ids(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    engine = engine_lib.LLMEngine(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(max_tpu_bytes=1*1024*1024),
    )

    vocab_size = vocab.GetPieceSize()
    num_allowed_tokens = vocab_size // 4
    forbidden_tokens = set(range(num_allowed_tokens, vocab_size))

    # EOS is forbidden so we are sure to get a full length generation.
    forbidden_tokens.add(vocab.eos_id())
    max_generation_steps = 10

    result = run_engine_complete(
        engine,
        ['input string'],
        max_generation_steps=max_generation_steps,
        return_logits=False,
        forbidden_tokens=forbidden_tokens,
        temperature=1.0,
        seed=123,
    )

    prompt_length = len(engine.tokenize('input string'))
    self.assertLen(result.tokens[0], max_generation_steps + prompt_length)
    self.assertNoCommonElements(result.tokens[0], forbidden_tokens)


if __name__ == '__main__':
  absltest.main()

