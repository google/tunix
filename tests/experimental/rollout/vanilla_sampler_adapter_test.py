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

"""Tests for VanillaSamplerAdapter with Tunix JAX Sampler."""

import asyncio
from absl.testing import absltest
from flax import nnx
import numpy as np
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.generate import sampler as generate_sampler_lib
from tunix.tests import test_common as tc


class VanillaSamplerAdapterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = tc.MockVocab()
    self.transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    self.cache_config = generate_sampler_lib.CacheConfig(
        cache_size=64,
        num_layers=4,
        num_kv_heads=4,
        head_dim=16,
    )
    self.vanilla_sampler = vanilla_sampler_adapter.VanillaSamplerAdapter(
        server_id="tpu_slice_01",
        transformer=self.transformer,
        tokenizer=self.vocab,
        cache_config=self.cache_config,
    )
    self.vanilla_sampler.initialize()

  def test_single_sampling_request(self):
    req = base_sampler_lib.SamplingRequest(
        request_id="req_01",
        prompt="input string",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=10,
            temperature=0.0,
        ),
    )
    response = asyncio.run(self.vanilla_sampler.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "req_01")
    self.assertIsNotNone(response.text)
    self.assertGreater(response.prompt_token_ids.size, 0)

  def test_sampling_request_with_logprobs(self):
    req = base_sampler_lib.SamplingRequest(
        request_id="req_logprobs",
        prompt="input string",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=10,
            temperature=0.0,
            return_logprobs=True,
        ),
    )
    response = asyncio.run(self.vanilla_sampler.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "req_logprobs")
    self.assertIsNotNone(response.logprobs)

  def test_batch_sampling_requests(self):
    reqs = [
        base_sampler_lib.SamplingRequest(
            request_id="req_a",
            prompt="input string 1",
            sampling_params=base_sampler_lib.SamplingParams(
                max_tokens=8,
                temperature=0.0,
            ),
        ),
        base_sampler_lib.SamplingRequest(
            request_id="req_b",
            prompt="hello world 2",
            sampling_params=base_sampler_lib.SamplingParams(
                max_tokens=8,
                temperature=0.0,
            ),
        ),
    ]
    responses = asyncio.run(self.vanilla_sampler.sample(reqs))
    self.assertIsInstance(responses, list)
    self.assertLen(responses, 2)
    self.assertEqual(responses[0].request_id, "req_a")
    self.assertEqual(responses[1].request_id, "req_b")
    self.assertIsNotNone(responses[0].text)
    self.assertIsNotNone(responses[1].text)
    self.assertGreater(responses[0].prompt_token_ids.size, 0)
    self.assertGreater(responses[1].prompt_token_ids.size, 0)

  def test_construct_with_integer_cache_size(self):
    sampler_adapter_direct = vanilla_sampler_adapter.VanillaSamplerAdapter(
        server_id="tpu_slice_02",
        transformer=self.transformer,
        tokenizer=self.vocab,
        cache_config=64,
    )
    req = base_sampler_lib.SamplingRequest(
        request_id="req_direct",
        prompt="direct prompt",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=6,
            temperature=0.0,
        ),
    )
    response = asyncio.run(sampler_adapter_direct.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "req_direct")
    self.assertIsNotNone(response.text)
    self.assertEqual(response.prompt_token_ids.dtype, np.int32)

  def test_uninitialized_sampler_raises(self):
    uninit_sampler = vanilla_sampler_adapter.VanillaSamplerAdapter(
        server_id="empty"
    )
    with self.assertRaises(RuntimeError):
      asyncio.run(
          uninit_sampler.sample(
              base_sampler_lib.SamplingRequest(prompt="hello")
          )
      )

  def test_sample_none_requests_raises(self):
    with self.assertRaises(ValueError):
      asyncio.run(self.vanilla_sampler.sample(None))


if __name__ == "__main__":
  absltest.main()
