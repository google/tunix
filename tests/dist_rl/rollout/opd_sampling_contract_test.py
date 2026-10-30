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

"""Checks actual worker/engine sampling settings rather than request fields."""

import dataclasses
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import nnx
import numpy as np
from tunix.common import configs
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.rollout import vanilla_sampler
from tunix.dist_rl.rollout import vllm_sampler
from tunix.tests import test_common


def _request():
  return sampler_lib.SamplingRequest(
      request_id="opd",
      prompt=np.array([2, 3], dtype=np.int32),
      sampling_params=sampler_lib.SamplingParams(
          max_tokens=2,
          temperature=1.0,
          top_p=1.0,
          top_k=0,
          return_logprobs=True,
      ),
  )


class OPDSamplingContractTest(unittest.IsolatedAsyncioTestCase):

  async def test_native_rejects_actual_worker_config_despite_neutral_request(
      self,
  ):
    config = configs.RolloutConfig(
        temperature=1.0, top_p=1.0, top_k=None, return_logprobs=True
    )
    for overrides in (
        {"temperature": 0.8},
        {"top_p": 0.9},
        {"top_p": None},
        {"top_k": 4},
        {"beam_size": 1},
        {"return_logprobs": False},
    ):
      with self.subTest(overrides=overrides):
        # The contract must fail before model/cache initialization or decoding.
        sampler = object.__new__(vanilla_sampler.VanillaSampler)
        sampler.config = dataclasses.replace(config, **overrides)
        with self.assertRaises(ValueError):
          await sampler.sample(_request(), require_unfiltered=True)

  async def test_native_unfiltered_sampling_runs_real_cpu_decoder(self):
    vocab = test_common.MockVocab()
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = vanilla_sampler.VanillaSampler(
        server_id="cpu-opd",
        config=configs.RolloutConfig(
            kv_cache_size=16,
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            return_logprobs=True,
            weight_sync_mode="none",
        ),
        transformer=model,
        tokenizer=vocab,
    )
    try:
      result = await sampler.sample(_request(), require_unfiltered=True)
      self.assertGreater(result.token_ids.size, 0)
      self.assertEqual(result.token_ids.shape, result.logprobs.shape)
      self.assertTrue(np.all(np.isfinite(result.logprobs)))
      with self.assertRaisesRegex(ValueError, "forbidden_tokens"):
        await sampler.sample(
            _request(), require_unfiltered=True, forbidden_tokens=[2]
        )
    finally:
      await sampler.stop()

  async def test_async_vllm_rejects_effective_filters_before_engine_dispatch(
      self,
  ):
    sampler = vllm_sampler.VllmSampler(
        engine_args=SimpleNamespace(model="fake"),
        config=configs.RolloutConfig(weight_sync_mode="none"),
    )
    generate = mock.Mock()
    sampler._engine = SimpleNamespace(generate=generate)
    sampler._is_running = True
    for overrides in (
        {"min_p": 0.25},
        {"repetition_penalty": 1.1},
        {"min_tokens": 2},
        {"logits_processors": [object()]},
        {"best_of": 2},
        {"logprobs": None},
    ):
      with self.subTest(overrides=tuple(overrides)):
        effective = SimpleNamespace(**{
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": 1,
            **overrides,
        })
        # Stub only the optional vLLM import/builder; sample() validates the
        # resulting actual engine object before engine.generate is called.
        with mock.patch.object(
            sampler, "_build_vllm_params", return_value=effective
        ):
          with self.assertRaises(ValueError):
            await sampler.sample(_request(), require_unfiltered=True)
        generate.assert_not_called()

    sampler.engine_args.logits_processors = ["custom.EngineProcessor"]
    neutral = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=-1, logprobs=1)
    with mock.patch.object(sampler, "_build_vllm_params", return_value=neutral):
      with self.assertRaisesRegex(ValueError, "engine logits processors"):
        await sampler.sample(_request(), require_unfiltered=True)
    generate.assert_not_called()


if __name__ == "__main__":
  unittest.main()
