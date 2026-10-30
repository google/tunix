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

"""Real async sampler output processing with a CPU-only fake vLLM stream."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from tunix.common import configs
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.rollout import vllm_sampler


class VllmOutputLogprobsTest(unittest.IsolatedAsyncioTestCase):

  async def _sample(self, scores, *, tokens=(7, 0), requested=True):
    async def generate(prompt, sampling_params, request_id):
      del prompt, sampling_params, request_id
      yield SimpleNamespace(
          outputs=[
              SimpleNamespace(
                  text="answer",
                  token_ids=list(tokens),
                  logprobs=scores,
                  finish_reason="stop",
              )
          ],
      )

    sampler = vllm_sampler.VllmSampler(
        engine_args=SimpleNamespace(model="fake"),
        config=configs.RolloutConfig(weight_sync_mode="none"),
    )
    sampler._engine = SimpleNamespace(generate=generate)
    sampler._is_running = True
    request = sampler_lib.SamplingRequest(
        request_id="request",
        prompt="prompt",
        sampling_params=sampler_lib.SamplingParams(return_logprobs=requested),
    )
    # vLLM imports are deferred to this unchanged parameter builder. Replacing
    # only it lets the actual sample -> output parser chain run on CPU.
    with mock.patch.object(
        sampler,
        "_build_vllm_params",
        return_value=SimpleNamespace(logprobs=1 if requested else None),
    ):
      return await sampler.sample(request)

  async def test_scores_follow_sampled_ids_independent_of_dict_order(self):
    response = await self._sample([
        {99: SimpleNamespace(logprob=-0.1), 7: SimpleNamespace(logprob=-3.0)},
        {42: SimpleNamespace(logprob=-0.2), 0: SimpleNamespace(logprob=0.0)},
    ])
    np.testing.assert_array_equal(response.token_ids, [7, 0])
    np.testing.assert_array_equal(response.logprobs, [-3.0, 0.0])
    self.assertIsNone(response.error)

  async def test_requested_missing_or_corrupt_scores_raise(self):
    for scores in (
        None,
        [],
        [None],
        [{}],
        [{99: SimpleNamespace(logprob=-0.1)}],
        [{7: SimpleNamespace(logprob=np.nan)}],
        [{7: SimpleNamespace(logprob=np.inf)}],
        [{7: SimpleNamespace(logprob=-np.inf)}],
        [{7: SimpleNamespace(logprob=1e300)}],
        [{7: SimpleNamespace()}],
    ):
      with self.subTest(scores=scores):
        with self.assertRaises(ValueError):
          await self._sample(scores, tokens=(7,))

  async def test_mismatched_token_and_score_counts_raise(self):
    with self.assertRaisesRegex(ValueError, "logprobs shape"):
      await self._sample([{7: SimpleNamespace(logprob=-0.5)}])

  async def test_unrequested_missing_scores_remain_optional(self):
    response = await self._sample(None, requested=False)
    self.assertIsNone(response.logprobs)
    np.testing.assert_array_equal(response.token_ids, [7, 0])


if __name__ == "__main__":
  unittest.main()
