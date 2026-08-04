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

"""CPU contracts for canonical re-score through both in-process vLLM modes."""

from absl.testing import absltest
import numpy as np
import types
from unittest import mock

from tunix.generate import vllm_sampler
from tunix.rl.rollout import vllm_rollout


class _Tokenizer:

  def pad_id(self):
    return 0

  def eos_id(self):
    return 7


class _RescoreSampler:

  def __init__(self):
    self.tokenizer = _Tokenizer()
    self.reset_calls = 0
    self.prompts = None

  def reset_prefix_cache_when_idle(self):
    self.reset_calls += 1

  def generate_request_outputs(
      self, prompts, sampling_params, *, reset_prefix_cache=False
  ):
    del sampling_params
    if reset_prefix_cache:
      self.reset_calls += 1
    self.prompts = prompts
    outputs = []
    for prompt in prompts:
      token_ids = prompt["prompt_token_ids"]
      prompt_logprobs = [None]
      for token_id in token_ids[1:]:
        prompt_logprobs.append({
            token_id: types.SimpleNamespace(logprob=-float(token_id))
        })
      outputs.append(types.SimpleNamespace(prompt_logprobs=prompt_logprobs))
    return outputs


class VllmRolloutCanonicalTest(absltest.TestCase):

  def test_sampler_driver_mode_uses_locked_reset_and_driver_generate(self):
    sampler = object.__new__(vllm_sampler.VllmSampler)
    sampler.llm = None
    sampler._driver = mock.Mock()
    sampler._generate_server_mode = mock.Mock(return_value=["driver-output"])
    prompts = [{"prompt_token_ids": [1, 2]}]
    params = object()

    self.assertEqual(
        sampler.generate_request_outputs(
            prompts, params, reset_prefix_cache=True
        ),
        ["driver-output"],
    )
    sampler._generate_server_mode.assert_called_once_with(
        prompts,
        params,
        reset_prefix_cache=True,
        reset_timeout_s=300.0,
    )

  def test_prefill_rescore_uses_mode_independent_sampler_contract(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_prefill_rescore_provenance = None
    result = rollout.get_prefill_rescore_logps(
        prompt_tokens=np.asarray([[0, 1, 2], [0, 0, 5]], np.int32),
        completion_tokens=np.asarray([[3, 4, 0], [6, 0, 0]], np.int32),
        processed=False,
        completion_lengths=np.asarray([2, 1], np.int32),
    )

    np.testing.assert_array_equal(
        result, np.asarray([[-3.0, -4.0, 0.0], [-6.0, 0.0, 0.0]], np.float32)
    )
    self.assertEqual(rollout._sampler.reset_calls, 1)
    self.assertEqual(
        [prompt["prompt_token_ids"] for prompt in rollout._sampler.prompts],
        [[1, 2, 3, 4], [5, 6]],
    )


if __name__ == "__main__":
  absltest.main()
