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
import os
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

  def test_p32_d2b_runs_decode_and_prefill_captures(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    adapter = mock.Mock()
    adapter._vocab_size = 8
    adapter.finish_p32_d2b_capture.side_effect = [
        {
            "raw_rows": np.ones((2, 8), np.float32),
            "processed_rows": np.full((2, 8), 2.0, np.float32),
            "dp_ranks": (0, 1),
        },
        {
            "raw_rows": np.ones((2, 8), np.float32),
            "processed_rows": np.full((2, 8), 2.0, np.float32),
            "dp_ranks": (0, 1),
        },
    ]
    rollout._canonical_engine_adapter = adapter
    rollout._last_sampling_transforms = {
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 1.0,
    }
    decode_outputs = [
        types.SimpleNamespace(
            outputs=[
                types.SimpleNamespace(token_ids=[5, 6], logprobs=object())
            ]
        ),
        types.SimpleNamespace(
            outputs=[
                types.SimpleNamespace(token_ids=[7, 4], logprobs=object())
            ]
        ),
    ]
    prefill_outputs = [
        types.SimpleNamespace(prompt_logprobs=[None, object(), object(), object()]),
        types.SimpleNamespace(prompt_logprobs=[None, object(), object(), object()]),
    ]
    sampler = mock.Mock()
    sampler.generate_request_outputs.side_effect = [decode_outputs, prefill_outputs]
    rollout._sampler = sampler

    def fake_logprobs(token_ids, rows):
      del rows
      return [-float(token) for token in token_ids]

    with mock.patch.dict(
        os.environ,
        {
            "CANON_P32_D2B_FULL_DISTRIBUTION": "1",
            "CANON_LOGPROB_M": "8",
        },
        clear=False,
    ), mock.patch.object(
        vllm_rollout.generate_utils,
        "get_logprobs_from_vllm_output",
        side_effect=fake_logprobs,
    ):
      result = rollout.run_p32_d2b_engine_sentinels([[1, 2], [2, 3]])

    np.testing.assert_array_equal(
        result["generated_tokens"], np.asarray([[5, 6], [7, 4]], np.int32)
    )
    np.testing.assert_array_equal(
        result["decode_target_logps"], np.asarray([-6.0, -4.0], np.float32)
    )
    self.assertEqual(
        [call.args[0] for call in adapter.arm_p32_d2b_capture.call_args_list],
        ["decode", "prefill"],
    )
    self.assertEqual(sampler.generate_request_outputs.call_count, 2)

  def test_p28_full_chain_is_forwarded(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = mock.Mock()
    rollout._canonical_engine_adapter.run_p28_full_chain_gate.return_value = (
        "ok"
    )

    self.assertEqual(rollout.run_p28_full_chain_gate(), "ok")
    rollout._canonical_engine_adapter.run_p28_full_chain_gate.assert_called_once_with()

  def test_p28_block_vjp_layer_index_is_forwarded(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = mock.Mock()
    rollout._canonical_engine_adapter.run_p28_block_vjp_gate.return_value = (
        "ok"
    )

    self.assertEqual(rollout.run_p28_block_vjp_gate(layer_index=17), "ok")
    rollout._canonical_engine_adapter.run_p28_block_vjp_gate.assert_called_once_with(
        layer_index=17
    )

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
