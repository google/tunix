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

"""Tests for LegacyVllmSamplerAdapter with Tunix VllmSampler."""

import asyncio
from unittest import mock
from absl.testing import absltest
import numpy as np

from tunix.experimental.rollout import legacy_vllm_sampler_adapter
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.generate import base_sampler


class LegacyVllmSamplerAdapterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_vllm_sampler = mock.MagicMock()
    self.mock_vllm_sampler.return_value = base_sampler.SamplerOutput(
        text=["completion 1"],
        logits=None,
        tokens=[np.array([10, 20, 30], dtype=np.int32)],
        padded_prompt_tokens=np.array([[1, 2]], dtype=np.int32),
        logprobs=None,
    )
    self.mock_vllm_lib = mock.MagicMock()
    self.mock_vllm_lib.VllmSampler.return_value = self.mock_vllm_sampler

    self.patcher = mock.patch.object(
        legacy_vllm_sampler_adapter,
        "_get_vllm_sampler_cls",
        return_value=self.mock_vllm_lib,
    )
    self.patcher.start()

    self.mock_tokenizer = mock.MagicMock()
    self.mock_config = mock.MagicMock()

    self.sampler_adapter = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(
        server_id="vllm_slice_01",
        tokenizer=self.mock_tokenizer,
        config=self.mock_config,
    )

  def tearDown(self):
    self.patcher.stop()
    super().tearDown()

  def test_single_sampling_request(self):
    req = base_sampler_lib.SamplingRequest(
        request_id="vllm_req_01",
        prompt="hello vllm",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=16,
            temperature=0.7,
        ),
    )
    response = asyncio.run(self.sampler_adapter.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "vllm_req_01")
    self.assertEqual(response.text, "completion 1")
    np.testing.assert_array_equal(response.token_ids, [10, 20, 30])
    self.mock_vllm_sampler.assert_called_once()

  def test_batch_sampling_requests(self):
    self.mock_vllm_sampler.return_value = base_sampler.SamplerOutput(
        text=["completion 1", "completion 2"],
        logits=None,
        tokens=[
            np.array([10, 20], dtype=np.int32),
            np.array([40, 50], dtype=np.int32),
        ],
        padded_prompt_tokens=np.array([[1, 2], [3, 4]], dtype=np.int32),
        logprobs=None,
    )
    reqs = [
        base_sampler_lib.SamplingRequest(
            request_id="req_1",
            prompt="prompt 1",
        ),
        base_sampler_lib.SamplingRequest(
            request_id="req_2",
            prompt="prompt 2",
        ),
    ]
    responses = asyncio.run(self.sampler_adapter.sample(reqs))
    self.assertIsInstance(responses, list)
    self.assertLen(responses, 2)
    self.assertEqual(responses[0].text, "completion 1")
    self.assertEqual(responses[1].text, "completion 2")

  def test_weight_sync(self):
    mock_weights = {"layer1": "weights"}
    res = asyncio.run(self.sampler_adapter.weight_sync(mock_weights))
    self.assertTrue(res)
    self.mock_vllm_sampler.update_params.assert_called_once_with(mock_weights)

  def test_uninitialized_sampler_raises(self):
    uninit = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(
        server_id="empty"
    )
    with self.assertRaises(RuntimeError):
      asyncio.run(
          uninit.sample(base_sampler_lib.SamplingRequest(prompt="test"))
      )


if __name__ == "__main__":
  absltest.main()
