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

"""Unit tests for RolloutWorker request_id and prompt_id handling."""

import asyncio
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.rollout import sampler
from tunix.experimental.worker import rollout_worker


class RolloutWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_sampler = mock.AsyncMock(spec=sampler.Sampler)
    self.worker = rollout_worker.RolloutWorker(
        worker_id="rollout_worker_test_0",
        sampler=self.mock_sampler,
        tokenizer=mock.MagicMock(),
        chat_parser=mock.MagicMock(),
    )

  def test_sampling_to_rollout_response_echoes_request_id_and_prompt_id(self):
    req = datatypes.RolloutRequest(
        request_id="req_custom_uuid_123",
        prompt_id="42",
        group_index=2,
        prompt="Solve 2+2",
    )

    resp = self.worker._sampling_to_rollout_response(
        request=req,
        text="4",
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        token_ids=np.array([3], dtype=np.int32),
        logprobs=np.array([-0.1], dtype=np.float32),
    )

    self.assertEqual(resp.request_id, "req_custom_uuid_123")
    self.assertEqual(resp.prompt_id, "42")
    self.assertEqual(resp.group_index, 2)
    self.assertEqual(resp.status, "COMPLETED")
    self.assertEqual(resp.metadata.get("text"), "4")

  def test_to_rollout_response_preserves_rollout_response(self):
    existing_resp = datatypes.RolloutResponse(
        request_id="req_existing_uuid",
        prompt_id="99",
        group_index=1,
        status="COMPLETED",
    )

    result = self.worker._to_rollout_response(existing_resp)
    self.assertIs(result, existing_resp)
    self.assertEqual(result.request_id, "req_existing_uuid")
    self.assertEqual(result.prompt_id, "99")
    self.assertEqual(result.group_index, 1)

  def test_generate_direct_echoes_request_id_and_prompt_id(self):
    async def _run():
      self.mock_sampler.sample.return_value = [
          sampler.SamplingResponse(
              prompt_token_ids=np.array([1, 2], dtype=np.int32),
              token_ids=np.array([3, 4], dtype=np.int32),
              text="4",
              logprobs=np.array([-0.1, -0.2], dtype=np.float32),
          )
      ]
      req = datatypes.RolloutRequest(
          request_id="req_test_uuid_999",
          prompt_id="42",
          group_index=3,
          prompt="Test prompt",
      )
      responses = await self.worker._generate_rollout_requests_direct([req])
      self.mock_sampler.sample.assert_called_once()
      sampled_requests = self.mock_sampler.sample.call_args[0][0]
      self.assertLen(sampled_requests, 1)
      self.assertEqual(sampled_requests[0].request_id, "req_test_uuid_999")

      self.assertLen(responses, 1)
      resp = responses[0]
      self.assertEqual(resp.request_id, "req_test_uuid_999")
      self.assertEqual(resp.prompt_id, "42")
      self.assertEqual(resp.group_index, 3)
      self.assertEqual(resp.status, "COMPLETED")

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
