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

"""Unit tests for RolloutWorker and lineage telemetry generation."""

import asyncio
import threading
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import lineage
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.worker import rollout_worker


class RolloutWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tokenizer = mocks.MockTokenizer()
    self.chat_parser = mocks.MockChatParser()
    self.sampler = mocks.MockBaseSamplerImpl(
        sampler_name="test_sampler", default_delay=0.0
    )
    self.env_pool = mocks.MockEnvironmentPool(pool_size=5, default_delay=0.0)
    self.worker = rollout_worker.RolloutWorker(
        worker_id="rollout_worker_42",
        sampler=self.sampler,
        env_pool=self.env_pool,
        agent_factory=mocks.MockAgent,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
    )

  def test_generate_appends_lineage_telemetry_event(self):
    async def _run():
      ctx = lineage.LineageContext(
          tracking_id="traj_prompt_1_0",
          parent_tracking_ids=["prompt_1"],
      )
      ctx.add_event(
          component="engine.dispatch",
          operation="rollout",
          attributes={"policy_version": 0, "group_index": 0},
      )

      req = datatypes.RolloutRequest(
          request_id="req_prompt_1_0",
          prompt="What is 2+2?",
          prompt_id="prompt_1",
          group_index=0,
          metadata={"lineage": ctx},
      )

      resp = await self.worker.generate(requests=req)
      self.assertIsInstance(resp, datatypes.RolloutResponse)
      self.assertIn("lineage", resp.metadata)
      resp_ctx = resp.metadata["lineage"]
      self.assertIs(resp_ctx, ctx)
      self.assertLen(resp_ctx.events, 2)

      dispatch_event = resp_ctx.events[0]
      self.assertEqual(dispatch_event.component, "engine.dispatch")
      self.assertEqual(dispatch_event.operation, "rollout")

      worker_event = resp_ctx.events[1]
      self.assertEqual(worker_event.component, "worker.rollout")
      self.assertEqual(worker_event.operation, "generate")
      self.assertEqual(
          worker_event.attributes.get("worker_id"), "rollout_worker_42"
      )

    asyncio.run(_run())

  def test_sampling_to_rollout_response_appends_lineage_event(self):
    ctx = lineage.LineageContext(
        tracking_id="traj_p2_0",
        parent_tracking_ids=["p2"],
    )
    req = datatypes.RolloutRequest(
        request_id="req_p2_0",
        prompt="Hello",
        prompt_id="p2",
        group_index=0,
        metadata={"lineage": ctx},
    )

    resp = self.worker._sampling_to_rollout_response(
        request=req,
        text="Hello there!",
        prompt_tokens=np.array([1, 2, 3], dtype=np.int32),
        token_ids=np.array([4, 5, 6], dtype=np.int32),
        logprobs=None,
    )

    self.assertIn("lineage", resp.metadata)
    resp_ctx = resp.metadata["lineage"]
    self.assertIs(resp_ctx, ctx)
    self.assertLen(resp_ctx.events, 1)
    self.assertEqual(resp_ctx.events[0].component, "worker.rollout")
    self.assertEqual(resp_ctx.events[0].operation, "generate")
    self.assertEqual(
        resp_ctx.events[0].attributes.get("worker_id"), "rollout_worker_42"
    )

  def test_initialize_only_runs_sampler_once_under_concurrency(self):
    enter_init = threading.Event()
    release_init = threading.Event()

    def _initialize():
      enter_init.set()
      release_init.wait(timeout=1.0)

    responses = []

    def _call_initialize():
      responses.append(self.worker.initialize())

    t1 = threading.Thread(target=_call_initialize)
    t2 = threading.Thread(target=_call_initialize)
    with mock.patch.object(self.sampler, "initialize", side_effect=_initialize) as init_mock:
      t1.start()
      self.assertTrue(enter_init.wait(timeout=1.0))
      t2.start()
      release_init.set()
      t1.join(timeout=1.0)
      t2.join(timeout=1.0)

    self.assertFalse(t1.is_alive())
    self.assertFalse(t2.is_alive())
    self.assertEqual(init_mock.call_count, 1)
    self.assertEqual(self.worker.state, datatypes.WorkerState.READY)
    self.assertLen(responses, 2)
    self.assertEqual(sum(bool(r.metadata.get("ready")) for r in responses), 1)


if __name__ == "__main__":
  absltest.main()
