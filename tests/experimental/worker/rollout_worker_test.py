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
from tunix.experimental.common import datatypes
from tunix.experimental.common import lineage
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.worker import rollout_worker


class _RecordingSampler(mocks.MockBaseSamplerImpl):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.seen_max_tokens = []

  async def sample(
      self,
      sampling_requests: (
          sampler_lib.SamplingRequest
          | list[sampler_lib.SamplingRequest]
          | tuple[sampler_lib.SamplingRequest, ...]
          | object
      ) = None,
      **kwargs,
  ):
    requests = (
        list(sampling_requests)
        if isinstance(sampling_requests, (list, tuple))
        else [sampling_requests]
    )
    for req in requests:
      self.seen_max_tokens.append(req.sampling_params.max_tokens)
    return await super().sample(sampling_requests=sampling_requests, **kwargs)


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

  def test_generate_request_runs_through_env_path(self):
    async def _run():
      req = datatypes.RolloutRequest(
          request_id="req_env_0",
          prompt="Solve task",
          prompt_id="prompt_env_0",
          group_index=0,
          generation_kwargs={"force_finish": True, "answer": "4"},
      )

      resp = await self.worker.generate(requests=req)

      self.assertIsInstance(resp, datatypes.RolloutResponse)
      self.assertEqual(resp.env_reward, 1.0)
      self.assertNotEmpty(resp.segments)

    asyncio.run(_run())

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
    with mock.patch.object(
        self.sampler, "initialize", side_effect=_initialize
    ) as init_mock:
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

  def test_generate_uses_request_max_generation_steps(self):
    sampler = _RecordingSampler(
        sampler_name="recording_sampler", default_delay=0.0
    )
    worker = rollout_worker.RolloutWorker(
        worker_id="rollout_worker_recording",
        sampler=sampler,
        env_pool=self.env_pool,
        agent_factory=mocks.MockAgent,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
    )
    request = datatypes.RolloutRequest(
        request_id="req_max_steps_0",
        prompt="What is 2+2?",
        prompt_id="prompt_max_steps_0",
        group_index=0,
        generation_kwargs={
            "max_generation_steps": 123,
            "force_finish": True,
            "answer": "4",
        },
    )

    asyncio.run(worker.generate(requests=request))

    self.assertNotEmpty(sampler.seen_max_tokens)
    self.assertEqual(sampler.seen_max_tokens[0], 123)


if __name__ == "__main__":
  absltest.main()
