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
from tunix.experimental.trajectory import trajectory as trajectory_lib
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
          generation_kwargs={"max_generation_steps": 64},
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

  def test_to_rollout_response_trajectory_error(self):
    err = trajectory_lib.TrajectoryError(
        trajectory_id="err_traj_1",
        prompt_id="p1",
        error_message="episode failed",
        error_type="RuntimeError",
    )
    resp = self.worker._to_rollout_response(err)
    self.assertEqual(resp.status, "ERROR")
    self.assertEqual(resp.request_id, "err_traj_1")
    self.assertIsNone(resp.payload)
    self.assertIsNotNone(resp.error)
    self.assertEqual(resp.error.message, "episode failed")
    self.assertEqual(resp.error.error_type, "TrajectoryError")

  def test_to_rollout_response_trajectory_item(self):
    item = datatypes.TrajectoryItem(
        prompt_id="p1",
        group_index=0,
        traj={},
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        completion_tokens=np.array([10, 20], dtype=np.int32),
        action_mask=np.array([1.0, 1.0], dtype=np.float32),
        logprobs=[np.array([-0.1, -0.2], dtype=np.float32)],
        metadata={"foo": "bar"},
    )
    resp = self.worker._to_rollout_response(item)
    self.assertEqual(resp.status, "COMPLETED")
    self.assertEqual(resp.request_id, item.traj_id)
    self.assertIs(resp.payload, item)
    self.assertEqual(resp.metadata.get("foo"), "bar")

  def test_to_rollout_response_unsupported_type_raises(self):
    with self.assertRaises(TypeError):
      self.worker._to_rollout_response("not_a_trajectory")  # pyrefly: ignore[bad-argument-type]
    traj = trajectory_lib.Trajectory(
        trajectory_id="traj_1",
        agent=trajectory_lib.Agent(name="agent", version="1.0"),
    )
    with self.assertRaises(TypeError):
      self.worker._to_rollout_response(traj)


if __name__ == "__main__":
  absltest.main()
