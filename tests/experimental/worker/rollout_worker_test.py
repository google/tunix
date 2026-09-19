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

"""Unit tests for RolloutWorker, lineage telemetry, and Trajectory Store."""

import asyncio
import tempfile
import threading
from unittest import mock

from absl.testing import absltest
from etils import epath
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import lineage
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing
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

  def test_sample_prompts_with_return_routed_experts(self):
    async def _run():
      mock_routed = np.ones((2, 4, 8), dtype=np.int32)
      mock_response = sampler_lib.SamplingResponse(
          request_id="req_1",
          text="hello",
          prompt_token_ids=np.array([1, 2, 3], dtype=np.int32),
          token_ids=np.array([4, 5], dtype=np.int32),
          logprobs=np.array([0.0, 0.0], dtype=np.float32),
          routed_experts=mock_routed,
      )
      with mock.patch.object(
          self.sampler, "sample", return_value=[mock_response]
      ) as mock_sample:
        output = await self.worker.sample_prompts(
            ["test prompt"], return_routed_experts=True
        )
      self.assertLen(mock_sample.call_args[0][0], 1)
      req = mock_sample.call_args[0][0][0]
      self.assertTrue(req.sampling_params.return_routed_experts)
      self.assertIsNotNone(output.routed_experts)
      self.assertLen(output.routed_experts, 1)
      np.testing.assert_array_equal(output.routed_experts[0], mock_routed)

    asyncio.run(_run())

  def test_sample_prompts_defaults_no_routed_experts(self):
    async def _run():
      mock_response = sampler_lib.SamplingResponse(
          request_id="req_1",
          text="hello",
          prompt_token_ids=np.array([1, 2, 3], dtype=np.int32),
          token_ids=np.array([4, 5], dtype=np.int32),
          logprobs=np.array([0.0, 0.0], dtype=np.float32),
      )
      with mock.patch.object(
          self.sampler, "sample", return_value=[mock_response]
      ) as mock_sample:
        output = await self.worker.sample_prompts(["test prompt"])
      self.assertLen(mock_sample.call_args[0][0], 1)
      req = mock_sample.call_args[0][0][0]
      self.assertFalse(req.sampling_params.return_routed_experts)
      self.assertIsNone(output.routed_experts)

    asyncio.run(_run())


def _worker(config=None):
  return rollout_worker.RolloutWorker(
      worker_id="w0",
      config=config,
      sampler=mocks.MockBaseSamplerImpl(sampler_name="mock_sampler"),
      tokenizer="mock",
      chat_parser="mock",
  )


class RolloutWorkerTrajectoryStoreTest(absltest.TestCase):

  def test_no_config_means_no_store(self):
    worker = _worker()
    self.assertIsNone(worker.trajectory_store)

  def test_config_without_trajectory_store_config_means_no_store(self):
    worker = _worker(config=rollout_worker.RolloutConfig())
    self.assertIsNone(worker.trajectory_store)

  def test_disabled_trajectory_store_config_means_no_store(self):
    config = rollout_worker.RolloutConfig(
        trajectory_store_config={"enabled": False, "backend": "file"}
    )
    worker = _worker(config=config)
    self.assertIsNone(worker.trajectory_store)

  def test_enabled_file_backend_builds_store_once(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = rollout_worker.RolloutConfig(
        trajectory_store_config={
            "enabled": True,
            "backend": "file",
            "root_dir": str(tmp_dir),
            "run_id": "worker_run",
        }
    )
    worker = _worker(config=config)
    self.assertIsInstance(
        worker.trajectory_store, file_store.FileTrajectoryStore
    )
    worker.stop()

  def test_two_workers_get_two_independent_store_instances(self):
    # Each process constructs its own store; two RolloutWorker instances in
    # this test process stand in for two separate worker pods, each of which
    # would build its own store exactly once, in its own __init__.
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = {
        "enabled": True,
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": "shared",
    }
    worker_a = _worker(
        config=rollout_worker.RolloutConfig(trajectory_store_config=config)
    )
    worker_b = _worker(
        config=rollout_worker.RolloutConfig(trajectory_store_config=config)
    )
    self.assertIsNot(
        worker_a.trajectory_store, worker_b.trajectory_store
    )
    worker_a.stop()
    worker_b.stop()

  def test_stop_closes_the_store(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = rollout_worker.RolloutConfig(
        trajectory_store_config={
            "enabled": True,
            "backend": "file",
            "root_dir": str(tmp_dir),
            "run_id": "worker_run",
        }
    )
    worker = _worker(config=config)
    store = worker.trajectory_store
    assert store is not None
    worker.stop()
    with self.assertRaises(RuntimeError):
      store.add_step(trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1)

  def test_stop_closes_the_store_even_when_cancel_all_raises(self):
    worker = _worker()
    worker._trajectory_store = mock.MagicMock()  # pylint: disable=protected-access
    worker.manager.cancel_all = mock.MagicMock(
        side_effect=RuntimeError("cancel_all failed")
    )
    with self.assertRaises(RuntimeError):
      worker.stop()
    worker._trajectory_store.close.assert_called_once()  # pylint: disable=protected-access

  def test_stop_without_a_store_does_not_raise(self):
    worker = _worker()
    worker.stop()


if __name__ == "__main__":
  absltest.main()
