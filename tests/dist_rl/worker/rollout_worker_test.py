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
import collections
import dataclasses
import tempfile
import threading
import types
from unittest import mock

from absl.testing import absltest
from etils import epath
import numpy as np
from opentelemetry.sdk import metrics as sdk_metrics
from opentelemetry.sdk.metrics import export as sdk_metrics_export
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import lineage
from tunix.dist_rl.common import registry
from tunix.dist_rl.common import test_utils as mocks
from tunix.dist_rl.testing import otel_testing
from tunix.dist_rl.trajectory import file_store
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.worker import otel_metrics
from tunix.dist_rl.worker import rollout_worker


class RolloutWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tokenizer = mocks.MockTokenizer()
    self.chat_parser = mocks.MockChatParser()
    self.sampler = mocks.MockBaseSamplerImpl(
        sampler_name="test_sampler", default_delay=0.0
    )
    self.worker = rollout_worker.RolloutWorker(
        worker_id="rollout_worker_42",
        sampler=self.sampler,
        env_name="mock_env",
        agent_name="mock_agent",
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
    )

  def test_missing_tokenizer_raises(self):
    with self.assertRaises(TypeError):
      rollout_worker.RolloutWorker(  # pyrefly: ignore[missing-argument]
          worker_id="rollout_worker_42",
          sampler=self.sampler,
          env_name="mock_env",
          agent_name="mock_agent",
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
      self.assertLen(resp_ctx.events, 3)

      dispatch_event = resp_ctx.events[0]
      self.assertEqual(dispatch_event.component, "engine.dispatch")
      self.assertEqual(dispatch_event.operation, "rollout")

      received_event = resp_ctx.events[1]
      self.assertEqual(received_event.component, "worker.received")
      self.assertEqual(received_event.operation, "generate")
      self.assertEqual(
          received_event.attributes.get("worker_id"), "rollout_worker_42"
      )
      self.assertNotIn("execution_time_s", received_event.attributes)
      self.assertGreaterEqual(
          received_event.timestamp_s, dispatch_event.timestamp_s
      )

      worker_event = resp_ctx.events[2]
      self.assertEqual(worker_event.component, "worker.rollout")
      self.assertEqual(worker_event.operation, "generate")
      self.assertEqual(
          worker_event.attributes.get("worker_id"), "rollout_worker_42"
      )
      self.assertGreaterEqual(
          worker_event.timestamp_s, received_event.timestamp_s
      )
      self.assertGreaterEqual(
          worker_event.attributes.get("execution_time_s", -1.0), 0.0
      )

      # Batched `generate([req_1, req_2])` also stamps `worker.received` and
      # `worker.rollout` on each request's lineage context.
      batch_ctxs = [
          lineage.LineageContext(tracking_id=f"traj_batch_{idx}")
          for idx in range(2)
      ]
      for b_ctx in batch_ctxs:
        b_ctx.add_event(component="engine.dispatch", operation="rollout")
      batch_reqs = [
          datatypes.RolloutRequest(
              request_id=f"req_batch_{idx}",
              prompt=f"Prompt {idx}",
              prompt_id=f"prompt_batch_{idx}",
              group_index=idx,
              generation_kwargs={"max_generation_steps": 64},
              metadata={"lineage": batch_ctxs[idx]},
          )
          for idx in range(2)
      ]
      batch_resps = await self.worker.generate(requests=batch_reqs)
      self.assertLen(batch_resps, 2)
      for idx, b_resp in enumerate(batch_resps):
        self.assertEqual(
            [e.component for e in batch_ctxs[idx].events],
            ["engine.dispatch", "worker.received", "worker.rollout"],
        )
        self.assertIs(b_resp.metadata["lineage"], batch_ctxs[idx])

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
    ctx = lineage.LineageContext(
        tracking_id="traj_p1_3",
        parent_tracking_ids=["p1"],
    )
    err = trajectory_lib.TrajectoryError(
        trajectory_id="err_traj_1",
        prompt_id="p1",
        error_message="episode failed",
        error_type="RuntimeError",
        metadata={
            "batch_idx": 2,
            "prompt_idx": 9,
            "intra_batch_idx": 1,
            "group_index": 3,
            "policy_version": 4,
            "lineage": ctx,
        },
    )
    resp = self.worker._to_rollout_response(err)
    self.assertEqual(resp.status, "ERROR")
    self.assertEqual(resp.request_id, "err_traj_1")
    self.assertIsNone(resp.payload)
    self.assertIsNotNone(resp.error)
    self.assertEqual(resp.error.message, "episode failed")
    self.assertEqual(resp.error.error_type, "TrajectoryError")
    self.assertEqual(
        resp.metadata,
        {
            "prompt_id": "p1",
            "batch_idx": 2,
            "prompt_idx": 9,
            "intra_batch_idx": 1,
            "group_index": 3,
            "policy_version": 4,
            "lineage": ctx,
        },
    )
    self.assertLen(ctx.events, 1)
    self.assertEqual(ctx.events[0].component, "worker.rollout")
    self.assertEqual(
        ctx.events[0].attributes.get("worker_id"), "rollout_worker_42"
    )

  def test_to_rollout_response_trajectory_item(self):
    item = datatypes.TrajectoryItem(
        prompt_id="p1",
        group_index=0,
        traj={"status": "SUCCEEDED"},
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

  def test_generate_rejects_non_rollout_requests(self):
    async def _run():
      with self.assertRaises(TypeError):
        await self.worker.generate("raw string prompt")  # pyrefly: ignore[bad-argument-type]
      with self.assertRaises(TypeError):
        await self.worker.generate(["raw string prompt"])  # pyrefly: ignore[bad-argument-type]

    asyncio.run(_run())

  def test_generate_accepts_non_list_sequence(self):
    async def _run():
      req = datatypes.RolloutRequest(
          request_id="req_deque_1",
          prompt="What is 2+2?",
          prompt_id="prompt_1",
          group_index=0,
          generation_kwargs={"max_generation_steps": 64},
      )
      responses = await self.worker.generate(collections.deque([req]))
      self.assertIsInstance(responses, list)
      self.assertLen(responses, 1)
      self.assertIsInstance(responses[0], datatypes.RolloutResponse)

    asyncio.run(_run())

  def test_start_async_starts_sampler_and_binds_weight_sync(self):
    class _AsyncSampler(mocks.MockBaseSamplerImpl):

      def __init__(self):
        super().__init__(sampler_name="async_sampler")
        self.started = False
        self.warmed = False

      async def start(self):
        self.started = True

      async def bind_weight_sync(self):
        self.warmed = True

    async_sampler = _AsyncSampler()
    worker = rollout_worker.RolloutWorker(
        worker_id="rollout_async",
        sampler=async_sampler,
        env_name="mock_env",
        agent_name="mock_agent",
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
    )

    async def _run():
      resp = await worker.start()
      self.assertEqual(resp.metadata["worker_id"], "rollout_async")
      self.assertTrue(resp.metadata["started"])
      self.assertTrue(async_sampler.started)
      self.assertTrue(async_sampler.warmed)

    asyncio.run(_run())

  def test_initialize_failure_sets_error_state(self):
    with mock.patch.object(
        self.sampler, "initialize", side_effect=RuntimeError("vllm init boom")
    ):
      with self.assertRaisesRegex(RuntimeError, "vllm init boom"):
        self.worker.initialize()
    self.assertEqual(self.worker.state, datatypes.WorkerState.ERROR)

  def test_start_failure_sets_error_state(self):
    with mock.patch.object(
        self.sampler, "start", side_effect=RuntimeError("vllm start boom")
    ):
      with self.assertRaisesRegex(RuntimeError, "vllm start boom"):
        asyncio.run(self.worker.start())
    self.assertEqual(self.worker.state, datatypes.WorkerState.ERROR)

  def test_bind_and_get_weight_sync_metadata_auto_initialize_from_pending(self):
    worker = rollout_worker.RolloutWorker(
        worker_id="rollout_pending_sync",
        sampler=self.sampler,
        env_name="mock_env",
        agent_name="mock_agent",
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
    )
    self.assertEqual(worker.state, datatypes.WorkerState.PENDING)
    asyncio.run(worker.bind_weight_sync())
    self.assertEqual(worker.state, datatypes.WorkerState.READY)

    worker2 = rollout_worker.RolloutWorker(
        worker_id="rollout_pending_meta",
        sampler=self.sampler,
        env_name="mock_env",
        agent_name="mock_agent",
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
    )
    self.assertEqual(worker2.state, datatypes.WorkerState.PENDING)
    with mock.patch.object(
        self.sampler,
        "get_weight_sync_metadata",
        new=mock.AsyncMock(return_value=[]),
    ):
      asyncio.run(worker2.get_weight_sync_metadata())
    self.assertEqual(worker2.state, datatypes.WorkerState.READY)

  def test_abort_after_commit_still_reports_committed(self):
    request = datatypes.WeightSyncRequest(
        controller_id="c",
        policy_version=1,
        extra_config={"req_id": "wsync-v1-r0", "uuid": 1},
    )

    async def _run():
      await self.worker.bind_weight_sync()
      await self.worker.pre_weight_sync(request)
      await self.worker.weight_sync(request)
      await self.worker.post_weight_sync(request)
      # A rollback abort racing a late commit must not hide the commit, nor may
      # an abort for an older round or one without a round key.
      await self.worker.abort_weight_sync(request)
      await self.worker.abort_weight_sync(
          dataclasses.replace(
              request, extra_config={"req_id": "wsync-v0-r0", "uuid": 0}
          )
      )
      await self.worker.abort_weight_sync(None)
      return await self.worker.get_weight_sync_status()

    status = asyncio.run(_run())
    self.assertEqual(status["phase"], "committed")
    self.assertEqual(status["req_id"], "wsync-v1-r0")
    self.assertEqual(status["uuid"], 1)

  def test_weight_sync_status_prefers_sampler_round_tracker(self):
    tracker_report = {"req_id": "wsync-v2-r1", "uuid": 2, "phase": "committed"}
    with mock.patch.object(
        self.sampler,
        "get_weight_sync_status",
        new=mock.AsyncMock(return_value=tracker_report),
        create=True,
    ):
      status = asyncio.run(self.worker.get_weight_sync_status())
    self.assertEqual(status, dict(tracker_report, policy_version=0))

  def test_weight_sync_status_falls_back_when_tracker_never_opened_round(self):
    unopened_report = {
        "req_id": None,
        "uuid": -1,
        "phase": "idle",
        "policy_version": -1,
    }
    request = datatypes.WeightSyncRequest(
        controller_id="c",
        policy_version=1,
        extra_config={"req_id": "wsync-v1-r0", "uuid": 1},
    )
    asyncio.run(self.worker.pre_weight_sync(request))
    with mock.patch.object(
        self.sampler,
        "get_weight_sync_status",
        new=mock.AsyncMock(return_value=unopened_report),
        create=True,
    ):
      status = asyncio.run(self.worker.get_weight_sync_status())
    self.assertEqual(
        status,
        {
            "req_id": "wsync-v1-r0",
            "uuid": 1,
            "phase": "prepared",
            "policy_version": 0,
        },
    )


def _worker(config=None):
  return rollout_worker.RolloutWorker(
      worker_id="w0",
      config=config,
      sampler=mocks.MockBaseSamplerImpl(sampler_name="mock_sampler"),
      env_name="mock_env",
      agent_name="mock_agent",
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

  def test_enabled_file_backend_builds_store_once(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = rollout_worker.RolloutConfig(
        trajectory_store_config={
            "backend": "file",
            "root_dir": str(tmp_dir),
            "run_id": "worker_run",
        }
    )
    worker = _worker(config=config)
    self.assertIsInstance(
        worker.trajectory_store, file_store.FileTrajectoryStore
    )
    self.assertIs(
        worker.trajectory_store.bound_metadata_cls,
        trajectory_lib.TunixTrajectoryMetadata,
    )
    worker.stop()

  def test_two_workers_get_two_independent_store_instances(self):
    # Each process constructs its own store; two RolloutWorker instances in
    # this test process stand in for two separate worker pods, each of which
    # would build its own store exactly once, in its own __init__.
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = {
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
    self.assertIsNot(worker_a.trajectory_store, worker_b.trajectory_store)
    worker_a.stop()
    worker_b.stop()

  def test_stop_closes_the_store(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    config = rollout_worker.RolloutConfig(
        trajectory_store_config={
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
      store.add_step(
          trajectory_testing.TUNIX_ENV_STEP_0,
          trajectory_testing.TUNIX_METADATA_1,
      )

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

  def test_stop_is_idempotent_and_invokes_on_stop_callback(self):
    worker = _worker()
    mock_store = mock.MagicMock()
    worker._trajectory_store = mock_store  # pylint: disable=protected-access
    worker.manager.cancel_all = mock.MagicMock()
    observed_aborts = []
    worker.set_on_stop(observed_aborts.append)

    first = worker.stop(abort=True)
    second = worker.stop(abort=False)

    self.assertEqual(worker.state, datatypes.WorkerState.STOPPED)
    self.assertEqual(
        first.metadata,
        {
            "worker_id": "w0",
            "policy_version": 0,
            "state": "STOPPED",
            "stopped": True,
            "abort": True,
        },
    )
    self.assertEqual(
        second.metadata,
        {
            "worker_id": "w0",
            "policy_version": 0,
            "state": "STOPPED",
            "stopped": True,
            "already_stopped": True,
            "abort": False,
        },
    )
    worker.manager.cancel_all.assert_called_once_with()
    mock_store.close.assert_called_once_with()
    # The callback's abort flag stays latched after a later non-abort stop.
    self.assertEqual(observed_aborts, [True, True])

  def test_stop_keeps_failure_signal_without_recleaning_if_cleanup_raises(
      self,
  ):
    worker = _worker()
    mock_store = mock.MagicMock()
    worker._trajectory_store = mock_store  # pylint: disable=protected-access
    worker.manager.cancel_all = mock.MagicMock(
        side_effect=RuntimeError("cancel_all failed")
    )
    observed_aborts = []
    worker.set_on_stop(observed_aborts.append)

    # A clean (non-abort) stop whose cleanup fails is escalated to an abort.
    with self.assertRaisesRegex(RuntimeError, "cancel_all failed"):
      worker.stop()

    self.assertEqual(worker.state, datatypes.WorkerState.STOPPED)
    mock_store.close.assert_called_once_with()
    self.assertEqual(observed_aborts, [True])

    # Local teardown does not re-run cleanup, but still surfaces the earlier
    # failure with the abort flag latched.
    with self.assertRaisesRegex(
        RuntimeError, "failed to stop cleanly earlier: cancel_all failed"
    ):
      worker.stop()
    worker.manager.cancel_all.assert_called_once_with()
    mock_store.close.assert_called_once_with()
    self.assertEqual(observed_aborts, [True, True])

  def test_info_and_initialize_report_trajectory_store_config(self):
    worker_disabled = _worker()
    self.assertIsNone(
        worker_disabled.info().resources["trajectory_store_config"]
    )
    self.assertIsNone(
        worker_disabled.initialize().metadata["trajectory_store_config"]
    )
    worker_disabled.stop()

    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    input_cfg = {
        "backend": "file",
        "root_dir": str(tmp_dir),
        "run_id": "handshake_run",
    }
    worker_enabled = _worker(
        config=rollout_worker.RolloutConfig(trajectory_store_config=input_cfg)
    )
    self.assertEqual(
        worker_enabled.info().resources["trajectory_store_config"],
        input_cfg,
    )
    self.assertEqual(
        worker_enabled.initialize().metadata["trajectory_store_config"],
        input_cfg,
    )
    worker_enabled.stop()

  def test_rollout_config_drain_timeout_forwarded_and_reported(self):
    worker_default = _worker()
    self.assertEqual(worker_default.manager.drain_timeout_s, 660.0)
    self.assertEqual(worker_default.info().resources["drain_timeout_s"], 660.0)
    self.assertFalse(worker_default.info().resources["partial_rollout"])
    init_meta = worker_default.initialize().metadata
    self.assertEqual(init_meta["drain_timeout_s"], 660.0)
    self.assertFalse(init_meta["partial_rollout"])
    worker_default.stop()

    worker_custom = _worker(
        config=rollout_worker.RolloutConfig(
            drain_timeout_s=1860.0,
            partial_rollout=True,
        )
    )
    self.assertEqual(worker_custom.manager.drain_timeout_s, 1860.0)
    self.assertEqual(worker_custom.info().resources["drain_timeout_s"], 1860.0)
    self.assertTrue(worker_custom.info().resources["partial_rollout"])
    custom_meta = worker_custom.initialize().metadata
    self.assertEqual(custom_meta["drain_timeout_s"], 1860.0)
    self.assertTrue(custom_meta["partial_rollout"])
    worker_custom.stop()

    for invalid in (0, -10.0, float("nan"), float("inf"), True, False):
      with self.assertRaisesRegex(
          ValueError, "drain_timeout_s must be a positive finite number"
      ):
        rollout_worker.RolloutConfig(drain_timeout_s=invalid)

  def test_resolves_registered_env_and_agent_with_configs(self):
    class _FakeEnv(mocks.MockEnvironment):

      def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    class _FakeAgent(mocks.MockAgent):

      def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    env_cfg = {"backend": "k8s"}
    agent_cfg = {"scaffold": "r2egym", "source": "worker"}
    with (
        mock.patch.object(
            registry.ENV_REGISTRY, "get", return_value=_FakeEnv
        ) as mock_env_get,
        mock.patch.object(
            registry.AGENT_REGISTRY, "get", return_value=_FakeAgent
        ) as mock_agent_get,
    ):
      worker = rollout_worker.RolloutWorker(
          worker_id="rollout_worker_reg",
          config=rollout_worker.RolloutConfig(),
          sampler=mocks.MockBaseSamplerImpl(
              sampler_name="test_sampler", default_delay=0.0
          ),
          tokenizer=mocks.MockTokenizer(),
          chat_parser=mocks.MockChatParser(),
          env_name="test_env",
          agent_name="test_agent",
          env_config=env_cfg,
          agent_config=agent_cfg,
      )

    mock_env_get.assert_called_once_with("test_env")
    mock_agent_get.assert_called_once_with("test_agent")
    self.assertIs(worker.manager.env_factory, _FakeEnv)
    self.assertIs(worker.manager.agent_factory, _FakeAgent)
    self.assertIsInstance(worker.manager.env_config, types.MappingProxyType)
    self.assertIsInstance(worker.manager.agent_config, types.MappingProxyType)
    env_cfg["backend"] = "mutated"
    agent_cfg["source"] = "mutated"
    self.assertEqual(worker.manager.env_config, {"backend": "k8s"})
    self.assertEqual(
        worker.manager.agent_config,
        {"scaffold": "r2egym", "source": "worker"},
    )

  def test_unregistered_env_or_agent_raises_key_error(self):
    with self.assertRaises(KeyError):
      rollout_worker.RolloutWorker(
          worker_id="rollout_worker_bad_env",
          sampler=mocks.MockBaseSamplerImpl(
              sampler_name="test_sampler", default_delay=0.0
          ),
          tokenizer=mocks.MockTokenizer(),
          chat_parser=mocks.MockChatParser(),
          env_name="unregistered_env_xyz",
          agent_name="mock_agent",
      )
    with self.assertRaises(KeyError):
      rollout_worker.RolloutWorker(
          worker_id="rollout_worker_bad_agent",
          sampler=mocks.MockBaseSamplerImpl(
              sampler_name="test_sampler", default_delay=0.0
          ),
          tokenizer=mocks.MockTokenizer(),
          chat_parser=mocks.MockChatParser(),
          env_name="mock_env",
          agent_name="unregistered_agent_xyz",
      )


class RolloutWorkerOTelMetricsTest(absltest.TestCase):
  """Verifies direct OpenTelemetry SDK instruments emitted by RolloutWorker."""

  def setUp(self):
    super().setUp()
    self.reader = sdk_metrics_export.InMemoryMetricReader()
    self.meter_provider = sdk_metrics.MeterProvider(
        metric_readers=[self.reader]
    )
    self.addCleanup(self.meter_provider.shutdown)
    self.sampler = mocks.MockBaseSamplerImpl(
        sampler_name="test_sampler", default_delay=0.0
    )
    self.worker = rollout_worker.RolloutWorker(
        worker_id="rollout_otel_0",
        sampler=self.sampler,
        env_name="mock_env",
        agent_name="mock_agent",
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
        otel_meter_provider=self.meter_provider,
    )

  def test_emits_all_rollout_worker_otel_instruments_from_trajectory_item(self):
    ctx = lineage.LineageContext(
        tracking_id="traj_p1_0",
        task_name="swe_bench",
        parent_tracking_ids=["p1"],
    )
    item = datatypes.TrajectoryItem(
        prompt_id="p1",
        group_index=0,
        traj={
            "status": "SUCCEEDED",
            "prompt_tokens": np.array([1, 2, 3, 4], dtype=np.int32),
            "conversation_tokens": np.array(
                [10, 11, 12, 13, 98, 99, 14, 15], dtype=np.int32
            ),
            "assistant_turn_lengths": [4, 2],
            "execution_time_s": 1.25,
            "sampler_time_s": 0.75,
            "env_time": {
                "reset_latency": 0.10,
                "step_latency": [0.20, 0.30],
                "close_latency": 0.05,
            },
            "reward_time": {
                "reward_latency": 0.15,
            },
        },
        metadata={
            "clipped": False,
            "lineage": ctx,
        },
    )

    resp = self.worker._to_rollout_response(item)
    self.assertEqual(resp.status, "COMPLETED")
    metrics = otel_testing.collect_metrics(self.reader)

    self.assertEqual(
        metrics["tunix.worker.trajectories.completed"],
        [(
            1,
            None,
            {
                "worker_id": "rollout_otel_0",
                "task_name": "swe_bench",
                "status": "SUCCEEDED",
                "clipped": "false",
                "has_error": "false",
            },
        )],
    )
    self.assertEqual(
        metrics["tunix.worker.turns.total"],
        [(
            2,
            None,
            {
                "worker_id": "rollout_otel_0",
                "task_name": "swe_bench",
            },
        )],
    )
    self.assertEqual(
        metrics["tunix.worker.prompt_tokens"],
        [(
            4,
            1,
            {
                "worker_id": "rollout_otel_0",
                "task_name": "swe_bench",
                "status": "SUCCEEDED",
            },
        )],
    )
    self.assertEqual(
        metrics["tunix.worker.completion_tokens"],
        [(
            6,
            1,
            {
                "worker_id": "rollout_otel_0",
                "task_name": "swe_bench",
                "status": "SUCCEEDED",
                "clipped": "false",
            },
        )],
    )
    exec_pts = metrics["tunix.worker.execution_sec"]
    self.assertLen(exec_pts, 1)
    self.assertAlmostEqual(exec_pts[0][0], 1.25)
    self.assertEqual(exec_pts[0][1], 1)
    self.assertEqual(
        exec_pts[0][2],
        {
            "worker_id": "rollout_otel_0",
            "task_name": "swe_bench",
            "status": "SUCCEEDED",
        },
    )

    sampler_pts = metrics["tunix.worker.sampler_time_s"]
    self.assertLen(sampler_pts, 1)
    self.assertAlmostEqual(sampler_pts[0][0], 0.75)
    self.assertEqual(sampler_pts[0][1], 1)
    self.assertEqual(
        sampler_pts[0][2],
        {
            "worker_id": "rollout_otel_0",
            "task_name": "swe_bench",
        },
    )

    env_by_op = {
        attrs["op"]: (val, cnt, attrs)
        for val, cnt, attrs in metrics["tunix.worker.env_time_sec"]
    }
    self.assertCountEqual(env_by_op.keys(), ["reset", "step", "close"])
    self.assertAlmostEqual(env_by_op["reset"][0], 0.10)
    self.assertEqual(env_by_op["reset"][1], 1)
    self.assertAlmostEqual(env_by_op["step"][0], 0.50)
    self.assertEqual(env_by_op["step"][1], 2)
    self.assertAlmostEqual(env_by_op["close"][0], 0.05)
    self.assertEqual(env_by_op["close"][1], 1)

    reward_pts = metrics["tunix.worker.reward_time_sec"]
    self.assertLen(reward_pts, 1)
    self.assertAlmostEqual(reward_pts[0][0], 0.15)
    self.assertEqual(reward_pts[0][1], 1)
    self.assertEqual(
        reward_pts[0][2],
        {
            "worker_id": "rollout_otel_0",
            "task_name": "swe_bench",
        },
    )

  def test_emits_clipped_error_and_single_count_with_on_complete(self):
    # 1. TrajectoryError without lineage uses task_name="" and status="ERROR".
    err = trajectory_lib.TrajectoryError(
        trajectory_id="err_1",
        prompt_id="p_err",
        error_message="boom",
        error_type="RuntimeError",
    )
    self.worker._to_rollout_response(err)

    # 2. Clipped TrajectoryItem with TIMEOUT status and prompt_length override.
    clipped_item = datatypes.TrajectoryItem(
        prompt_id="p_clip",
        group_index=1,
        traj={
            "status": "TIMEOUT",
            "prompt_tokens": np.array([0, 0, 1, 2, 3], dtype=np.int32),
            "prompt_length": 3,
            "assistant_turn_lengths": [4],
        },
        metadata={"clipped": True},
    )
    self.worker._to_rollout_response(clipped_item)

    # 3. Budget termination (MAX_CONTEXT_LIMIT_REACHED) sets has_error="false";
    # untokenized empty prompt_tokens=[], timed-out close_latency=0.0, and
    # unexecuted pre-seeded reward_latency=0.0 are skipped rather than recorded
    # as spurious 0 observations.
    budget_item = datatypes.TrajectoryItem(
        prompt_id="p_budget",
        group_index=0,
        traj={
            "status": "MAX_CONTEXT_LIMIT_REACHED",
            "prompt_tokens": [],
            "assistant_turn_lengths": [3, 2],
            "env_time": {
                "reset_latency": 0.12,
                "step_latency": [],
                "close_latency": 0.0,
            },
            "reward_time": {
                "reward_latency": 0.0,
            },
        },
        metadata={"clipped": True},
    )
    self.worker._to_rollout_response(budget_item)

    # 4. End-to-end generate() with on_complete callback must record once.
    callback_responses = []
    req = datatypes.RolloutRequest(
        request_id="req_gen_1",
        prompt="What is 2+2?",
        prompt_id="prompt_gen_1",
        group_index=0,
        generation_kwargs={"max_generation_steps": 64},
    )
    resp = asyncio.run(
        self.worker.generate(req, on_complete=callback_responses.append)
    )
    self.assertLen(callback_responses, 1)
    self.assertIs(callback_responses[0], resp)

    metrics = otel_testing.collect_metrics(self.reader)
    completed_by_status = {
        attrs["status"]: (val, attrs)
        for val, _, attrs in metrics["tunix.worker.trajectories.completed"]
    }
    self.assertEqual(
        completed_by_status["ERROR"],
        (
            1,
            {
                "worker_id": "rollout_otel_0",
                "task_name": "",
                "status": "ERROR",
                "clipped": "false",
                "has_error": "true",
            },
        ),
    )
    self.assertEqual(
        completed_by_status["TIMEOUT"],
        (
            1,
            {
                "worker_id": "rollout_otel_0",
                "task_name": "",
                "status": "TIMEOUT",
                "clipped": "true",
                "has_error": "true",
            },
        ),
    )
    self.assertEqual(
        completed_by_status["MAX_CONTEXT_LIMIT_REACHED"],
        (
            1,
            {
                "worker_id": "rollout_otel_0",
                "task_name": "",
                "status": "MAX_CONTEXT_LIMIT_REACHED",
                "clipped": "true",
                "has_error": "false",
            },
        ),
    )
    gen_status = str(resp.payload.traj["status"])
    self.assertEqual(completed_by_status[gen_status][0], 1)

    prompt_by_status = {
        attrs["status"]: val
        for val, _, attrs in metrics["tunix.worker.prompt_tokens"]
    }
    # Uses unpadded prompt_length=3 rather than len(prompt_tokens)=5, and skips
    # untokenized empty prompt_tokens=[] on MAX_CONTEXT_LIMIT_REACHED.
    self.assertEqual(prompt_by_status["TIMEOUT"], 3)
    self.assertNotIn("MAX_CONTEXT_LIMIT_REACHED", prompt_by_status)

    env_ops_recorded = {
        attrs["op"] for _, _, attrs in metrics["tunix.worker.env_time_sec"]
    }
    # budget_item had reset=0.12, close=0.0 (timed-out), while generate() used
    # MockEnvironment with positive reset/step/close; verify budget_item did not
    # add a second close observation if only generate() closed cleanly.
    close_cnt = sum(
        cnt
        for _, cnt, attrs in metrics["tunix.worker.env_time_sec"]
        if attrs["op"] == "close"
    )
    self.assertIn("reset", env_ops_recorded)
    self.assertLessEqual(close_cnt, 1)
    # Neither budget_item (masked_out reward_latency=0.0) nor MockEnvironment
    # (final_reward_fn=None -> pre-seeded reward_latency=0.0) ran a final reward
    # callback, so reward_time_sec must record zero observations.
    self.assertNotIn("tunix.worker.reward_time_sec", metrics)

    completion_by_status = {
        attrs["status"]: val
        for val, _, attrs in metrics["tunix.worker.completion_tokens"]
    }
    self.assertEqual(completion_by_status["TIMEOUT"], 4)
    self.assertEqual(completion_by_status["MAX_CONTEXT_LIMIT_REACHED"], 5)

    # 5. Store-rehydrated empty timing dicts `{}` emit zero timing points, and
    # contract violations fail loudly before mutating counters and when OTel is
    # disabled.
    rehydrated_item = datatypes.TrajectoryItem(
        prompt_id="p_rehydrated",
        group_index=0,
        traj={"status": "SUCCEEDED", "env_time": {}, "reward_time": {}},
    )
    self.worker._to_rollout_response(rehydrated_item)
    with self.assertRaises(TypeError):
      asyncio.run(
          self.worker.generate(
              datatypes.RolloutRequest(
                  request_id="req_bad_lineage",
                  prompt="bad",
                  prompt_id="p_bad",
                  metadata={"lineage": "not_a_lineage_context"},
              )
          )
      )
    with self.assertRaises(TypeError):
      self.worker._to_rollout_response(
          datatypes.TrajectoryItem(
              prompt_id="p",
              group_index=0,
              traj={"status": datatypes.TrajectoryStatus.SUCCEEDED},
          )
      )
    with self.assertRaises(TypeError):
      self.worker._to_rollout_response(
          datatypes.TrajectoryItem(
              prompt_id="p",
              group_index=0,
              traj={"status": "SUCCEEDED", "env_time": 0.5},
          )
      )
    with self.assertRaises(TypeError):
      self.worker._to_rollout_response(
          datatypes.TrajectoryItem(
              prompt_id="p",
              group_index=0,
              traj={"status": "SUCCEEDED", "reward_time": 0.5},
          )
      )
    with mock.patch.object(self.worker._otel, "_enabled", False):
      with self.assertRaises(TypeError):
        self.worker._to_rollout_response(
            datatypes.TrajectoryItem(prompt_id="p", group_index=0, traj=None)  # pyrefly: ignore[bad-argument-type]
        )
      with self.assertRaises(KeyError):
        self.worker._to_rollout_response(
            datatypes.TrajectoryItem(prompt_id="p", group_index=0, traj={})
        )
      with self.assertRaises(TypeError):
        self.worker._to_rollout_response(
            datatypes.TrajectoryItem(
                prompt_id="p",
                group_index=0,
                traj={"status": "SUCCEEDED", "reward_time": 0.5},
            )
        )
      with self.assertRaises(TypeError):
        self.worker._to_rollout_response(
            datatypes.TrajectoryItem(
                prompt_id="p",
                group_index=0,
                traj={"status": "SUCCEEDED"},
                metadata={"lineage": "not_a_lineage_context"},
            )
        )

  def test_emits_weight_sync_durations_and_requests_inflight_gauge(self):
    enter_sample = asyncio.Event()
    release_sample = asyncio.Event()

    class _BlockingSampler(mocks.MockBaseSamplerImpl):

      async def sample(self, req, **kwargs):
        enter_sample.set()
        await release_sample.wait()
        return await super().sample(req, **kwargs)

    worker_1 = rollout_worker.RolloutWorker(
        worker_id="rollout_otel_1",
        sampler=_BlockingSampler(sampler_name="blocking_sampler"),
        env_name="mock_env",
        agent_name="mock_agent",
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
        otel_meter_provider=self.meter_provider,
    )

    async def _run_sync_and_inflight():
      req_sync = datatypes.WeightSyncRequest(
          controller_id="c",
          policy_version=2,
          extra_config={"req_id": "wsync-v2-r0", "uuid": 1},
      )
      await self.worker.pre_weight_sync(req_sync)
      # Idempotent duplicate pre_weight_sync must not overwrite pause start.
      first_pause_start = self.worker._otel._sync_pause_start_s  # pylint: disable=protected-access
      await self.worker.pre_weight_sync(req_sync)
      self.assertEqual(
          self.worker._otel._sync_pause_start_s, first_pause_start  # pylint: disable=protected-access
      )
      await self.worker.weight_sync(req_sync)
      await self.worker.post_weight_sync(req_sync)
      # Stale abort for an already-committed round must not emit aborted_paused.
      await self.worker.abort_weight_sync(req_sync)

      # Second round: genuine abort emits aborted_paused.
      req_abort = datatypes.WeightSyncRequest(
          controller_id="c",
          policy_version=3,
          extra_config={"req_id": "wsync-v3-r0", "uuid": 2},
      )
      await self.worker.pre_weight_sync(req_abort)
      await self.worker.abort_weight_sync(req_abort)

      # Observe live inflight gauge across both workers while worker_1 has 1
      # active request suspended in sampler.sample().
      gen_task = asyncio.create_task(
          worker_1.generate(
              datatypes.RolloutRequest(
                  request_id="req_inflight_1",
                  prompt="What is 2+2?",
                  prompt_id="p_inflight",
                  group_index=0,
                  generation_kwargs={"max_generation_steps": 16},
              )
          )
      )
      await enter_sample.wait()
      active_metrics = otel_testing.collect_metrics(self.reader)
      release_sample.set()
      await gen_task
      idle_metrics = otel_testing.collect_metrics(self.reader)
      return active_metrics, idle_metrics

    active_metrics, idle_metrics = asyncio.run(_run_sync_and_inflight())

    ws_by_phase = {
        attrs["phase"]: (val, cnt, attrs)
        for val, cnt, attrs in idle_metrics["tunix.worker.weight_sync.duration"]
    }
    self.assertCountEqual(
        ws_by_phase.keys(),
        ["pre", "h2d", "post", "total_paused", "aborted_paused"],
    )
    self.assertEqual(ws_by_phase["pre"][1], 3)
    self.assertEqual(ws_by_phase["h2d"][1], 1)
    self.assertEqual(ws_by_phase["post"][1], 1)
    self.assertEqual(ws_by_phase["total_paused"][1], 1)
    self.assertEqual(ws_by_phase["aborted_paused"][1], 1)
    for val, _, attrs in ws_by_phase.values():
      self.assertGreaterEqual(val, 0.0)
      self.assertEqual(attrs["worker_id"], "rollout_otel_0")

    active_inflight = {
        attrs["worker_id"]: val
        for val, _, attrs in active_metrics["tunix.worker.requests_inflight"]
    }
    self.assertEqual(
        active_inflight,
        {"rollout_otel_0": 0, "rollout_otel_1": 1},
    )
    idle_inflight = {
        attrs["worker_id"]: val
        for val, _, attrs in idle_metrics["tunix.worker.requests_inflight"]
    }
    self.assertEqual(
        idle_inflight,
        {"rollout_otel_0": 0, "rollout_otel_1": 0},
    )

    worker_1.stop()
    post_stop_metrics = otel_testing.collect_metrics(self.reader)
    post_stop_inflight = {
        attrs["worker_id"]: val
        for val, _, attrs in post_stop_metrics["tunix.worker.requests_inflight"]
    }
    self.assertEqual(post_stop_inflight, {"rollout_otel_0": 0})

  def test_records_nothing_when_opentelemetry_is_not_installed(self):
    with mock.patch.object(otel_metrics, "otel_api", None):
      disabled_worker = rollout_worker.RolloutWorker(
          worker_id="rollout_otel_disabled",
          sampler=self.sampler,
          env_name="mock_env",
          agent_name="mock_agent",
          tokenizer=mocks.MockTokenizer(),
          chat_parser=mocks.MockChatParser(),
          otel_meter_provider=self.meter_provider,
      )
    self.assertFalse(disabled_worker._otel._enabled)

    resp = disabled_worker._to_rollout_response(
        datatypes.TrajectoryItem(
            prompt_id="p",
            group_index=0,
            traj={"status": "SUCCEEDED", "assistant_turn_lengths": [2]},
        )
    )
    self.assertEqual(resp.status, "COMPLETED")

    async def _run_sync_round():
      req_sync = datatypes.WeightSyncRequest(
          controller_id="c",
          policy_version=2,
          extra_config={"req_id": "wsync-v2-r0", "uuid": 1},
      )
      await disabled_worker.pre_weight_sync(req_sync)
      await disabled_worker.weight_sync(req_sync)
      await disabled_worker.post_weight_sync(req_sync)

    asyncio.run(_run_sync_round())
    disabled_worker._otel.unregister_inflight()

    # Only the enabled worker from `setUp` is observed, and only through the
    # `requests_inflight` gauge it registered.
    metrics = otel_testing.collect_metrics(self.reader)
    self.assertNotIn("tunix.worker.trajectories.completed", metrics)
    self.assertNotIn("tunix.worker.weight_sync.duration", metrics)
    self.assertEqual(
        [attrs for _, _, attrs in metrics["tunix.worker.requests_inflight"]],
        [{"worker_id": "rollout_otel_0"}],
    )


if __name__ == "__main__":
  absltest.main()
