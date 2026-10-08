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

import asyncio
import time
import types
import unittest
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.rl.agentic import registry
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.weight_sync import weight_sync


class _FakeSampler(sampler_lib.Sampler):

  def __init__(self, metadata):
    self._metadata = metadata
    self.calls = []

  async def get_weight_sync_metadata(self, **kwargs):
    self.calls.append(kwargs)
    return self._metadata

  async def bind_weight_sync(self, **kwargs):
    self.calls.append("bind")
    return None

  async def pre_weight_sync(self, sync_request=None, **kwargs):
    return "ok"


class GetWeightSyncMetadataTest(unittest.IsolatedAsyncioTestCase):

  async def test_delegates_to_sampler(self):
    sampler = _FakeSampler([{"unit": "sampler0"}])
    manager = manager_lib.RolloutManager(
        sampler=sampler, tokenizer="mock", chat_parser="mock"
    )
    result = await manager.get_weight_sync_metadata()
    self.assertEqual(result, [{"unit": "sampler0"}])

  async def test_forwards_kwargs(self):
    sampler = _FakeSampler([])
    manager = manager_lib.RolloutManager(
        sampler=sampler, tokenizer="mock", chat_parser="mock"
    )
    await manager.get_weight_sync_metadata(timeout_s=5)
    self.assertEqual(sampler.calls, [{"timeout_s": 5}])

  async def test_default_sampler_raises_not_implemented(self):
    manager = manager_lib.RolloutManager(tokenizer="mock", chat_parser="mock")
    with self.assertRaises(NotImplementedError):
      await manager.get_weight_sync_metadata()


class _FakeSyncSampler(_FakeSampler):

  async def pre_weight_sync(self, sync_request=None, **kwargs):
    return "ok"

  async def post_weight_sync(self, sync_request=None, **kwargs):
    return "ok"


class _CaptureEnv:

  last_init_kwargs = None

  def __init__(self, **kwargs):
    self.init_kwargs = dict(kwargs)
    _CaptureEnv.last_init_kwargs = self.init_kwargs


if not registry.ENV_REGISTRY.contains("manager_capture_env"):
  registry.ENV_REGISTRY.register("manager_capture_env")(_CaptureEnv)


class _NoopCollector:

  def __init__(
      self,
      traj_id,
      request,
      sampler,
      env_client,
      agent,
      tokenizer,
      chat_parser,
      eos_ids=None,
      partial_rollout=False,
      policy_version_fn=None,
  ):
    del (
        request,
        sampler,
        agent,
        tokenizer,
        chat_parser,
        eos_ids,
        partial_rollout,
        policy_version_fn,
    )
    self.traj_id = traj_id
    self.env = env_client

  async def run_episode(self):
    return datatypes.TrajectoryItem(
        prompt_id=self.traj_id,
        group_index=0,
        traj={},
    )


class RegisteredEnvMetadataTest(unittest.IsolatedAsyncioTestCase):

  async def test_generate_forwards_request_metadata_to_registered_env(self):
    _CaptureEnv.last_init_kwargs = None
    manager = manager_lib.RolloutManager(
        config=types.SimpleNamespace(env_name="manager_capture_env"),
        sampler=_FakeSyncSampler([]),
        agent_factory=lambda: object(),
        tokenizer="mock",
        chat_parser="mock",
    )
    request = datatypes.RolloutRequest(
        request_id="req_1",
        prompt="prompt text",
        prompt_id="prompt_1",
        group_index=2,
        target_policy_version=7,
        metadata={
            "question": "ignored top level",
            "answer": "ignored top level",
            "gold_answer": "ignored top level",
            "env_config": {
                "prompt": "prompt text",
                "prompt_id": "prompt_1",
                "question": "How many?",
                "answer": "42",
                "gold_answer": "42",
                "max_steps": 3,
            },
        },
    )

    with mock.patch.object(
        manager_lib.collector_lib,
        "TrajectoryCollectorEngine",
        _NoopCollector,
    ):
      await manager.generate(request)

    self.assertIsNotNone(_CaptureEnv.last_init_kwargs)
    self.assertEqual(_CaptureEnv.last_init_kwargs["prompt"], "prompt text")
    self.assertEqual(_CaptureEnv.last_init_kwargs["prompt_id"], "prompt_1")
    self.assertEqual(_CaptureEnv.last_init_kwargs["group_index"], 2)
    self.assertEqual(_CaptureEnv.last_init_kwargs["policy_version"], 7)
    self.assertEqual(_CaptureEnv.last_init_kwargs["question"], "How many?")
    self.assertEqual(_CaptureEnv.last_init_kwargs["answer"], "42")
    self.assertEqual(_CaptureEnv.last_init_kwargs["gold_answer"], "42")
    self.assertEqual(_CaptureEnv.last_init_kwargs["max_steps"], 3)


class AdmissionGateTest(unittest.IsolatedAsyncioTestCase):

  def _manager(self, **kwargs):
    return manager_lib.RolloutManager(
        sampler=_FakeSyncSampler([]),
        tokenizer="mock",
        chat_parser="mock",
        **kwargs,
    )

  async def test_pre_closes_admission(self):
    manager = self._manager()
    await manager.pre_weight_sync()
    self.assertFalse(manager._traffic.is_admission_open())

  async def test_post_reopens_admission(self):
    manager = self._manager()
    await manager.pre_weight_sync()
    await manager.post_weight_sync()
    self.assertTrue(manager._traffic.is_admission_open())

  async def test_reopen_admission_after_abort(self):
    manager = self._manager()
    await manager.pre_weight_sync()
    self.assertTrue(manager.reopen_admission())
    self.assertTrue(manager._traffic.is_admission_open())

  async def test_bind_delegates_to_sampler(self):
    sampler = _FakeSyncSampler([])
    manager = manager_lib.RolloutManager(
        sampler=sampler, tokenizer="mock", chat_parser="mock")
    await manager.bind_weight_sync()

  async def test_abort_weight_sync_delegates_and_reopens_admission(self):
    sampler = mock.AsyncMock(spec=sampler_lib.Sampler)
    sampler.abort_weight_sync.return_value = "aborted"
    manager = manager_lib.RolloutManager(
        sampler=sampler, tokenizer="mock", chat_parser="mock"
    )
    manager._traffic.transition_to_syncing()
    res = await manager.abort_weight_sync()
    self.assertEqual(res, "aborted")
    sampler.abort_weight_sync.assert_awaited_once()
    self.assertTrue(manager._traffic.is_admission_open())

  async def test_repeated_pre_is_allowed(self):
    manager = self._manager()
    await manager.pre_weight_sync()
    await manager.pre_weight_sync()
    self.assertFalse(manager._traffic.is_admission_open())

  async def test_pre_waits_for_inflight_work(self):
    manager = self._manager()
    done = asyncio.Event()

    async def work():
      await done.wait()

    task = asyncio.create_task(work())
    manager._active_tasks["t0"] = task
    manager._traffic.track(task)
    pre = asyncio.create_task(manager.pre_weight_sync())
    await asyncio.sleep(0.01)
    self.assertFalse(pre.done())
    done.set()
    await task
    manager._active_tasks.pop("t0", None)
    await pre

  async def test_drain_timeout_returns(self):
    manager = self._manager(drain_timeout_s=0.05)
    task = asyncio.create_task(asyncio.Event().wait())
    manager._active_tasks["t0"] = task
    manager._traffic.track(task)
    await manager.pre_weight_sync()
    task.cancel()
    manager._active_tasks.pop("t0", None)


class AgentConfigTest(unittest.IsolatedAsyncioTestCase):

  async def test_request_agent_config_overrides_worker_config(self):
    created_configs = []

    class FakeAgent:

      def __init__(self, **kwargs):
        created_configs.append(kwargs)

    class FakeEnv:

      def __init__(self, **kwargs):
        del kwargs

    config = types.SimpleNamespace(
        agent_name="fake_agent_for_manager_test",
        agent_config={"source": "worker"},
        env_name="fake_env_for_manager_test",
        env_config={},
    )
    manager = manager_lib.RolloutManager(
        config=config,
        sampler=_FakeSyncSampler([]),
        tokenizer="mock",
        chat_parser="mock",
    )
    request = datatypes.RolloutRequest(
        prompt="prompt",
        prompt_id="p0",
        metadata={
            "agent_config": {"source": "request", "scaffold": "r2egym"},
        },
    )

    with mock.patch.object(
        registry.AGENT_REGISTRY, "contains", return_value=True
    ), mock.patch.object(
        registry.AGENT_REGISTRY, "get", return_value=FakeAgent
    ), mock.patch.object(
        registry.ENV_REGISTRY, "contains", return_value=True
    ), mock.patch.object(
        registry.ENV_REGISTRY, "get", return_value=FakeEnv
    ), mock.patch.object(
        manager_lib.collector_lib, "TrajectoryCollectorEngine"
    ) as collector_cls:
      collector = collector_cls.return_value
      collector.traj_id = request.traj_id
      collector.env = None
      collector.run_episode = mock.AsyncMock(return_value="trajectory")

      result = await manager._generate_one(request)

    self.assertEqual(result, "trajectory")
    self.assertEqual(
        created_configs, [{"source": "request", "scaffold": "r2egym"}]
    )

  async def test_worker_agent_config_is_fallback(self):
    created_configs = []

    class FakeAgent:

      def __init__(self, **kwargs):
        created_configs.append(kwargs)

    class FakeEnv:

      def __init__(self, **kwargs):
        del kwargs

    config = types.SimpleNamespace(
        agent_name="fake_agent_for_manager_test",
        agent_config={"source": "worker"},
        env_name="fake_env_for_manager_test",
        env_config={},
    )
    manager = manager_lib.RolloutManager(
        config=config,
        sampler=_FakeSyncSampler([]),
        tokenizer="mock",
        chat_parser="mock",
    )
    request = datatypes.RolloutRequest(prompt="prompt", prompt_id="p0")

    with mock.patch.object(
        registry.AGENT_REGISTRY, "contains", return_value=True
    ), mock.patch.object(
        registry.AGENT_REGISTRY, "get", return_value=FakeAgent
    ), mock.patch.object(
        registry.ENV_REGISTRY, "contains", return_value=True
    ), mock.patch.object(
        registry.ENV_REGISTRY, "get", return_value=FakeEnv
    ), mock.patch.object(
        manager_lib.collector_lib, "TrajectoryCollectorEngine"
    ) as collector_cls:
      collector = collector_cls.return_value
      collector.traj_id = request.traj_id
      collector.env = None
      collector.run_episode = mock.AsyncMock(return_value="trajectory")

      result = await manager._generate_one(request)

    self.assertEqual(result, "trajectory")
    self.assertEqual(created_configs, [{"source": "worker"}])

  async def test_configured_eos_tokens_reach_the_collector(self):
    # The collector scores truncation against this set. If it stops arriving,
    # the collector silently falls back to the tokenizer's own EOS and
    # clip_ratio starts counting normal terminations as truncations.
    config = types.SimpleNamespace(eos_tokens=[151643, 151645])
    manager = manager_lib.RolloutManager(
        config=config,
        sampler=_FakeSyncSampler([]),
        tokenizer="mock",
        chat_parser="mock",
    )
    request = datatypes.RolloutRequest(prompt="prompt", prompt_id="p0")

    with mock.patch.object(
        manager_lib.collector_lib, "TrajectoryCollectorEngine"
    ) as collector_cls:
      collector = collector_cls.return_value
      collector.traj_id = request.traj_id
      collector.env = None
      collector.run_episode = mock.AsyncMock(return_value="trajectory")

      await manager._generate_one(request)

    self.assertEqual(
        collector_cls.call_args.kwargs["eos_ids"], [151643, 151645]
    )

  async def test_unset_eos_tokens_leave_the_collector_on_its_default(self):
    manager = manager_lib.RolloutManager(
        config=types.SimpleNamespace(),
        sampler=_FakeSyncSampler([]),
        tokenizer="mock",
        chat_parser="mock",
    )
    request = datatypes.RolloutRequest(prompt="prompt", prompt_id="p0")

    with mock.patch.object(
        manager_lib.collector_lib, "TrajectoryCollectorEngine"
    ) as collector_cls:
      collector = collector_cls.return_value
      collector.traj_id = request.traj_id
      collector.env = None
      collector.run_episode = mock.AsyncMock(return_value="trajectory")

      await manager._generate_one(request)

    self.assertIsNone(collector_cls.call_args.kwargs["eos_ids"])

  async def test_generate_stamps_request_metadata_on_trajectory_error(self):
    manager = manager_lib.RolloutManager(
        config=types.SimpleNamespace(),
        sampler=_FakeSyncSampler([]),
        tokenizer="mock",
        chat_parser="mock",
    )
    request = datatypes.RolloutRequest(
        prompt="prompt",
        prompt_id="prompt_7",
        group_index=3,
        target_policy_version=5,
        metadata={
            "batch_idx": 1,
            "prompt_idx": 7,
            "intra_batch_idx": 3,
        },
    )

    with mock.patch.object(
        manager_lib.collector_lib, "TrajectoryCollectorEngine"
    ) as collector_cls:
      collector = collector_cls.return_value
      collector.traj_id = request.traj_id
      collector.env = None
      collector.run_episode = mock.AsyncMock(
          side_effect=RuntimeError("sandbox crashed")
      )

      result = await manager._generate_one(request)

    self.assertEqual(result.prompt_id, "prompt_7")
    self.assertEqual(result.error_message, "sandbox crashed")
    self.assertEqual(result.error_type, "RuntimeError")
    self.assertEqual(
        result.metadata,
        {
            "batch_idx": 1,
            "prompt_idx": 7,
            "intra_batch_idx": 3,
            "prompt_id": "prompt_7",
            "group_index": 3,
            "policy_version": 5,
        },
    )


class WeightSyncModeTest(absltest.TestCase):

  @mock.patch(
      "tunix.experimental.weight_sync.raiden_weight_sync_delegate.RaidenWeightSyncDelegate"
  )
  def test_config_weight_sync_mode_raiden(self, mock_delegate_cls):
    config = types.SimpleNamespace(
        sampler_type="vanilla",
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
    )
    manager = manager_lib.RolloutManager(
        config=config, tokenizer="mock", chat_parser="mock"
    )
    self.assertTrue(getattr(manager.sampler, "enable_raiden", False))
    delegate = getattr(manager.sampler, "raiden_sync_delegate", None)
    self.assertIsNotNone(delegate)
    mock_delegate_cls.assert_called_once_with(server_id="vanilla_sampler")

  def test_config_weight_sync_mode_fallback(self):
    config = types.SimpleNamespace(
        sampler_type="vanilla",
        weight_sync_mode=weight_sync.WeightSyncMode.FALLBACK,
    )
    manager = manager_lib.RolloutManager(
        config=config, tokenizer="mock", chat_parser="mock"
    )
    self.assertFalse(getattr(manager.sampler, "enable_raiden", False))
    self.assertIsNone(getattr(manager.sampler, "raiden_sync_delegate", None))

  @mock.patch(
      "tunix.experimental.weight_sync.raiden_weight_sync_delegate.RaidenWeightSyncDelegate"
  )
  @mock.patch(
      "tunix.experimental.rollout.inprocess_vllm_sampler_adapter._get_vllm_sampler_cls"
  )
  def test_config_weight_sync_mode_inprocess_vllm_raiden(
      self, mock_get_vllm, mock_delegate_cls
  ):
    mock_lib = mock.MagicMock()
    mock_lib.VllmSampler.return_value = mock.MagicMock()
    mock_get_vllm.return_value = mock_lib
    config = types.SimpleNamespace(
        sampler_type="inprocess_vllm",
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
    )
    manager = manager_lib.RolloutManager(
        config=config, tokenizer="mock", chat_parser="mock"
    )
    self.assertTrue(getattr(manager.sampler, "enable_raiden", False))
    delegate = getattr(manager.sampler, "raiden_sync_delegate", None)
    self.assertIsNotNone(delegate)
    mock_delegate_cls.assert_called_once_with(
        server_id="inprocess_vllm_sampler"
    )


class PartialRolloutTest(unittest.IsolatedAsyncioTestCase):

  async def test_pre_weight_sync_skips_drain_when_partial_rollout_enabled(self):
    sampler = mock.AsyncMock(spec=sampler_lib.Sampler)
    sampler.pre_weight_sync.return_value = "ok"
    sampler.weight_sync.return_value = 3
    sampler.post_weight_sync.return_value = 3
    config = types.SimpleNamespace(partial_rollout=True)
    manager = manager_lib.RolloutManager(
        config=config,
        sampler=sampler,
        tokenizer="mock",
        chat_parser="mock",
        drain_timeout_s=10.0,
    )

    never_done = asyncio.Event()
    task = asyncio.create_task(never_done.wait())
    manager._active_tasks["t0"] = task
    manager._traffic.track(task)
    fake_collector = mock.MagicMock()
    manager._active_collectors["t0"] = fake_collector

    try:
      await asyncio.wait_for(manager.pre_weight_sync(), timeout=0.5)
      fake_collector.pause.assert_called_once()
      sampler.pre_weight_sync.assert_awaited_once_with(
          None, partial_rollout=True
      )

      sync_req = sampler_lib.WeightSyncRequest(policy_version=3)
      await manager.weight_sync(sync_req)
      self.assertEqual(manager._policy_version, 3)

      await manager.post_weight_sync(sync_req)
      fake_collector.resume.assert_called_once()
      self.assertEqual(manager._policy_version, 3)
    finally:
      task.cancel()
      manager._active_tasks.pop("t0", None)
      manager._active_collectors.pop("t0", None)


class _PauseTrackingSampler(_FakeSyncSampler):

  def __init__(self):
    super().__init__([])
    self.policy_version = 0
    self.pause_calls = []
    self.resume_calls = 0
    self.pre_kwargs = []
    self.sync_kwargs = []
    self.post_kwargs = []
    self._unpaused = asyncio.Event()
    self._unpaused.set()

  async def pause(
      self, mode: str = "keep", *, clear_cache: bool = False, **kwargs
  ):
    del kwargs
    self.pause_calls.append((mode, clear_cache))
    self._unpaused.clear()

  async def resume(self, **kwargs):
    del kwargs
    self.resume_calls += 1
    self._unpaused.set()

  async def pre_weight_sync(self, sync_request=None, **kwargs):
    del sync_request
    use_partial = bool(kwargs.get("partial_rollout", False))
    await self.pause(
        mode="keep" if use_partial else "wait",
        clear_cache=not use_partial,
    )
    self.pre_kwargs.append(dict(kwargs))
    return "ok"

  async def weight_sync(self, sync_request=None, **kwargs):
    self.sync_kwargs.append(dict(kwargs))
    new_v = getattr(sync_request, "version", None)
    if new_v is None:
      new_v = getattr(sync_request, "policy_version", self.policy_version + 1)
    self.policy_version = int(new_v)
    return self.policy_version

  async def post_weight_sync(self, sync_request=None, **kwargs):
    del sync_request
    self.post_kwargs.append(dict(kwargs))
    await self.resume()
    return "ok"


class PartialRolloutWeightSyncCasesTest(unittest.IsolatedAsyncioTestCase):

  async def test_queued_admission_during_syncing_does_not_raise(self):
    sampler = _PauseTrackingSampler()
    manager = manager_lib.RolloutManager(
        config=types.SimpleNamespace(partial_rollout=True),
        sampler=sampler,
        tokenizer="mock",
        chat_parser="mock",
    )

    await manager.pre_weight_sync()
    self.assertFalse(manager._traffic.is_admission_open())

    entered_episode = asyncio.Event()
    observed_target_versions = []

    class _RecordingCollector(_NoopCollector):

      def __init__(self, traj_id, request, *args, **kwargs):
        super().__init__(traj_id, request, *args, **kwargs)
        self.request = request

      async def run_episode(self):
        observed_target_versions.append(self.request.target_policy_version)
        entered_episode.set()
        return datatypes.TrajectoryItem(
            prompt_id=self.request.prompt_id,
            group_index=self.request.group_index,
            traj={},
            policy_version=self.request.target_policy_version,
        )

    req = datatypes.RolloutRequest(
        prompt="queued prompt",
        prompt_id="p_queued",
        group_index=0,
        target_policy_version=0,
    )
    with mock.patch.object(
        manager_lib.collector_lib,
        "TrajectoryCollectorEngine",
        _RecordingCollector,
    ):
      gen_task = asyncio.create_task(manager.generate(req))
      await asyncio.sleep(0.02)
      # Must stay queued on wait_for_admission() without raising
      # AdmissionClosedError.
      self.assertFalse(gen_task.done())
      self.assertFalse(entered_episode.is_set())

      await manager.weight_sync(types.SimpleNamespace(version=2))
      await manager.post_weight_sync()
      result = await asyncio.wait_for(gen_task, timeout=2.0)

    self.assertTrue(entered_episode.is_set())
    self.assertEqual(observed_target_versions, [2])
    self.assertEqual(result.policy_version, 2)

  async def test_case1_single_turn_partial_rollout_off_drains_and_pauses_wait(
      self,
  ):
    sampler = _PauseTrackingSampler()
    manager = manager_lib.RolloutManager(
        config=types.SimpleNamespace(partial_rollout=False),
        sampler=sampler,
        tokenizer="mock",
        chat_parser="mock",
    )
    finish_turn = asyncio.Event()

    class _SlowTurnCollector(_NoopCollector):

      def __init__(self, traj_id, request, *args, **kwargs):
        super().__init__(traj_id, request, *args, **kwargs)
        self.paused = False

      def pause(self):
        self.paused = True

      def resume(self):
        self.paused = False

      async def run_episode(self):
        await finish_turn.wait()
        return datatypes.TrajectoryItem(
            prompt_id="p0", group_index=0, traj={}, policy_version=0
        )

    with mock.patch.object(
        manager_lib.collector_lib,
        "TrajectoryCollectorEngine",
        _SlowTurnCollector,
    ):
      gen_task = asyncio.create_task(
          manager.generate(datatypes.RolloutRequest(prompt="p", prompt_id="p0"))
      )
      await asyncio.sleep(0.01)

      pre_task = asyncio.create_task(manager.pre_weight_sync())
      await asyncio.sleep(0.02)
      # Case 1 must wait for the in-flight turn to drain before pausing sampler.
      self.assertFalse(pre_task.done())
      self.assertEqual(sampler.pause_calls, [])

      finish_turn.set()
      await asyncio.wait_for(pre_task, timeout=2.0)
      traj = await asyncio.wait_for(gen_task, timeout=2.0)

    self.assertEqual(traj.policy_version, 0)
    self.assertEqual(sampler.pause_calls, [("wait", True)])
    await manager.post_weight_sync()
    self.assertEqual(sampler.resume_calls, 1)

  async def test_case2_single_turn_partial_rollout_on_skips_drain_and_preserves_kv(
      self,
  ):
    sampler = _PauseTrackingSampler()
    manager = manager_lib.RolloutManager(
        config=types.SimpleNamespace(partial_rollout=True),
        sampler=sampler,
        tokenizer="mock",
        chat_parser="mock",
    )
    turn_started = asyncio.Event()

    class _StitchedSingleTurnCollector(_NoopCollector):

      def __init__(self, traj_id, request, *args, **kwargs):
        super().__init__(traj_id, request, *args, **kwargs)
        self.paused = False

      def pause(self):
        self.paused = True

      def resume(self):
        self.paused = False

      async def run_episode(self):
        start_v = sampler.policy_version
        turn_started.set()
        # Simulate mid-decode pause/resume across weight sync: waits until
        # weight_sync + post_weight_sync have updated weights and resumed.
        while (
            sampler.policy_version == start_v or not sampler._unpaused.is_set()
        ):
          await asyncio.sleep(0.005)
        return datatypes.TrajectoryItem(
            prompt_id="p0",
            group_index=0,
            traj={},
            policy_version=start_v,
        )

    with mock.patch.object(
        manager_lib.collector_lib,
        "TrajectoryCollectorEngine",
        _StitchedSingleTurnCollector,
    ):
      gen_task = asyncio.create_task(
          manager.generate(datatypes.RolloutRequest(prompt="p", prompt_id="p0"))
      )
      await asyncio.wait_for(turn_started.wait(), timeout=2.0)

      # Case 2: pre_weight_sync returns immediately without draining in-flight
      # turn.
      await asyncio.wait_for(manager.pre_weight_sync(), timeout=0.5)
      self.assertEqual(sampler.pause_calls, [("keep", False)])
      self.assertEqual(sampler.pre_kwargs, [{"partial_rollout": True}])
      self.assertFalse(gen_task.done())

      await manager.weight_sync(types.SimpleNamespace(version=1))
      await manager.post_weight_sync()
      self.assertEqual(sampler.resume_calls, 1)

      traj = await asyncio.wait_for(gen_task, timeout=2.0)
      # Start-of-turn stamping: started under v0, resumed under v1 -> v0.
      self.assertEqual(traj.policy_version, 0)

  async def test_case3_multi_turn_partial_rollout_off_drains_full_episode(self):
    sampler = _PauseTrackingSampler()
    manager = manager_lib.RolloutManager(
        config=types.SimpleNamespace(partial_rollout=False),
        sampler=sampler,
        tokenizer="mock",
        chat_parser="mock",
    )
    in_env_step = asyncio.Event()
    finish_turn2 = asyncio.Event()

    class _MultiTurnCollector(_NoopCollector):

      def pause(self):
        pass

      def resume(self):
        pass

      async def run_episode(self):
        v1 = sampler.policy_version
        in_env_step.set()
        await finish_turn2.wait()
        v2 = sampler.policy_version
        return datatypes.TrajectoryItem(
            prompt_id="p0",
            group_index=0,
            traj={},
            policy_version=min(v1, v2),
        )

    with mock.patch.object(
        manager_lib.collector_lib,
        "TrajectoryCollectorEngine",
        _MultiTurnCollector,
    ):
      gen_task = asyncio.create_task(
          manager.generate(datatypes.RolloutRequest(prompt="p", prompt_id="p0"))
      )
      await asyncio.wait_for(in_env_step.wait(), timeout=2.0)

      pre_task = asyncio.create_task(manager.pre_weight_sync())
      await asyncio.sleep(0.02)
      self.assertFalse(pre_task.done())

      finish_turn2.set()
      await asyncio.wait_for(pre_task, timeout=2.0)
      await manager.weight_sync(types.SimpleNamespace(version=1))
      await manager.post_weight_sync()
      traj = await asyncio.wait_for(gen_task, timeout=2.0)

    self.assertEqual(traj.policy_version, 0)
    self.assertEqual(sampler.pause_calls, [("wait", True)])

  async def test_case4_multi_turn_partial_rollout_on_parks_mid_env_at_turn_boundary(
      self,
  ):
    sampler = _PauseTrackingSampler()
    in_env_step = asyncio.Event()
    release_env_step = asyncio.Event()
    turn2_called_sampler = asyncio.Event()
    turn_versions_seen = []

    async def _sample_fn(req, **kwargs):
      del req, kwargs
      v = sampler.policy_version
      turn_versions_seen.append(v)
      if len(turn_versions_seen) == 2:
        turn2_called_sampler.set()
      return "ans"

    sampler.sample = _sample_fn

    class _FakeInnerCollectEngine:

      def __init__(self, *, model_call, **kwargs):
        del kwargs
        self._model_call = model_call

      async def collect(self, mode="Token"):
        del mode
        # Turn 1 under v0
        await self._model_call([{"role": "user", "content": "q1"}])
        # Environment step in progress when weight sync begins
        in_env_step.set()
        await release_env_step.wait()
        # Turn 2 must block on collector._unpaused until post_weight_sync
        await self._model_call([{"role": "user", "content": "q2"}])
        return {"conversation_tokens": [1, 2, 3, 4], "status": "DONE"}

    manager = manager_lib.RolloutManager(
        config=types.SimpleNamespace(partial_rollout=True),
        sampler=sampler,
        env_pool=types.SimpleNamespace(acquire_env=lambda _: object()),
        agent_factory=object,
        tokenizer=types.SimpleNamespace(eos_token_id=2),
        chat_parser=types.SimpleNamespace(
            parse=lambda msgs, **kw: "prompt_str"
        ),
    )

    with mock.patch.object(
        manager_lib.collector_lib.rl_collect_engine,
        "TrajectoryCollectEngine",
        _FakeInnerCollectEngine,
    ), mock.patch.object(
        manager_lib.collector_lib,
        "_build_prompt",
        return_value=[1, 2],
    ):
      req = datatypes.RolloutRequest(
          prompt="multi-turn q",
          prompt_id="p_case4",
          group_index=0,
          max_response_length=32,
          generation_kwargs={"max_generation_steps": 32},
      )
      gen_task = asyncio.create_task(manager.generate(req))
      await asyncio.wait_for(in_env_step.wait(), timeout=2.0)

      # Trigger pre_weight_sync while episode is inside env.step().
      await asyncio.wait_for(manager.pre_weight_sync(), timeout=0.5)
      self.assertEqual(sampler.pause_calls, [("keep", False)])

      # Let env.step() finish while still in SYNCING; Turn 2 must park at
      # collector._unpaused before calling sampler.sample().
      release_env_step.set()
      await asyncio.sleep(0.02)
      self.assertFalse(turn2_called_sampler.is_set())

      # Complete weight sync to v1 and resume.
      await manager.weight_sync(types.SimpleNamespace(version=1))
      await manager.post_weight_sync()

      traj = await asyncio.wait_for(gen_task, timeout=2.0)

    self.assertTrue(turn2_called_sampler.is_set())
    self.assertEqual(turn_versions_seen, [0, 1])
    self.assertEqual(traj.policy_version, 0)
    self.assertEqual(traj.metadata["policy_versions"], [0, 1])

  async def test_fr1_synthetic_straggler_ab_benchmark(self):
    async def _run_scenario(partial_rollout: bool) -> tuple[float, list[int]]:
      sampler = _PauseTrackingSampler()
      manager = manager_lib.RolloutManager(
          config=types.SimpleNamespace(partial_rollout=partial_rollout),
          sampler=sampler,
          tokenizer="mock",
          chat_parser="mock",
      )

      class _StragglerCollector(_NoopCollector):

        def __init__(self, traj_id, request, *args, **kwargs):
          super().__init__(traj_id, request, *args, **kwargs)
          self.request = request

        def pause(self):
          pass

        def resume(self):
          pass

        async def run_episode(self):
          start_v = sampler.policy_version
          delay_s = float(self.request.metadata.get("delay_s", 0.01))
          await asyncio.sleep(delay_s)
          while not sampler._unpaused.is_set():
            await asyncio.sleep(0.002)
          return datatypes.TrajectoryItem(
              prompt_id=self.request.prompt_id,
              group_index=self.request.group_index,
              traj={},
              policy_version=start_v,
          )

      requests = []
      # Group 0: 4 fast rollouts (10ms)
      for g_idx in range(4):
        requests.append(
            datatypes.RolloutRequest(
                prompt="g0",
                prompt_id="group_0",
                group_index=g_idx,
                metadata={"delay_s": 0.01},
            )
        )
      # Group 1: 3 fast rollouts (10ms) + 1 straggler rollout (200ms)
      for g_idx in range(4):
        requests.append(
            datatypes.RolloutRequest(
                prompt="g1",
                prompt_id="group_1",
                group_index=g_idx,
                metadata={"delay_s": 0.20 if g_idx == 3 else 0.01},
            )
        )

      with mock.patch.object(
          manager_lib.collector_lib,
          "TrajectoryCollectorEngine",
          _StragglerCollector,
      ):
        tasks = [asyncio.create_task(manager.generate(r)) for r in requests]
        # Wait for Group 0 (first 4 tasks) to finish (~10ms).
        await asyncio.gather(*tasks[:4])

        t0 = time.perf_counter()
        await manager.pre_weight_sync()
        await manager.weight_sync(types.SimpleNamespace(version=1))
        await manager.post_weight_sync()
        sync_wall_s = time.perf_counter() - t0

        results = await asyncio.gather(*tasks)
        versions = [r.policy_version for r in results]
        return sync_wall_s, versions

    wait_sync_s, wait_versions = await _run_scenario(partial_rollout=False)
    keep_sync_s, keep_versions = await _run_scenario(partial_rollout=True)

    # With partial_rollout=False (mode="wait"), pre_weight_sync blocks for the
    # remaining ~190ms of Group 1's straggler.
    self.assertGreaterEqual(wait_sync_s, 0.12)
    # With partial_rollout=True (mode="keep"), weight sync completes without
    # waiting for the straggler (>80% latency reduction).
    self.assertLess(keep_sync_s, wait_sync_s * 0.2)
    self.assertEqual(wait_versions, [0] * 8)
    self.assertEqual(keep_versions, [0] * 8)


if __name__ == "__main__":
  absltest.main()

