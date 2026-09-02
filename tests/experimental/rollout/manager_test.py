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
import types
import unittest
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.rl.agentic import registry
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
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
  ):
    del request, sampler, agent, tokenizer, chat_parser
    self.traj_id = traj_id
    self.env = env_client

  async def run_episode(self):
    return trajectory_lib.Trajectory(
        trajectory_id=self.traj_id,
        agent=trajectory_lib.Agent(name="test-agent", version="1.0"),
    )


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


class WeightSyncModeTest(absltest.TestCase):

  def test_config_weight_sync_mode_raiden(self):
    config = types.SimpleNamespace(
        sampler_type="vanilla",
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
    )
    manager = manager_lib.RolloutManager(
        config=config, tokenizer="mock", chat_parser="mock"
    )
    self.assertTrue(getattr(manager.sampler, "enable_raiden", False))
    self.assertIsNotNone(getattr(manager.sampler, "raiden_sync_delegate", None))

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
      "tunix.experimental.rollout.inprocess_vllm_sampler_adapter._get_vllm_sampler_cls"
  )
  def test_config_weight_sync_mode_inprocess_vllm_raiden(self, mock_get_vllm):
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
    self.assertIsNotNone(getattr(manager.sampler, "raiden_sync_delegate", None))


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


if __name__ == "__main__":
  absltest.main()

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
  ):
    del request, sampler, agent, tokenizer, chat_parser
    self.traj_id = traj_id
    self.env = env_client

  async def run_episode(self):
    return trajectory_lib.Trajectory(
        trajectory_id=self.traj_id,
        agent=trajectory_lib.Agent(name="test-agent", version="1.0"),
    )


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


class WeightSyncModeTest(absltest.TestCase):

  def test_config_weight_sync_mode_raiden(self):
    config = types.SimpleNamespace(
        sampler_type="vanilla",
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
    )
    manager = manager_lib.RolloutManager(
        config=config, tokenizer="mock", chat_parser="mock"
    )
    self.assertTrue(getattr(manager.sampler, "enable_raiden", False))
    self.assertIsNotNone(getattr(manager.sampler, "raiden_sync_delegate", None))

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
      "tunix.experimental.rollout.inprocess_vllm_sampler_adapter._get_vllm_sampler_cls"
  )
  def test_config_weight_sync_mode_inprocess_vllm_raiden(self, mock_get_vllm):
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
    self.assertIsNotNone(getattr(manager.sampler, "raiden_sync_delegate", None))


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
          "prompts": "prompt text",
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
    self.assertEqual(_CaptureEnv.last_init_kwargs["prompts"], "prompt text")
    self.assertEqual(_CaptureEnv.last_init_kwargs["prompt_id"], "prompt_1")
    self.assertEqual(_CaptureEnv.last_init_kwargs["group_index"], 2)
    self.assertEqual(_CaptureEnv.last_init_kwargs["policy_version"], 7)
    self.assertEqual(_CaptureEnv.last_init_kwargs["question"], "How many?")
    self.assertEqual(_CaptureEnv.last_init_kwargs["answer"], "42")
    self.assertEqual(_CaptureEnv.last_init_kwargs["gold_answer"], "42")
    self.assertEqual(_CaptureEnv.last_init_kwargs["max_steps"], 3)


if __name__ == "__main__":
  absltest.main()
