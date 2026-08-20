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
import unittest

from absl.testing import absltest
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib


class _FakeSampler(sampler_lib.Sampler):

  def __init__(self, metadata):
    self._metadata = metadata
    self.calls = []

  async def get_weight_sync_metadata(self, **kwargs):
    self.calls.append(kwargs)
    return self._metadata


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


if __name__ == "__main__":
  absltest.main()
