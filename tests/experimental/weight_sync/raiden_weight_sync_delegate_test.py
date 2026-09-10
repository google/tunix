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

"""Tests verifying destination-side Raiden weight sync delegation."""

import unittest
from unittest import mock

from absl.testing import absltest
from tunix.experimental.weight_sync import raiden_weight_sync_delegate


class _FakeWorker:

  def __init__(self, job_name, worker_index=0, **kwargs):
    self.job_name = job_name
    self.worker_index = worker_index
    self.kwargs = kwargs
    self.bound = False
    self.active = False
    self.bound_state = None
    self.arrays = []
    self.h2d_calls = 0
    self.bind_calls = 0

  def bind(self, state):
    self.bound = True
    self.active = True
    self.bind_calls += 1
    self.bound_state = state
    self.arrays = list(state.values()) if isinstance(state, dict) else [state]

  def work_unit_metadata(self):
    return {"unit": self.job_name}

  def h2d(self):
    self.h2d_calls += 1

  def metrics(self):
    return {}

  def checksums(self):
    return {}


class _Request:

  def __init__(self, policy_version, req_id=None):
    self.policy_version = policy_version
    self.extra_config = {"req_id": req_id} if req_id else {}


class RaidenWeightSyncDelegateTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.object(
        raiden_weight_sync_delegate.raiden_synchronizer,
        "RaidenSynchronizer",
        _FakeWorker,
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def _delegate(self):
    return raiden_weight_sync_delegate.RaidenWeightSyncDelegate()

  async def test_bind_binds_state(self):
    delegate = self._delegate()
    fake_state = {"w": 1}
    await delegate.bind_weight_sync(state=fake_state)
    worker = delegate._synchronizers[0]
    self.assertIs(worker.bound_state, fake_state)

  async def test_worker_uses_the_validated_config(self):
    delegate = self._delegate()
    self.assertIs(delegate._synchronizers[0].kwargs["auto_h2d"], True)
    self.assertEqual(delegate._synchronizers[0].kwargs, {"auto_h2d": True})

  async def test_repeat_phases_bind_exactly_once(self):
    delegate = self._delegate()
    fake_state = {"w": 1}
    await delegate.bind_weight_sync(state=fake_state, sampler=mock.MagicMock())
    await delegate.get_weight_sync_metadata()
    await delegate.pre_weight_sync()
    await delegate.weight_sync()
    self.assertEqual(delegate._synchronizers[0].bind_calls, 1)

  async def test_metadata_returns_one_entry_per_worker(self):
    delegate = self._delegate()
    md = await delegate.get_weight_sync_metadata()
    self.assertEqual(md, [{"unit": "rollout"}])

  async def test_weight_sync_installs_and_tracks_version(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1})
    version = await delegate.weight_sync(_Request(policy_version=5))
    self.assertEqual(version, 5)
    self.assertEqual(delegate._synchronizers[0].h2d_calls, 1)

  async def test_weight_sync_without_request_bumps_version(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1})
    self.assertEqual(await delegate.weight_sync(), 1)
    self.assertEqual(await delegate.weight_sync(), 2)

  async def test_weight_sync_with_zero_version_bumps(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1})
    self.assertEqual(await delegate.weight_sync(_Request(policy_version=0)), 1)

  async def test_weight_sync_before_bind_raises(self):
    delegate = self._delegate()
    with self.assertRaisesRegex(RuntimeError, "bind_weight_sync"):
      await delegate.weight_sync()

  async def test_post_weight_sync_returns_true(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1}, sampler=mock.MagicMock())
    self.assertTrue(await delegate.post_weight_sync())

  async def test_pre_weight_sync_without_sampler_raises(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1})
    with self.assertRaisesRegex(RuntimeError, "Sampler is not available"):
      await delegate.pre_weight_sync()

  async def test_pre_weight_sync_clears_prefix_and_kv_cache(self):
    sampler = mock.MagicMock()
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
    await delegate.bind_weight_sync(sampler=sampler, state={"w": 1})
    req = _Request(policy_version=1, req_id="req-1")
    await delegate.pre_weight_sync(sync_request=req)
    sampler.delete_cache.assert_called_once()

  async def test_post_weight_sync_reinitializes_kv_cache(self):
    sampler = mock.MagicMock()
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
    await delegate.bind_weight_sync(sampler=sampler, state={"w": 1})
    req = _Request(policy_version=1, req_id="req-1")
    # Only a pre (or an abort) may open a round on the tracker, so the commit
    # phase has to follow one that carries the same round key.
    await delegate.pre_weight_sync(sync_request=req)
    await delegate.post_weight_sync(sync_request=req)
    sampler.reinitialize_cache.assert_called_once()

  async def test_round_tracker_lifecycle_and_status(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1}, sampler=mock.MagicMock())
    req = _Request(policy_version=1, req_id="req-1")
    await delegate.pre_weight_sync(sync_request=req)
    await delegate.weight_sync(sync_request=req)
    await delegate.post_weight_sync(sync_request=req)
    report = delegate.get_weight_sync_status()
    self.assertIsNotNone(report)

  async def test_abort_weight_sync_marks_aborted(self):
    delegate = self._delegate()
    req = _Request(policy_version=1, req_id="req-1")
    res = await delegate.abort_weight_sync(sync_request=req)
    self.assertTrue(res)


if __name__ == "__main__":
  absltest.main()
