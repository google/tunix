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

import threading
import unittest
from unittest import mock

from absl.testing import absltest
from tunix.common import configs
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import raiden_weight_sync_delegate


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

  def h2d(self, sync_request=None):
    self.h2d_calls += 1
    self.last_h2d_request = sync_request

  def apply_to_sampler(self, sampler):
    self.applied_to = sampler

  def abort(self, sync_request=None):
    self.aborted = sync_request

  def metrics(self):
    return {}

  def checksums(self):
    return {}


def _request(policy_version=0, req_id=None):
  return datatypes.WeightSyncRequest(
      policy_version=policy_version,
      extra_config={"req_id": req_id} if req_id else {},
  )


def _sampler(**config_kwargs):
  sampler = mock.MagicMock(
      spec=(
          "config",
          "delete_cache",
          "reset_prefix_cache",
          "reinitialize_cache",
          "refresh_state_leaves",
      )
  )
  sampler.config = configs.RolloutConfig(**config_kwargs)
  return sampler


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
    await delegate.bind_weight_sync(state=fake_state, sampler=_sampler())
    await delegate.get_weight_sync_metadata()
    await delegate.pre_weight_sync(_request())
    await delegate.weight_sync(_request())
    self.assertEqual(delegate._synchronizers[0].bind_calls, 1)

  async def test_metadata_returns_one_entry_per_worker(self):
    delegate = self._delegate()
    md = await delegate.get_weight_sync_metadata()
    self.assertEqual(md, [{"unit": "rollout"}])

  async def test_weight_sync_installs_and_tracks_version(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1})
    version = await delegate.weight_sync(_request(policy_version=5))
    self.assertEqual(version, 5)
    self.assertEqual(delegate._synchronizers[0].h2d_calls, 1)

  async def test_weight_sync_with_zero_version_preserves_zero(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1})
    self.assertEqual(await delegate.weight_sync(_request(policy_version=0)), 0)

  async def test_weight_sync_before_bind_raises(self):
    delegate = self._delegate()
    with self.assertRaisesRegex(RuntimeError, "bind_weight_sync"):
      await delegate.weight_sync(_request())

  def test_default_server_id_is_rollout(self):
    delegate = self._delegate()
    worker = delegate._synchronizers[0]
    self.assertEqual(worker.job_name, "rollout")
    self.assertEqual(worker.worker_index, 0)

  def test_custom_server_id_and_worker_index_propagated(self):
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate(
        server_id="replica_worker_1", worker_index=3
    )
    worker = delegate._synchronizers[0]
    self.assertEqual(worker.job_name, "replica_worker_1")
    self.assertEqual(worker.worker_index, 3)

  def test_server_id_in_kwargs_propagated(self):
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate(
        **{"server_id": "custom_server_id"}
    )
    worker = delegate._synchronizers[0]
    self.assertEqual(worker.job_name, "custom_server_id")

  async def test_is_bounded_lifecycle(self):
    delegate = self._delegate()
    self.assertFalse(delegate.is_bounded())
    await delegate.bind_weight_sync(state={"w": 1})
    self.assertTrue(delegate.is_bounded())

  async def test_post_weight_sync_returns_true(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1}, sampler=_sampler())
    self.assertTrue(await delegate.post_weight_sync(_request()))

  async def test_pre_weight_sync_without_sampler_raises(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1})
    with self.assertRaisesRegex(RuntimeError, "Sampler is not available"):
      await delegate.pre_weight_sync(_request())

  async def test_pre_weight_sync_clears_prefix_and_kv_cache(self):
    sampler = _sampler()
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
    await delegate.bind_weight_sync(sampler=sampler, state={"w": 1})
    req = _request(policy_version=1, req_id="req-1")
    await delegate.pre_weight_sync(sync_request=req)
    sampler.delete_cache.assert_called_once()

  async def test_post_weight_sync_reinitializes_kv_cache(self):
    sampler = _sampler()
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
    await delegate.bind_weight_sync(sampler=sampler, state={"w": 1})
    req = _request(policy_version=1, req_id="req-1")
    # Only a pre (or an abort) may open a round on the tracker, so the commit
    # phase has to follow one that carries the same round key.
    await delegate.pre_weight_sync(sync_request=req)
    await delegate.post_weight_sync(sync_request=req)
    sampler.reinitialize_cache.assert_called_once()

  async def test_weight_sync_keeps_kv_cache_when_sampler_opts_out(self):
    sampler = _sampler(rollout_vllm_free_kv_cache_during_weight_sync=False)
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
    await delegate.bind_weight_sync(sampler=sampler, state={"w": 1})
    req = _request(policy_version=1, req_id="req-1")
    await delegate.pre_weight_sync(sync_request=req)
    await delegate.post_weight_sync(sync_request=req)
    sampler.delete_cache.assert_not_called()
    sampler.reinitialize_cache.assert_not_called()
    sampler.reset_prefix_cache.assert_called_once()
    sampler.refresh_state_leaves.assert_called_once()

  async def test_partial_rollout_preserves_kv_and_prefix_cache(self):
    sampler = _sampler(
        rollout_vllm_free_kv_cache_during_weight_sync=True,
        partial_rollout=True,
    )
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
    await delegate.bind_weight_sync(sampler=sampler, state={"w": 1})
    req = _request(policy_version=1, req_id="req-1")
    await delegate.pre_weight_sync(sync_request=req)
    await delegate.post_weight_sync(sync_request=req)
    sampler.delete_cache.assert_not_called()
    sampler.reinitialize_cache.assert_not_called()
    sampler.reset_prefix_cache.assert_not_called()
    sampler.refresh_state_leaves.assert_called_once()

  async def test_round_tracker_lifecycle_and_status(self):
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1}, sampler=_sampler())
    req = _request(policy_version=1, req_id="req-1")
    await delegate.pre_weight_sync(sync_request=req)
    await delegate.weight_sync(sync_request=req)
    await delegate.post_weight_sync(sync_request=req)
    report = delegate.get_weight_sync_status()
    self.assertIsNotNone(report)

  async def test_abort_weight_sync_marks_aborted(self):
    delegate = self._delegate()
    req = _request(policy_version=1, req_id="req-1")
    res = await delegate.abort_weight_sync(sync_request=req)
    self.assertTrue(res)
    self.assertIs(delegate._synchronizers[0].aborted, req)

  async def test_injected_synchronizer_runs_full_destination_contract(self):
    sync = _FakeWorker("gcs_rollout")
    built = []
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate(
        synchronizer_factory=lambda state: built.append(state) or sync
    )
    self.assertEqual(delegate._synchronizers, [])
    self.assertFalse(delegate.is_bounded())
    with self.assertRaisesRegex(RuntimeError, "bind_weight_sync must run"):
      await delegate.weight_sync(_request(policy_version=3))
    sampler = _sampler()
    req = _request(policy_version=3, req_id="req-3")
    await delegate.bind_weight_sync(state={"w": 1}, sampler=sampler)
    self.assertTrue(delegate.is_bounded())
    self.assertEqual(
        await delegate.get_weight_sync_metadata(), [{"unit": "gcs_rollout"}]
    )
    await delegate.pre_weight_sync(sync_request=req)
    self.assertEqual(await delegate.weight_sync(sync_request=req), 3)
    self.assertTrue(await delegate.post_weight_sync(sync_request=req))
    self.assertEqual(delegate._synchronizers, [sync])
    self.assertEqual(built, [{"w": 1}])
    self.assertIs(sync.last_h2d_request, req)
    self.assertIs(sync.applied_to, sampler)


  async def test_abort_after_pre_reinitializes_freed_kv_cache_and_keeps_version(
      self,
  ):
    sampler = _sampler(rollout_vllm_free_kv_cache_during_weight_sync=True)
    delegate = self._delegate()
    await delegate.bind_weight_sync(state={"w": 1}, sampler=sampler)

    # Commit version 1 first.
    req1 = _request(policy_version=1, req_id="req-1")
    req1.extra_config["uuid"] = 1
    await delegate.pre_weight_sync(sync_request=req1)
    await delegate.weight_sync(sync_request=req1)
    self.assertEqual(delegate._version, 0)
    await delegate.post_weight_sync(sync_request=req1)
    self.assertEqual(delegate._version, 1)
    self.assertEqual(sampler.reinitialize_cache.call_count, 1)

    # Stage version 2 then abort: KV cache must be reinitialized once and
    # committed version must stay at 1.
    req2 = _request(policy_version=2, req_id="req-2")
    req2.extra_config["uuid"] = 2
    await delegate.pre_weight_sync(sync_request=req2)
    self.assertEqual(await delegate.weight_sync(sync_request=req2), 2)
    self.assertEqual(delegate._version, 1)
    self.assertTrue(await delegate.abort_weight_sync(sync_request=req2))
    self.assertEqual(sampler.reinitialize_cache.call_count, 2)
    self.assertEqual(delegate._version, 1)
    # Duplicate abort is a no-op and does not double-reinitialize KV cache.
    self.assertFalse(await delegate.abort_weight_sync(sync_request=req2))
    self.assertEqual(sampler.reinitialize_cache.call_count, 2)

  async def test_h2d_and_apply_to_sampler_run_off_event_loop_thread(self):
    loop_thread_id = threading.get_ident()
    observed_threads: dict[str, int] = {}

    class _ThreadRecordingWorker(_FakeWorker):

      def h2d(self, sync_request=None):
        observed_threads["h2d"] = threading.get_ident()
        super().h2d(sync_request=sync_request)

      def apply_to_sampler(self, sampler):
        observed_threads["apply"] = threading.get_ident()
        super().apply_to_sampler(sampler)

    worker = _ThreadRecordingWorker("gcs_rollout")
    delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate(
        synchronizer_factory=lambda _state: worker
    )
    sampler = _sampler()
    req = _request(policy_version=4, req_id="req-4")
    req.extra_config["uuid"] = 4
    await delegate.bind_weight_sync(state={"w": 1}, sampler=sampler)
    await delegate.pre_weight_sync(sync_request=req)
    await delegate.weight_sync(sync_request=req)
    await delegate.post_weight_sync(sync_request=req)
    self.assertNotEqual(observed_threads["h2d"], loop_thread_id)
    self.assertNotEqual(observed_threads["apply"], loop_thread_id)
    self.assertEqual(delegate._version, 4)


if __name__ == "__main__":
  absltest.main()

