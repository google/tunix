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

"""Tests for RaidenSamplerAdapter."""

import unittest
from unittest import mock

from absl.testing import absltest
from tunix.experimental.rollout import raiden_sampler_adapter


class _FakeWorker:

  def __init__(self, job_name, worker_index=0, **kwargs):
    self.job_name = job_name
    self.worker_index = worker_index
    self.kwargs = kwargs
    self.bound = False
    self.bound_state = None
    self.h2d_calls = 0

  def bind(self, state):
    self.bound = True
    self.bind_calls = getattr(self, "bind_calls", 0) + 1
    self.bound_state = state

  def work_unit_metadata(self):
    return {"unit": self.job_name}

  def h2d(self):
    self.h2d_calls += 1

  def metrics(self):
    return {}

  def checksums(self):
    return {}


class _FakeSampler:
  transformer_state = {"w": 1}


class _Request:

  def __init__(self, policy_version):
    self.policy_version = policy_version


class RaidenSamplerAdapterTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.object(
        raiden_sampler_adapter.raiden_synchronizer,
        "RaidenSynchronizer",
        _FakeWorker,
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def _adapter(self):
    adapter = raiden_sampler_adapter.RaidenSamplerAdapter(
        server_id="test_sampler"
    )
    adapter.sampler = _FakeSampler()
    return adapter

  async def test_bind_binds_sampler_state(self):
    adapter = self._adapter()
    await adapter.bind_weight_sync()
    worker = adapter._synchronizers[0]
    self.assertIs(worker.bound_state, adapter.sampler.transformer_state)

  async def test_worker_uses_the_validated_config(self):
    adapter = self._adapter()
    self.assertIs(adapter._synchronizers[0].kwargs["auto_h2d"], True)

  async def test_repeat_phases_bind_exactly_once(self):
    adapter = self._adapter()
    await adapter.bind_weight_sync()
    await adapter.get_weight_sync_metadata()
    await adapter.pre_weight_sync()
    await adapter.weight_sync()
    self.assertEqual(adapter._synchronizers[0].bind_calls, 1)

  async def test_metadata_returns_one_entry_per_worker(self):
    adapter = self._adapter()
    md = await adapter.get_weight_sync_metadata()
    self.assertEqual(md, [{"unit": "rollout"}])

  async def test_weight_sync_installs_and_tracks_version(self):
    adapter = self._adapter()
    await adapter.bind_weight_sync()
    version = await adapter.weight_sync(_Request(policy_version=5))
    self.assertEqual(version, 5)
    self.assertEqual(adapter._synchronizers[0].h2d_calls, 1)

  async def test_weight_sync_without_request_bumps_version(self):
    adapter = self._adapter()
    await adapter.bind_weight_sync()
    self.assertEqual(await adapter.weight_sync(), 1)
    self.assertEqual(await adapter.weight_sync(), 2)

  async def test_weight_sync_with_zero_version_bumps(self):
    adapter = self._adapter()
    await adapter.bind_weight_sync()
    self.assertEqual(await adapter.weight_sync(_Request(policy_version=0)), 1)

  async def test_weight_sync_before_bind_raises(self):
    adapter = self._adapter()
    with self.assertRaisesRegex(RuntimeError, "bind_weight_sync"):
      await adapter.weight_sync()

  async def test_bind_without_sampler_raises(self):
    adapter = raiden_sampler_adapter.RaidenSamplerAdapter(
        server_id="test_sampler"
    )
    with self.assertRaisesRegex(RuntimeError, "initialize"):
      await adapter.bind_weight_sync()

  async def test_post_weight_sync_returns_true(self):
    adapter = self._adapter()
    self.assertTrue(await adapter.post_weight_sync())


if __name__ == "__main__":
  absltest.main()