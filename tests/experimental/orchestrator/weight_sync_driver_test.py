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

"""Tests for WeightSyncDriver."""

import asyncio

from absl.testing import absltest
from tunix.experimental.orchestrator import weight_sync_driver


class _FakeCoordinator:

  def __init__(self):
    self.calls = []
    self.loops = []
    self.fail_next = False

  async def sync(self, policy_version=0, **kwargs):
    self.calls.append(policy_version)
    self.loops.append(asyncio.get_running_loop())
    if self.fail_next:
      self.fail_next = False
      raise RuntimeError("round failed")
    return f"committed-v{policy_version}"


class _FakeComponents:

  def __init__(self, coordinator):
    self.coordinator = coordinator
    self.closed = False
    self.built_on = asyncio.get_running_loop()

  async def close(self):
    self.closed = True


class WeightSyncDriverTest(absltest.TestCase):

  def _driver(self, coordinator):
    self.factory_uuids = []

    async def factory(initial_uuid):
      self.factory_uuids.append(initial_uuid)
      return _FakeComponents(coordinator)

    return weight_sync_driver.WeightSyncDriver(
        factory, initial_uuid=7, initial_policy_version=3
    )

  def test_factory_gets_initial_uuid(self):
    driver = self._driver(_FakeCoordinator())
    self.assertEqual(self.factory_uuids, [7])
    driver.close()

  def test_sync_advances_version(self):
    coordinator = _FakeCoordinator()
    driver = self._driver(coordinator)
    self.assertEqual(driver.sync_weights(), "committed-v4")
    self.assertEqual(driver.sync_weights(), "committed-v5")
    self.assertEqual(driver.policy_version, 5)
    self.assertEqual(coordinator.calls, [4, 5])
    driver.close()

  def test_failed_round_keeps_version(self):
    coordinator = _FakeCoordinator()
    coordinator.fail_next = True
    driver = self._driver(coordinator)
    with self.assertRaises(RuntimeError):
      driver.sync_weights()
    self.assertEqual(driver.policy_version, 3)
    self.assertEqual(driver.sync_weights(), "committed-v4")
    driver.close()

  def test_everything_runs_on_one_loop(self):
    coordinator = _FakeCoordinator()
    driver = self._driver(coordinator)
    driver.sync_weights()
    driver.sync_weights()
    loops = set(coordinator.loops)
    self.assertLen(loops, 1)
    driver.close()

  def test_close_closes_components(self):
    coordinator = _FakeCoordinator()
    driver = self._driver(coordinator)
    components = driver._components
    driver.close()
    self.assertTrue(components.closed)


if __name__ == "__main__":
  absltest.main()
