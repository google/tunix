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

"""Tests for RaidenTpuWorker."""

from unittest import mock

from absl.testing import absltest

from tunix.experimental.orchestrator import weight_sync
from tunix.experimental.worker import raiden_synchronizer
from tunix.experimental.worker import raiden_tpu_worker


class _FakeSynchronizer:

  def __init__(self, job_name, state, *, auto_h2d=False, parallelism=4,
               bind_ip=None):
    del parallelism, bind_ip
    self.job_name = job_name
    self.state = state
    self.auto_h2d = auto_h2d
    self.rebound_with = None

  def rebind(self, state):
    self.rebound_with = state

  def work_unit_metadata(self):
    return weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name=self.job_name),
        shards=("addr:1",),
    )


class RaidenTpuWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.object(
        raiden_synchronizer, "RaidenSynchronizer", _FakeSynchronizer
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_bind_once_then_rebind(self):
    worker = raiden_tpu_worker.RaidenTpuWorker("rollout")
    worker.bind({"w": 1})
    first = worker._synchronizer
    worker.bind({"w": 2})
    self.assertIs(worker._synchronizer, first)
    self.assertEqual(first.rebound_with, {"w": 2})

  def test_auto_h2d_defaults_off(self):
    worker = raiden_tpu_worker.RaidenTpuWorker("rollout")
    worker.bind({"w": 1})
    self.assertIs(worker._synchronizer.auto_h2d, False)

  def test_worker_index_stamps_replica_id(self):
    worker = raiden_tpu_worker.RaidenTpuWorker("rollout", worker_index=3)
    worker.bind({"w": 1})
    self.assertEqual(worker.work_unit_metadata().unit.job_replica_id, "3")
    base = raiden_tpu_worker.RaidenTpuWorker("rollout")
    base.bind({"w": 1})
    self.assertEqual(base.work_unit_metadata().unit.job_replica_id, "")

  def test_host_stage_pulls_state_to_host(self):
    sentinel = {"pulled": True}
    with mock.patch.object(
        raiden_synchronizer, "to_host_cpu_state", return_value=sentinel
    ) as pull:
      worker = raiden_tpu_worker.RaidenTpuWorker("trainer", host_stage=True)
      worker.bind({"w": 1})
    pull.assert_called_once_with({"w": 1})
    self.assertIs(worker._synchronizer.state, sentinel)


if __name__ == "__main__":
  absltest.main()