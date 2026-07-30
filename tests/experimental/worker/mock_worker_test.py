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

"""Tests for mock_worker."""

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.worker import mock_worker

WorkerState = datatypes.WorkerState


class MockWorkerTest(absltest.TestCase):

  def test_initial_state(self):
    worker = mock_worker.MockWorker("test_id", {"trainer"})
    self.assertEqual(worker.state, WorkerState.PENDING)
    self.assertEqual(worker.call_counts["initialize"], 0)
    self.assertEqual(worker.call_counts["compile"], 0)
    self.assertEqual(worker.call_counts["start"], 0)
    self.assertEqual(worker.call_counts["stop"], 0)

  def test_initialize(self):
    worker = mock_worker.MockWorker("test_id", {"trainer"})
    worker.initialize()
    self.assertEqual(worker.state, WorkerState.INITIALIZING)
    self.assertEqual(worker.call_counts["initialize"], 1)

  def test_compile(self):
    worker = mock_worker.MockWorker("test_id", {"trainer"})
    worker.initialize()
    worker.compile()
    self.assertEqual(worker.state, WorkerState.READY)
    self.assertEqual(worker.call_counts["compile"], 1)

  def test_start(self):
    worker = mock_worker.MockWorker("test_id", {"trainer"})
    worker.initialize()
    worker.compile()
    worker.start()
    self.assertEqual(worker.state, WorkerState.READY)
    self.assertEqual(worker.call_counts["start"], 1)

  def test_stop(self):
    worker = mock_worker.MockWorker("test_id", {"trainer"})
    worker.stop()
    self.assertEqual(worker.state, WorkerState.STOPPED)
    self.assertEqual(worker.call_counts["stop"], 1)

  def test_heartbeat_reports_current_state(self):
    worker = mock_worker.MockWorker("test_id", {"trainer"})
    worker.initialize()
    worker.compile()
    worker.state = WorkerState.COMPILING
    self.assertEqual(worker.heartbeat().state, WorkerState.COMPILING)

  def test_invalid_state_transition(self):
    worker = mock_worker.MockWorker("test_id", {"trainer"})
    with self.assertRaisesRegex(
        RuntimeError,
        "Invalid transition from PENDING to READY",
    ):
      worker.state = WorkerState.READY


if __name__ == "__main__":
  absltest.main()
