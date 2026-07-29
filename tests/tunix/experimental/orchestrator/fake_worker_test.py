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

"""Tests for fake_worker."""

from absl.testing import absltest
from tunix.tunix.experimental.orchestrator import fake_worker


class FakeWorkerTest(absltest.TestCase):

  def test_initial_state(self):
    worker = fake_worker.FakeWorker("test_id", {"trainer"})
    self.assertEqual(worker.state, "PENDING")
    self.assertEqual(worker.call_counts["initialize"], 0)
    self.assertEqual(worker.call_counts["compile"], 0)
    self.assertEqual(worker.call_counts["start"], 0)
    self.assertEqual(worker.call_counts["stop"], 0)

  def test_initialize(self):
    worker = fake_worker.FakeWorker("test_id", {"trainer"})
    worker.initialize()
    self.assertEqual(worker.state, "INITIALIZED")
    self.assertEqual(worker.call_counts["initialize"], 1)

  def test_compile(self):
    worker = fake_worker.FakeWorker("test_id", {"trainer"})
    worker.compile()
    self.assertEqual(worker.state, "COMPILED")
    self.assertEqual(worker.call_counts["compile"], 1)

  def test_start(self):
    worker = fake_worker.FakeWorker("test_id", {"trainer"})
    worker.start()
    self.assertEqual(worker.state, "READY")
    self.assertEqual(worker.call_counts["start"], 1)

  def test_stop(self):
    worker = fake_worker.FakeWorker("test_id", {"trainer"})
    worker.stop()
    self.assertEqual(worker.state, "STOPPED")
    self.assertEqual(worker.call_counts["stop"], 1)


if __name__ == "__main__":
  absltest.main()
