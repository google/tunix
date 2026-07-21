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

"""Smoke tests for the main-based FakeRolloutWorker."""

import asyncio

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.testing import fake_rollout_worker


class FakeRolloutWorkerTest(absltest.TestCase):

  def _started_worker(self) -> fake_rollout_worker.FakeRolloutWorker:
    worker = fake_rollout_worker.FakeRolloutWorker(worker_id="fake-rollout")
    worker.initialize()
    worker.compile(
        datatypes.ShapeConfig(max_prompt_length=2, max_response_tokens=3)
    )
    worker.start()
    return worker

  def _request(self, request_id: str = "r0") -> datatypes.RolloutRequest:
    return datatypes.RolloutRequest(
        request_id=request_id,
        prompt_id="p0",
        prompt_text="hi",
        sampling_params=datatypes.SamplingParams(max_tokens=8),
    )

  def test_lifecycle_health_and_info(self):
    worker = self._started_worker()
    self.assertEqual(worker.health().state, "READY")
    info = worker.info()
    self.assertEqual(info.worker_id, "fake-rollout")
    self.assertIn("rollout", info.roles)
    worker.stop()
    self.assertEqual(worker.health().state, "STOPPED")

  def test_generate_returns_rollout_results(self):
    worker = self._started_worker()
    results = asyncio.run(worker.generate([self._request("r0")]))
    self.assertLen(results, 1)
    self.assertEqual(results[0].request_id, "r0")
    self.assertEqual(results[0].status, "COMPLETED")

  def test_generate_invokes_on_complete_callback(self):
    worker = self._started_worker()
    seen = []
    asyncio.run(worker.generate([self._request("a"), self._request("b")],
                                on_complete=seen.append))
    self.assertEqual([r.request_id for r in seen], ["a", "b"])

  def test_sync_weights_advances_version(self):
    worker = self._started_worker()
    version = worker.sync_weights(
        datatypes.WeightSyncMetadata(version=1, method="in_process")
    )
    self.assertEqual(version, 1)
    self.assertEqual(worker.health().policy_version, 1)


if __name__ == "__main__":
  absltest.main()
