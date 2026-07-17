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

"""Runs the RolloutWorker contract suite against FakeRolloutWorker."""

from absl.testing import absltest
from tunix.experimental.testing import fake_rollout_worker
from tunix.experimental.testing import rollout_worker_contract


class FakeRolloutWorkerContractTest(
    rollout_worker_contract.RolloutWorkerContractSuite, absltest.TestCase
):

  def make_worker(self):
    return fake_rollout_worker.FakeRolloutWorker(worker_id="fake-rollout")


if __name__ == "__main__":
  absltest.main()
