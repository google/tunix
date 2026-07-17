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

"""Runs the TrainerWorker contract suite against FakeTrainerWorker."""

from absl.testing import absltest
from tunix.experimental.testing import fake_trainer_worker
from tunix.experimental.testing import trainer_worker_contract


class FakeTrainerWorkerContractTest(
    trainer_worker_contract.TrainerWorkerContractSuite, absltest.TestCase
):

  def make_worker(self):
    return fake_trainer_worker.FakeTrainerWorker(worker_id="fake-trainer")


if __name__ == "__main__":
  absltest.main()
