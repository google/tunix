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

"""Runs the shared worker contract suites against the in-tree workers.

These are the acceptance suites a worker implementation is held to. Running
them here against the trainer worker (over the toy trainer) and the inference
worker (over the stub scoring core) keeps the suites honest: if a suite drifts
from the contract the real workers implement, this fails.
"""

import asyncio
from typing import Any

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.testing import fake_rollout_worker
from tunix.experimental.testing import inference_worker_contract
from tunix.experimental.testing import toy_trainer
from tunix.experimental.testing import trainer_worker_contract
from tunix.experimental.worker import inference_worker
from tunix.experimental.worker import trainer_worker


class TrainerWorkerContractTest(
    trainer_worker_contract.TrainerWorkerContractSuite, absltest.TestCase
):
  """The trainer worker, over the toy trainer, satisfies the shared contract."""

  def make_worker(self) -> Any:
    return trainer_worker.TrainerWorker(
        lambda: toy_trainer.ToyAbstractTrainer({"vocab_size": 16}),
        worker_id="toy-trainer-worker",
    )

  def make_payload(self) -> Any:
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    return datatypes.RLTrainerPayload(
        token_ids=ids,
        token_mask=np.ones_like(ids),
        loss_mask=np.ones_like(ids),
        advantages=np.array([1.0], dtype=np.float32),
    )


class InferenceWorkerContractTest(
    inference_worker_contract.InferenceWorkerContractSuite, absltest.TestCase
):
  """The inference worker, over the stub core, satisfies the shared contract."""

  def make_worker(self, chunk_size: int | None = None) -> Any:
    return inference_worker.InferenceWorker(
        inference_worker_contract.StubReferenceScoringCore(),
        worker_id="stub-inference-worker",
        pad_id=0,
        eos_id=1,
        chunk_size=chunk_size,
    )


class FakeRolloutWorkerTest(absltest.TestCase):
  """The deterministic rollout worker behaves as the contract requires."""

  def _request(self, prompt_id: str) -> datatypes.RolloutRequest:
    return datatypes.RolloutRequest(
        prompt=f"prompt {prompt_id}",
        prompt_id=prompt_id,
        request_id=f"req-{prompt_id}",
    )

  def test_results_are_stable_across_runs(self):
    first = asyncio.run(
        fake_rollout_worker.FakeRolloutWorker().generate(self._request("p0"))
    )
    second = asyncio.run(
        fake_rollout_worker.FakeRolloutWorker().generate(self._request("p0"))
    )

    self.assertEqual(first.status, "SUCCEEDED")
    np.testing.assert_array_equal(
        first.segments[0].tokens, second.segments[0].tokens
    )
    self.assertEqual(first.env_reward, second.env_reward)

  def test_batch_streams_each_result_as_it_lands(self):
    worker = fake_rollout_worker.FakeRolloutWorker()
    streamed = []

    responses = asyncio.run(
        worker.generate(
            [self._request(f"p{i}") for i in range(3)],
            on_complete=streamed.append,
        )
    )

    self.assertLen(responses, 3)
    self.assertEqual(
        [r.request_id for r in streamed], ["req-p0", "req-p1", "req-p2"]
    )

  def test_failure_is_reported_in_band(self):
    worker = fake_rollout_worker.FakeRolloutWorker(fail_prompt_ids=["p1"])

    responses = asyncio.run(
        worker.generate([self._request("p0"), self._request("p1")])
    )

    self.assertEqual(responses[0].status, "SUCCEEDED")
    self.assertEqual(responses[1].status, "FAILED")
    self.assertIsNotNone(responses[1].error)

  def test_weight_sync_advances_the_reported_policy_version(self):
    worker = fake_rollout_worker.FakeRolloutWorker()

    version = worker.sync_weights(
        datatypes.WeightSyncRequest(policy_version=7)
    )

    self.assertEqual(version, 7)
    self.assertEqual(worker.heartbeat().policy_version, 7)
    response = asyncio.run(worker.generate(self._request("p0")))
    self.assertEqual(response.policy_version, 7)

  def test_stalled_request_completes_after_release(self):
    async def _run():
      worker = fake_rollout_worker.FakeRolloutWorker(stall_prompt_ids=["slow"])
      pending = asyncio.create_task(worker.generate(self._request("slow")))
      await asyncio.sleep(0.01)
      self.assertFalse(pending.done())

      worker.release()
      return await asyncio.wait_for(pending, timeout=5.0)

    self.assertEqual(asyncio.run(_run()).status, "SUCCEEDED")


if __name__ == "__main__":
  absltest.main()
