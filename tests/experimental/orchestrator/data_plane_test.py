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

"""Integration test: the data-plane components composed against a fake worker.

Exercises the full off-loop path for one step -- acquire dispatch credits, open
groups in the assembler, generate with the FakeRolloutWorker, admit results,
drain complete groups, release credits on terminal, and enqueue for training --
to prove the components fit together.
"""

import asyncio

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import dispatch_credits
from tunix.experimental.orchestrator import group_assembler
from tunix.experimental.orchestrator import request_ledger
from tunix.experimental.orchestrator import train_batch_queue
from tunix.experimental.testing import fake_rollout_worker


def _make_records(group_id, group_size):
  records = []
  for sample_index in range(group_size):
    request = datatypes.RolloutRequest(
        request_id=f"{group_id}:{sample_index}",
        prompt_id=group_id,
        prompt_text="hi",
        sampling_params=datatypes.SamplingParams(max_tokens=4),
    )
    records.append(
        request_ledger.RequestRecord(
            request=request, group_id=group_id, sample_index=sample_index
        )
    )
  return records


class DataPlaneIntegrationTest(absltest.TestCase):

  def test_one_step_dispatch_assemble_train(self):
    group_size = 2
    group_ids = ["train/0/0", "train/0/1"]
    all_records = {g: _make_records(g, group_size) for g in group_ids}
    total_requests = group_size * len(group_ids)

    credits = dispatch_credits.DispatchCredits(capacity=total_requests)
    assembler = group_assembler.GroupAssembler(min_group_size=group_size)
    queue = train_batch_queue.TrainBatchQueue(maxsize=len(group_ids))
    worker = fake_rollout_worker.FakeRolloutWorker(worker_id="r0")
    worker.start()

    # Dispatch: acquire a credit per request and open each group.
    requests = []
    for records in all_records.values():
      for record in records:
        self.assertTrue(credits.try_acquire(1))
        requests.append(record.request)
      assembler.open_group(records)
    self.assertEqual(credits.in_use(), total_requests)

    # Generate and admit every result.
    results = asyncio.run(worker.generate(requests))
    for result in results:
      self.assertEqual(assembler.admit(result), request_ledger.Admission.ACCEPTED)

    # Drain complete groups, release credits on terminal, enqueue for training.
    ready = assembler.drain_ready()
    self.assertLen(ready, len(group_ids))
    for group in ready:
      credits.release(len(group))
      self.assertTrue(queue.put(group))

    self.assertEqual(sorted(g.group_id for g in ready), sorted(group_ids))
    self.assertEqual(credits.in_use(), 0)  # all terminal -> credits returned
    self.assertLen(queue, len(group_ids))
    self.assertEqual(assembler.ledger.inflight_group_count(), 0)

  def test_duplicate_delivery_is_deduped(self):
    records = _make_records("train/0/0", 2)
    assembler = group_assembler.GroupAssembler()
    assembler.open_group(records)
    worker = fake_rollout_worker.FakeRolloutWorker(worker_id="r0")
    worker.start()

    results = asyncio.run(worker.generate([r.request for r in records]))
    self.assertEqual(assembler.admit(results[0]), request_ledger.Admission.ACCEPTED)
    # At-least-once redelivery of the same result is dropped.
    self.assertEqual(assembler.admit(results[0]), request_ledger.Admission.DUPLICATE)
    self.assertEqual(assembler.admit(results[1]), request_ledger.Admission.ACCEPTED)
    self.assertLen(assembler.drain_ready(), 1)


if __name__ == "__main__":
  absltest.main()
