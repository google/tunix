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

"""Reusable contract test suite for the TrainerWorker receipt protocol.

Mix `TrainerWorkerContractSuite` into an `absltest.TestCase` and implement
`make_worker()` (and, for a real trainer, `make_payload()`); the shared tests
pin the gradient-accumulation protocol: per-micro-step dedup, {0..N-1}
verification, cached update on retry, unknown-accum-id failure,
reset_accumulation, eval-does-not-disturb-accumulation, and checkpoint
round-trip. This is the protocol contract (bookkeeping); the numeric
accumulation contract is covered separately against a real trainer.
"""

from tunix.experimental.worker import abstract_worker


class TrainerWorkerContractSuite:
  """Contract tests for the gradient-accumulation receipt protocol."""

  def make_worker(self) -> abstract_worker.Worker:
    raise NotImplementedError("Subclasses must provide make_worker().")

  def make_payload(self):
    """Payload passed to fwd_bwd/eval_step; overridden for a real trainer."""
    return None

  def _started_worker(self):
    worker = self.make_worker()
    worker.initialize()
    worker.start()
    return worker

  def test_lifecycle_and_info(self):
    worker = self.make_worker()
    worker.initialize()
    worker.start()
    self.assertEqual(worker.health().state, "READY")
    self.assertIn("trainer", worker.info().roles)
    worker.stop()
    self.assertEqual(worker.health().state, "STOPPED")

  def test_fwd_bwd_returns_receipt(self):
    worker = self._started_worker()
    receipt = worker.fwd_bwd(
        self.make_payload(), accum_id="a", micro_index=0, loss_scale=1.0
    )
    self.assertEqual(receipt.accum_id, "a")
    self.assertEqual(receipt.micro_index, 0)
    self.assertFalse(receipt.applied)

  def test_duplicate_micro_step_is_not_reaccumulated(self):
    worker = self._started_worker()
    first = worker.fwd_bwd(self.make_payload(), accum_id="a", micro_index=0)
    second = worker.fwd_bwd(self.make_payload(), accum_id="a", micro_index=0)
    self.assertEqual(first, second)

  def test_update_applies_and_advances_step(self):
    worker = self._started_worker()
    worker.fwd_bwd(self.make_payload(), accum_id="a", micro_index=0)
    worker.fwd_bwd(self.make_payload(), accum_id="a", micro_index=1)
    result = worker.update(accum_id="a", expected_micro_steps=2)
    self.assertTrue(result.applied)
    self.assertEqual(result.step, 1)

  def test_update_retry_returns_cached_result(self):
    worker = self._started_worker()
    worker.fwd_bwd(self.make_payload(), accum_id="a", micro_index=0)
    first = worker.update(accum_id="a", expected_micro_steps=1)
    second = worker.update(accum_id="a", expected_micro_steps=1)
    self.assertEqual(first, second)

  def test_update_with_missing_micro_step_raises(self):
    worker = self._started_worker()
    worker.fwd_bwd(self.make_payload(), accum_id="a", micro_index=0)
    with self.assertRaises(ValueError):
      worker.update(accum_id="a", expected_micro_steps=2)

  def test_update_unknown_accum_id_raises(self):
    worker = self._started_worker()
    with self.assertRaises(KeyError):
      worker.update(accum_id="no-such-accum", expected_micro_steps=1)

  def test_reset_accumulation_discards_receipts(self):
    worker = self._started_worker()
    worker.fwd_bwd(self.make_payload(), accum_id="a", micro_index=0)
    worker.reset_accumulation("a")
    with self.assertRaises(KeyError):
      worker.update(accum_id="a", expected_micro_steps=1)

  def test_eval_step_does_not_disturb_accumulation(self):
    worker = self._started_worker()
    worker.fwd_bwd(self.make_payload(), accum_id="a", micro_index=0)
    worker.eval_step(self.make_payload())
    result = worker.update(accum_id="a", expected_micro_steps=1)
    self.assertTrue(result.applied)

  def test_checkpoint_round_trip(self):
    worker = self._started_worker()
    worker.save_checkpoint({"custom": 42})
    restored = worker.restore_checkpoint()
    self.assertEqual(restored["custom"], 42)
    self.assertIn("step", restored)
