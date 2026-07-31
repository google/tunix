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

"""Reusable contract suite for trainer-worker implementations.

Mix `TrainerWorkerContractSuite` into an `absltest.TestCase` and implement
`make_worker()` and `make_payload()`. The tests pin the worker-level behavior
an orchestrator relies on: the lifecycle states and role advertised to the
control plane, accumulation driven by the caller (N forward/backward passes
then one update), an update step counter that advances, and evaluation that
leaves a pending accumulation alone.

The numeric side of accumulation is `abstract_trainer_contract`; this suite is
about the worker wrapper's bookkeeping and lifecycle.
"""

from typing import Any

from tunix.experimental.common import datatypes

WorkerState = datatypes.WorkerState


class TrainerWorkerContractSuite:
  """Contract tests shared across trainer-worker implementations."""

  def make_worker(self) -> Any:
    """Returns a fresh trainer worker under test."""
    raise NotImplementedError("Subclasses must provide make_worker().")

  def make_payload(self) -> Any:
    """Returns a payload the worker's trainer accepts."""
    raise NotImplementedError("Subclasses must provide make_payload().")

  def _started_worker(self) -> Any:
    worker = self.make_worker()
    worker.initialize()
    worker.start()
    return worker

  def test_reports_its_role_to_the_control_plane(self):
    worker = self.make_worker()
    self.assertIn("trainer", worker.info().roles)

  def test_lifecycle_reaches_ready_then_stopped(self):
    worker = self.make_worker()

    worker.initialize()
    worker.start()
    self.assertEqual(worker.heartbeat().state, WorkerState.READY)

    worker.stop()
    self.assertEqual(worker.heartbeat().state, WorkerState.STOPPED)

  def test_compile_leaves_the_worker_ready(self):
    worker = self.make_worker()
    worker.initialize()

    worker.compile(self.make_payload())

    self.assertEqual(worker.heartbeat().state, WorkerState.READY)

  def test_update_advances_the_step_count(self):
    worker = self._started_worker()

    worker.fwd_bwd(self.make_payload())
    first = worker.update()
    worker.fwd_bwd(self.make_payload())
    second = worker.update()

    self.assertEqual(first, 1)
    self.assertEqual(second, 2)

  def test_accumulates_across_micro_batches_before_updating(self):
    worker = self._started_worker()

    worker.fwd_bwd(self.make_payload())
    worker.fwd_bwd(self.make_payload())
    step = worker.update()

    # Two forward/backward passes are one optimizer step, not two.
    self.assertEqual(step, 1)

  def test_eval_step_leaves_a_pending_accumulation_alone(self):
    worker = self._started_worker()

    worker.fwd_bwd(self.make_payload())
    worker.eval_step(self.make_payload())
    step = worker.update()

    self.assertEqual(step, 1)

  def test_checkpoint_round_trip_reports_the_step(self):
    worker = self._started_worker()
    worker.fwd_bwd(self.make_payload())
    worker.update()

    worker.save_checkpoint({"custom": 42})
    restored = worker.restore_checkpoint()

    self.assertEqual(restored["custom"], 42)
    self.assertIn("step", restored)
