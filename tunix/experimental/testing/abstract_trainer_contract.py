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

"""Reusable numeric contract suite for trainer implementations.

Mix `AbstractTrainerContractSuite` into an `absltest.TestCase`, implement
`make_trainer()` and `read_params()`, and the implementation is held to the
behaviors the step-level trainer API promises: caller-driven accumulation that
reproduces the equivalent full-batch update, evaluation that mutates nothing,
an update step counter that advances, and a checkpoint that round-trips.

Accumulation is only exact across micro-batches carrying the same number of
contributing tokens, because the API accumulates the mean of per-micro-batch
gradients; the suite uses equal-sized micro-batches for that reason.

Trainer calls are at-most-once in this contract: there is no dedup or receipt
protocol, so a retried `fwd_bwd` accumulates twice and a retried `update`
applies twice. Callers must not blindly retry them.
"""

from typing import Any

import chex
import jax
import numpy as np

from tunix.experimental.common import datatypes


def detach(tree: Any) -> Any:
  """Copies a parameter pytree so later updates cannot alias the snapshot."""
  return jax.tree.map(np.array, tree)


class AbstractTrainerContractSuite:
  """Contract tests shared across trainer implementations."""

  def make_trainer(self) -> Any:
    """Returns a fresh trainer under test."""
    raise NotImplementedError("Subclasses must provide make_trainer().")

  def read_params(self, trainer: Any) -> Any:
    """Returns the trainer's live parameters as a pytree."""
    raise NotImplementedError("Subclasses must provide read_params().")

  def _ready_trainer(self) -> Any:
    trainer = self.make_trainer()
    trainer.with_loss_fn(lambda *a, **k: None)
    trainer.compile(None)
    return trainer

  def _payload(
      self, token_ids, loss_mask, advantages
  ) -> datatypes.RLTrainerPayload:
    ids = np.asarray(token_ids, dtype=np.int32)
    mask = np.asarray(loss_mask, dtype=np.int32)
    return datatypes.RLTrainerPayload(
        token_ids=ids,
        token_mask=np.ones_like(ids),
        loss_mask=mask,
        advantages=np.asarray(advantages, dtype=np.float32),
    )

  def test_accumulation_equals_full_batch(self):
    """Two micro-batches then one update == the same rows in one batch."""
    ids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    mask = np.ones_like(ids)
    adv = np.array([1.0, -1.0, 0.5, 2.0])

    accumulated = self._ready_trainer()
    accumulated.fwd_bwd(self._payload(ids[:2], mask[:2], adv[:2]))
    accumulated.fwd_bwd(self._payload(ids[2:], mask[2:], adv[2:]))
    accumulated.update()

    full_batch = self._ready_trainer()
    full_batch.fwd_bwd(self._payload(ids, mask, adv))
    full_batch.update()

    chex.assert_trees_all_close(
        self.read_params(accumulated),
        self.read_params(full_batch),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_update_advances_the_step_count(self):
    payload = self._payload([[1, 2, 3]], [[1, 1, 1]], [1.0])
    trainer = self._ready_trainer()

    trainer.fwd_bwd(payload)
    first = trainer.update()
    trainer.fwd_bwd(payload)
    second = trainer.update()

    self.assertEqual(second, first + 1)

  def test_eval_step_mutates_nothing(self):
    payload = self._payload([[1, 2, 3]], [[1, 1, 1]], [1.0])
    trainer = self._ready_trainer()
    trainer.fwd_bwd(payload)
    trainer.update()
    before = detach(self.read_params(trainer))

    trainer.eval_step(payload)

    chex.assert_trees_all_close(before, self.read_params(trainer))

  def test_eval_step_leaves_accumulation_intact(self):
    """Evaluating mid-accumulation must not disturb the pending update."""
    payload = self._payload([[1, 2, 3]], [[1, 1, 1]], [1.0])

    evaluated = self._ready_trainer()
    evaluated.fwd_bwd(payload)
    evaluated.eval_step(payload)
    evaluated.update()

    untouched = self._ready_trainer()
    untouched.fwd_bwd(payload)
    untouched.update()

    chex.assert_trees_all_close(
        self.read_params(evaluated), self.read_params(untouched)
    )

  def test_checkpoint_round_trip(self):
    payload = self._payload([[1, 2, 3]], [[1, 1, 1]], [1.0])
    trainer = self._ready_trainer()
    trainer.fwd_bwd(payload)
    trainer.update()
    trainer.save_checkpoint({"custom": 7})
    saved = detach(self.read_params(trainer))

    trainer.fwd_bwd(payload)
    trainer.update()
    metadata = trainer.restore_checkpoint()

    self.assertEqual(metadata["custom"], 7)
    self.assertIn("step", metadata)
    chex.assert_trees_all_close(saved, self.read_params(trainer))
