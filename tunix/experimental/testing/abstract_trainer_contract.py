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

"""Reusable numeric contract suite for AbstractTrainer implementations.

Mix `AbstractTrainerContractSuite` into an `absltest.TestCase` and implement
`make_trainer()`. The headline test is the accumulation numeric contract: N
micro-batches with caller-supplied token-mean `loss_scale`s must reproduce the
single full-batch update exactly. Also covers duplicate-micro-step dedup, cached
update on retry, eval-mutates-nothing, and checkpoint round-trip. A real trainer
runs the same suite; parameters are read via the in-process weight-sync locator,
so the suite stays implementation-agnostic.
"""

import chex
import jax
import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.train import abstract_trainer


class AbstractTrainerContractSuite:
  """Contract tests shared across AbstractTrainer implementations."""

  def make_trainer(self) -> abstract_trainer.AbstractTrainer:
    raise NotImplementedError("Subclasses must provide make_trainer().")

  def _ready_trainer(self) -> abstract_trainer.AbstractTrainer:
    trainer = self.make_trainer()
    trainer.with_loss_fn(lambda *a, **k: None)
    trainer.compile(
        datatypes.ShapeConfig(max_prompt_length=1, max_response_tokens=3)
    )
    return trainer

  def _example(
      self, completion_ids, loss_mask, advantages
  ) -> datatypes.TrainExampleV1:
    ids = np.asarray(completion_ids, dtype=np.int32)
    rows = ids.shape[0]
    return datatypes.TrainExampleV1(
        loss_mask=np.asarray(loss_mask, dtype=np.int32),
        prompt_ids=np.zeros((rows, 1), dtype=np.int32),
        prompt_mask=np.ones((rows, 1), dtype=np.int32),
        completion_ids=ids,
        advantages=np.asarray(advantages, dtype=np.float32),
    )

  def _params(self, trainer):
    return trainer.prepare_weight_sync(
        datatypes.WeightSyncSpec(version=0)
    ).locator

  def _snapshot(self, trainer):
    return jax.tree.map(np.array, self._params(trainer))

  def test_accumulation_equals_full_batch(self):
    ids = np.array([[1, 2, 3], [4, 5, 0], [6, 7, 8], [9, 10, 0]])
    mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]])
    adv = np.array([1.0, -1.0, 0.5, 2.0])
    total = float(mask.sum())

    accumulated = self._ready_trainer()
    accumulated.fwd_bwd(
        self._example(ids[:2], mask[:2], adv[:2]),
        accum_id="a",
        micro_index=0,
        loss_scale=float(mask[:2].sum()) / total,
    )
    accumulated.fwd_bwd(
        self._example(ids[2:], mask[2:], adv[2:]),
        accum_id="a",
        micro_index=1,
        loss_scale=float(mask[2:].sum()) / total,
    )
    accumulated.update(accum_id="a", expected_micro_steps=2)

    full_batch = self._ready_trainer()
    full_batch.fwd_bwd(
        self._example(ids, mask, adv),
        accum_id="b",
        micro_index=0,
        loss_scale=1.0,
    )
    full_batch.update(accum_id="b", expected_micro_steps=1)

    chex.assert_trees_all_close(
        self._params(accumulated),
        self._params(full_batch),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_duplicate_micro_step_not_reaccumulated(self):
    ex = self._example([[1, 2, 3]], [[1, 1, 1]], [1.0])

    dedup = self._ready_trainer()
    dedup.fwd_bwd(ex, accum_id="a", micro_index=0, loss_scale=1.0)
    dedup.fwd_bwd(ex, accum_id="a", micro_index=0, loss_scale=1.0)
    dedup.update(accum_id="a", expected_micro_steps=1)

    once = self._ready_trainer()
    once.fwd_bwd(ex, accum_id="b", micro_index=0, loss_scale=1.0)
    once.update(accum_id="b", expected_micro_steps=1)

    chex.assert_trees_all_close(self._params(dedup), self._params(once))

  def test_update_retry_returns_cached_result(self):
    ex = self._example([[1, 2, 3]], [[1, 1, 1]], [1.0])
    trainer = self._ready_trainer()
    trainer.fwd_bwd(ex, accum_id="a", micro_index=0, loss_scale=1.0)

    first = trainer.update(accum_id="a", expected_micro_steps=1)
    after_first = self._snapshot(trainer)
    second = trainer.update(accum_id="a", expected_micro_steps=1)

    self.assertEqual(first, second)
    chex.assert_trees_all_close(after_first, self._params(trainer))

  def test_eval_step_does_not_mutate_params(self):
    ex = self._example([[1, 2, 3]], [[1, 1, 1]], [1.0])
    trainer = self._ready_trainer()
    trainer.fwd_bwd(ex, accum_id="a", micro_index=0, loss_scale=1.0)
    trainer.update(accum_id="a", expected_micro_steps=1)

    before = self._snapshot(trainer)
    trainer.eval_step(ex)

    chex.assert_trees_all_close(before, self._params(trainer))

  def test_checkpoint_round_trip(self):
    ex = self._example([[1, 2, 3]], [[1, 1, 1]], [1.0])
    trainer = self._ready_trainer()
    trainer.fwd_bwd(ex, accum_id="a", micro_index=0, loss_scale=1.0)
    trainer.update(accum_id="a", expected_micro_steps=1)
    trainer.save_checkpoint({"custom": 7})
    saved = self._snapshot(trainer)

    trainer.fwd_bwd(ex, accum_id="c", micro_index=0, loss_scale=1.0)
    trainer.update(accum_id="c", expected_micro_steps=1)
    metadata = trainer.restore_checkpoint()

    self.assertEqual(metadata["custom"], 7)
    self.assertIn("step", metadata)
    chex.assert_trees_all_close(saved, self._params(trainer))
