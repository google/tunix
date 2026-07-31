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

"""Tests for the toy trainer, including the shared trainer contract suite."""

from typing import Any

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.testing import abstract_trainer_contract
from tunix.experimental.testing import toy_trainer


def _payload(token_ids, loss_mask, advantages) -> datatypes.RLTrainerPayload:
  ids = np.asarray(token_ids, dtype=np.int32)
  return datatypes.RLTrainerPayload(
      token_ids=ids,
      token_mask=np.ones_like(ids),
      loss_mask=np.asarray(loss_mask, dtype=np.int32),
      advantages=np.asarray(advantages, dtype=np.float32),
  )


class ToyTrainerContractTest(
    abstract_trainer_contract.AbstractTrainerContractSuite, absltest.TestCase
):
  """The toy trainer must satisfy the shared trainer contract."""

  def make_trainer(self) -> Any:
    return toy_trainer.ToyAbstractTrainer({"vocab_size": 16})

  def read_params(self, trainer: Any) -> Any:
    return trainer.params


class ToyTrainerTest(absltest.TestCase):
  """Behavior specific to the toy trainer itself."""

  def test_update_moves_weights_of_the_tokens_it_saw(self):
    trainer = toy_trainer.ToyAbstractTrainer({"vocab_size": 8})

    trainer.fwd_bwd(_payload([[1, 2]], [[1, 1]], [1.0]))
    trainer.update()

    # A positive advantage increases the score of the tokens in the sequence.
    self.assertGreater(float(trainer.params["w"][1]), 0.0)
    self.assertGreater(float(trainer.params["w"][2]), 0.0)
    # Untouched tokens stay put.
    self.assertEqual(float(trainer.params["w"][5]), 0.0)

  def test_masked_positions_do_not_contribute(self):
    masked = toy_trainer.ToyAbstractTrainer({"vocab_size": 8})
    masked.fwd_bwd(_payload([[1, 7]], [[1, 0]], [1.0]))
    masked.update()

    self.assertEqual(float(masked.params["w"][7]), 0.0)
    self.assertGreater(float(masked.params["w"][1]), 0.0)

  def test_update_without_accumulated_gradients_is_an_error(self):
    trainer = toy_trainer.ToyAbstractTrainer()
    with self.assertRaises(RuntimeError):
      trainer.update()

  def test_gen_model_input_fn_adapts_a_foreign_payload(self):
    trainer = toy_trainer.ToyAbstractTrainer({"vocab_size": 8})
    trainer.with_gen_model_input_fn(
        lambda payload: {
            "token_ids": np.asarray(payload["ids"]),
            "loss_mask": np.asarray(payload["mask"]),
            "advantages": np.asarray(payload["adv"], dtype=np.float32),
        }
    )

    trainer.fwd_bwd({"ids": [[3]], "mask": [[1]], "adv": [1.0]})
    trainer.update()

    self.assertGreater(float(trainer.params["w"][3]), 0.0)

  def test_prepare_weight_sync_stages_a_detached_copy(self):
    trainer = toy_trainer.ToyAbstractTrainer({"vocab_size": 8})
    trainer.fwd_bwd(_payload([[1]], [[1]], [1.0]))
    trainer.update()
    trainer.prepare_weight_sync()
    staged = trainer.staged_weights

    trainer.fwd_bwd(_payload([[1]], [[1]], [1.0]))
    trainer.update()

    self.assertNotAlmostEqual(
        float(staged["w"][1]), float(trainer.params["w"][1])
    )

  def test_metrics_report_then_clear(self):
    trainer = toy_trainer.ToyAbstractTrainer({"vocab_size": 8})
    # Weights start at zero, so the first loss is exactly zero; take one step
    # to move them before asking for a non-trivial metric.
    trainer.fwd_bwd(_payload([[1]], [[1]], [1.0]))
    trainer.update()
    trainer.get_metrics()

    trainer.fwd_bwd(_payload([[1]], [[1]], [1.0]))
    first = trainer.get_metrics()
    second = trainer.get_metrics()

    self.assertNotEqual(float(first.scalar_metrics["loss"]), 0.0)
    self.assertEqual(float(second.scalar_metrics["loss"]), 0.0)


if __name__ == "__main__":
  absltest.main()
