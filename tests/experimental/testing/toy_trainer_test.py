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

"""Runs the AbstractTrainer numeric contract suite against ToyAbstractTrainer."""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.testing import abstract_trainer_contract
from tunix.experimental.testing import toy_trainer


class ToyAbstractTrainerContractTest(
    abstract_trainer_contract.AbstractTrainerContractSuite, absltest.TestCase
):

  def make_trainer(self):
    return toy_trainer.ToyAbstractTrainer()

  def test_custom_gen_model_input_fn_is_honored(self):
    trainer = toy_trainer.ToyAbstractTrainer()
    trainer.with_loss_fn(lambda *a, **k: None)

    def zero_advantage_adapter(payload):
      return {
          "completion_ids": jnp.asarray(payload.completion_ids),
          "loss_mask": jnp.asarray(payload.loss_mask),
          "advantages": jnp.zeros_like(
              jnp.asarray(payload.advantages, dtype=jnp.float32)
          ),
      }

    trainer.with_gen_model_input_fn(zero_advantage_adapter)
    trainer.compile(
        datatypes.ShapeConfig(max_prompt_length=1, max_response_tokens=3)
    )
    example = datatypes.TrainExampleV1(
        loss_mask=np.ones((1, 3), np.int32),
        prompt_ids=np.zeros((1, 1), np.int32),
        prompt_mask=np.ones((1, 1), np.int32),
        completion_ids=np.array([[1, 2, 3]], np.int32),
        advantages=np.array([5.0], np.float32),
    )
    before = jax.tree.map(
        np.array,
        trainer.prepare_weight_sync(
            datatypes.WeightSyncSpec(version=0)
        ).locator,
    )
    trainer.fwd_bwd(example, accum_id="a", micro_index=0, loss_scale=1.0)
    trainer.update(accum_id="a", expected_micro_steps=1)
    after = trainer.prepare_weight_sync(
        datatypes.WeightSyncSpec(version=0)
    ).locator

    # Zeroed advantages -> zero gradient -> params unchanged, proving the custom
    # adapter (not the default) drove fwd_bwd.
    chex.assert_trees_all_close(before, after)


if __name__ == "__main__":
  absltest.main()
