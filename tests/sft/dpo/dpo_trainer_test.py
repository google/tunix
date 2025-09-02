# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from grain import python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import common
from tunix.sft.dpo import dpo_trainer as dpo_lib
from tunix.tests import test_common as tc

jax.config.update("jax_threefry_partitionable", False)
# jax.config.update("jax_debug_nans", True) # useful for debugging NaN


class MySource(grain.RandomAccessDataSource):

  def __init__(self, data):
    self._data = data

  def __getitem__(self, idx):
    return self._data[idx]

  def __len__(self):
    return len(self._data)


def _dummy_dataset(
    source: MySource,
    prompt_ids: np.ndarray,
    prompt_mask: np.ndarray,
    chosen_ids: np.ndarray,
    chosen_mask: np.ndarray,
    rejected_ids: np.ndarray,
    rejected_mask: np.ndarray,
):
  return grain.MapDataset.source(source).map(
      lambda x: dpo_lib.TrainingInput(
          prompt_ids=prompt_ids,
          prompt_mask=prompt_mask,
          chosen_ids=chosen_ids,
          chosen_mask=chosen_mask,
          rejected_ids=rejected_ids,
          rejected_mask=rejected_mask,
      )
  )


class DpoTrainerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="chosen_reject_equal_length",
          prompt_ids=np.arange(0, 10).reshape(2, 5),
          prompt_mask=np.ones((2, 5)),
          chosen_ids=np.arange(10, 20).reshape(2, 5),
          chosen_mask=np.ones((2, 5)),
          rejected_ids=np.arange(20, 30).reshape(2, 5),
          rejected_mask=np.ones((2, 5)),
      ),
      dict(
          testcase_name="chosen_reject_unequal_length",
          prompt_ids=np.arange(0, 10).reshape(2, 5),
          prompt_mask=np.ones((2, 5)),
          chosen_ids=np.arange(10, 20).reshape(2, 5),
          chosen_mask=np.ones((2, 5)),
          rejected_ids=np.arange(20, 26).reshape(2, 3),
          rejected_mask=np.ones((2, 3)),
      ),
  )
  def test_dpo_trainer(
      self,
      prompt_ids,
      prompt_mask,
      chosen_ids,
      chosen_mask,
      rejected_ids,
      rejected_mask,
  ):
    model = tc.ToyTransformer(rngs=nnx.Rngs(0))
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = tc.ToyTransformer(rngs=nnx.Rngs(0))
    dpo_config = dpo_lib.DpoTrainingConfig(
        eval_every_n_steps=10,
        max_steps=10,
    )
    dpo_trainer = dpo_lib.DpoTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_config,
    )
    train_ds = _dummy_dataset(
        MySource(np.arange(10)),
        prompt_ids,
        prompt_mask,
        chosen_ids,
        chosen_mask,
        rejected_ids,
        rejected_mask,
    )
    dpo_trainer.train(train_ds, None)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    for metric_name in [
        "chosen_rewards",
        "rejected_rewards",
        "rewards_margin",
        "rewards_accuracy",
    ]:
      self.assertLen(
          dpo_trainer.metrics_logger.get_metric_history(metric_name, "train"),
          dpo_trainer._train_steps,
      )

  def test_dpo_loss_fn(self):
    np.random.seed(0)
    model = tc.ToyTransformer(rngs=nnx.Rngs(0))
    per_token_logps = np.random.normal(0, 5, size=(8, 4))
    ref_per_token_logps = np.random.normal(0, 5, size=(8, 4)).sum(axis=-1)
    train_example = dpo_lib.TrainExample(
        input_ids=jnp.arange(0, 32).reshape(8, 4),
        positions=jnp.ones((8, 4)),
        attention_mask=jnp.ones((8, 4, 4)),
        ref_chosen_logps=ref_per_token_logps[:4],
        ref_rejected_logps=ref_per_token_logps[4:],
        logits_to_keep=4,
        completion_mask=jnp.ones((8, 4)),
    )

    with mock.patch.object(
        common, "get_per_token_logps", return_value=jnp.array(per_token_logps)
    ):
      loss, _ = dpo_lib.dpo_loss_fn(model, train_example, 0.1, 0)
      np.testing.assert_allclose(loss, 0.753059, atol=1e-5)

      loss, _ = dpo_lib.dpo_loss_fn(model, train_example, 0.1, 0.3)
      np.testing.assert_allclose(loss, 0.925447, atol=1e-5)

  def test_dpo_prepare_inputs(self):
    model = tc.ToyTransformer(rngs=nnx.Rngs(0))
    ref_model = tc.ToyTransformer(rngs=nnx.Rngs(0))
    dpo_trainer = dpo_lib.DpoTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_lib.DpoTrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
        ),
    )

    training_input = dpo_lib.TrainingInput(
        prompt_ids=np.array([[1, 2, 3, 4, 5], [0, 0, 1, 2, 3]]),
        prompt_mask=np.array([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
        chosen_ids=np.array([[10, 11, 12, 0], [13, 14, 15, 16]]),
        chosen_mask=np.array([[1, 1, 1, 0], [1, 1, 1, 1]]),
        rejected_ids=np.array([[20, 21, 22], [23, 0, 0]]),
        rejected_mask=np.array([[1, 1, 1], [1, 0, 0]]),
    )
    out = dpo_trainer._prepare_inputs(training_input)
    expected_input_ids = np.array([
        [1, 2, 3, 4, 5, 10, 11, 12, 0],
        [0, 0, 1, 2, 3, 13, 14, 15, 16],
        [1, 2, 3, 4, 5, 20, 21, 22, 0],
        [0, 0, 1, 2, 3, 23, 0, 0, 0],
    ])
    np.testing.assert_array_equal(out.input_ids, expected_input_ids)
    self.assertEqual(np.sum(out.attention_mask[0]), 44)
    self.assertEqual(np.sum(out.attention_mask[1]), 28)
    self.assertEqual(np.sum(out.attention_mask[2]), 44)
    self.assertEqual(np.sum(out.attention_mask[3]), 22)
    np.testing.assert_allclose(
        out.ref_chosen_logps, np.array([-20.536058, -20.905323]), atol=1e-5
    )
    np.testing.assert_allclose(
        out.ref_rejected_logps, np.array([-18.149311, -8.219014]), atol=1e-5
    )
    expected_completion_mask = np.array(
        [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]]
    )
    np.testing.assert_array_equal(out.completion_mask, expected_completion_mask)
    self.assertEqual(out.logits_to_keep, 4)


if __name__ == "__main__":
  absltest.main()
