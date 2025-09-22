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
from tunix.sft.dpo import vl_dpo_trainer as vl_dpo_lib
from tunix.sft.dpo import dpo_trainer as dpo_lib
from tunix.tests import test_common as tc


# Patch ToyTransformer to accept pixel_values
class PatchedToyTransformer(tc.ToyTransformer):

  def __call__(
      self,
      input_tokens,
      positions=None,
      attention_mask=None,
      cache=None,
      pixel_values=None,
  ):
    return super().__call__(
        input_tokens,
        positions=positions,
        attention_mask=attention_mask,
        cache=cache,
    )


class MySource(grain.RandomAccessDataSource):

  def __init__(self, data):
    self._data = data

  def __getitem__(self, idx):
    return self._data[idx]

  def __len__(self):
    return len(self._data)


def _dummy_dataset_with_images(
    source: MySource,
    prompt_ids: np.ndarray,
    prompt_mask: np.ndarray,
    chosen_ids: np.ndarray,
    chosen_mask: np.ndarray,
    rejected_ids: np.ndarray,
    rejected_mask: np.ndarray,
    images: np.ndarray,
):
  return grain.MapDataset.source(source).map(
      lambda x: vl_dpo_lib.VLMTrainingInput(
          prompt_ids=prompt_ids,
          prompt_mask=prompt_mask,
          chosen_ids=chosen_ids,
          chosen_mask=chosen_mask,
          rejected_ids=rejected_ids,
          rejected_mask=rejected_mask,
          pixel_values=images,
      )
  )


class VlDpoTrainerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="with_images",
          prompt_ids=np.arange(0, 10).reshape(2, 5),
          prompt_mask=np.ones((2, 5)),
          chosen_ids=np.arange(10, 20).reshape(2, 5),
          chosen_mask=np.ones((2, 5)),
          rejected_ids=np.arange(20, 30).reshape(2, 5),
          rejected_mask=np.ones((2, 5)),
          images=np.random.rand(2, 3, 224, 224),  # Example image embeddings
      ),
      dict(
          testcase_name="without_images",
          prompt_ids=np.arange(0, 10).reshape(2, 5),
          prompt_mask=np.ones((2, 5)),
          chosen_ids=np.arange(10, 20).reshape(2, 5),
          chosen_mask=np.ones((2, 5)),
          rejected_ids=np.arange(20, 30).reshape(2, 5),
          rejected_mask=np.ones((2, 5)),
          images=None,
      ),
  )
  def test_vl_dpo_trainer(
      self,
      prompt_ids,
      prompt_mask,
      chosen_ids,
      chosen_mask,
      rejected_ids,
      rejected_mask,
      images,
  ):
    model = PatchedToyTransformer(rngs=nnx.Rngs(0))
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = PatchedToyTransformer(rngs=nnx.Rngs(0))
    dpo_config = vl_dpo_lib.VlmDpoTrainingConfig(
        eval_every_n_steps=10,
        max_steps=10,
    )
    vl_dpo_trainer = vl_dpo_lib.VLM_DpoTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_config,
    )
    train_ds = _dummy_dataset_with_images(
        MySource(np.arange(10)),
        prompt_ids,
        prompt_mask,
        chosen_ids,
        chosen_mask,
        rejected_ids,
        rejected_mask,
        images if images is not None else np.zeros((2, 3, 224, 224)),
    )
    vl_dpo_trainer.train(train_ds, None)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    for metric_name in [
        "chosen_rewards",
        "rejected_rewards",
        "rewards_margin",
        "rewards_accuracy",
    ]:
      self.assertLen(
          vl_dpo_trainer.metrics_logger.get_metric_history(
              metric_name, "train"
          ),
          vl_dpo_trainer._train_steps,
      )

  def test_vl_dpo_loss_fn(self):
    np.random.seed(0)
    model = PatchedToyTransformer(rngs=nnx.Rngs(0))
    per_token_logps = np.random.normal(0, 5, size=(8, 4))
    ref_per_token_logps = np.random.normal(0, 5, size=(8, 4)).sum(axis=-1)
    train_example = vl_dpo_lib.VLMTrainExample(
        input_ids=jnp.arange(0, 32).reshape(8, 4),
        positions=jnp.ones((8, 4)),
        attention_mask=jnp.ones((8, 4, 4)),
        ref_chosen_logps=ref_per_token_logps[:4],
        ref_rejected_logps=ref_per_token_logps[4:],
        completion_mask=jnp.ones((8, 4)),
        pixel_values=jnp.ones((8, 3, 224, 224)),  # Example image embeddings
    )
    with mock.patch.object(
        common, "get_per_token_logps", return_value=jnp.array(per_token_logps)
    ):
      loss, _ = vl_dpo_lib.vl_dpo_loss_fn(model, train_example, 0.1, 0)
      np.testing.assert_allclose(loss, 0.753059, atol=1e-5)
      loss, _ = vl_dpo_lib.vl_dpo_loss_fn(model, train_example, 0.1, 0.3)
      np.testing.assert_allclose(loss, 0.925447, atol=1e-5)

  def test_vl_dpo_prepare_inputs(self):
    model = PatchedToyTransformer(rngs=nnx.Rngs(0))
    ref_model = PatchedToyTransformer(rngs=nnx.Rngs(0))
    vl_dpo_trainer = vl_dpo_lib.VLM_DpoTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=vl_dpo_lib.VlmDpoTrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
        ),
    )
    training_input = vl_dpo_lib.VLMTrainingInput(
        prompt_ids=np.array([[1, 2, 3, 4, 5], [0, 0, 1, 2, 3]]),
        prompt_mask=np.array([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
        chosen_ids=np.array([[10, 11, 12, 0], [13, 14, 15, 16]]),
        chosen_mask=np.array([[1, 1, 1, 0], [1, 1, 1, 1]]),
        rejected_ids=np.array([[20, 21, 22], [23, 0, 0]]),
        rejected_mask=np.array([[1, 1, 1], [1, 0, 0]]),
        pixel_values=np.random.rand(2, 3, 224, 224),  # Example image embeddings
    )
    out = vl_dpo_trainer._prepare_inputs(training_input)
    self.assertEqual(
        out.pixel_values.shape, (4, 3, 224, 224)
    )  # Check image duplication

  def test_vl_dpo_vs_dpo_no_images(self):
    model = PatchedToyTransformer(rngs=nnx.Rngs(0))
    ref_model = PatchedToyTransformer(rngs=nnx.Rngs(0))
    dpo_trainer = dpo_lib.DpoTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_lib.DpoTrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
        ),
    )
    vl_dpo_trainer = vl_dpo_lib.VLM_DpoTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=vl_dpo_lib.VlmDpoTrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
        ),
    )
    training_input = vl_dpo_lib.VLMTrainingInput(
        prompt_ids=np.array([[1, 2, 3, 4, 5], [0, 0, 1, 2, 3]]),
        prompt_mask=np.array([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
        chosen_ids=np.array([[10, 11, 12, 0], [13, 14, 15, 16]]),
        chosen_mask=np.array([[1, 1, 1, 0], [1, 1, 1, 1]]),
        rejected_ids=np.array([[20, 21, 22], [23, 0, 0]]),
        rejected_mask=np.array([[1, 1, 1], [1, 0, 0]]),
        pixel_values=np.zeros((2, 3, 224, 224)),  # No images
    )

    dpo_out = dpo_trainer._prepare_inputs(training_input)
    vl_dpo_out = vl_dpo_trainer._prepare_inputs(training_input)

    np.testing.assert_array_equal(dpo_out.input_ids, vl_dpo_out.input_ids)
    np.testing.assert_array_equal(
        dpo_out.attention_mask, vl_dpo_out.attention_mask
    )


if __name__ == "__main__":
  absltest.main()
