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
from tunix.sft.dpo import dpo_trainer as orpo_lib
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
      lambda x: orpo_lib.TrainingInput(
          prompt_ids=prompt_ids,
          prompt_mask=prompt_mask,
          chosen_ids=chosen_ids,
          chosen_mask=chosen_mask,
          rejected_ids=rejected_ids,
          rejected_mask=rejected_mask,
      )
  )


def _dummy_string_dataset(
    source: MySource,
    prompts: list[str],
    chosen_responses: list[str],
    rejected_responses: list[str],
    return_dict=False,
):
  ds = grain.MapDataset.source(source)
  if return_dict:
    return ds.map(
        lambda x: {
            "prompts": prompts,
            "chosen_responses": chosen_responses,
            "rejected_responses": rejected_responses,
        }
    )
  else:
    return ds.map(
        lambda x: orpo_lib.DataInput(
            prompts=prompts,
            chosen_responses=chosen_responses,
            rejected_responses=rejected_responses,
        )
    )


class ORPOTrainerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="basic_training",
          prompt_ids=np.arange(0, 10).reshape(2, 5),
          prompt_mask=np.ones((2, 5)),
          chosen_ids=np.arange(10, 20).reshape(2, 5),
          chosen_mask=np.ones((2, 5)),
          rejected_ids=np.arange(20, 30).reshape(2, 5),
          rejected_mask=np.ones((2, 5)),
      ),
  )
  def test_orpo_trainer(
      self,
      prompt_ids,
      prompt_mask,
      chosen_ids,
      chosen_mask,
      rejected_ids,
      rejected_mask,
  ):
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    orpo_config = orpo_lib.ORPOTrainingConfig(
        algorithm="orpo",
        eval_every_n_steps=5,
        max_steps=10,
    )
    orpo_trainer = orpo_lib.ORPOTrainer(
        model=model,
        ref_model=None,
        optimizer=optax.sgd(1e-3),
        training_config=orpo_config,
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
    eval_ds = _dummy_dataset(
        MySource(np.arange(2)),
        prompt_ids,
        prompt_mask,
        chosen_ids,
        chosen_mask,
        rejected_ids,
        rejected_mask,
    )
    orpo_trainer.train(train_ds, eval_ds=eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    for metric_name in [
        "rewards/chosen",
        "rewards/rejected",
        "rewards/margin",
        "rewards/accuracy",
        "log_probs/chosen",
        "log_probs/rejected",
        "odds_ratio",
    ]:
      self.assertLen(
          orpo_trainer.metrics_logger.get_metric_history(  # pyrefly: ignore[missing-attribute]
              "", metric_name, "train"
          ),
          orpo_trainer._train_steps,
      )
      self.assertLen(
          orpo_trainer.metrics_logger.get_metric_history(  # pyrefly: ignore[missing-attribute]
              "", metric_name, "eval"
          ),
          3,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="dataclass_inputs",
          train_ds=_dummy_string_dataset(
              MySource(np.arange(10)),
              prompts=["Tunix", "Parallax"],
              chosen_responses=["PT", "distributed training"],
              rejected_responses=["optimizer library", "quantization"],
          ),
      ),
      dict(
          testcase_name="dict_inputs",
          train_ds=_dummy_string_dataset(
              MySource(np.arange(10)),
              prompts=["Tunix", "Parallax"],
              chosen_responses=["PT", "distributed training"],
              rejected_responses=["optimizer library", "quantization"],
              return_dict=True,
          ),
      ),
  )
  def test_orpo_trainer_with_string_inputs(self, train_ds):
    tokenizer = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    orpo_config = orpo_lib.ORPOTrainingConfig(
        algorithm="orpo",
        eval_every_n_steps=10,
        max_steps=10,
        max_prompt_length=3,
        max_response_length=3,
    )
    orpo_trainer = orpo_lib.ORPOTrainer(
        model=model,
        ref_model=None,
        optimizer=optax.sgd(1e-3),
        training_config=orpo_config,
        tokenizer=tokenizer,
    )
    orpo_trainer.train(train_ds, None)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    for metric_name in [
        "rewards/chosen",
        "rewards/rejected",
        "rewards/margin",
        "rewards/accuracy",
    ]:
      self.assertLen(
          orpo_trainer.metrics_logger.get_metric_history(  # pyrefly: ignore[missing-attribute]
              "", metric_name, "train"
          ),
          orpo_trainer._train_steps,
      )

  def test_orpo_loss_fn(self):
    """Test ORPO loss function directly with mocked logps."""
    np.random.seed(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    # Use negative log probs (as they should be in reality)
    per_token_logps = -np.abs(np.random.rand(8, 4))
    completion_mask = jnp.ones((8, 4))
    token_logps = (per_token_logps * completion_mask).sum(axis=-1)

    batch_size = token_logps.shape[0]
    chosen_logps = token_logps[: batch_size // 2]
    rejected_logps = token_logps[batch_size // 2 :]

    train_example = orpo_lib.TrainExample(
        input_ids=jnp.arange(0, 32).reshape(8, 4),
        positions=jnp.ones((8, 4)),
        attention_mask=jnp.ones((8, 4, 4)),
        ref_chosen_logps=None,
        ref_rejected_logps=None,
        completion_mask=completion_mask,
        logits_to_keep=4,
    )

    with mock.patch.object(
        orpo_lib,
        "compute_logps",
        return_value=(jnp.array(chosen_logps), jnp.array(rejected_logps), None),
    ):
      loss, aux = orpo_lib.dpo_loss_fn(
          model,
          train_example,
          algorithm="orpo",
          lambda_orpo=0.1,
          label_smoothing=0,
      )
      # Loss should be a scalar and finite
      self.assertEqual(loss.shape, ())
      self.assertTrue(jnp.isfinite(loss))

      # Check that aux metrics exist
      self.assertIn("rewards/chosen", aux)
      self.assertIn("rewards/rejected", aux)
      self.assertIn("rewards/margin", aux)
      self.assertIn("rewards/accuracy", aux)
      self.assertIn("log_probs/chosen", aux)
      self.assertIn("log_probs/rejected", aux)
      self.assertIn("odds_ratio", aux)

      # Check that accuracy is between 0 and 1
      self.assertGreaterEqual(aux["rewards/accuracy"], 0.0)
      self.assertLessEqual(aux["rewards/accuracy"], 1.0)

  def test_compute_logps_with_prompt_loss(self):
    """Test compute_logps directly to ensure correct slicing when enable_prompt_loss_orpo=True."""
    tokenizer = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    # 2 prompts (one is left-padded with 0), 2 chosen completions, 2 rejected completions
    # Prompt 1: [1, 2, 3, 4]
    # Prompt 2: [0, 1, 2, 3] (left-padded)
    # Completions (length 3):
    # Chosen 1: [10, 11, 0]
    # Chosen 2: [12, 13, 14]
    # Rejected 1: [20, 21, 0]
    # Rejected 2: [15, 0, 0]
    input_ids = jnp.array([
        [1, 2, 3, 4, 10, 11, 0],  # Prompt + Chosen
        [0, 1, 2, 3, 12, 13, 14],  # Prompt (left-padded) + Chosen
        [1, 2, 3, 4, 20, 21, 0],  # Prompt + Rejected
        [0, 1, 2, 3, 15, 0, 0],  # Prompt (left-padded) + Rejected
    ])

    completion_mask = jnp.array([
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
    ])

    full_mask = (input_ids != 0).astype(jnp.int32)
    positions = orpo_lib.common.build_positions_from_mask(full_mask)
    attention_mask = orpo_lib.common.make_causal_attn_mask(full_mask)

    chosen_logps, rejected_logps, prompt_chosen_logps = orpo_lib.compute_logps(
        model,
        input_ids,
        positions,
        attention_mask,
        logits_to_keep=3,
        completion_mask=completion_mask,
        enable_prompt_loss_orpo=True,
        full_mask=full_mask,
    )

    # Verify shapes
    self.assertEqual(chosen_logps.shape, (2,))
    self.assertEqual(rejected_logps.shape, (2,))
    self.assertEqual(prompt_chosen_logps.shape, (2,))

    # Let's also verify that they are finite
    self.assertTrue(jnp.all(jnp.isfinite(chosen_logps)))
    self.assertTrue(jnp.all(jnp.isfinite(rejected_logps)))
    self.assertTrue(jnp.all(jnp.isfinite(prompt_chosen_logps)))

    # Verify that prompt_chosen_logps (prompt + completion) is mathematically distinct
    # and strictly less than chosen_logps (completion only) due to summing negative prompt log-probabilities.
    self.assertFalse(jnp.allclose(chosen_logps, prompt_chosen_logps))
    self.assertTrue(jnp.all(prompt_chosen_logps < chosen_logps))

  def test_orpo_loss_fn_with_prompt_loss(self):
    """Test ORPO loss function directly with enable_prompt_loss_orpo=True."""
    np.random.seed(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    # Use negative log probs
    per_token_logps = -np.abs(np.random.rand(8, 4))
    completion_mask = jnp.ones((8, 4))
    token_logps = (per_token_logps * completion_mask).sum(axis=-1)

    batch_size = token_logps.shape[0]
    chosen_logps = token_logps[: batch_size // 2]
    rejected_logps = token_logps[batch_size // 2 :]
    prompt_chosen_logps = chosen_logps * 1.5

    train_example = orpo_lib.TrainExample(
        input_ids=jnp.arange(0, 32).reshape(8, 4),
        positions=jnp.ones((8, 4)),
        attention_mask=jnp.ones((8, 4, 4)),
        ref_chosen_logps=None,
        ref_rejected_logps=None,
        completion_mask=completion_mask,
        logits_to_keep=4,
        full_mask=jnp.ones((8, 8)),
    )

    with mock.patch.object(
        orpo_lib,
        "compute_logps",
        return_value=(
            jnp.array(chosen_logps),
            jnp.array(rejected_logps),
            jnp.array(prompt_chosen_logps),
        ),
    ):
      loss, aux = orpo_lib.dpo_loss_fn(
          model,
          train_example,
          algorithm="orpo",
          lambda_orpo=0.1,
          label_smoothing=0,
          enable_prompt_loss_orpo=True,
      )

      # Assert against mathematically-verified golden values
      self.assertEqual(loss.shape, ())
      self.assertTrue(jnp.isfinite(loss))
      np.testing.assert_allclose(loss, 3.494651, atol=1e-4)
      self.assertIn("sft_loss", aux)
      np.testing.assert_allclose(aux["sft_loss"], 3.423784, atol=1e-4)
      np.testing.assert_allclose(aux["or_loss"], 0.708663, atol=1e-4)

  def test_orpo_loss_fn_with_average_log_prob(self):
    """Test ORPO loss function directly with average_log_prob_orpo=True."""
    np.random.seed(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    # Use negative log probs
    per_token_logps = -np.abs(np.random.rand(8, 4))
    # Let's make completion masks have different lengths to truly test division
    completion_mask = jnp.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
    ])
    token_logps = (per_token_logps * completion_mask).sum(axis=-1)

    batch_size = token_logps.shape[0]
    chosen_logps = token_logps[: batch_size // 2]
    rejected_logps = token_logps[batch_size // 2 :]

    train_example = orpo_lib.TrainExample(
        input_ids=jnp.arange(0, 32).reshape(8, 4),
        positions=jnp.ones((8, 4)),
        attention_mask=jnp.ones((8, 4, 4)),
        ref_chosen_logps=None,
        ref_rejected_logps=None,
        completion_mask=completion_mask,
        logits_to_keep=4,
    )

    with mock.patch.object(
        orpo_lib,
        "compute_logps",
        return_value=(jnp.array(chosen_logps), jnp.array(rejected_logps), None),
    ):
      loss, aux = orpo_lib.dpo_loss_fn(
          model,
          train_example,
          algorithm="orpo",
          lambda_orpo=0.1,
          label_smoothing=0,
          average_log_prob_orpo=True,
      )

      # Assert against mathematically-verified golden values
      self.assertEqual(loss.shape, ())
      self.assertTrue(jnp.isfinite(loss))
      np.testing.assert_allclose(loss, 0.671129, atol=1e-4)
      self.assertIn("sft_loss", aux)
      np.testing.assert_allclose(aux["sft_loss"], 0.592339, atol=1e-4)
      np.testing.assert_allclose(aux["or_loss"], 0.787900, atol=1e-4)



  def test_orpo_prepare_inputs_for_strings(self):
    tokenizer = tc.MockVocab()

    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    orpo_trainer = orpo_lib.ORPOTrainer(
        model=model,
        ref_model=None,
        optimizer=optax.sgd(1e-3),
        training_config=orpo_lib.ORPOTrainingConfig(
            algorithm="orpo",
            eval_every_n_steps=10,
            max_steps=10,
            max_prompt_length=3,
            max_response_length=3,
        ),
        tokenizer=tokenizer,
    )

    # These are random strings, they hold no meaning.
    training_input = orpo_lib.DataInput(
        prompts=["Tunix", "Parallax"],
        chosen_responses=["PT", "distributed training"],
        rejected_responses=["optimizer library", "quantization"],
    )
    out = orpo_trainer._prepare_inputs(training_input)

    expected_input_ids = np.array([
        [0, 1, 14, 1, 16, 0],
        [0, 1, 15, 1, 18, 19],
        [0, 1, 14, 1, 20, 17],
        [0, 1, 15, 1, 21, 0],
    ])
    np.testing.assert_array_equal(out.input_ids, expected_input_ids)
    self.assertEqual(np.sum(out.attention_mask[0]), 14)
    self.assertEqual(np.sum(out.attention_mask[1]), 15)
    self.assertEqual(np.sum(out.attention_mask[2]), 15)
    self.assertEqual(np.sum(out.attention_mask[3]), 14)
    expected_completion_mask = np.array(
        [[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0]]
    )
    np.testing.assert_array_equal(out.completion_mask, expected_completion_mask)
    self.assertEqual(out.logits_to_keep, 3)

  def test_orpo_prepare_inputs(self):
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    orpo_trainer = orpo_lib.ORPOTrainer(
        model=model,
        ref_model=None,
        optimizer=optax.sgd(1e-3),
        training_config=orpo_lib.ORPOTrainingConfig(
            algorithm="orpo",
            eval_every_n_steps=10,
            max_steps=10,
        ),
    )

    training_input = orpo_lib.TrainingInput(
        prompt_ids=np.array([[1, 2, 3, 4, 5], [0, 0, 1, 2, 3]]),
        prompt_mask=np.array([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
        chosen_ids=np.array([[10, 11, 12, 0], [13, 14, 15, 16]]),
        chosen_mask=np.array([[1, 1, 1, 0], [1, 1, 1, 1]]),
        rejected_ids=np.array([[20, 21, 22, 0], [23, 0, 0, 0]]),
        rejected_mask=np.array([[1, 1, 1, 0], [1, 0, 0, 0]]),
    )
    out = orpo_trainer._prepare_inputs(training_input)
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
    expected_completion_mask = np.array(
        [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]]
    )
    np.testing.assert_array_equal(out.completion_mask, expected_completion_mask)
    self.assertEqual(out.logits_to_keep, 4)


if __name__ == "__main__":
  absltest.main()
