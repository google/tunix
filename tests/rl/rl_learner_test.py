"""Tests for `rl_learner`."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import optax
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner

class DummyModel(nnx.Module):
  pass


class DummyConfig(algo_config_lib.AlgorithmConfig):
  pass


class DummyLearner(rl_learner.RLLearner[DummyConfig]):

  def _generate_and_compute_advantage(self, training_input, mode):
    pass

  def _compute_trajectory_ids(self, example, steps):
    return [''] * len(example['prompts'])

  def _num_iterations(self):
    return 1

  def _num_generations(self):
    return 1


class RLLearnerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('1', None, None, None, None, [32, 32], False),
      ('2', 8, None, None, None, [8, 8], False),
      ('3', 8, 2, None, None, [2, 2], False),
      ('4', 8, 4, 4, 4, [4, 4], False),
      ('5', 8, 4, 3, 4, [], True),
      ('6', 8, 4, 4, 3, [], True),
  )
  def test_micro_batching(
      self,
      mini_batch_size,
      train_micro_batch_size,
      rollout_micro_batch_size,
      compute_logps_micro_batch_size,
      expected_values,
      expect_failure,
  ):
    config = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        mini_batch_size=mini_batch_size,
        train_micro_batch_size=train_micro_batch_size,
        rollout_micro_batch_size=rollout_micro_batch_size,
        compute_logps_micro_batch_size=compute_logps_micro_batch_size,
        eval_every_n_steps=1,
        max_steps=1,
    )

    actor_model = DummyModel()
    rollout_model = DummyModel()
    mock_cluster = mock.MagicMock()
    mock_cluster.actor_trainer.model = actor_model
    mock_cluster.rollout.model.return_value = rollout_model
    mock_cluster.cluster_config.training_config = config
    mock_cluster.actor_trainer.train_steps = 0
    mock_cluster.actor_trainer.iter_steps = 0

    learner = DummyLearner(
        rl_cluster=mock_cluster,
        algo_config=DummyConfig(),
        reward_fns=lambda prompts, completions, **kwargs: [1.0] * len(prompts),
    )

    full_batch_size = 32
    train_ds = [{'prompts': [''] * full_batch_size}]

    if expect_failure:
      with self.assertRaises(ValueError):
        learner.train(train_ds)
    else:
      learner.train(train_ds)
      (
          expected_rollout_micro,
          expected_compute_logps_micro,
      ) = expected_values

      self.assertEqual(
          learner._rollout_micro_batch_size, expected_rollout_micro
      )
      self.assertEqual(
          learner._compute_logps_micro_batch_size, expected_compute_logps_micro
      )


if __name__ == '__main__':
  absltest.main()
