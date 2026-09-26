"""Tests for `rl_learner`."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import numpy as np
import optax
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import rl_learner
from tunix.rl.queue import data_queue

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


class ShuffleLearner(DummyLearner):
  def _generate_and_compute_advantage(self, training_input, mode):
    return training_input


_N = 4
_DATA = [f'{g}{i}' for g in 'AB' for i in range(_N)]


def _make_shuffle_learner(
    data_shuffle_seed=0, restored_iter_steps=0, restored_metadata=None
):
  engine = mock.MagicMock()
  engine.cluster_config.training_config = rl_engine_lib.RLTrainingConfig(
      actor_optimizer=optax.sgd(1e-3), eval_every_n_steps=1, max_steps=2
  )
  engine.actor_trainer.model = DummyModel()
  engine.rollout.model.return_value = DummyModel()
  engine.actor_trainer.iter_steps = restored_iter_steps
  restored_metadata = restored_metadata or {}
  engine.actor_trainer.restored_global_step.return_value = (
      restored_metadata.get('global_step', 0)
  )
  engine.actor_trainer.restored_checkpoint_metadata.return_value = (
      restored_metadata
  )
  return ShuffleLearner(
      rl_engine=engine,
      algo_config=DummyConfig(),
      reward_fns=lambda prompts, completions, **kwargs: [1.0] * len(prompts),
      data_shuffle_seed=data_shuffle_seed,
  )


def _prepare_mini_batch_step(learner, iterator):
  queue = data_queue.SimpleDataQueue(maxsize=0)
  learner._prepare_data(
      iterator,
      proceed_num_steps=_N,
      sample_repeat=1,
      batch_repeat=1,
      service_target_batch_size=1,
      data_queue=queue,
  )
  return [str(x[0]['prompts'][0]) for x in iter(queue.get, None)]


def _batches():
  return iter([{'prompts': np.array([x])} for x in _DATA])


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
    config = rl_engine_lib.RLTrainingConfig(
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
    mock_engine = mock.MagicMock()
    mock_engine.actor_trainer.model = actor_model
    mock_engine.rollout.model.return_value = rollout_model
    mock_engine.cluster_config.training_config = config
    mock_engine.actor_trainer.train_steps = 0
    mock_engine.actor_trainer.iter_steps = 0

    learner = DummyLearner(
        rl_engine=mock_engine,
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

  def test_resume_restores_data_shuffle_state(self):
    continuous = _make_shuffle_learner()
    iterator = _batches()
    _prepare_mini_batch_step(continuous, iterator)
    expected_state = np.asarray(continuous._data_shuffle_seed).copy()
    expected_order = _prepare_mini_batch_step(continuous, iterator)

    resumed = _make_shuffle_learner(
        restored_iter_steps=_N,
        restored_metadata={
            'global_step': 1,
            'role': 'actor',
            'data_shuffle_prng_state': expected_state.tolist(),
        },
    )
    np.testing.assert_array_equal(resumed._data_shuffle_seed, expected_state)
    actual_order = _prepare_mini_batch_step(resumed, _batches())
    self.assertCountEqual(actual_order, _DATA[_N:])
    self.assertEqual(actual_order, expected_order)

  def test_checkpoint_without_shuffle_state_starts_from_seed(self):
    learner = _make_shuffle_learner(
        restored_iter_steps=_N,
        restored_metadata={'global_step': 1, 'role': 'actor'},
    )
    np.testing.assert_array_equal(
        learner._data_shuffle_seed, jax.random.PRNGKey(0)
    )


if __name__ == '__main__':
  absltest.main()
