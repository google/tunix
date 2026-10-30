"""Tests for `rl_learner`."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import numpy as np
import optax
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import rl_learner
from tunix.rl.queue import data_queue as queue_lib


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

  def test_sequence_packing_ignores_train_micro_batch_size(self):
    config = rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        mini_batch_size=8,
        train_micro_batch_size=3,
        rollout_micro_batch_size=None,
        compute_logps_micro_batch_size=None,
        max_seq_token_per_tpu=128,
        eval_every_n_steps=1,
        max_steps=1,
    )

    actor_model = DummyModel()
    rollout_model = DummyModel()
    mock_engine = mock.MagicMock()
    mock_engine.actor_trainer.model = actor_model
    mock_engine.rollout.model.return_value = rollout_model
    mock_engine.cluster_config.training_config = config
    mock_engine.cluster_config.rollout_config.max_prompt_length = 32
    mock_engine.cluster_config.rollout_config.max_tokens_to_generate = 32
    mock_engine.actor_trainer.train_steps = 0
    mock_engine.actor_trainer.iter_steps = 0

    learner = DummyLearner(
        rl_engine=mock_engine,
        algo_config=DummyConfig(),
        reward_fns=lambda prompts, completions, **kwargs: [1.0] * len(prompts),
    )
    self.assertTrue(learner._packing_enabled)

    full_batch_size = 32
    train_ds = [{'prompts': [''] * full_batch_size}]
    with mock.patch.object(learner, '_run_global_step') as mock_global_step:
      mock_engine.actor_trainer.train_steps = 1
      learner.train(train_ds)

    self.assertEqual(learner._rollout_micro_batch_size, 8)
    self.assertEqual(learner._compute_logps_micro_batch_size, 8)
    mock_global_step.assert_called_once()
    # Arg 3 of _run_global_step is iterator_steps_per_mini_batch (8 // 8 == 1)
    self.assertEqual(mock_global_step.call_args.args[3], 1)

  def test_train_closes_engine_on_exception(self):
    config = rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        eval_every_n_steps=1,
        max_steps=10,
    )
    mock_engine = mock.MagicMock()
    mock_engine.actor_trainer.model = DummyModel()
    mock_engine.rollout.model.return_value = DummyModel()
    mock_engine.cluster_config.training_config = config
    mock_engine.actor_trainer.train_steps = 0
    mock_engine.actor_trainer.iter_steps = 0

    learner = DummyLearner(
        rl_engine=mock_engine,
        algo_config=DummyConfig(),
        reward_fns=lambda prompts, completions, **kwargs: [1.0] * len(prompts),
    )

    full_batch_size = 8
    train_ds = [{'prompts': [''] * full_batch_size}]

    with mock.patch.object(
        learner, '_run_global_step', side_effect=RuntimeError('Loop crash')
    ):
      with self.assertRaises(RuntimeError):
        learner.train(train_ds)

    mock_engine.close.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name='one_iteration',
          num_iterations=1,
          steps_per_mini_batch=1,
          restored_iter_steps=2,
          expected=['C'],
      ),
      dict(
          testcase_name='two_iterations',
          num_iterations=2,
          steps_per_mini_batch=1,
          restored_iter_steps=2,
          expected=['B', 'B'],
      ),
      dict(
          testcase_name='two_iterations_two_micro_batches',
          num_iterations=2,
          steps_per_mini_batch=2,
          restored_iter_steps=4,
          expected=['C', 'D', 'C', 'D'],
      ),
      dict(
          testcase_name='between_iterations_replays_mini_batch',
          num_iterations=2,
          steps_per_mini_batch=1,
          restored_iter_steps=1,
          expected=['A', 'A'],
      ),
      dict(
          testcase_name='mid_mini_batch_replays_mini_batch',
          num_iterations=2,
          steps_per_mini_batch=2,
          restored_iter_steps=2,
          expected=['A', 'B', 'A', 'B'],
      ),
      # Packed iter_steps don't track input micro-batches; train_steps does.
      dict(
          testcase_name='packing',
          num_iterations=2,
          steps_per_mini_batch=1,
          restored_iter_steps=7,
          restored_train_steps=2,
          packing_enabled=True,
          expected=['B', 'B'],
      ),
  )
  def test_resume_fast_forwards_consumed_micro_batches(
      self,
      num_iterations,
      steps_per_mini_batch,
      restored_iter_steps,
      expected,
      restored_train_steps=0,
      packing_enabled=False,
  ):
    class Learner(DummyLearner):

      def _generate_and_compute_advantage(self, training_input, mode):
        return training_input

      def _num_iterations(self):
        return num_iterations

    mock_engine = mock.MagicMock()
    mock_engine.cluster_config.training_config.max_seq_token_per_tpu = None
    mock_engine.actor_trainer.model = None
    mock_engine.actor_trainer.iter_steps = restored_iter_steps
    mock_engine.actor_trainer.train_steps = restored_train_steps
    learner = Learner(
        rl_engine=mock_engine,
        algo_config=DummyConfig(),
        reward_fns=lambda prompts, completions, **kwargs: [1.0] * len(prompts),
    )
    learner._packing_enabled = packing_enabled
    iterator = iter([{'prompts': np.array([x])} for x in 'ABCDEFGH'])

    def prepare():
      data_queue = queue_lib.SimpleDataQueue(maxsize=0)
      learner._prepare_data(
          iterator,
          proceed_num_steps=steps_per_mini_batch,
          sample_repeat=1,
          batch_repeat=num_iterations,
          service_target_batch_size=1,
          data_queue=data_queue,
      )
      return [str(x[0]['prompts'][0]) for x in iter(data_queue.get, None)]

    self.assertEqual(prepare(), expected)
    self.assertEqual(
        learner._iter_steps, restored_iter_steps + steps_per_mini_batch
    )
    # The fast-forward only happens once.
    next_expected = [chr(ord(x) + steps_per_mini_batch) for x in expected]
    self.assertEqual(prepare(), next_expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='zero_steps_per_mini_batch',
          steps_per_mini_batch=0,
          num_iterations=1,
      ),
      dict(
          testcase_name='zero_num_iterations',
          steps_per_mini_batch=1,
          num_iterations=0,
      ),
  )
  def test_num_micro_batches_to_skip_rejects_non_positive(
      self, steps_per_mini_batch, num_iterations
  ):
    mock_engine = mock.MagicMock()
    mock_engine.cluster_config.training_config.max_seq_token_per_tpu = None
    mock_engine.actor_trainer.model = None
    mock_engine.actor_trainer.iter_steps = 0
    mock_engine.actor_trainer.train_steps = 0
    learner = DummyLearner(
        rl_engine=mock_engine,
        algo_config=DummyConfig(),
        reward_fns=lambda prompts, completions, **kwargs: [1.0] * len(prompts),
    )
    with self.assertRaisesRegex(ValueError, 'Expected positive'):
      learner._num_micro_batches_to_skip(steps_per_mini_batch, num_iterations)

  @parameterized.parameters(([],), (None,))
  def test_empty_or_none_reward_fns_fails_fast_for_sequence_level(
      self, empty_fns
  ):
    config = rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        eval_every_n_steps=1,
        max_steps=1,
    )
    mock_engine = mock.MagicMock()
    mock_engine.actor_trainer.model = DummyModel()
    mock_engine.rollout.model.return_value = DummyModel()
    mock_engine.cluster_config.training_config = config

    with self.assertRaisesRegex(ValueError, 'reward_fns cannot be empty'):
      DummyLearner(
          rl_engine=mock_engine,
          algo_config=DummyConfig(),
          reward_fns=empty_fns,
      )

  def test_env_reward_mode_with_reward_fns_none_builds_manager(self):
    config = rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        eval_every_n_steps=1,
        max_steps=1,
    )
    mock_engine = mock.MagicMock()
    mock_engine.actor_trainer.model = DummyModel()
    mock_engine.rollout.model.return_value = DummyModel()
    mock_engine.cluster_config.training_config = config

    learner = DummyLearner(
        rl_engine=mock_engine,
        algo_config=DummyConfig(reward_mode='env'),
        reward_fns=None,
    )
    self.assertIsNotNone(learner.reward_manager)
    rewards = learner._compute_rewards(
        prompts=['p'],
        completions=['c'],
        mode=rl_engine_lib.Mode.TRAIN,
        trajectory_rewards=[3.5],
    )
    self.assertEqual(list(rewards), [3.5])

  def test_use_reward_model_skips_reward_manager_and_fails_on_compute(self):
    config = rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        eval_every_n_steps=1,
        max_steps=1,
    )
    mock_engine = mock.MagicMock()
    mock_engine.actor_trainer.model = DummyModel()
    mock_engine.rollout.model.return_value = DummyModel()
    mock_engine.cluster_config.training_config = config
    mock_engine.inference_worker._models = {'reward': DummyModel()}

    learner = DummyLearner(
        rl_engine=mock_engine,
        algo_config=DummyConfig(algo_variant='ppo'),
        reward_fns=None,
    )
    self.assertIsNone(learner.reward_manager)
    with self.assertRaisesRegex(
        ValueError, 'reward_manager is not initialized'
    ):
      learner._compute_rewards(
          prompts=['p'],
          completions=['c'],
          mode=rl_engine_lib.Mode.TRAIN,
      )


if __name__ == '__main__':
  absltest.main()
