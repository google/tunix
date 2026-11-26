"""Tests for `rl_learner`."""

import contextlib
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import numpy as np
import optax
from tunix.common import test_utils
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import rl_cluster as rl_engine_lib
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


_PER_HOST_SWITCHES = ('TUNIX_PER_HOST_LEARNER', 'TUNIX_PER_HOST_SAFE_LOGPS')


@contextlib.contextmanager
def _per_host_env(process_count, **switches):
  """Sets exactly `switches` and fakes `process_count` JAX processes."""
  with mock.patch.dict(os.environ), mock.patch.object(
      rl_learner.jax, 'process_count', return_value=process_count
  ):
    for name in _PER_HOST_SWITCHES:
      os.environ.pop(name, None)
    os.environ.update(switches)
    yield


def _mesh(fsdp):
  mesh = mock.MagicMock()
  mesh.shape = {'fsdp': fsdp}
  return mesh


def _mock_engine(actor_mesh, rollout_mesh, **training_config):
  config = rl_engine_lib.RLTrainingConfig(
      actor_optimizer=optax.sgd(1e-3),
      eval_every_n_steps=1,
      max_steps=1,
      **training_config,
  )
  engine = mock.MagicMock()
  engine.actor_trainer.model = DummyModel()
  engine.rollout.model.return_value = DummyModel()
  engine.cluster_config.training_config = config
  engine.cluster_config.role_to_mesh = {
      rl_engine_lib.Role.ACTOR: actor_mesh,
      rl_engine_lib.Role.ROLLOUT: rollout_mesh,
  }
  # Separate mesh objects stand for actor and rollout on different devices.
  engine.is_colocated.return_value = actor_mesh is rollout_mesh
  engine.actor_trainer.train_steps = 0
  engine.actor_trainer.iter_steps = 0
  return engine


def _learner(engine):
  return DummyLearner(
      rl_engine=engine,
      algo_config=DummyConfig(),
      reward_fns=lambda prompts, completions, **kwargs: [1.0] * len(prompts),
  )


class PerHostLearnerTest(parameterized.TestCase):
  """The per-host learner (TUNIX_PER_HOST_LEARNER=1) with a mock engine."""

  def test_separate_meshes_load_data_synchronously(self):
    engine = _mock_engine(_mesh(2), _mesh(2))
    with _per_host_env(2):
      learner = _learner(engine)
    # Not colocated: asynchronous rollout is on without the per-host learner.
    self.assertTrue(learner.can_enable_async_rollout)

    def fake_submit(fn, **kwargs):
      del fn
      kwargs['data_queue'].put(None)  # An empty mini-batch.
      return mock.MagicMock()

    engine.actor_trainer.train_steps = 1  # Stop after one global step.
    with _per_host_env(2, TUNIX_PER_HOST_LEARNER='1'), mock.patch.object(
        learner.executor, 'submit', side_effect=fake_submit
    ) as submit:
      with self.assertLogs(level='WARNING') as logs:
        learner.train([{'prompts': [''] * 8}])
    self.assertFalse(learner.can_enable_async_rollout)
    self.assertIn('asynchronous rollout is turned off', '\n'.join(logs.output))
    submit.assert_called_once()
    self.assertIs(submit.call_args.kwargs['async_loading'], False)

  def test_shard_batch_by_process_slices_tensor_like_columns(self):
    values = np.arange(4, dtype=np.float32)
    batch = {
        'prompts': ['a', 'b', 'c', 'd'],
        'array': values,
        'tensor': test_utils.CustomTensor(values.reshape(4, 1)),
        'scalar': test_utils.CustomTensor(np.array(0.5, dtype=np.float32)),
        'tag': 'shared',
    }
    local = rl_learner.shard_batch_by_process(batch, 1, 2)
    self.assertEqual(local['prompts'], ['b', 'd'])
    np.testing.assert_array_equal(local['array'], [1.0, 3.0])
    np.testing.assert_array_equal(local['tensor'], [[1.0], [3.0]])
    self.assertIs(local['scalar'], batch['scalar'])  # Not batch-aligned.
    self.assertEqual(local['tag'], 'shared')
    # Explicit rows, e.g. a length-balanced shard.
    local = rl_learner.shard_batch_by_process(batch, 0, 2, indices=[0, 3])
    self.assertEqual(local['prompts'], ['a', 'd'])
    np.testing.assert_array_equal(local['tensor'], [[0.0], [3.0]])

  @parameterized.named_parameters(
      ('none', {}, []),
      (
          'trainer_logps',
          {'use_rollout_logps': False},
          ['trainer logprobs (use_rollout_logps=False)'],
      ),
      # GRPOLearner skips trainer logprobs under score centering.
      (
          'score_centering',
          {'use_rollout_logps': False, 'score_centering': True},
          [],
      ),
  )
  def test_trainer_side_logps_mirror_grpo(self, config, want):
    mesh = _mesh(2)
    learner = DummyLearner(
        rl_engine=_mock_engine(mesh, mesh),
        algo_config=DummyConfig(**config),
        reward_fns=lambda prompts, completions, **kwargs: [1.0] * len(prompts),
    )
    self.assertEqual(learner._per_host_trainer_side_logps(), want)

  @parameterized.named_parameters(
      # TUNIX_PER_HOST_SAFE_LOGPS=1: rows x 2 processes must divide fsdp=4.
      ('global_rows', {'TUNIX_PER_HOST_SAFE_LOGPS': '1'}, 4),
      # Otherwise the process-local rows must divide fsdp=2 by themselves.
      ('process_local_rows', {}, 2),
  )
  def test_eval_batches_must_divide_the_data_axis(self, switches, fsdp):
    mesh = _mesh(fsdp)
    with _per_host_env(2, TUNIX_PER_HOST_LEARNER='1', **switches):
      learner = _learner(_mock_engine(mesh, mesh))
      # 4 prompts: 2 rows per process.
      local = learner._shard_eval_batch_for_process(
          {'prompts': ['a', 'b', 'c', 'd']}
      )
      self.assertEqual(list(local['prompts']), ['a', 'c'])
      # A last eval batch of 2 prompts: 1 row per process.
      with self.assertRaisesRegex(ValueError, 'eval batch has 1 rows'):
        learner._shard_eval_batch_for_process({'prompts': ['e', 'f']})

  def test_train_checks_eval_batches_and_unbatched_train_tails(self):
    mesh = _mesh(4)
    engine = _mock_engine(mesh, mesh)
    with _per_host_env(
        2, TUNIX_PER_HOST_LEARNER='1', TUNIX_PER_HOST_SAFE_LOGPS='1'
    ):
      learner = _learner(engine)
      engine.actor_trainer.train_steps = 1  # Stop after one global step.
      with mock.patch.object(learner, '_run_global_step') as run_global_step:
        learner.train(
            [{'prompts': [''] * 8}, {'prompts': [''] * 2}],
            [{'prompts': [''] * 4}, {'prompts': [''] * 2}],
        )
      train_iterator, eval_ds = run_global_step.call_args.args[4:6]
      # The configured train micro-batch (8 prompts) is fine, the smaller last
      # batch is not.
      self.assertLen(next(train_iterator)['prompts'], 4)
      with self.assertRaisesRegex(ValueError, 'train batch has 1 rows'):
        next(train_iterator)
      eval_batches = iter(eval_ds)
      self.assertLen(next(eval_batches)['prompts'], 2)
      with self.assertRaisesRegex(ValueError, 'eval batch has 1 rows'):
        next(eval_batches)

  def test_packing_skips_the_train_batch_row_checks(self):
    # 2 processes, fsdp=8: 2 prompts per process are 4 global rows, which
    # `shard_input` would replicate, so unpacked training is refused. Packed
    # chunks hold a multiple of `per_host_pack_unit` rows instead.
    mesh = _mesh(8)
    batches = [{'prompts': [''] * 4}]
    with _per_host_env(
        2, TUNIX_PER_HOST_LEARNER='1', TUNIX_PER_HOST_SAFE_LOGPS='1'
    ):
      with self.assertRaisesRegex(ValueError, 'train micro-batch has 2 rows'):
        _learner(_mock_engine(mesh, mesh)).train(batches)

      engine = _mock_engine(mesh, mesh, max_seq_tokens_per_device=64)
      engine.cluster_config.rollout_config = mock.MagicMock(
          max_prompt_length=8, max_tokens_to_generate=8
      )
      learner = _learner(engine)
      engine.actor_trainer.train_steps = 1  # Stop after one global step.
      with mock.patch.object(learner, '_run_global_step') as run_global_step:
        learner.train(batches)
      train_iterator = run_global_step.call_args.args[4]
      self.assertLen(next(train_iterator)['prompts'], 2)
      # Eval batches are not packed: they keep their check.
      with self.assertRaisesRegex(ValueError, 'eval batch has 1 rows'):
        learner._shard_eval_batch_for_process({'prompts': ['e', 'f']})


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
        max_seq_tokens_per_device=128,
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

  def test_create_micro_batch_iterator_flattens_tensor_like(self):
    learner = DummyLearner.__new__(DummyLearner)
    values = np.arange(4, dtype=np.float32)
    batch = {
        'prompts': ['a', 'b', 'c', 'd'],
        'vector': test_utils.CustomTensor(values),
        'column': test_utils.CustomTensor(values.reshape(4, 1)),
    }
    micro_batches = list(
        learner._create_micro_batch_iterator(iter([batch]), 2)
    )
    self.assertLen(micro_batches, 2)
    self.assertEqual(list(micro_batches[1]['prompts']), ['c', 'd'])
    for key in ('vector', 'column'):
      flattened = np.concatenate([mb[key] for mb in micro_batches])
      np.testing.assert_array_equal(flattened, [0.0, 1.0, 2.0, 3.0])
      self.assertEqual(micro_batches[0][key].shape, (2,))
      self.assertEqual(micro_batches[0][key].dtype, np.float32)

    scalar_batches = [
        {
            'prompts': 'p0',
            'weight': test_utils.CustomTensor(np.array(0.5, dtype=np.float32)),
            'step': test_utils.CustomTensor(np.array(3, dtype=np.int32)),
        },
        {
            'prompts': 'p1',
            'weight': test_utils.CustomTensor(np.array(1.5, dtype=np.float32)),
            'step': test_utils.CustomTensor(np.array(4, dtype=np.int32)),
        },
    ]
    scalar_out = list(
        learner._create_micro_batch_iterator(iter(scalar_batches), 2)
    )
    self.assertLen(scalar_out, 1)
    np.testing.assert_array_equal(scalar_out[0]['weight'], [0.5, 1.5])
    self.assertEqual(scalar_out[0]['weight'].dtype, np.float32)
    np.testing.assert_array_equal(scalar_out[0]['step'], [3, 4])
    self.assertEqual(scalar_out[0]['step'].dtype, np.int32)


if __name__ == '__main__':
  absltest.main()
