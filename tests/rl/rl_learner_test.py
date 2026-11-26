"""Tests for `rl_learner`."""

import gc
import os
import threading
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

# Full collections the tests run themselves, e.g. in place of the one after
# the weight sync. The `gc.collect` spy in GcFreezeTest does not count them.
_real_collect = gc.collect
_real_freeze = gc.freeze
_real_unfreeze = gc.unfreeze
_GC_FREEZE_ENV_VARS = (
    'TUNIX_GC_FREEZE',
    'TUNIX_GC_FREEZE_AT',
    'TUNIX_GC_FREEZE_EVERY',
    'TUNIX_GC_FREEZE_REFREEZE',
    'TUNIX_GC_FREEZE_DISABLE',
    'TUNIX_GC_FREEZE_SAFETY',
)


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


def _learner_with_mock_engine(max_steps):
  config = rl_engine_lib.RLTrainingConfig(
      actor_optimizer=optax.sgd(1e-3),
      eval_every_n_steps=1,
      max_steps=max_steps,
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
  return learner, mock_engine


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

  def test_train_drives_gc_freeze_after_every_global_step(self):
    self.enter_context(mock.patch.dict(os.environ, {'TUNIX_GC_FREEZE': '1'}))
    gc_freeze_cls = self.enter_context(
        mock.patch.object(rl_learner, '_GcFreeze')
    )
    gc_freeze = gc_freeze_cls.return_value
    learner, mock_engine = _learner_with_mock_engine(max_steps=3)
    events = []
    gc_freeze.step_end.side_effect = lambda: events.append('step_end')
    gc_freeze.close.side_effect = lambda: events.append('gc_freeze.close')
    mock_engine.close.side_effect = lambda: events.append('engine.close')

    def run_global_step(*args, **kwargs):
      del args, kwargs
      events.append('global_step')
      mock_engine.actor_trainer.train_steps += 1

    with mock.patch.object(
        learner, '_run_global_step', side_effect=run_global_step
    ):
      learner.train([{'prompts': [''] * 8}])

    gc_freeze_cls.assert_called_once_with()
    self.assertEqual(
        events,
        ['global_step', 'step_end'] * 3 + ['gc_freeze.close', 'engine.close'],
    )

  def test_train_closes_gc_freeze_when_the_loop_raises(self):
    self.enter_context(mock.patch.dict(os.environ, {'TUNIX_GC_FREEZE': '1'}))
    gc_freeze = self.enter_context(
        mock.patch.object(rl_learner, '_GcFreeze')
    ).return_value
    learner, _ = _learner_with_mock_engine(max_steps=3)

    with mock.patch.object(
        learner, '_run_global_step', side_effect=RuntimeError('Loop crash')
    ):
      with self.assertRaises(RuntimeError):
        learner.train([{'prompts': [''] * 8}])

    gc_freeze.step_end.assert_not_called()
    gc_freeze.close.assert_called_once_with()

  def test_train_without_the_flag_does_not_touch_gc(self):
    self.enter_context(mock.patch.dict(os.environ))
    os.environ.pop('TUNIX_GC_FREEZE', None)
    gc_freeze_cls = self.enter_context(
        mock.patch.object(rl_learner, '_GcFreeze')
    )
    learner, mock_engine = _learner_with_mock_engine(max_steps=1)

    def run_global_step(*args, **kwargs):
      del args, kwargs
      mock_engine.actor_trainer.train_steps += 1

    with mock.patch.object(
        learner, '_run_global_step', side_effect=run_global_step
    ):
      learner.train([{'prompts': [''] * 8}])

    gc_freeze_cls.assert_not_called()


class GcFreezeTest(absltest.TestCase):
  """Drives `_GcFreeze` with patched env and checks the interpreter's GC state.

  Some CPython builds freeze objects at startup, so a freeze is checked against
  the freeze count in setUp. After a freeze, `close()` leaves the count at 0:
  `gc.unfreeze()` also releases what the interpreter froze. A frozen object
  freed by reference counting lowers the count, so where nothing should be
  frozen or unfrozen the tests check the `gc.freeze` and `gc.unfreeze` spies
  instead. `gc.collect` is wrapped to count the collections `_GcFreeze` runs.
  Each test restores the environment and, as far as possible, the GC state it
  started with.
  """

  def setUp(self):
    super().setUp()
    self.freeze_count_before = gc.get_freeze_count()
    self.gc_enabled_before = gc.isenabled()
    self.callbacks_before = list(gc.callbacks)
    self.enter_context(mock.patch.dict(os.environ))
    for name in _GC_FREEZE_ENV_VARS:
      os.environ.pop(name, None)
    self.collect = self.enter_context(
        mock.patch.object(gc, 'collect', wraps=_real_collect)
    )
    self.freeze = self.enter_context(
        mock.patch.object(gc, 'freeze', wraps=_real_freeze)
    )
    self.unfreeze = self.enter_context(
        mock.patch.object(gc, 'unfreeze', wraps=_real_unfreeze)
    )

  def tearDown(self):
    gc.callbacks[:] = self.callbacks_before
    if self.gc_enabled_before:
      gc.enable()
    else:
      gc.disable()
    # `close()` unfreezes everything, including what the interpreter froze at
    # startup. Freezing again freezes every live object rather than that exact
    # set, which cannot be recreated; only the tests in this class look at
    # freeze counts.
    if gc.get_freeze_count() != self.freeze_count_before:
      _real_unfreeze()
      if self.freeze_count_before:
        _real_freeze()
    super().tearDown()

  def _make_gc_freeze(self, env):
    os.environ.update(env)
    return rl_learner._GcFreeze()

  def test_freezes_after_warmup_and_close_restores_the_gc_state(self):
    gc.enable()
    gc_freeze = self._make_gc_freeze({'TUNIX_GC_FREEZE_AT': '3'})
    self.assertLen(gc.callbacks, len(self.callbacks_before) + 1)
    for _ in range(2):
      gc_freeze.step_end()
      self.assertTrue(gc.isenabled())
    self.collect.assert_not_called()
    self.freeze.assert_not_called()

    gc_freeze.step_end()

    self.collect.assert_called_once_with()
    self.freeze.assert_called_once_with()
    self.assertFalse(gc.isenabled())
    self.assertGreater(gc.get_freeze_count(), self.freeze_count_before)

    gc_freeze.close()

    self.assertTrue(gc.isenabled())
    self.assertEqual(gc.get_freeze_count(), 0)
    self.assertEqual(gc.callbacks, self.callbacks_before)

  def test_close_leaves_automatic_gc_off_if_it_was_off(self):
    gc.disable()
    gc_freeze = self._make_gc_freeze({'TUNIX_GC_FREEZE_AT': '1'})
    gc_freeze.step_end()
    self.assertGreater(gc.get_freeze_count(), self.freeze_count_before)

    gc_freeze.close()

    self.assertFalse(gc.isenabled())
    self.assertEqual(gc.get_freeze_count(), 0)
    self.assertEqual(gc.callbacks, self.callbacks_before)

  def test_close_before_the_freeze_only_removes_the_callback(self):
    gc.enable()
    gc_freeze = self._make_gc_freeze({'TUNIX_GC_FREEZE_AT': '5'})
    gc_freeze.step_end()

    gc_freeze.close()
    gc_freeze.close()  # Idempotent.

    self.collect.assert_not_called()
    # Nothing was frozen, so close() unfreezes nothing.
    self.freeze.assert_not_called()
    self.unfreeze.assert_not_called()
    self.assertTrue(gc.isenabled())
    self.assertEqual(gc.callbacks, self.callbacks_before)

  def test_disable_0_keeps_automatic_gc_on(self):
    gc.enable()
    gc_freeze = self._make_gc_freeze(
        {'TUNIX_GC_FREEZE_AT': '1', 'TUNIX_GC_FREEZE_DISABLE': '0'}
    )
    gc_freeze.step_end()

    self.assertTrue(gc.isenabled())
    self.assertGreater(gc.get_freeze_count(), self.freeze_count_before)
    gc_freeze.close()
    self.assertEqual(gc.get_freeze_count(), 0)

  def test_turns_automatic_gc_off_again_if_something_reenables_it(self):
    gc.enable()
    gc_freeze = self._make_gc_freeze({'TUNIX_GC_FREEZE_AT': '1'})
    gc_freeze.step_end()
    gc.enable()

    gc_freeze.step_end()

    self.assertFalse(gc.isenabled())
    gc_freeze.close()
    self.assertTrue(gc.isenabled())

  def test_collects_every_n_steps_after_the_freeze(self):
    gc_freeze = self._make_gc_freeze({
        'TUNIX_GC_FREEZE_AT': '1',
        'TUNIX_GC_FREEZE_EVERY': '2',
        'TUNIX_GC_FREEZE_SAFETY': '0',
    })
    gc_freeze.step_end()  # Freezes.
    self.collect.reset_mock()

    collections = []
    for _ in range(4):  # Steps 2 to 5.
      gc_freeze.step_end()
      collections.append(self.collect.call_count)

    self.assertEqual(collections, [1, 1, 2, 2])  # On steps 2 and 4.
    gc_freeze.close()

  def test_refreeze_unfreezes_collects_and_freezes_again(self):
    gc_freeze = self._make_gc_freeze({
        'TUNIX_GC_FREEZE_AT': '1',
        'TUNIX_GC_FREEZE_EVERY': '2',
        'TUNIX_GC_FREEZE_REFREEZE': '1',
    })
    gc_freeze.step_end()  # Freezes.
    calls = mock.Mock()
    calls.attach_mock(self.collect, 'collect')
    calls.attach_mock(self.freeze, 'freeze')
    calls.attach_mock(self.unfreeze, 'unfreeze')
    calls.reset_mock()

    gc_freeze.step_end()  # Step 2: periodic.

    self.assertEqual(
        calls.mock_calls,
        [mock.call.unfreeze(), mock.call.collect(), mock.call.freeze()],
    )
    # Every live object is frozen again, which can be fewer objects than in
    # setUp if frozen ones died in the meantime.
    self.assertGreater(gc.get_freeze_count(), 0)
    gc_freeze.close()
    self.assertEqual(gc.get_freeze_count(), 0)

  def test_safety_collect_only_in_steps_without_a_full_collection(self):
    gc_freeze = self._make_gc_freeze(
        {'TUNIX_GC_FREEZE_AT': '1', 'TUNIX_GC_FREEZE_EVERY': '100'}
    )
    gc_freeze.step_end()  # Freezes.
    self.collect.reset_mock()

    _real_collect()  # E.g. the collection after the weight sync.
    gc_freeze.step_end()
    self.collect.assert_not_called()

    gc_freeze.step_end()
    self.collect.assert_called_once_with()
    gc_freeze.close()

  def test_no_collection_after_the_freeze_with_safety_0(self):
    gc_freeze = self._make_gc_freeze({
        'TUNIX_GC_FREEZE_AT': '1',
        'TUNIX_GC_FREEZE_EVERY': '100',
        'TUNIX_GC_FREEZE_SAFETY': '0',
    })
    gc_freeze.step_end()  # Freezes.
    self.collect.reset_mock()

    for _ in range(3):
      gc_freeze.step_end()

    self.collect.assert_not_called()
    gc_freeze.close()

  def test_counts_a_full_collection_on_another_thread(self):
    gc_freeze = self._make_gc_freeze(
        {'TUNIX_GC_FREEZE_AT': '1', 'TUNIX_GC_FREEZE_EVERY': '100'}
    )
    gc_freeze.step_end()  # Freezes.
    self.collect.reset_mock()

    # E.g. a data loader thread.
    collector = threading.Thread(target=_real_collect)
    collector.start()
    collector.join()
    gc_freeze.step_end()

    self.collect.assert_not_called()
    gc_freeze.close()

  def test_young_generation_collection_is_not_a_full_collection(self):
    gc_freeze = self._make_gc_freeze(
        {'TUNIX_GC_FREEZE_AT': '1', 'TUNIX_GC_FREEZE_EVERY': '100'}
    )
    gc_freeze.step_end()  # Freezes.
    self.collect.reset_mock()

    _real_collect(0)
    gc_freeze.step_end()

    self.collect.assert_called_once_with()
    gc_freeze.close()

  def test_rejects_step_counts_below_1(self):
    for name in ('TUNIX_GC_FREEZE_AT', 'TUNIX_GC_FREEZE_EVERY'):
      with self.subTest(name=name):
        with mock.patch.dict(os.environ, {name: '0'}):
          with self.assertRaises(ValueError):
            rl_learner._GcFreeze()
    self.assertEqual(gc.callbacks, self.callbacks_before)

  def test_train_restores_the_gc_state(self):
    gc.enable()
    os.environ.update({'TUNIX_GC_FREEZE': '1', 'TUNIX_GC_FREEZE_AT': '1'})
    learner, mock_engine = _learner_with_mock_engine(max_steps=2)
    frozen_during_step = []

    def run_global_step(*args, **kwargs):
      del args, kwargs
      frozen_during_step.append(self.freeze.called)
      mock_engine.actor_trainer.train_steps += 1

    with mock.patch.object(
        learner, '_run_global_step', side_effect=run_global_step
    ):
      learner.train([{'prompts': [''] * 8}])

    # Frozen at the end of the first global step, restored when train()
    # returns.
    self.assertEqual(frozen_during_step, [False, True])
    self.assertTrue(gc.isenabled())
    self.assertEqual(gc.get_freeze_count(), 0)
    self.assertEqual(gc.callbacks, self.callbacks_before)


if __name__ == '__main__':
  absltest.main()
