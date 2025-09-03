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

import itertools
from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
from grain import python as grain
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import optax
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.ppo import ppo_learner as ppo_lib
from tunix.rl.queue import data_queue as queue_lib
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc
from typing_extensions import override


_DUMMY_DATA = [
    'input string',
    'hello world',
    'My name',
    'hello there',
]


def reward_1(completions, **kargs):  # pylint: disable=unused-argument
  return jnp.arange(len(completions))


def reward_2(prompts, answer, **kargs):  # pylint: disable=unused-argument
  return jnp.arange(len(answer))


class MySource(grain.RandomAccessDataSource):

  def __init__(self, data=None, repeat=1):
    if data is None:
      data = _DUMMY_DATA
    self._data = data * repeat

  def __getitem__(self, idx):
    return self._data[idx]

  def __len__(self):
    return len(self._data)


def _dummy_dataset(source=MySource(), batch_size: int = 1):
  return (
      grain.MapDataset.source(source)
      .batch(batch_size)
      .map(lambda x: {'prompts': x, 'answer': x})
  )


class PpoLearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_cpus = 2
    chex.set_n_cpu_devices(self.num_cpus)
    assert len(jax.devices()) == self.num_cpus

  def test_iterator(self):

    class _EmptyTrainer(ppo_lib.PpoLearner):
      """A trainer used to test the iterator preparation."""

      def __init__(self):
        self._data_shuffle_key = jax.random.PRNGKey(42)
        self.rollout_worker_mesh = pxla.thread_resources.env.physical_mesh
        self._train_steps = 0
        self._eval_steps = 0
        self._last_train_step = 0

      @override
      def _generate_and_compute_advantage(self, example, mode='train'):
        return example

    empty_trainer = _EmptyTrainer()

    def _prepare(dataset, batch_repeat, mini_batch_size=None):
      iterator = iter(dataset)
      while True:
        try:
          data_queue = queue_lib.SimpleDataQueue(maxsize=2)
          empty_trainer._prepare_data(
              iterator=iterator,
              micro_batch_size=mini_batch_size,
              batch_repeat=batch_repeat,
              data_queue=data_queue,
              async_loading=False,
              training=True,
              shuffle_data=True,
          )
          yield data_queue.get(block=True)
        except StopIteration:
          break

    # Case 1: batch repeat = 1, grad_acc_steps = 1
    dataset = _dummy_dataset([i for i in range(8)], 4)
    res = [
        d.get('prompts').tolist()
        for d in itertools.chain.from_iterable(_prepare(dataset, 1))
    ]
    expected = [[3, 2, 1, 0], [5, 6, 4, 7]]
    self.assertEqual(res, expected)

    # Case 2: batch repeat = 3, grad_acc_steps = 1
    dataset = _dummy_dataset([i for i in range(8)], 4)
    res = [
        d.get('prompts').tolist()
        for d in itertools.chain.from_iterable(_prepare(dataset, 3))
    ]
    expected = [
        [3, 1, 2, 0],
        [1, 2, 0, 3],
        [0, 1, 3, 2],
        [5, 6, 4, 7],
        [4, 5, 7, 6],
        [5, 7, 4, 6],
    ]
    self.assertEqual(res, expected)

    # Case 3: batch repeat = 1, mini_batch_size = 2
    dataset = _dummy_dataset([i for i in range(8)], 4)
    res = [
        d.get('prompts').tolist()
        for d in itertools.chain.from_iterable(_prepare(dataset, 1, 2))
    ]
    expected = [[0, 1], [3, 2], [5, 7], [4, 6]]
    self.assertEqual(res, expected)

    # Case 4: batch repeat = 3, grad_acc_steps = 1, mini_batch_size = 2
    dataset = _dummy_dataset([i for i in range(8)], 4)
    res = [
        d.get('prompts').tolist()
        for d in itertools.chain.from_iterable(_prepare(dataset, 3, 2))
    ]
    expected = [
        [3, 1],
        [2, 0],
        [1, 0],
        [2, 3],
        [0, 3],
        [1, 2],
        [5, 4],
        [6, 7],
        [4, 7],
        [5, 6],
        [5, 6],
        [4, 7],
    ]
    self.assertEqual(res, expected)

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_reward_model',
          'use_reward_model': True,
          'reward_fns': None,
          'use_different_rollout_config': True,
      },
      {
          'testcase_name': 'with_reward_fn',
          'use_reward_model': False,
          'reward_fns': [reward_1, reward_2],
          'use_different_rollout_config': False,
      },
      {
          'testcase_name': 'with_reward_model_diff_rollout_config',
          'use_reward_model': True,
          'reward_fns': None,
          'use_different_rollout_config': True,
      },
  )
  def test_ppo_learner(
      self, use_reward_model, reward_fns, use_different_rollout_config
  ):
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    original_model_variables = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.Param)
    )

    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    value_model = utils.create_critic_model(model)
    var_filter = nnx.All(nnx.Param, lambda path, x: 'output' not in path)
    original_value_model_variables = jax.tree.map(
        jnp.copy, nnx.state(value_model, var_filter)
    )

    if use_reward_model:
      reward_model = tc.ToyTransformer(
          rngs=nnx.Rngs(2), vocab_size=vocab.GetPieceSize()
      )
      reward_model = tc.MockTransformerWithScoreHead(reward_model, nnx.Rngs(1))

    mesh = pxla.thread_resources.env.physical_mesh
    default_rollout_config = base_rollout.RolloutConfig(
        max_tokens_to_generate=10,
        max_prompt_length=256,
        kv_cache_size=1024,
    )
    if use_different_rollout_config:
      another_rollout_config = base_rollout.RolloutConfig(
          max_tokens_to_generate=10,
          max_prompt_length=256,
          kv_cache_size=1024,
          temperature=0.5,
      )
      rollout_config = {
          rl_cluster_lib.Mode.TRAIN: default_rollout_config,
          rl_cluster_lib.Mode.EVAL: another_rollout_config,
      }
    else:
      rollout_config = default_rollout_config
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
            rl_cluster_lib.Role.REWARD: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=2,
            max_steps=10,
            gradient_accumulation_steps=1,
        ),
        rollout_config=rollout_config,
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        reward=reward_model if use_reward_model else None,  # pylint: disable=undefined-variable
        tokenizer=vocab,
        cluster_config=cluster_config,
    )
    ppo_config = ppo_lib.PpoConfig(num_ppo_epochs=1)
    ppo_learner = ppo_lib.PpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        ppo_config=ppo_config,
    )
    self.assertFalse(ppo_learner.should_sync_weights)
    self.assertFalse(ppo_learner.can_enable_async_rollout)

    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    eval_ds = _dummy_dataset(batch_size=2)
    ppo_learner.train(train_ds, eval_ds)

    model_variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        tc.assert_not_equal, original_model_variables, model_variables
    )
    value_model_variables = nnx.state(value_model, var_filter)

    jax.tree.map_with_path(
        tc.assert_not_equal,
        original_value_model_variables,
        value_model_variables,
    )

    self.assertEqual(ppo_learner._train_steps, 10)
    # max_steps / eval_every_n_steps * (#_rows_in_eval_ds / eval_batch_size)
    # = 10 / 2 * (4 / 2) = 10
    self.assertEqual(ppo_learner._eval_steps, 10)
    self.assertEqual(
        ppo_learner.rl_cluster.actor_trainer._train_steps,
        ppo_learner._train_steps,
    )
    self.assertEqual(
        ppo_learner.rl_cluster.critic_trainer._train_steps,
        ppo_learner._train_steps,
    )

    actor_metric_logger = ppo_learner._actor_metrics_logger
    self.assertNotEqual(
        actor_metric_logger.get_metric('reward/mean', 'train'), 0
    )
    for metric_name in [
        'score/mean',
        'reward/mean',
        'loss',  # policy loss
        'reward_kl_penalty',
        'pg_clipfrac',
    ]:
      self.assertLen(
          actor_metric_logger.get_metric_history(metric_name, 'train'),
          ppo_learner._train_steps,
      )
      if metric_name not in ('loss', 'kl'):  # KL is not logged in eval mode.
        self.assertLen(
            actor_metric_logger.get_metric_history(metric_name, 'eval'),
            ppo_learner._eval_steps,
        )
      elif metric_name == 'loss':
        self.assertLen(
            actor_metric_logger.get_metric_history(metric_name, 'eval'),
            5,  # eval loss is aggregated, so # equal to # of eval invocations.
        )

    for metric_name in ['loss', 'vpred_mean', 'vf_clipfrac']:
      self.assertLen(
          ppo_learner._critic_metrics_logger.get_metric_history(
              metric_name, 'train'
          ),
          ppo_learner._train_steps,
      )
      eval_steps = 5 if metric_name == 'loss' else ppo_learner._eval_steps
      self.assertLen(
          ppo_learner._critic_metrics_logger.get_metric_history(
              metric_name, 'eval'
          ),
          eval_steps,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='multi_iter',
          input_data_size=2,
          batch_size=1,
          mini_batch_size=None,
          num_ppo_epochs=2,
          beta=0.04,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_inference_worker_logps_fn_call_at_step=[
              0,
              1,
              2,
              3,
              4,
              5,
              6,
              7,
          ],
          expected_rollout_worker_logps_fn_call_at_step=[
              0,
              1,
              2,
              3,
              4,
              5,
              6,
              7,
          ],
          raise_error=False,
      ),
      dict(
          testcase_name='multi_iter_with_gradient_accumulation',
          input_data_size=2,
          batch_size=1,
          mini_batch_size=None,
          num_ppo_epochs=2,
          beta=0.04,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=None,
          expected_inference_worker_logps_fn_call_at_step=None,
          expected_rollout_worker_logps_fn_call_at_step=None,
          raise_error=True,
      ),
      dict(
          testcase_name='multi_iter_with_mini_batching',
          input_data_size=6,
          batch_size=8,
          mini_batch_size=2,
          num_ppo_epochs=2,
          beta=0.04,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2],
          expected_inference_worker_logps_fn_call_at_step=[0, 1, 2],
          expected_rollout_worker_logps_fn_call_at_step=[0, 1, 2],
          raise_error=False,
      ),
      dict(
          testcase_name=(
              'multi_iter_with_mini_batching_and_gradient_accumulation'
          ),
          input_data_size=6,
          batch_size=8,
          mini_batch_size=4,
          num_ppo_epochs=2,
          beta=0.04,
          gradient_accumulation_steps=4,
          expected_gen_fn_call_at_step=[0, 1, 2],
          expected_inference_worker_logps_fn_call_at_step=[0, 1, 2],
          expected_rollout_worker_logps_fn_call_at_step=[0, 1, 2],
          raise_error=False,
      ),
      dict(
          testcase_name='single_iter_with_gradient_accumulation',
          input_data_size=2,
          batch_size=1,
          mini_batch_size=None,
          num_ppo_epochs=1,
          beta=0.04,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=None,
          expected_inference_worker_logps_fn_call_at_step=None,
          expected_rollout_worker_logps_fn_call_at_step=None,
          raise_error=True,
      ),
      dict(
          testcase_name='single_iter',
          input_data_size=2,
          batch_size=1,
          mini_batch_size=None,
          num_ppo_epochs=1,
          beta=0.04,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_inference_worker_logps_fn_call_at_step=(
              [0, 1, 2, 3, 4, 5, 6, 7]
          ),
          expected_rollout_worker_logps_fn_call_at_step=(
              [0, 1, 2, 3, 4, 5, 6, 7]
          ),
          raise_error=False,
      ),
      dict(
          testcase_name='single_iter_without_kl',
          input_data_size=2,
          batch_size=1,
          mini_batch_size=None,
          num_ppo_epochs=1,
          beta=0.0,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_inference_worker_logps_fn_call_at_step=[],
          expected_rollout_worker_logps_fn_call_at_step=[
              0,
              1,
              2,
              3,
              4,
              5,
              6,
              7,
          ],
          raise_error=False,
      ),
      dict(
          testcase_name='single_iter_with_mini_batching',
          input_data_size=6,
          batch_size=8,
          mini_batch_size=2,
          num_ppo_epochs=1,
          beta=0.04,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2],
          expected_inference_worker_logps_fn_call_at_step=[0, 1, 2],
          expected_rollout_worker_logps_fn_call_at_step=[0, 1, 2],
          raise_error=False,
      ),
  )
  def test_multi_iteration_training(
      self,
      input_data_size,
      batch_size,
      mini_batch_size,
      num_ppo_epochs,
      beta,
      gradient_accumulation_steps,
      expected_gen_fn_call_at_step,
      expected_inference_worker_logps_fn_call_at_step,
      expected_rollout_worker_logps_fn_call_at_step,
      raise_error,
  ):
    gen_fn_call_at_step = []
    rollout_worker_logps_fn_call_at_step = []
    inference_worker_logps_fn_call_at_step = []

    def wrap_fn(fn, fn_call_at_step, trainer):
      def wrapper(*args, **kwargs):
        fn_call_at_step.append(trainer.train_steps)
        return fn(*args, **kwargs)

      return wrapper

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    original_model_variables = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.Param)
    )

    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    value_model = utils.create_critic_model(model)
    var_filter = nnx.All(nnx.Param, lambda path, x: 'output' not in path)
    original_value_model_variables = jax.tree.map(
        jnp.copy, nnx.state(value_model, var_filter)
    )

    reward_model = tc.ToyTransformer(
        rngs=nnx.Rngs(2), vocab_size=vocab.GetPieceSize()
    )
    reward_model = tc.MockTransformerWithScoreHead(reward_model, nnx.Rngs(1))

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
            rl_cluster_lib.Role.REWARD: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=12,
            max_steps=10,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        reward=reward_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )

    ppo_config = ppo_lib.PpoConfig(
        num_ppo_epochs=num_ppo_epochs,
        mini_batch_size=mini_batch_size,
        beta=beta,
    )
    ppo_learner = ppo_lib.PpoLearner(
        rl_cluster=rl_cluster,
        ppo_config=ppo_config,
    )

    ppo_learner._generate_and_compute_advantage = wrap_fn(
        ppo_learner._generate_and_compute_advantage,
        gen_fn_call_at_step,
        ppo_learner.rl_cluster.actor_trainer,
    )

    rl_cluster.rollout.get_per_token_logps = wrap_fn(
        rl_cluster.rollout.get_per_token_logps,
        rollout_worker_logps_fn_call_at_step,
        ppo_learner.rl_cluster.actor_trainer,
    )
    rl_cluster.inference_worker.get_ref_per_token_logps = wrap_fn(
        rl_cluster.inference_worker.get_ref_per_token_logps,
        inference_worker_logps_fn_call_at_step,
        ppo_learner.rl_cluster.actor_trainer,
    )

    train_ds = _dummy_dataset(
        _DUMMY_DATA * input_data_size, batch_size=batch_size
    )

    if raise_error:
      with self.assertRaises(ValueError):
        ppo_learner.train(train_ds, None)
      return
    ppo_learner.train(train_ds, None)

    model_variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        tc.assert_not_equal, original_model_variables, model_variables
    )
    value_model_variables = nnx.state(value_model, var_filter)

    jax.tree.map_with_path(
        tc.assert_not_equal,
        original_value_model_variables,
        value_model_variables,
    )

    self.assertEqual(gen_fn_call_at_step, expected_gen_fn_call_at_step)
    self.assertEqual(
        inference_worker_logps_fn_call_at_step,
        expected_inference_worker_logps_fn_call_at_step,
    )
    self.assertEqual(
        rollout_worker_logps_fn_call_at_step,
        expected_rollout_worker_logps_fn_call_at_step,
    )


if __name__ == '__main__':
  absltest.main()
