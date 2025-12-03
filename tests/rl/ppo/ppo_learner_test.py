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

# import itertools
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
# from tunix.rl.queue import data_queue as queue_lib
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc
# from typing_extensions import override
import os


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

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


class PPOLearnerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    num_cpus = int(os.environ.get('DEVICE_COUNTS', 2))
    chex.set_n_cpu_devices(num_cpus)
    print(f'Setting up test with {num_cpus} devices')

  # def test_iterator(self):

  #   class _EmptyTrainer(ppo_lib.PPOLearner):
  #     """A trainer used to test the iterator preparation."""

  #     def __init__(self):
  #       self._data_shuffle_key = jax.random.PRNGKey(42)
  #       self.rollout_worker_mesh = pxla.thread_resources.env.physical_mesh
  #       self._iter_steps = 0
  #       self._eval_steps = 0
  #       self._last_iter_step = 0

  #     @override
  #     def _generate_and_compute_advantage(self, example, mode='train'):
  #       return example

  #   empty_trainer = _EmptyTrainer()

  #   def _prepare(dataset, batch_repeat, grad_acc_steps, mini_batch_size=None):
  #     iterator = iter(dataset)
  #     while True:
  #       try:
  #         data_queue = queue_lib.SimpleDataQueue(maxsize=2)
  #         empty_trainer._prepare_data(
  #             iterator=iterator,
  #             mini_batch_size=mini_batch_size,
  #             proceed_num_steps=grad_acc_steps,
  #             batch_repeat=batch_repeat,
  #             data_queue=data_queue,
  #             async_loading=False,
  #             shuffle_data=True,
  #         )
  #         yield data_queue.get(block=True)
  #       except StopIteration:
  #         break

  #   # Case 1: batch repeat = 1, grad_acc_steps = 1
  #   dataset = _dummy_dataset([i for i in range(8)], 4)
  #   res = [
  #       d.get('prompts').tolist()
  #       for d in itertools.chain.from_iterable(_prepare(dataset, 1, 1))
  #   ]
  #   expected = [[3, 2, 1, 0], [5, 6, 4, 7]]
  #   self.assertEqual(res, expected)

  #   # Case 2: batch repeat = 3, grad_acc_steps = 1
  #   dataset = _dummy_dataset([i for i in range(8)], 4)
  #   res = [
  #       d.get('prompts').tolist()
  #       for d in itertools.chain.from_iterable(_prepare(dataset, 3, 1))
  #   ]
  #   expected = [
  #       [3, 1, 2, 0],
  #       [1, 2, 0, 3],
  #       [0, 1, 3, 2],
  #       [5, 6, 4, 7],
  #       [4, 5, 7, 6],
  #       [5, 7, 4, 6],
  #   ]
  #   self.assertEqual(res, expected)

  #   # Case 3: batch repeat = 1, grad_acc_steps = 3
  #   empty_trainer._data_shuffle_key = jax.random.PRNGKey(42)
  #   dataset = _dummy_dataset([i for i in range(16)], 4)
  #   res = [
  #       d.get('prompts').tolist()
  #       for d in itertools.chain.from_iterable(_prepare(dataset, 1, 3))
  #   ]
  #   expected = [
  #       [3, 2, 1, 0],
  #       [5, 6, 4, 7],
  #       [11, 9, 10, 8],
  #       # [12, 13, 14, 15]  # dropped, cannot meet size of grad_acc_steps
  #   ]
  #   self.assertEqual(res, expected)

  #   # Case 4: batch repeat = 2, grad_acc_steps = 3
  #   dataset = _dummy_dataset([i for i in range(16)], 4)
  #   res = [
  #       d.get('prompts').tolist()
  #       for d in itertools.chain.from_iterable(_prepare(dataset, 2, 3))
  #   ]
  #   expected = [
  #       [1, 2, 0, 3],
  #       [7, 5, 6, 4],
  #       [9, 10, 8, 11],
  #       [0, 1, 3, 2],
  #       [5, 7, 4, 6],
  #       [11, 9, 10, 8],
  #   ]
  #   self.assertEqual(res, expected)

  #   # Case 5: batch repeat = 1, grad_acc_steps = 1, mini_batch_size = 2
  #   dataset = _dummy_dataset([i for i in range(8)], 4)
  #   res = [
  #       d.get('prompts').tolist()
  #       for d in itertools.chain.from_iterable(_prepare(dataset, 1, 1, 2))
  #   ]
  #   expected = [[3, 1], [2, 0], [5, 6], [4, 7]]
  #   self.assertEqual(res, expected)

  #   # Case 6: batch repeat = 3, grad_acc_steps = 1, mini_batch_size = 2
  #   dataset = _dummy_dataset([i for i in range(8)], 4)
  #   res = [
  #       d.get('prompts').tolist()
  #       for d in itertools.chain.from_iterable(_prepare(dataset, 3, 1, 2))
  #   ]
  #   expected = [
  #       [0, 1],
  #       [3, 2],
  #       [1, 3],
  #       [0, 2],
  #       [3, 1],
  #       [2, 0],
  #       [5, 7],
  #       [4, 6],
  #       [7, 5],
  #       [6, 4],
  #       [5, 4],
  #       [6, 7],
  #   ]
  #   self.assertEqual(res, expected)

  #   # Case 7: batch repeat = 1, grad_acc_steps = 3, mini_batch_size = 2
  #   dataset = _dummy_dataset([i for i in range(16)], 4)
  #   res = [
  #       d.get('prompts').tolist()
  #       for d in itertools.chain.from_iterable(_prepare(dataset, 1, 3, 2))
  #   ]
  #   expected = [
  #       [3, 1],
  #       [2, 0],
  #       [5, 4],
  #       [6, 7],
  #       [8, 11],
  #       [9, 10],
  #       # [12, 13], [14, 15],  # dropped, cannot meet size of grad_acc_steps
  #   ]
  #   self.assertEqual(res, expected)

  #   # Case 8: batch repeat = 2, grad_acc_steps = 3, mini_batch_size = 2
  #   dataset = _dummy_dataset([i for i in range(16)], 4)
  #   res = [
  #       d.get('prompts').tolist()
  #       for d in itertools.chain.from_iterable(_prepare(dataset, 2, 3, 2))
  #   ]
  #   expected = [
  #       [1, 0],
  #       [2, 3],
  #       [4, 7],
  #       [5, 6],
  #       [9, 10],
  #       [8, 11],
  #       [2, 3],
  #       [0, 1],
  #       [7, 5],
  #       [4, 6],
  #       [11, 10],
  #       [9, 8],
  #   ]
  #   self.assertEqual(res, expected)

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_reward_model',
          'use_reward_model': True,
          'reward_fns': None,
          'use_different_rollout_config': True,
          'epsilon_c': None,
          'entropy_coef': None,
      },
      {
          'testcase_name': 'dual_clip',
          'use_reward_model': True,
          'reward_fns': None,
          'use_different_rollout_config': True,
          'epsilon_c': 2.0,
          'entropy_coef': None,
      },
      {
          'testcase_name': 'dual_clip_with_entropy',
          'use_reward_model': True,
          'reward_fns': None,
          'use_different_rollout_config': True,
          'epsilon_c': 2.0,
          'entropy_coef': 0.1,
      },
      {
          'testcase_name': 'with_reward_fn',
          'use_reward_model': False,
          'reward_fns': [reward_1, reward_2],
          'use_different_rollout_config': False,
          'epsilon_c': None,
          'entropy_coef': None,
      },
      {
          'testcase_name': 'with_reward_model_diff_rollout_config',
          'use_reward_model': True,
          'reward_fns': None,
          'use_different_rollout_config': True,
          'epsilon_c': None,
          'entropy_coef': None,
      },
  )
  def test_ppo_learner(
      self,
      use_reward_model,
      reward_fns,
      use_different_rollout_config,
      epsilon_c,
      entropy_coef,
  ):
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    original_model_variables = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.Param)
    )

    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )

    value_model = utils.create_critic_model(model)
    var_filter = nnx.All(nnx.Param, lambda path, x: 'output' not in path)
    original_value_model_variables = jax.tree.map(
        jnp.copy, nnx.state(value_model, var_filter)
    )

    if use_reward_model:
      reward_model = tc.ToyTransformer(
          config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(2),
      )
      reward_model = tc.ToyTransformerWithScoreHead(reward_model, nnx.Rngs(1))

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
            gradient_accumulation_steps=None,
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
    ppo_config = ppo_lib.PPOConfig(
        num_iterations=1, epsilon_c=epsilon_c, entropy_coef=entropy_coef
    )
    ppo_learner = ppo_lib.PPOLearner(
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

    self.assertEqual(ppo_learner._iter_steps, 10)
    self.assertEqual(ppo_learner._eval_iter_steps, 2)  # num eval batches
    self.assertEqual(
        ppo_learner.rl_cluster.actor_trainer._iter_steps,
        ppo_learner._iter_steps,
    )
    self.assertEqual(
        ppo_learner.rl_cluster.critic_trainer._iter_steps,
        ppo_learner._iter_steps,
    )

    for metric_name in [
        'score/mean',
        'reward/mean',
        'reward_kl_penalty',
    ]:
      self.assertLen(
          ppo_learner.rl_cluster._rl_metrics_logger.get_metric_history(
              'global', metric_name, 'train'
          ),
          ppo_learner.rl_cluster.global_steps,
      )

    actor_metric_logger = ppo_learner.rl_cluster.actor_trainer.metrics_logger
    expected_metrics = ['loss', 'pg_clipfrac']  # policy loss
    if epsilon_c is not None:
      expected_metrics.append('pg_clipfrac_lower')
    for metric_name in expected_metrics:
      self.assertLen(
          actor_metric_logger.get_metric_history('actor', metric_name, 'train'),
          ppo_learner._iter_steps,
      )
      self.assertLen(
          actor_metric_logger.get_metric_history('actor', metric_name, 'eval'),
          ppo_learner.rl_cluster.actor_trainer.train_steps
          / cluster_config.training_config.eval_every_n_steps,
          msg=f'metric_name: {metric_name}',
      )

    critic_metric_logger = ppo_learner.rl_cluster.critic_trainer.metrics_logger
    for metric_name in ['loss', 'vpred_mean', 'vf_clipfrac']:
      self.assertLen(
          critic_metric_logger.get_metric_history(
              'critic', metric_name, 'train'
          ),
          ppo_learner.rl_cluster.critic_trainer.train_steps,
      )
      self.assertLen(
          critic_metric_logger.get_metric_history(
              'critic', metric_name, 'eval'
          ),
          ppo_learner.rl_cluster.critic_trainer.train_steps
          / cluster_config.training_config.eval_every_n_steps,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='multi_iter_without_gradient_accumulation',
          name='multi_iter_without_gradient_accumulation',
          num_iterations=2,
          beta=0.04,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 2, 4, 6, 8],
          expected_inference_worker_logps_fn_call_at_step=[0, 2, 4, 6, 8],
          expected_rollout_worker_logps_fn_call_at_step=[0, 2, 4, 6, 8],
      ),
      dict(
          testcase_name='multi_iter_with_gradient_accumulation',
          name='multi_iter_with_gradient_accumulation',
          num_iterations=2,
          beta=0.04,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 0, 0, 6, 6, 6, 12, 12],
          expected_inference_worker_logps_fn_call_at_step=[
              0,
              0,
              0,
              6,
              6,
              6,
              12,
              12,
          ],
          expected_rollout_worker_logps_fn_call_at_step=[
              0,
              0,
              0,
              6,
              6,
              6,
              12,
              12,
          ],
      ),
      dict(
          testcase_name='single_iter_with_gradient_accumulation',
          name='single_iter_with_gradient_accumulation',
          num_iterations=1,
          beta=0.04,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 0, 0, 3, 3, 3, 6, 6],
          expected_inference_worker_logps_fn_call_at_step=(
              [0, 0, 0, 3, 3, 3, 6, 6]
          ),
          expected_rollout_worker_logps_fn_call_at_step=(
              [0, 0, 0, 3, 3, 3, 6, 6]
          ),
      ),
      dict(
          testcase_name='single_iter_without_gradient_accumulation',
          name='single_iter_without_gradient_accumulation',
          num_iterations=1,
          beta=0.04,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_inference_worker_logps_fn_call_at_step=(
              [0, 1, 2, 3, 4, 5, 6, 7]
          ),
          expected_rollout_worker_logps_fn_call_at_step=(
              [0, 1, 2, 3, 4, 5, 6, 7]
          ),
      ),
      dict(
          testcase_name='single_iter_without_kl',
          name='single_iter_without_kl',
          num_iterations=1,
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
      ),
  )
  def test_multi_iteration_training(
      self,
      name,
      num_iterations,
      beta,
      gradient_accumulation_steps,
      expected_gen_fn_call_at_step,
      expected_inference_worker_logps_fn_call_at_step,
      expected_rollout_worker_logps_fn_call_at_step,
  ):
    # TODO(b/446969561): Re-enable these test cases. Due to the change in
    # cl/810188417, the current test case will fail.
    if name in (
        'multi_iter_with_gradient_accumulation',
        'single_iter_with_gradient_accumulation',
    ):
      self.skipTest(
          'Skipping failing test cases with gradient accumulation. See'
          ' b/446969561 for details.'
      )

    gen_fn_call_at_step = []
    rollout_worker_logps_fn_call_at_step = []
    inference_worker_logps_fn_call_at_step = []

    def wrap_fn(fn, fn_call_at_step, trainer):
      def wrapper(*args, **kwargs):
        fn_call_at_step.append(trainer.iter_steps)
        return fn(*args, **kwargs)

      return wrapper

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    original_model_variables = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.Param)
    )

    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )

    value_model = utils.create_critic_model(model)
    var_filter = nnx.All(nnx.Param, lambda path, x: 'output' not in path)
    original_value_model_variables = jax.tree.map(
        jnp.copy, nnx.state(value_model, var_filter)
    )

    reward_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(2)
    )
    reward_model = tc.ToyTransformerWithScoreHead(reward_model, nnx.Rngs(1))

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

    ppo_config = ppo_lib.PPOConfig(num_iterations=num_iterations, beta=beta)
    ppo_learner = ppo_lib.PPOLearner(
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

    train_ds = _dummy_dataset(_DUMMY_DATA * 2, batch_size=1)
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
