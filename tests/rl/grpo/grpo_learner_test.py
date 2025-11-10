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
import os
import shutil
import tempfile
import types
import uuid
from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
from flax.nnx import filterlib
from grain import python as grain
import jax
from jax import sharding
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.perf import trace as trace_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.rl.queue import data_queue as queue_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import profiler
from tunix.tests import test_common as tc
from typing_extensions import override

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

Mesh = sharding.Mesh

# Use tokens defined in MockVocab in test_common.py
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


class GRPOLearnerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    num_cpus = int(os.environ.get('DEVICE_COUNTS', 2))
    chex.set_n_cpu_devices(num_cpus)
    print(f'Setting up test with {num_cpus} devices')

    cls.device_count = jax.device_count()

  def test_iterator(self):

    class _EmptyTrainer(grpo_lib.GRPOLearner):
      """A trainer that does nothing but used to test the iterator preparation."""

      def __init__(self, grpo_config):
        self._iter_steps = 0
        self._eval_iter_steps = 0
        self.rollout_worker_mesh = pxla.thread_resources.env.physical_mesh
        self._last_iter_step = 0
        self.grpo_config = grpo_config
        self._data_shuffle_seed = None
        self.rl_cluster = types.SimpleNamespace(
            cluster_config=types.SimpleNamespace(
                training_config=types.SimpleNamespace(
                    rollout_micro_batch_size=1,
                    compute_logps_micro_batch_size=1,
                )
            ),
            buffer_metrics=lambda x, mode: None,
            perf=trace_lib.NoopTracer(),
        )

        self._rollout_micro_batch_size = 1
        self._compute_logps_micro_batch_size = 1

      @override
      def _generate_and_compute_advantage(self, example, mode='train'):
        if 'trajectory_ids' in example:
          del example['trajectory_ids']

        prompts = example['prompts']
        num_samples = len(prompts)

        # Return a SimpleNamespace to mimic TrainExample attributes
        return types.SimpleNamespace(
            prompt_ids=np.array(prompts),
            prompt_mask=np.ones((num_samples, 1), dtype=np.int32),
            completion_ids=np.zeros((num_samples, 1), dtype=np.int32),
            completion_mask=np.zeros((num_samples, 1), dtype=np.int32),
            ref_per_token_logps=None,
            advantages=np.zeros(num_samples, dtype=np.float32),
            old_per_token_logps=None,
        )

    empty_trainer = _EmptyTrainer(
        grpo_lib.GRPOConfig(num_generations=2, num_iterations=1)
    )

    def _prepare(dataset, sample_repeat, batch_repeat, grad_acc_steps):
      iterator = iter(dataset)
      while True:
        try:
          queue_size = batch_repeat * grad_acc_steps + 1
          data_queue = queue_lib.SimpleDataQueue(maxsize=queue_size)
          empty_trainer._prepare_data(
              iterator=iterator,
              proceed_num_steps=grad_acc_steps,
              sample_repeat=sample_repeat,
              batch_repeat=batch_repeat,
              data_queue=data_queue,
              async_loading=False,
              service_target_batch_size=1,
          )
          while True:
            item = data_queue.get(block=True)
            if item is None:
              break
            yield item
        except StopIteration:
          break

    dataset = _dummy_dataset([i for i in range(4)], 2)
    res = [
        d.prompt_ids.tolist()
        for d in itertools.chain.from_iterable(_prepare(dataset, 5, 3, 1))
    ]
    expected = [
        # sample repeat
        # < -------- >
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # ^
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # |
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # |  batch repeat
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],  # |
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],  # |
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],  # v
    ]
    self.assertEqual(res, expected)

    dataset = _dummy_dataset([i for i in range(16)], 2)
    res = [
        d.prompt_ids.tolist()
        for d in itertools.chain.from_iterable(_prepare(dataset, 2, 2, 3))
    ]
    expected = [
        [0, 0, 1, 1],  # ^            ^
        [2, 2, 3, 3],  # |  grad accu |
        [4, 4, 5, 5],  # v            |
        [0, 0, 1, 1],  #              |  batch repeat
        [2, 2, 3, 3],  #              |
        [4, 4, 5, 5],  #              v
        [6, 6, 7, 7],
        [8, 8, 9, 9],
        [10, 10, 11, 11],
        [6, 6, 7, 7],
        [8, 8, 9, 9],
        [10, 10, 11, 11],
        # [12, 12, 13, 13], drop due to that it cannot meet size of grad accu
        # [14, 14, 15, 15],
    ]
    self.assertEqual(res, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_reward_fn',
          reward_fns=reward_1,
          loss_algo='grpo',
      ),
      dict(
          testcase_name='multiple_reward_fns',
          reward_fns=[
              reward_1,
              reward_2,
          ],
          loss_algo='grpo',
      ),
      dict(
          testcase_name='single_reward_fn_gspo',
          reward_fns=reward_1,
          loss_algo='gspo-token',
      ),
  )
  def test_grpo_learner(self, reward_fns, loss_algo):
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=2,
            max_steps=10,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )
    rl_cluster.with_external_metrics_logger(print)

    grpo_config = grpo_lib.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        loss_algo=loss_algo,
    )
    grpo_learner = grpo_lib.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        grpo_config=grpo_config,
        metric_fns=[lambda **kwargs: {'test_metric': (1.0, np.mean)}],
    )
    self.assertFalse(grpo_learner.should_sync_weights)
    self.assertFalse(grpo_learner.can_enable_async_rollout)
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    eval_ds = _dummy_dataset(batch_size=1)

    def wrap_prepare_data(fn, fn_call_at_step, learner):
      def wrapper(*args, **kwargs):
        if str(kwargs['mode']) == 'train':
          fn_call_at_step['train'].append(learner._iter_steps)
        else:
          fn_call_at_step['eval'].append(learner._eval_iter_steps)
        return fn(*args, **kwargs)

      return wrapper

    prepare_data_call_at_step = {'train': [], 'eval': []}
    grpo_learner._prepare_data = wrap_prepare_data(
        grpo_learner._prepare_data,
        prepare_data_call_at_step,
        grpo_learner,
    )

    grpo_learner.train(train_ds, eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    self.assertEqual(grpo_learner._iter_steps, 10)  # max_steps
    self.assertEqual(grpo_learner._eval_iter_steps, 4)  # num eval batches
    self.assertEqual(
        grpo_learner.rl_cluster.actor_trainer.iter_steps,
        grpo_learner._iter_steps,
    )
    expected_prepare_data_call_at_step = {
        'train': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'eval': [0, 0, 0, 0, 0],  # eval step is being reset to 0 at each time
    }
    self.assertEqual(
        prepare_data_call_at_step,
        expected_prepare_data_call_at_step,
    )
    self.assertEqual(
        grpo_learner.rl_cluster.global_steps,
        10,  # max_steps / num_iterations
    )

    rl_metric_logger = grpo_learner.rl_cluster._rl_metrics_logger

    rewards_metrics = (
        ('rewards/' + f.__name__ for f in reward_fns)
        if isinstance(reward_fns, list)
        else ('rewards/' + reward_fns.__name__,)
    )
    # Metric 'prompts' and 'completions' are not logged in native metric logger because jax.monitoring does not support string values.
    for metric_name in [
        'rewards/sum',
        'rewards/min',
        'rewards/max',
        *rewards_metrics,
        'test_metric',
    ]:
      if metric_name == 'rewards/reward_2' and not isinstance(reward_fns, list):
        continue

      self.assertLen(
          rl_metric_logger.get_metric_history(metric_name, 'train'),
          grpo_learner.rl_cluster.global_steps,
          msg=f'metric_name: {metric_name}',
      )
      self.assertLen(
          rl_metric_logger.get_metric_history(metric_name, 'eval'),
          grpo_learner.rl_cluster.actor_trainer.train_steps
          / cluster_config.training_config.eval_every_n_steps,
          msg=f'metric_name: {metric_name}',
      )

    metric_logger = grpo_learner.rl_cluster.actor_trainer.metrics_logger
    for metric_name in ['loss', 'kl']:
      self.assertLen(
          metric_logger.get_metric_history(metric_name, 'train'),
          grpo_learner.rl_cluster.actor_trainer.train_steps,
          msg=f'metric_name: {metric_name}',
      )
      self.assertLen(
          metric_logger.get_metric_history(metric_name, 'eval'),
          grpo_learner.rl_cluster.actor_trainer.train_steps
          / cluster_config.training_config.eval_every_n_steps,
          msg=f'metric_name: {metric_name}',
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
          testcase_name='multi_iter_without_kl',
          name='multi_iter_without_kl',
          num_iterations=2,
          beta=0,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 0, 0, 6, 6, 6, 12, 12],
          expected_inference_worker_logps_fn_call_at_step=[],
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
          testcase_name='singler_iter_with_gradient_accumulation',
          name='singler_iter_with_gradient_accumulation',
          num_iterations=1,
          beta=0.04,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 0, 0, 3, 3, 3, 6, 6],
          expected_inference_worker_logps_fn_call_at_step=[
              0,
              0,
              0,
              3,
              3,
              3,
              6,
              6,
          ],
          expected_rollout_worker_logps_fn_call_at_step=[],
      ),
      dict(
          testcase_name='singler_iter_without_gradient_accumulation',
          name='singler_iter_without_gradient_accumulation',
          num_iterations=1,
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
          expected_rollout_worker_logps_fn_call_at_step=[],
      ),
      dict(
          testcase_name='singler_iter_without_kl',
          name='singler_iter_without_kl',
          num_iterations=1,
          beta=0,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_inference_worker_logps_fn_call_at_step=[],
          expected_rollout_worker_logps_fn_call_at_step=[],
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
        'multi_iter_without_kl',
        'singler_iter_with_gradient_accumulation',
    ):
      self.skipTest(
          'Skipping failing test cases with gradient accumulation > 1. See'
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
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
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
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )

    grpo_config = grpo_lib.GRPOConfig(
        num_generations=2,
        num_iterations=num_iterations,
        beta=beta,
    )

    grpo_learner = grpo_lib.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_1,
        grpo_config=grpo_config,
    )
    self.assertEqual(grpo_learner.rl_cluster.global_steps, 0)

    grpo_learner._generate_and_compute_advantage = wrap_fn(
        grpo_learner._generate_and_compute_advantage,
        gen_fn_call_at_step,
        grpo_learner.rl_cluster.actor_trainer,
    )

    rl_cluster.rollout.get_per_token_logps = wrap_fn(
        rl_cluster.rollout.get_per_token_logps,
        rollout_worker_logps_fn_call_at_step,
        grpo_learner.rl_cluster.actor_trainer,
    )
    rl_cluster.inference_worker.get_ref_per_token_logps = wrap_fn(
        rl_cluster.inference_worker.get_ref_per_token_logps,
        inference_worker_logps_fn_call_at_step,
        grpo_learner.rl_cluster.actor_trainer,
    )

    train_ds = _dummy_dataset(_DUMMY_DATA * 2, batch_size=1)
    grpo_learner.train(train_ds, None)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    self.assertEqual(gen_fn_call_at_step, expected_gen_fn_call_at_step)
    self.assertEqual(
        inference_worker_logps_fn_call_at_step,
        expected_inference_worker_logps_fn_call_at_step,
    )
    self.assertEqual(
        rollout_worker_logps_fn_call_at_step,
        expected_rollout_worker_logps_fn_call_at_step,
    )
    self.assertEqual(
        grpo_learner.rl_cluster.actor_trainer.train_steps,
        grpo_learner.rl_cluster.actor_trainer.iter_steps
        // cluster_config.training_config.get_with_default(
            'gradient_accumulation_steps', 1
        ),
    )
    self.assertLen(
        grpo_learner.rl_cluster.actor_trainer.metrics_logger.get_metric_history(
            'kl', 'train'
        ),
        grpo_learner.rl_cluster.actor_trainer.train_steps,
    )

  def test_grpo_with_lora_model(self):
    # reshard through default device_put.
    split_index = self.device_count // 2
    mesh1 = Mesh(
        np.array(
            sorted(jax.devices(), key=lambda d: d.id)[:split_index]
        ).reshape(split_index, 1),
        ('fsdp', 'tp'),
    )
    mesh2 = Mesh(
        np.array(
            sorted(jax.devices(), key=lambda d: d.id)[split_index:]
        ).reshape(1, split_index),
        ('fsdp', 'tp'),
    )
    vocab = tc.MockVocab()
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )
    actor_model = tc.get_lora_model(
        ref_model,
        mesh=mesh1,
    )
    original_base_params = jax.tree.map(
        jnp.copy, nnx.state(actor_model, filterlib.Not(nnx.LoRAParam))
    )
    original_lora_variables = jax.tree.map(
        jnp.copy, nnx.state(actor_model, nnx.LoRAParam)
    )
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh1,
            rl_cluster_lib.Role.REFERENCE: mesh1,
            rl_cluster_lib.Role.ROLLOUT: mesh2,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=10,
            max_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )
    grpo_config = grpo_lib.GRPOConfig(
        num_generations=2,
        num_iterations=1,
    )

    grpo_learner = grpo_lib.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_1,
        grpo_config=grpo_config,
    )
    self.assertTrue(grpo_learner.should_sync_weights)
    self.assertTrue(grpo_learner.can_enable_async_rollout)
    train_ds = _dummy_dataset(batch_size=2)
    grpo_learner.train(train_ds, None)

    base_params = nnx.state(
        rl_cluster.actor_trainer.model, filterlib.Not(nnx.LoRAParam)
    )
    lora_params = nnx.state(rl_cluster.actor_trainer.model, nnx.LoRAParam)
    lora_params_from_sampler = nnx.state(
        grpo_learner.rl_cluster.rollout.model(), nnx.LoRAParam
    )
    jax.tree.map_with_path(
        tc.assert_not_equal, original_lora_variables, lora_params
    )
    jax.tree.map_with_path(
        tc.assert_equal, lora_params_from_sampler, lora_params
    )
    jax.tree.map_with_path(tc.assert_equal, original_base_params, base_params)

  def test_exception_from_data_preparation(self):

    class _TrainerWithException(grpo_lib.GRPOLearner):

      @override
      def _generate_and_compute_advantage(self, example, mode='train'):
        raise ValueError('test exception')

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=10,
            max_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )

    grpo_config = grpo_lib.GRPOConfig(
        num_generations=2,
        num_iterations=1,
    )

    grpo_learner = _TrainerWithException(
        rl_cluster=rl_cluster,
        reward_fns=reward_1,
        grpo_config=grpo_config,
    )

    train_ds = _dummy_dataset(batch_size=2)
    with self.assertRaises(ValueError):
      grpo_learner.train(train_ds, None)

  def test_resume_training(self):
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=10,
            max_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )

    grpo_config = grpo_lib.GRPOConfig(
        num_generations=2,
        num_iterations=1,
    )

    grpo_learner = grpo_lib.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_1,
        grpo_config=grpo_config,
    )
    self.assertEqual(grpo_learner.rl_cluster.global_steps, 0)

    train_ds_full = _dummy_dataset(batch_size=2)
    grpo_learner.train(train_ds_full, None)
    self.assertEqual(grpo_learner.rl_cluster.global_steps, 2)

    try:
      temp_path = self.create_tempdir().full_path
    except Exception:
      temp_path = tempfile.TemporaryDirectory().name

    model2 = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    cluster_config2 = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=10,
            max_steps=10,
            checkpoint_root_directory=temp_path,
            checkpointing_options=ocp.CheckpointManagerOptions(
                save_interval_steps=1,
                max_to_keep=10,
            ),
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )
    rl_cluster2 = rl_cluster_lib.RLCluster(
        actor=model2,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config2,
    )

    grpo_learner2 = grpo_lib.GRPOLearner(
        rl_cluster=rl_cluster2,
        reward_fns=reward_1,
        grpo_config=grpo_config,
    )

    grpo_learner2.train(train_ds_full[0:1], None)
    rl_cluster2 = rl_cluster_lib.RLCluster(
        actor=model2,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config2,
    )
    grpo_learner2 = grpo_lib.GRPOLearner(
        rl_cluster=rl_cluster2,
        reward_fns=reward_1,
        grpo_config=grpo_config,
    )
    self.assertEqual(grpo_learner2.rl_cluster.global_steps, 1)
    assert grpo_learner2._last_iter_step == 1

    grpo_learner2.train(train_ds_full, None)
    self.assertEqual(grpo_learner2.rl_cluster.global_steps, 2)

    variables1 = nnx.state(model, nnx.Param)
    variables2 = nnx.state(model2, nnx.Param)
    jax.tree.map_with_path(tc.assert_equal, variables1, variables2)

  def test_trajectory_ids(self):
    def my_reward_fn(trajectories, prompts, **kwargs):
      for t_id, prompt in zip(kwargs['trajectory_ids'], prompts):
        trajectories[kwargs['mode']][t_id] = prompt
      return [1.0] * len(prompts)

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())

    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    mesh = pxla.thread_resources.env.physical_mesh

    def create_rl_cluster(grad_accu_steps, mini_batch_size):
      cluster_config = rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine='vanilla',
          offload_to_cpu=False,
          training_config=rl_cluster_lib.RLTrainingConfig(
              actor_optimizer=optax.sgd(1e-3),
              eval_every_n_steps=grad_accu_steps * 2,
              max_steps=10,
              # We can't set grad_acc_steps directly, so we do it through
              # mini_batch_size and training_micro_batch_size.
              mini_batch_size=mini_batch_size,
              train_micro_batch_size=mini_batch_size // grad_accu_steps,
          ),
          rollout_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10,
              max_prompt_length=256,
              kv_cache_size=1024,
          ),
      )
      rl_cluster = rl_cluster_lib.RLCluster(
          actor=model,
          reference=ref_model,
          tokenizer=vocab,
          cluster_config=cluster_config,
      )
      return rl_cluster.with_external_metrics_logger(print)

    grpo_config = grpo_lib.GRPOConfig(
        num_generations=2,
        num_iterations=1,
    )
    first_trajectories = {'train': {}, 'eval': {}}
    grpo_learner = grpo_lib.GRPOLearner(
        rl_cluster=create_rl_cluster(1, 4),
        reward_fns=lambda **kwargs: my_reward_fn(
            trajectories=first_trajectories, **kwargs
        ),
        grpo_config=grpo_config,
        metric_fns=[lambda **kwargs: {'test_metric': (1.0, np.mean)}],
    )
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=4)
    eval_ds = _dummy_dataset(batch_size=1)

    grpo_learner.train(train_ds, eval_ds)

    # Execute with different batch size and gradient accumulation steps.
    second_trajectories = {'train': {}, 'eval': {}}
    grpo_learner = grpo_lib.GRPOLearner(
        rl_cluster=create_rl_cluster(4, 8),
        reward_fns=lambda **kwargs: my_reward_fn(
            trajectories=second_trajectories, **kwargs
        ),
        grpo_config=grpo_config,
        metric_fns=[lambda **kwargs: {'test_metric': (1.0, np.mean)}],
    )
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=8)
    eval_ds = _dummy_dataset(batch_size=1)

    grpo_learner.train(train_ds, eval_ds)

    # Check that the trajectories are the same.
    self.assertEqual(first_trajectories, second_trajectories)
    self.assertLen(
        first_trajectories['train'], 80
    )  # max_steps * batch_size * num_generations
    self.assertLen(first_trajectories['eval'], 8)  # eval_rows * num_generations

  @parameterized.named_parameters(
      dict(
          testcase_name='single_update',
          batch_size=8,
          mini_batch_size=8,
          train_micro_batch_size=4,
          rollout_micro_batch_size=4,
          compute_logps_micro_batch_size=4,
      ),
      dict(
          testcase_name='multi_update',
          batch_size=8,
          mini_batch_size=4,
          train_micro_batch_size=2,
          rollout_micro_batch_size=2,
          compute_logps_micro_batch_size=2,
      ),
      dict(
          testcase_name='single_update_with_bigger_rollout_and_compute_logps',
          batch_size=8,
          mini_batch_size=8,
          train_micro_batch_size=4,
          rollout_micro_batch_size=8,
          compute_logps_micro_batch_size=8,
      ),
      dict(
          testcase_name='only_rollout_and_compute_logps',
          batch_size=8,
          mini_batch_size=None,
          train_micro_batch_size=None,
          rollout_micro_batch_size=4,
          compute_logps_micro_batch_size=4,
      ),
      dict(
          testcase_name='individible_batch_size',
          batch_size=20,
          mini_batch_size=20,
          train_micro_batch_size=10,
          rollout_micro_batch_size=4,
          compute_logps_micro_batch_size=4,
      ),
  )
  def test_micro_batch_training(
      self,
      batch_size,
      mini_batch_size,
      train_micro_batch_size,
      rollout_micro_batch_size,
      compute_logps_micro_batch_size,
  ):
    def my_reward_fn(trajectories, prompts, **kwargs):
      for t_id, prompt in zip(kwargs['trajectory_ids'], prompts):
        trajectories[kwargs['mode']][t_id] = prompt
      return jnp.arange(len(prompts))

    def create_learner(
        mini_batch_size,
        train_micro_batch_size,
        rollout_micro_batch_size,
        compute_logps_micro_batch_size,
        trajectories,
    ):
      vocab = tc.MockVocab()
      model = tc.ToyTransformer(
          rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
      )
      ref_model = tc.ToyTransformer(
          rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
      )

      mesh = pxla.thread_resources.env.physical_mesh
      cluster_config = rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine='vanilla',
          offload_to_cpu=False,
          training_config=rl_cluster_lib.RLTrainingConfig(
              actor_optimizer=optax.sgd(1e-3),
              eval_every_n_steps=2,
              max_steps=20,
              mini_batch_size=mini_batch_size,
              train_micro_batch_size=train_micro_batch_size,
              rollout_micro_batch_size=rollout_micro_batch_size,
              compute_logps_micro_batch_size=compute_logps_micro_batch_size,
              profiler_options=profiler.ProfilerOptions(
                  profiler_steps=2,
                  skip_first_n_steps=1,
                  log_dir='/tmp/profiler',
              ),
          ),
          rollout_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10,
              max_prompt_length=32,
              kv_cache_size=256,
              temperature=0.5,
          ),
      )
      rl_cluster = rl_cluster_lib.RLCluster(
          actor=model,
          reference=ref_model,
          tokenizer=vocab,
          cluster_config=cluster_config,
      )

      grpo_config = grpo_lib.GRPOConfig(
          num_generations=2,
          num_iterations=1,
      )
      grpo_learner = grpo_lib.GRPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=lambda **kwargs: my_reward_fn(
              trajectories=trajectories, **kwargs
          ),
          grpo_config=grpo_config,
      )
      return grpo_learner, model

    #  80 rows with repeat=20.
    train_ds = _dummy_dataset(MySource(repeat=20), batch_size=batch_size)
    eval_ds = _dummy_dataset(batch_size=1)

    # Baseline with no micro batching.
    base_trajectories = {'train': {}, 'eval': {}}
    grpo_learner, model = create_learner(
        mini_batch_size=None,
        train_micro_batch_size=None,
        rollout_micro_batch_size=None,
        compute_logps_micro_batch_size=None,
        trajectories=base_trajectories,
    )
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    grpo_learner.train(train_ds, eval_ds)
    self.assertEqual(
        80 // batch_size, grpo_learner.rl_cluster.actor_trainer.train_steps
    )

    base_variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        tc.assert_not_equal, original_variables, base_variables
    )

    # Train with micro batching.
    micro_batch_trajectories = {'train': {}, 'eval': {}}
    grpo_learner, model = create_learner(
        mini_batch_size=mini_batch_size,
        train_micro_batch_size=train_micro_batch_size,
        rollout_micro_batch_size=rollout_micro_batch_size,
        compute_logps_micro_batch_size=compute_logps_micro_batch_size,
        trajectories=micro_batch_trajectories,
    )
    grpo_learner.train(train_ds, eval_ds)
    micro_batch_variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        tc.assert_not_equal, original_variables, micro_batch_variables
    )
    self.assertEqual(base_trajectories, micro_batch_trajectories)
    self.assertEqual(
        80 // (mini_batch_size or batch_size),
        grpo_learner.rl_cluster.actor_trainer.train_steps,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='single_mini_batch',
          max_steps=8,
          batch_size=8,
          mini_batch_size=8,
      ),
      dict(
          testcase_name='multi_mini_batch',
          max_steps=8,
          batch_size=8,
          mini_batch_size=4,
      ),
  )
  def test_checkpoint_with_mini_batch(
      self, max_steps, batch_size, mini_batch_size
  ):
    ckpt_dir = f'/tmp/{self.id()}/{uuid.uuid4()}/checkpoint'
    if os.path.exists(ckpt_dir):
      shutil.rmtree(ckpt_dir)

    def create_learner(
        ckpt_dir,
        max_steps,
    ):
      vocab = tc.MockVocab()
      model = tc.ToyTransformer(
          rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
      )
      ref_model = tc.ToyTransformer(
          rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
      )

      mesh = pxla.thread_resources.env.physical_mesh
      cluster_config = rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine='vanilla',
          offload_to_cpu=False,
          training_config=rl_cluster_lib.RLTrainingConfig(
              actor_optimizer=optax.sgd(1e-3),
              eval_every_n_steps=2,
              max_steps=max_steps,
              mini_batch_size=mini_batch_size,
              train_micro_batch_size=mini_batch_size,
              rollout_micro_batch_size=mini_batch_size,
              compute_logps_micro_batch_size=mini_batch_size,
              checkpointing_options=ocp.CheckpointManagerOptions(
                  save_interval_steps=4,
              ),
              checkpoint_root_directory=ckpt_dir,
          ),
          rollout_config=base_rollout.RolloutConfig(
              max_tokens_to_generate=10,
              max_prompt_length=32,
              kv_cache_size=256,
              temperature=0.5,
          ),
      )
      rl_cluster = rl_cluster_lib.RLCluster(
          actor=model,
          reference=ref_model,
          tokenizer=vocab,
          cluster_config=cluster_config,
      )

      grpo_config = grpo_lib.GRPOConfig(
          num_generations=2,
          num_iterations=1,
      )
      grpo_learner = grpo_lib.GRPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=reward_1,
          grpo_config=grpo_config,
      )
      return grpo_learner

    # make sure we have enough rows
    train_ds = _dummy_dataset(MySource(repeat=100), batch_size=batch_size)

    grpo_learner = create_learner(ckpt_dir, max_steps=max_steps)
    self.assertEqual(grpo_learner.rl_cluster.global_steps, 0)
    grpo_learner.train(train_ds, None)
    self.assertEqual(
        grpo_learner.rl_cluster.global_steps,
        max_steps * mini_batch_size / batch_size,
    )

    # Increase max_steps and so we continue training from checkpoint.
    grpo_learner2 = create_learner(ckpt_dir, max_steps=max_steps * 2)
    self.assertEqual(
        grpo_learner2.rl_cluster.global_steps,
        grpo_learner.rl_cluster.global_steps,
    )
    self.assertEqual(
        grpo_learner2.rl_cluster.actor_trainer._restored_custom_metadata,
        {'global_step': grpo_learner.rl_cluster.global_steps},
    )

    # double the batch size it should also work with checkpoint resumption.
    batch_size *= 2
    train_ds = _dummy_dataset(MySource(repeat=100), batch_size=batch_size)
    grpo_learner2.train(train_ds, None)
    self.assertEqual(
        grpo_learner2.rl_cluster.global_steps,
        grpo_learner.rl_cluster.global_steps
        + max_steps * mini_batch_size / batch_size,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='single_reward_fn',
          reward_fns=reward_1,
      ),
      dict(
          testcase_name='multiple_reward_fns',
          reward_fns=[
              reward_1,
              reward_2,
          ],
      ),
      dict(
          testcase_name='single_reward_fn_gspo',
          reward_fns=reward_1,
      ),
  )
  def test_compute_rewards_shape(self, reward_fns):
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=2,
            max_steps=10,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )
    rl_cluster.with_external_metrics_logger(print)

    grpo_config = grpo_lib.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        loss_algo='grpo',
    )
    grpo_learner = grpo_lib.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        grpo_config=grpo_config,
        metric_fns=[lambda **kwargs: {'test_metric': (1.0, np.mean)}],
    )
    prompts = ['p0', 'p1', 'p2']
    completions = ['c1', 'c2', 'c3']
    answers = ['a1', 'a2', 'a3']
    rewards = grpo_learner._compute_rewards(
        prompts=prompts,
        completions=completions,
        answer=answers,
        mode=rl_cluster_lib.Mode.TRAIN,
    )
    self.assertLen(rewards, len(prompts))


if __name__ == '__main__':
  absltest.main()
