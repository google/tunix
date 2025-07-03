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
import queue

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
from grain import python as grain
import jax
from jax import sharding
from jax.interpreters import pxla
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from tunix.generate import sampler as sampler_lib
from tunix.rl import common
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.rl.rollout import vanilla_rollout
from tunix.tests import test_common as tc
from typing_extensions import override


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

  def __init__(self, data=None):
    if data is None:
      data = _DUMMY_DATA
    self._data = data

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


class GrpoLearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_cpus = 2
    chex.set_n_cpu_devices(self.num_cpus)
    assert len(jax.devices()) == self.num_cpus

  def test_iterator(self):

    class _EmptyTrainer(grpo_lib.GrpoLearner):
      """A trainer that does nothing but used to test the iterator preparation."""

      def __init__(self):
        self._train_steps = 0
        self._eval_steps = 0
        self.rollout_worker_mesh = pxla.thread_resources.env.physical_mesh
        self._last_train_step = 0

      @override
      def _generate_and_compute_advantage(self, example, mode='train'):
        return example

    empty_trainer = _EmptyTrainer()

    def _prepare(dataset, sample_repeat, batch_repeat, grad_acc_steps):
      iterator = iter(dataset)
      while True:
        try:
          data_queue = queue.Queue(maxsize=2)
          empty_trainer.prepare_dataset(
              iterator=iterator,
              proceed_num_steps=grad_acc_steps,
              sample_repeat=sample_repeat,
              batch_repeat=batch_repeat,
              data_queue=data_queue,
              async_loading=False,
          )
          yield data_queue.get(block=True)
        except StopIteration:
          break

    dataset = _dummy_dataset([i for i in range(4)], 2)
    res = [
        d.get('prompts').tolist()
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
        d.get('prompts').tolist()
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
      ),
      dict(
          testcase_name='multiple_reward_fns',
          reward_fns=[
              reward_1,
              reward_2,
          ],
      ),
  )
  def test_grpo_trainer(self, reward_fns):
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )
    rollout_worker = vanilla_rollout.VanillaRollout(
        model=model,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    grpo_config = grpo_lib.GrpoConfig(
        total_generation_steps=10,
        num_generations=2,
        eval_every_n_steps=1,
        max_steps=10,
        max_prompt_length=256,
        num_iterations=1,
    )
    grpo_trainer = grpo_lib.GrpoLearner(
        model=model,
        ref_model=ref_model,
        rollout_worker=rollout_worker,
        reward_fns=reward_fns,
        optimizer=optax.sgd(1e-3),
        training_config=grpo_config,
    )
    self.assertFalse(grpo_trainer.need_sync_sampler_weights)
    train_ds = eval_ds = _dummy_dataset(batch_size=2)
    grpo_trainer.train(train_ds, eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    self.assertEqual(grpo_trainer._train_steps, 2)
    self.assertEqual(grpo_trainer._eval_steps, 4)
    self.assertEqual(
        grpo_trainer.trainer._train_steps, grpo_trainer._train_steps
    )

    metric_logger = grpo_trainer._metrics_logger
    self.assertNotEqual(metric_logger.get_metric('rewards/overall', 'train'), 0)
    for metric_name in [
        'rewards/overall',
        'rewards/reward_1',
        'rewards/reward_2',
        'completions/mean_length',
        'completions/max_length',
        'completions/min_length',
        'kl',
    ]:
      if metric_name == 'rewards/reward_2' and not isinstance(reward_fns, list):
        continue
      self.assertLen(
          metric_logger.get_metric_history(metric_name, 'train'),
          grpo_trainer._train_steps,
      )
      if metric_name != 'kl':  # KL is not logged in eval mode.
        self.assertLen(
            metric_logger.get_metric_history(metric_name, 'eval'),
            grpo_trainer._eval_steps,
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='multi_iter_without_gradient_accumulation',
          num_iterations=2,
          beta=0.04,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 2, 4, 6, 8],
          expected_logps_fn_call_at_step=[0, 0, 2, 4, 6, 8],
          expected_rollout_worker_logps_fn_call_at_step=[0, 2, 4, 6, 8],
      ),
      dict(
          testcase_name='multi_iter_with_gradient_accumulation',
          num_iterations=2,
          beta=0.04,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 0, 0, 6, 6, 6],
          expected_logps_fn_call_at_step=[0, 0, 0, 0, 6, 6, 6],
          expected_rollout_worker_logps_fn_call_at_step=[0, 0, 0, 6, 6, 6],
      ),
      dict(
          testcase_name='multi_iter_without_kl',
          num_iterations=2,
          beta=0,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 0, 0, 6, 6, 6],
          expected_logps_fn_call_at_step=[0],
          expected_rollout_worker_logps_fn_call_at_step=[0, 0, 0, 6, 6, 6],
      ),
      dict(
          testcase_name='singler_iter_with_gradient_accumulation',
          num_iterations=1,
          beta=0.04,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 0, 0, 3, 3, 3, 6, 6],
          expected_logps_fn_call_at_step=[0, 0, 0, 3, 3, 3, 6, 6],
          expected_rollout_worker_logps_fn_call_at_step=[],
      ),
      dict(
          testcase_name='singler_iter_without_gradient_accumulation',
          num_iterations=1,
          beta=0.04,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_logps_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_rollout_worker_logps_fn_call_at_step=[],
      ),
      dict(
          testcase_name='singler_iter_without_kl',
          num_iterations=1,
          beta=0,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_logps_fn_call_at_step=[],
          expected_rollout_worker_logps_fn_call_at_step=[],
      ),
  )
  def test_multi_iteration_training(
      self,
      num_iterations,
      beta,
      gradient_accumulation_steps,
      expected_gen_fn_call_at_step,
      expected_logps_fn_call_at_step,
      expected_rollout_worker_logps_fn_call_at_step,
  ):
    gen_fn_call_at_step = []
    logps_fn_call_at_step = []
    rollout_worker_logps_fn_call_at_step = []

    def wrap_fn(fn, fn_call_at_step, trainer):
      def wrapper(*args, **kwargs):
        fn_call_at_step.append(trainer.trainer.train_steps)
        return fn(*args, **kwargs)

      return wrapper, fn

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )
    rollout_worker = vanilla_rollout.VanillaRollout(
        model=model,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    grpo_config = grpo_lib.GrpoConfig(
        total_generation_steps=10,
        num_generations=2,
        beta=beta,
        num_iterations=num_iterations,
        eval_every_n_steps=10,
        max_steps=10,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_prompt_length=256,
    )
    grpo_trainer = grpo_lib.GrpoLearner(
        model=model,
        ref_model=ref_model,
        rollout_worker=rollout_worker,
        reward_fns=reward_1,
        optimizer=optax.sgd(1e-3),
        training_config=grpo_config,
    )

    grpo_trainer._generate_and_compute_advantage = wrap_fn(
        grpo_trainer._generate_and_compute_advantage,
        gen_fn_call_at_step,
        grpo_trainer,
    )[0]
    wrapped_get_per_token_logps, old_get_per_token_logps = wrap_fn(
        common.get_per_token_logps, logps_fn_call_at_step, grpo_trainer
    )
    common.get_per_token_logps = wrapped_get_per_token_logps
    rollout_worker.get_per_token_logps = wrap_fn(
        rollout_worker.get_per_token_logps,
        rollout_worker_logps_fn_call_at_step,
        grpo_trainer,
    )[0]

    try:
      train_ds = _dummy_dataset(_DUMMY_DATA * 2, batch_size=1)
      grpo_trainer.train(train_ds, None)

      variables = nnx.state(model, nnx.Param)
      jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

      self.assertEqual(gen_fn_call_at_step, expected_gen_fn_call_at_step)
      self.assertEqual(
          logps_fn_call_at_step[1:], expected_logps_fn_call_at_step
      )
      self.assertEqual(
          rollout_worker_logps_fn_call_at_step,
          expected_rollout_worker_logps_fn_call_at_step,
      )
    finally:
      common.get_per_token_logps = old_get_per_token_logps

  # def test_grpo_with_lora_model(self):
  #   mesh1 = Mesh(
  #       np.array(jax.devices()[: self.num_cpus // 2]).reshape(1, 1),
  #       ('fsdp', 'tp'),
  #   )
  #   mesh2 = Mesh(
  #       np.array(jax.devices()[self.num_cpus // 2 :]).reshape(1, 1),
  #       ('fsdp', 'tp'),
  #   )
  #   vocab = tc.MockVocab()
  #   model = tc.get_lora_model(
  #       tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()),
  #       mesh=mesh1,
  #   )
  #   sampler_model = tc.get_lora_model(
  #       tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()),
  #       mesh=mesh2,
  #   )
  #   original_base_params = jax.tree.map(
  #       jnp.copy, nnx.state(model, filterlib.Not(nnx.LoRAParam))
  #   )
  #   original_lora_variables = jax.tree.map(
  #       jnp.copy, nnx.state(model, nnx.LoRAParam)
  #   )
  #   ref_model = tc.ToyTransformer(
  #       rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
  #   )
  #   rollout_worker = vanilla_rollout.VanillaRollout(
  #       model=sampler_model,
  #       tokenizer=vocab,
  #       cache_config=sampler_lib.CacheConfig(
  #           cache_size=1024,
  #           num_layers=4,
  #           num_kv_heads=4,
  #           head_dim=16,
  #       ),
  #   )
  #   grpo_config = grpo_lib.GrpoConfig(
  #       total_generation_steps=10,
  #       num_generations=2,
  #       eval_every_n_steps=10,
  #       max_steps=10,
  #       max_prompt_length=256,
  #   )
  #   grpo_trainer = grpo_lib.GrpoLearner(
  #       model=model,
  #       ref_model=ref_model,
  #       rollout_worker=rollout_worker,
  #       reward_fns=reward_1,
  #       optimizer=optax.sgd(1e-3),
  #       training_config=grpo_config,
  #       trainer_mesh=mesh1,
  #       rollout_worker_mesh=mesh2,
  #   )
  #   self.assertTrue(grpo_trainer.need_sync_sampler_weights)
  #   train_ds = _dummy_dataset(batch_size=2)
  #   grpo_trainer.train(train_ds, None)

  #   base_params = nnx.state(model, filterlib.Not(nnx.LoRAParam))
  #   lora_params = nnx.state(model, nnx.LoRAParam)
  #   lora_params_from_sampler = nnx.state(
  #       grpo_trainer.rollout_worker.model(), nnx.LoRAParam
  #   )
  #   jax.tree.map_with_path(
  #       tc.assert_not_equal, original_lora_variables, lora_params
  #   )
  #   jax.tree.map_with_path(
  #       tc.assert_equal, lora_params_from_sampler, lora_params
  #   )
  #   jax.tree.map_with_path(tc.assert_equal, original_base_params, base_params)

  def test_exception_from_data_preparation(self):
    class _TrainerWithException(grpo_lib.GrpoLearner):

      @override
      def _generate_and_compute_advantage(self, example, mode='train'):
        raise ValueError('test exception')

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )
    rollout_worker = vanilla_rollout.VanillaRollout(
        model=model,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    grpo_config = grpo_lib.GrpoConfig(
        total_generation_steps=10,
        num_generations=2,
        eval_every_n_steps=1,
        max_steps=10,
        max_prompt_length=256,
        num_iterations=1,
    )
    grpo_trainer = _TrainerWithException(
        model=model,
        ref_model=ref_model,
        rollout_worker=rollout_worker,
        reward_fns=reward_1,
        optimizer=optax.sgd(1e-3),
        training_config=grpo_config,
    )
    train_ds = _dummy_dataset(batch_size=2)
    with self.assertRaises(ValueError):
      grpo_trainer.train(train_ds, None)

  def test_resume_training(self):
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )
    rollout_worker = vanilla_rollout.VanillaRollout(
        model=model,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    grpo_config = grpo_lib.GrpoConfig(
        total_generation_steps=10,
        num_generations=2,
        eval_every_n_steps=1,
        max_steps=10,
        max_prompt_length=256,
        num_iterations=1,
    )
    grpo_trainer = grpo_lib.GrpoLearner(
        model=model,
        ref_model=ref_model,
        rollout_worker=rollout_worker,
        reward_fns=reward_1,
        optimizer=optax.sgd(1e-3),
        training_config=grpo_config,
    )
    train_ds_full = _dummy_dataset(batch_size=2)
    grpo_trainer.train(train_ds_full, None)

    temp_path = self.create_tempdir().full_path
    grpo_config2 = grpo_lib.GrpoConfig(
        total_generation_steps=10,
        num_generations=2,
        eval_every_n_steps=1,
        max_steps=10,
        max_prompt_length=256,
        num_iterations=1,
        checkpoint_root_directory=temp_path,
        checkpointing_options=ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            max_to_keep=10,
        ),
    )

    model2 = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )
    rollout_worker2 = vanilla_rollout.VanillaRollout(
        model=model2,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    grpo_trainer2 = grpo_lib.GrpoLearner(
        model=model2,
        ref_model=ref_model,
        rollout_worker=rollout_worker2,
        reward_fns=reward_1,
        optimizer=optax.sgd(1e-3),
        training_config=grpo_config2,
    )
    grpo_trainer2.train(train_ds_full[0:1], None)
    grpo_trainer2 = grpo_lib.GrpoLearner(
        model=model2,
        ref_model=ref_model,
        rollout_worker=rollout_worker2,
        reward_fns=reward_1,
        optimizer=optax.sgd(1e-3),
        training_config=grpo_config2,
    )
    assert grpo_trainer2._last_train_step == 1
    grpo_trainer2.train(train_ds_full, None)

    variables1 = nnx.state(model, nnx.Param)
    variables2 = nnx.state(model2, nnx.Param)
    jax.tree.map_with_path(tc.assert_equal, variables1, variables2)


if __name__ == '__main__':
  absltest.main()
