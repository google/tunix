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

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.nnx import filterlib
from grain import python as grain
import jax
import jax.numpy as jnp
import optax
from tunix.generate import sampler as sampler_lib
from tunix.rl import common
from tunix.rl.grpo import grpo_trainer as grpo_lib
from tunix.tests import test_common as tc

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


class GrpoTrainerTest(parameterized.TestCase):

  def test_repeat_training_input_iter(self):
    itr = grpo_lib.RepeatTrainingInputIter(
        iter(_dummy_dataset([i for i in range(4)], 2)),
        sample_repeat=5,
        batch_repeat=3,
    )
    res = [d.get('prompts').tolist() for d in itr]
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

  def test_repeat_training_input_iter_with_grad_accumulation(self):
    itr = grpo_lib.RepeatTrainingInputIter(
        iter(_dummy_dataset([i for i in range(16)], 2)),
        sample_repeat=2,
        batch_repeat=2,
        gradient_accumulation_steps=3,
    )
    res = [d.get('prompts').tolist() for d in itr]

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
        [12, 12, 13, 13],
        [14, 14, 15, 15],
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
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    grpo_config = grpo_lib.GrpoTrainingConfig(
        total_generation_steps=10,
        num_generations=2,
        eval_every_n_steps=1,
        max_steps=10,
    )
    grpo_trainer = grpo_lib.GrpoTrainer(
        model=model,
        ref_model=ref_model,
        sampler=sampler,
        reward_fns=reward_fns,
        optimizer=optax.sgd(1e-3),
        training_config=grpo_config,
    )
    train_ds = eval_ds = _dummy_dataset(batch_size=2)
    grpo_trainer.train(train_ds, eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    self.assertEqual(grpo_trainer._train_steps, 2)
    self.assertEqual(grpo_trainer._eval_steps, 4)

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
          expected_logps_fn_call_at_step=[0, 0, 2, 2, 4, 4, 6, 6, 8, 8],
      ),
      dict(
          testcase_name='multi_iter_with_gradient_accumulation',
          num_iterations=2,
          beta=0.04,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 1, 2, 6, 7, 8],
          expected_logps_fn_call_at_step=[0, 0, 1, 1, 2, 2, 6, 6, 7, 7, 8, 8],
      ),
      dict(
          testcase_name='multi_iter_without_kl',
          num_iterations=2,
          beta=0,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 1, 2, 6, 7, 8],
          expected_logps_fn_call_at_step=[0, 1, 2, 6, 7, 8],
      ),
      dict(
          testcase_name='singler_iter_with_gradient_accumulation',
          num_iterations=1,
          beta=0.04,
          gradient_accumulation_steps=3,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_logps_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
      ),
      dict(
          testcase_name='singler_iter_without_gradient_accumulation',
          num_iterations=1,
          beta=0.04,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_logps_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
      ),
      dict(
          testcase_name='singler_iter_without_kl',
          num_iterations=1,
          beta=0,
          gradient_accumulation_steps=None,
          expected_gen_fn_call_at_step=[0, 1, 2, 3, 4, 5, 6, 7],
          expected_logps_fn_call_at_step=[],
      ),
  )
  def test_multi_iteration_training(
      self,
      num_iterations,
      beta,
      gradient_accumulation_steps,
      expected_gen_fn_call_at_step,
      expected_logps_fn_call_at_step,
  ):
    gen_fn_call_at_step = []
    logps_fn_call_at_step = []

    def wrap_fn(fn, fn_call_at_step, trainer):
      def wrapper(*args, **kwargs):
        fn_call_at_step.append(trainer._train_steps)
        return fn(*args, **kwargs)

      return wrapper

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    grpo_config = grpo_lib.GrpoTrainingConfig(
        total_generation_steps=10,
        num_generations=2,
        beta=beta,
        num_iterations=num_iterations,
        eval_every_n_steps=10,
        max_steps=10,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    grpo_trainer = grpo_lib.GrpoTrainer(
        model=model,
        ref_model=ref_model,
        sampler=sampler,
        reward_fns=reward_1,
        optimizer=optax.sgd(1e-3),
        training_config=grpo_config,
    )
    grpo_trainer._generate_and_compute_advantage = wrap_fn(
        grpo_trainer._generate_and_compute_advantage,
        gen_fn_call_at_step,
        grpo_trainer,
    )
    common.get_per_token_logps = wrap_fn(
        common.get_per_token_logps, logps_fn_call_at_step, grpo_trainer
    )

    train_ds = _dummy_dataset(_DUMMY_DATA * 2, batch_size=1)
    grpo_trainer.train(train_ds, None)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    self.assertEqual(gen_fn_call_at_step, expected_gen_fn_call_at_step)
    self.assertEqual(logps_fn_call_at_step[1:], expected_logps_fn_call_at_step)

  def test_grpo_with_lora_model(self):
    vocab = tc.MockVocab()
    model = tc.get_lora_model(
        tc.ToyTransformer(rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize())
    )
    original_base_params = jax.tree.map(
        jnp.copy, nnx.state(model, filterlib.Not(nnx.LoRAParam))
    )
    original_lora_variables = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.LoRAParam)
    )
    ref_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=vocab.GetPieceSize()
    )
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
            num_layers=4,
            num_kv_heads=4,
            head_dim=16,
        ),
    )
    grpo_config = grpo_lib.GrpoTrainingConfig(
        total_generation_steps=10,
        num_generations=2,
        eval_every_n_steps=10,
        max_steps=10,
    )
    grpo_trainer = grpo_lib.GrpoTrainer(
        model=model,
        ref_model=ref_model,
        sampler=sampler,
        reward_fns=reward_1,
        optimizer=optax.sgd(1e-3),
        training_config=grpo_config,
    )
    train_ds = _dummy_dataset(batch_size=2)
    grpo_trainer.train(train_ds, None)

    base_params = nnx.state(model, filterlib.Not(nnx.LoRAParam))
    lora_params = nnx.state(model, nnx.LoRAParam)
    jax.tree.map_with_path(
        tc.assert_not_equal, original_lora_variables, lora_params
    )
    jax.tree.map_with_path(tc.assert_equal, original_base_params, base_params)


if __name__ == '__main__':
  absltest.main()
