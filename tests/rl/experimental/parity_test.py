# Copyright 2026 Google LLC
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


import copy
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import grain.python as grain
import jax
from jax import sharding
from jax.interpreters import pxla
import numpy as np
import optax
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.experimental import agentic_grpo_learner
from tunix.rl.grpo import grpo_learner as grpo_learner_lib
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
Mesh = sharding.Mesh


class MockChatParser:

  def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
    # This mock parser ensures that the chat messages are flattened to a simple
    # string that matches what the standard GRPO learner receives (raw prompt).
    if not messages:
      return ''
    # We assume messages are like [{"role": "user", "content": "..."}]
    # We just return the content of the last message (user message).
    content = messages[-1]['content']
    # Prepend <s> to match the BOS token added by VanillaRollout/Sampler only
    # for the first message (which is the System message in Agentic).
    if is_first_msg:
      return '<s> ' + content
    return content

  @property
  def assistant_token(self):
    return ''


def _dummy_dataset(source, batch_size: int = 1) -> grain.MapDataset:
  return (
      grain.MapDataset.source(source)
      .batch(batch_size, drop_remainder=True)
      .map(lambda x: {'prompts': x, 'answer': x, 'question': x})
  )


def reward_fn(completions: list[str], **kargs: Any) -> list[float]:  # pylint: disable=unused-argument
  return [float(len(c)) for c in completions]


class ParityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='multi_iter_without_gradient_accumulation',
          name='multi_iter_without_gradient_accumulation',
          full_batch_size=4,
          num_generations=2,
          mini_batch_size=1,
          train_micro_batch_size=1,
          num_iterations=2,
          beta=0.04,
      ),
      dict(
          testcase_name='multi_iter_with_gradient_accumulation',
          name='multi_iter_with_gradient_accumulation',
          full_batch_size=4,
          num_generations=2,
          mini_batch_size=4,
          train_micro_batch_size=1,
          num_iterations=2,
          beta=0.04,
      ),
      dict(
          testcase_name='multi_iter_without_kl',
          name='multi_iter_without_kl',
          full_batch_size=4,
          num_generations=2,
          mini_batch_size=1,
          train_micro_batch_size=1,
          num_iterations=2,
          beta=0,
      ),
      dict(
          testcase_name='single_iter_without_gradient_accumulation',
          name='single_iter_without_gradient_accumulation',
          full_batch_size=4,
          num_generations=2,
          mini_batch_size=1,
          train_micro_batch_size=1,
          num_iterations=1,
          beta=0.04,
      ),
      dict(
          testcase_name='single_iter_with_gradient_accumulation',
          name='single_iter_with_gradient_accumulation',
          full_batch_size=4,
          num_generations=2,
          mini_batch_size=4,
          train_micro_batch_size=1,
          num_iterations=1,
          beta=0.04,
      ),
      dict(
          testcase_name='single_iter_without_kl',
          name='single_iter_without_kl',
          full_batch_size=4,
          num_generations=2,
          mini_batch_size=1,
          train_micro_batch_size=1,
          num_iterations=1,
          beta=0,
      ),
      dict(
          testcase_name='global2_mini2_micro2',
          name='global4_mini2_micro2',
          full_batch_size=2,
          num_generations=2,
          mini_batch_size=2,
          train_micro_batch_size=2,
          num_iterations=2,
          beta=0.04,
      ),
      dict(
          testcase_name='global4_mini4_micro2',
          name='global4_mini4_micro2',
          full_batch_size=4,
          num_generations=4,
          mini_batch_size=4,
          train_micro_batch_size=2,
          num_iterations=1,
          beta=0.04,
      ),
      dict(
          testcase_name='global4_mini2_micro2',
          name='global4_mini2_micro2',
          full_batch_size=4,
          num_generations=2,
          mini_batch_size=2,
          train_micro_batch_size=2,
          num_iterations=1,
          beta=0.04,
      ),
      dict(
          testcase_name='global4_mini2_micro1',
          name='global4_mini2_micro1',
          full_batch_size=4,
          num_generations=2,
          mini_batch_size=2,
          train_micro_batch_size=1,
          num_iterations=1,
          beta=0.04,
      ),
  )
  def test_model_weights_parity(
      self,
      name,
      full_batch_size,
      num_generations,
      mini_batch_size,
      train_micro_batch_size,
      num_iterations,
      beta,
  ):
    vocab = tc.MockVocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)

    # Ensure tokenizer has apply_chat_template which behaves like MockChatParser
    # so Agentic's chat-formatted prompt matches Standard's raw prompt.
    if not hasattr(tokenizer, 'apply_chat_template'):

      def mock_apply_chat_template(messages, **kwargs):
        return messages[-1]['content']

      tokenizer.apply_chat_template = mock_apply_chat_template

    # Patch tokenizer.encode to ensure consistent tokenization (e.g. no BOS/EOS)
    # between Agentic (which calls encode with add_special_tokens=False) and
    # Standard (which uses VanillaRollout defaults).
    def mock_encode(text, add_special_tokens=False):
      del add_special_tokens  # Ignore flag to enforce parity
      return vocab.EncodeAsIds(text)

    tokenizer.encode = mock_encode

    model1 = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    model2 = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    # Initialize weights identically
    variables1 = nnx.state(model1, nnx.Param)
    variables2 = nnx.state(model2, nnx.Param)
    jax.tree.map_with_path(tc.assert_close, variables1, variables2)

    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    # Common Configs
    eval_every_n_steps = 12
    max_prompt_length = 256
    max_generation_steps = 10

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
            eval_every_n_steps=eval_every_n_steps,
            max_steps=8 * num_iterations // mini_batch_size,
            gradient_accumulation_steps=None,
            # Ensure batch sizes match
            mini_batch_size=mini_batch_size,
            train_micro_batch_size=train_micro_batch_size,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=max_generation_steps,
            max_prompt_length=max_prompt_length,
            kv_cache_size=1024,
            temperature=0.0,  # Deterministic sampling
        ),
    )

    # 1. Setup Standard GRPO Learner
    rl_cluster1 = rl_cluster_lib.RLCluster(
        actor=model1,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    grpo_config1 = grpo_learner_lib.GRPOConfig(
        num_generations=num_generations,
        num_iterations=num_iterations,
        beta=beta,
        loss_algo='grpo',
    )

    grpo_learner = grpo_learner_lib.GRPOLearner(
        rl_cluster=rl_cluster1,
        reward_fns=reward_fn,
        algo_config=grpo_config1,
    )

    # 2. Setup Agentic GRPO Learner
    rl_cluster2 = rl_cluster_lib.RLCluster(
        actor=model2,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    grpo_config2 = agentic_grpo_learner.GRPOConfig(
        num_generations=num_generations,
        num_iterations=num_iterations,
        beta=beta,
        loss_algo='grpo',
        system_prompt='',
        max_concurrency=1,
    )

    agentic_learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster2,
        reward_fns=reward_fn,
        algo_config=grpo_config2,
        chat_parser=MockChatParser(),
    )

    # Data
    prompts = ['input string', 'hello world', 'My name', 'hello there']
    # Repeat data to ensure enough steps
    train_ds = _dummy_dataset(prompts * 2, batch_size=full_batch_size)

    # Run Training
    with mock.patch.object(
        rl_cluster1,
        'update_actor',
        wraps=rl_cluster1.update_actor,
        autospec=True,
    ) as mock_update1, mock.patch.object(
        rl_cluster1,
        'sync_weights',
        wraps=rl_cluster1.sync_weights,
        autospec=True,
    ) as mock_sync1, mock.patch.object(
        rl_cluster2,
        'update_actor',
        wraps=rl_cluster2.update_actor,
        autospec=True,
    ) as mock_update2, mock.patch.object(
        rl_cluster2,
        'sync_weights',
        wraps=rl_cluster2.sync_weights,
        autospec=True,
    ) as mock_sync2:
      grpo_learner.train(list(train_ds), None)
      agentic_learner.train(list(train_ds), None)

      with self.subTest('Call Counts Parity'):
        self.assertEqual(
            mock_update1.call_count,
            mock_update2.call_count,
            msg='update_actor call count mismatch: %d != %d'
            % (mock_update1.call_count, mock_update2.call_count),
        )
        self.assertEqual(
            mock_sync1.call_count,
            mock_sync2.call_count,
            msg='sync_weights call count mismatch: %d != %d'
            % (mock_sync1.call_count, mock_sync2.call_count),
        )

      with self.subTest('Data Parity'):
        # Verify update_actor arguments (Data Parity)
        def get_train_examples(mock_update):
          examples = []
          for call in mock_update.call_args_list:
            # args[0] is the batch (List[TrainExample])
            batch = call.args[0]
            examples.extend(batch)
          return examples

        examples1 = sorted(
            get_train_examples(mock_update1),
            key=lambda ex: (
                ex.prompt_ids.tobytes(),
                ex.completion_ids.tobytes(),
            ),
        )
        examples2 = sorted(
            get_train_examples(mock_update2),
            key=lambda ex: (
                ex.prompt_ids.tobytes(),
                ex.completion_ids.tobytes(),
            ),
        )

        self.assertEqual(
            len(examples1),
            len(examples2),
            msg='Number of training examples passed to update_actor mismatch',
        )

        for i, (ex1, ex2) in enumerate(zip(examples1, examples2)):
          np.testing.assert_array_equal(
              ex1.prompt_ids,
              ex2.prompt_ids,
              err_msg=f'prompt_ids mismatch at index {i}',
          )
          np.testing.assert_array_equal(
              ex1.completion_ids,
              ex2.completion_ids,
              err_msg=f'completion_ids mismatch at index {i}',
          )
          np.testing.assert_allclose(
              ex1.advantages,
              ex2.advantages,
              atol=1e-5,
              err_msg=f'advantages mismatch at index {i}',
          )

    with self.subTest('Model Weights Parity'):
      variables1 = nnx.state(model1, nnx.Param)
      variables2 = nnx.state(model2, nnx.Param)
      jax.tree.map_with_path(tc.assert_equal, variables1, variables2)


if __name__ == '__main__':
  absltest.main()
