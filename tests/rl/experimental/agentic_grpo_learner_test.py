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

"""Tests for agentic_grpo_learner."""

import asyncio
import os
import random
import shutil
import tempfile
import types
from typing import Iterable
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
from flax.nnx import filterlib
import grain.python as grain
import jax
from jax import sharding
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.generate import tokenizer_adapter
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.experimental import agentic_grpo_learner
from tunix.rl.queue import data_queue as queue_lib
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common
from typing_extensions import override

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
Mesh = sharding.Mesh
TrainingInputT = agentic_grpo_learner.TrainingInputT


def reward_fn_1(prompts, completions, **kwargs):
  del prompts, kwargs
  return [float(i) for i in range(len(completions))]


def reward_fn_2(answer, **kwargs):
  del kwargs
  return [float(i) for i in range(len(answer))]


_MOCK_RESPONSES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly changing the world.",
    "Flax is a neural network library for JAX.",
    "Reinforcement learning can be used to train agents.",
    "Hello there! How can I help you today?",
    "This is a sample response from the model.",
]


def _mock_generate(
    prompts: list[str] | list[list[dict[str, str]]],
    apply_chat_template: bool = False,
    mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
    micro_batch_size: int | None = None,
) -> base_rollout.RolloutOutput:
  del apply_chat_template, mode, micro_batch_size
  batch_size = len(prompts)
  return base_rollout.RolloutOutput(
      text=[random.choice(_MOCK_RESPONSES) for _ in range(batch_size)],
      tokens=np.ones((batch_size, 10), dtype=np.int32),
      left_padded_prompt_tokens=np.ones((batch_size, 8), dtype=np.int32),
      logits=None,
      logprobs=None,
  )


class MySource(grain.RandomAccessDataSource):

  def __init__(self, data=None, repeat=1):
    if data is None:
      data = ["input string", "hello world", "My name is", "hello there"]
    self._data = data * repeat

  def __getitem__(self, idx):
    return self._data[idx]

  def __len__(self):
    return len(self._data)


def _dummy_dataset(source=MySource(), batch_size: int = 1):
  return (
      grain.MapDataset.source(source)
      .batch(batch_size)
      .map(lambda x: {"prompts": x, "answer": x, "question": x})
  )


class MockChatParser:

  def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
    del add_generation_prompt, is_first_msg
    if not messages:
      return ""
    if messages[0]["role"] == "system":
      return f"System: {messages[0]['content']}"
    if messages[0]["role"] == "user":
      return f"User: {messages[0]['content']}"
    if messages[0]["role"] == "assistant":
      return f"Assistant: {messages[0]['content']}"
    return ""

  @property
  def assistant_token(self):
    return ""


class _LearnerWithException(agentic_grpo_learner.GRPOLearner):

  def _batch_to_train_example(
      self, batch_results, cached_inputs_for_window, mode
  ):
    raise ValueError("test exception in producer")


class AgenticGrpoLearnerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    chex.set_n_cpu_devices(2)
    cls.device_count = jax.device_count()

  def setUp(self):
    super().setUp()
    random.seed(42)

  def test_iterator(self):
    class _MockTrainer(agentic_grpo_learner.GRPOLearner):

      def __init__(self, algo_config):
        self.algo_config = algo_config
        self.rl_cluster = mock.Mock()
        self.rl_cluster.buffer_metrics = mock.Mock()
        self.metric_fns = []

      def _create_micro_batch_iterator(self, iterator, batch_size):
        # The dataset batch size is 2, and we want to test micro-batching
        # of size 1, as consumed by _orchestrator_producer.
        for batch in iterator:
          for i in range(len(batch["prompts"])):
            yield jax.tree.map(lambda x: x[i : i + 1], batch)

      @override
      def _batch_to_train_example(
          self, batch_results, cached_inputs_for_window, mode
      ):
        del batch_results, mode
        examples = []
        for _ in range(self.algo_config.num_generations):
          examples.append(
              types.SimpleNamespace(
                  prompt_ids=np.array(
                      [cached_inputs_for_window[0]["prompts"][0]]
                  ),
              )
          )
        return examples

      @override
      def _compute_trajectory_ids(
          self, example: TrainingInputT, prompt_index: int
      ) -> list[str]:
        return [
            f"{prompt_index}_{i}"
            for i in range(self.algo_config.num_generations)
        ]

      @override
      async def _orchestrator_producer(
          self,
          orchestrator,
          prompt_iterator: Iterable[TrainingInputT],
          num_generations: int = 1,
          collect_mode: str = "Token",
      ):
        for i, example in enumerate(prompt_iterator):
          group = [
              types.SimpleNamespace(
                  pair_index=i * self.algo_config.num_generations + j
              )
              for j in range(self.algo_config.num_generations)
          ]
          yield group, [example]

    algo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2, num_iterations=2
    )
    trainer = _MockTrainer(algo_config)

    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)
    dataset = _dummy_dataset(MySource(data=[i for i in range(2)]), batch_size=2)

    asyncio.run(trainer._producer(mock.Mock(), iter(dataset), train_data_queue))

    results = []
    while True:
      item = train_data_queue.get(block=True)
      if item is None:
        break
      results.append(item)

    prompt_ids = [r.prompt_ids[0] for r in results]
    self.assertEqual(prompt_ids, [0, 0, 0, 0, 1, 1, 1, 1])

  def test_grpo_config_validation(self):
    with self.assertRaisesRegex(
        ValueError, "num_generations must be greater than 1"
    ):
      agentic_grpo_learner.GRPOConfig(num_generations=1)
    with self.assertRaisesRegex(
        ValueError, "loss_algo should be either grpo or gspo-token"
    ):
      agentic_grpo_learner.GRPOConfig(loss_algo="invalid")

  def test_num_iterations_greater_than_1(self):
    vocab = test_common.MockVocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=100,  # do not run eval
            max_steps=10,
            gradient_accumulation_steps=None,
            mini_batch_size=1,
            train_micro_batch_size=1,  # to control calls to update_actor
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
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=2,  # > 1
        loss_algo="grpo",
    )
    grpo_learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=grpo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
    )

    train_ds = _dummy_dataset(
        MySource(data=["1", "2", "3", "4"], repeat=1), batch_size=1
    )

    with (
        mock.patch.object(
            grpo_learner,
            "_batch_to_train_example",
            wraps=grpo_learner._batch_to_train_example,
        ) as mock_b2te,
        mock.patch.object(
            rl_cluster, "update_actor", wraps=rl_cluster.update_actor
        ) as mock_update_actor,
    ):
      grpo_learner.train(train_ds)

      # 4 prompts, so _batch_to_train_example is called 4 times.
      self.assertEqual(mock_b2te.call_count, 4)
      # Each prompt (_b2te call) produces num_generations=2 examples.
      # For each example, producer loops num_iterations=2 times.
      # Total examples in train_data_queue = 4 * 2 * 2 = 16 examples.
      # train_micro_batch_size=1, num_generations=2.
      # _data_consumer_batch_generator batch size = 1*2=2 elements from queue.
      # 16 examples are grouped into 16/2 = 8 batches for update_actor.
      self.assertGreater(mock_update_actor.call_count, mock_b2te.call_count)
      self.assertEqual(mock_update_actor.call_count, 8)

  @parameterized.parameters("grpo", "gspo-token")
  def test_grpo_loss_fn(self, loss_algo):
    batch_size, seq_len = 2, 8
    prompt_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    completion_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    completion_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
    advantages = jnp.ones((batch_size,), dtype=jnp.float32)
    ref_per_token_logps = jnp.full(
        (batch_size, seq_len), -0.1, dtype=jnp.float32
    )

    train_example = agentic_grpo_learner.TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_ids > -1,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=None,
    )

    class MockModel(nnx.Module):

      def __init__(self, *, rngs: nnx.Rngs):
        self.lm_head = 1

      def __call__(self, inputs, positions, cache, attention_mask):
        return (
            jnp.full(
                (*inputs.shape, 32),
                0.1,
                dtype=jnp.float32,
            ),
            None,
        )
    algo_config = agentic_grpo_learner.GRPOConfig(
        beta=0.1,
        epsilon=0.2,
        loss_algo=loss_algo,
    )
    policy_loss_fn = function_registry.get_policy_loss_fn(
        algo_config.policy_loss_fn
    )
    loss, aux = policy_loss_fn(
        model=MockModel(rngs=nnx.Rngs(0)),
        train_example=train_example,
        algo_config=algo_config,
        pad_id=0,
        eos_id=2,
    )
    chex.assert_shape(loss, ())
    self.assertIn("kl", aux)

  def test_checkpointing(self):
    ckpt_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, ckpt_dir)
    mini_batch_size = 1

    def create_learner(
        ckpt_dir,
        max_steps,
    ):
      vocab = test_common.MockVocab()
      tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
      model = test_common.ToyTransformer(
          config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      )
      ref_model = test_common.ToyTransformer(
          config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      )

      mesh = pxla.thread_resources.env.physical_mesh
      cluster_config = rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine="vanilla",
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
                  save_interval_steps=1,
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
          tokenizer=tokenizer,
          cluster_config=cluster_config,
      )

      grpo_config = agentic_grpo_learner.GRPOConfig(
          num_generations=2,
          num_iterations=1,
      )
      grpo_learner = agentic_grpo_learner.GRPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=reward_fn_1,
          algo_config=grpo_config,
          chat_parser=MockChatParser(),
      )
      return grpo_learner

    train_ds = [
        {"prompts": [str(i)], "answer": [str(i)], "question": [str(i)]}
        for i in range(4)
    ]

    grpo_learner = create_learner(ckpt_dir, max_steps=10)
    self.assertEqual(grpo_learner.rl_cluster.global_steps, 0)
    # Train for 1 step.
    grpo_learner.train(train_ds[0:1])
    self.assertEqual(grpo_learner.rl_cluster.global_steps, 1)

    # Resume training with a new learner.
    grpo_learner2 = create_learner(ckpt_dir, max_steps=3)
    self.assertEqual(grpo_learner2.rl_cluster.global_steps, 1)

    grpo_learner2.train(train_ds)
    self.assertEqual(grpo_learner2.rl_cluster.global_steps, 3)

  @parameterized.named_parameters(
      dict(
          testcase_name="single_update",
          batch_size=8,
          mini_batch_size=8,
          train_micro_batch_size=4,
      ),
      dict(
          testcase_name="multi_update",
          batch_size=8,
          mini_batch_size=4,
          train_micro_batch_size=2,
      ),
  )
  def test_micro_batch_training(
      self,
      batch_size,
      mini_batch_size,
      train_micro_batch_size,
  ):
    def reward_fn_for_tracking(trajectories, prompts, **kwargs):
      for t_id, prompt in zip(kwargs["trajectory_ids"], prompts):
        trajectories[kwargs["mode"]][t_id] = prompt
      return [1.0] * len(prompts)

    def create_learner(
        mini_batch_size,
        train_micro_batch_size,
        trajectories,
    ):
      vocab = test_common.MockVocab()
      tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
      model = test_common.ToyTransformer(
          config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      )
      ref_model = test_common.ToyTransformer(
          config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      )

      mesh = pxla.thread_resources.env.physical_mesh
      cluster_config = rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine="vanilla",
          offload_to_cpu=False,
          training_config=rl_cluster_lib.RLTrainingConfig(
              actor_optimizer=optax.sgd(1e-3),
              eval_every_n_steps=10,
              max_steps=20,
              mini_batch_size=mini_batch_size,
              train_micro_batch_size=train_micro_batch_size,
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
          tokenizer=tokenizer,
          cluster_config=cluster_config,
      )

      grpo_config = agentic_grpo_learner.GRPOConfig(
          num_generations=2,
          num_iterations=1,
      )
      grpo_learner = agentic_grpo_learner.GRPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=lambda **kwargs: reward_fn_for_tracking(
              trajectories=trajectories, **kwargs
          ),
          algo_config=grpo_config,
          chat_parser=MockChatParser(),
      )
      return grpo_learner

    train_ds = [{
        "prompts": [str(i) for i in range(batch_size)],
        "answer": [str(i) for i in range(batch_size)],
        "question": [str(i) for i in range(batch_size)],
    }]

    # Baseline with no micro batching for train updates.
    base_trajectories = {"train": {}, "eval": {}}
    grpo_learner_base = create_learner(
        mini_batch_size=None,
        train_micro_batch_size=None,
        trajectories=base_trajectories,
    )
    grpo_learner_base.train(train_ds)

    # Train with micro batching for train updates.
    micro_batch_trajectories = {"train": {}, "eval": {}}
    grpo_learner_micro = create_learner(
        mini_batch_size=mini_batch_size,
        train_micro_batch_size=train_micro_batch_size,
        trajectories=micro_batch_trajectories,
    )
    grpo_learner_micro.train(train_ds)

    self.assertEqual(base_trajectories, micro_batch_trajectories)
    self.assertEqual(
        grpo_learner_base.rl_cluster.global_steps,
        grpo_learner_micro.rl_cluster.global_steps,
    )
    self.assertEqual(grpo_learner_base.rl_cluster.global_steps, 1)

  def test_trajectory_ids(self):
    def reward_fn_for_tracking(trajectories, prompts, **kwargs):
      for t_id, prompt in zip(kwargs["trajectory_ids"], prompts):
        trajectories[kwargs["mode"]][t_id] = prompt
      return [1.0] * len(prompts)

    def create_learner(
        mini_batch_size,
        train_micro_batch_size,
        trajectories,
    ):
      vocab = test_common.MockVocab()
      tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
      model = test_common.ToyTransformer(
          config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      )
      ref_model = test_common.ToyTransformer(
          config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      )

      mesh = pxla.thread_resources.env.physical_mesh
      cluster_config = rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine="vanilla",
          offload_to_cpu=False,
          training_config=rl_cluster_lib.RLTrainingConfig(
              actor_optimizer=optax.sgd(1e-3),
              eval_every_n_steps=10,
              max_steps=20,
              mini_batch_size=mini_batch_size,
              train_micro_batch_size=train_micro_batch_size,
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
          tokenizer=tokenizer,
          cluster_config=cluster_config,
      )

      grpo_config = agentic_grpo_learner.GRPOConfig(
          num_generations=2,
          num_iterations=1,
      )
      grpo_learner = agentic_grpo_learner.GRPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=lambda **kwargs: reward_fn_for_tracking(
              trajectories=trajectories, **kwargs
          ),
          algo_config=grpo_config,
          chat_parser=MockChatParser(),
      )
      return grpo_learner, model

    train_ds = [{
        "prompts": [str(i) for i in range(8)],
        "answer": [str(i) for i in range(8)],
        "question": [str(i) for i in range(8)],
    }]

    # Config 1: mini_batch_size=4, train_micro_batch_size=4
    trajectories1 = {"train": {}, "eval": {}}
    learner1, model1 = create_learner(4, 4, trajectories1)
    learner1.train(train_ds)

    # Config 2: mini_batch_size=8, train_micro_batch_size=2
    trajectories2 = {"train": {}, "eval": {}}
    learner2, model2 = create_learner(8, 2, trajectories2)
    learner2.train(train_ds)

    params1 = nnx.state(model1, nnx.Param)
    params2 = nnx.state(model2, nnx.Param)
    jax.tree.map_with_path(test_common.assert_close, params1, params2)

  def test_resume_training(self):
    ckpt_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, ckpt_dir)
    mini_batch_size = 1

    def create_learner(
        ckpt_dir,
        max_steps,
        reward_fn=reward_fn_1,
    ):
      vocab = test_common.MockVocab()
      tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
      model = test_common.ToyTransformer(
          config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      )
      ref_model = test_common.ToyTransformer(
          config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      )

      mesh = pxla.thread_resources.env.physical_mesh
      if ckpt_dir:
        checkpointing_options = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
        )
      else:
        checkpointing_options = None
      training_config = rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-3),
          eval_every_n_steps=10,  # avoid eval
          max_steps=max_steps,
          mini_batch_size=mini_batch_size,
          train_micro_batch_size=mini_batch_size,
          rollout_micro_batch_size=mini_batch_size,
          compute_logps_micro_batch_size=mini_batch_size,
          checkpointing_options=checkpointing_options,
          checkpoint_root_directory=ckpt_dir,
      )
      cluster_config = rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine="vanilla",
          offload_to_cpu=False,
          training_config=training_config,
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
          tokenizer=tokenizer,
          cluster_config=cluster_config,
      )

      grpo_config = agentic_grpo_learner.GRPOConfig(
          num_generations=2,
          num_iterations=1,
      )
      grpo_learner = agentic_grpo_learner.GRPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=reward_fn,
          algo_config=grpo_config,
          chat_parser=MockChatParser(),
      )
      return grpo_learner, model

    train_ds = [
        {"prompts": [str(i)], "answer": [str(i)], "question": [str(i)]}
        for i in range(2)
    ]

    # 1. Train in one go
    grpo_learner_full, model_full = create_learner(ckpt_dir=None, max_steps=2)
    grpo_learner_full.train(train_ds)
    self.assertEqual(grpo_learner_full.rl_cluster.global_steps, 2)

    # 2. Train interrupted
    grpo_learner_interrupt, _ = create_learner(ckpt_dir=ckpt_dir, max_steps=1)
    grpo_learner_interrupt.train(train_ds)
    self.assertEqual(grpo_learner_interrupt.rl_cluster.global_steps, 1)

    # 3. Resume training
    grpo_learner_resume, model_resume = create_learner(
        ckpt_dir=ckpt_dir, max_steps=2
    )
    self.assertEqual(grpo_learner_resume.rl_cluster.global_steps, 1)
    grpo_learner_resume.train(train_ds)
    self.assertEqual(grpo_learner_resume.rl_cluster.global_steps, 2)

    # 4. Compare weights
    params1 = nnx.state(model_full, nnx.Param)
    params2 = nnx.state(model_resume, nnx.Param)
    jax.tree.map_with_path(test_common.assert_close, params1, params2)

  def test_exception_handling(self):
    vocab = test_common.MockVocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            max_steps=2,
            eval_every_n_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=32,
            kv_cache_size=256,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    grpo_config = agentic_grpo_learner.GRPOConfig()
    learner = _LearnerWithException(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=grpo_config,
        chat_parser=MockChatParser(),
    )
    train_ds = [{"prompts": ["1"], "answer": ["1"], "question": ["1"]}]
    with self.assertRaisesRegex(ValueError, "test exception in producer"):
      learner.train(train_ds)

  @parameterized.named_parameters(
      dict(
          testcase_name="single_reward_fn",
          reward_fns=reward_fn_1,
          loss_algo="grpo",
      ),
      dict(
          testcase_name="multiple_reward_fns",
          reward_fns=[
              reward_fn_1,
              reward_fn_2,
          ],
          loss_algo="grpo",
      ),
      dict(
          testcase_name="single_reward_fn_gspo",
          reward_fns=reward_fn_1,
          loss_algo="gspo-token",
      ),
  )
  def test_grpo_learner(self, reward_fns, loss_algo):
    vocab = test_common.MockVocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=2,
            max_steps=20,
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
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    rl_cluster.with_external_metrics_logger(print)

    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        loss_algo=loss_algo,
    )
    grpo_learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=grpo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
    )
    self.assertFalse(grpo_learner.should_sync_weights)
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    eval_ds = _dummy_dataset(batch_size=1)

    with mock.patch.object(rl_cluster, "generate", side_effect=_mock_generate):
      grpo_learner.train(train_ds, eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_variables, variables
    )

    self.assertEqual(
        grpo_learner.rl_cluster.global_steps,
        20,
    )

    rl_metric_logger = grpo_learner.rl_cluster._rl_metrics_logger

    rewards_metrics = (
        ("rewards/" + f.__name__ for f in reward_fns)
        if isinstance(reward_fns, list)
        else ("rewards/" + reward_fns.__name__,)
    )
    for metric_name in [
        "rewards/sum",
        *rewards_metrics,
        "completions/mean_length",
        "completions/max_length",
        "completions/min_length",
        "test_metric",
    ]:
      if metric_name == "rewards/reward_fn_2" and not isinstance(
          reward_fns, list
      ):
        continue
      # We log metrics per step, and sometimes one extra step is logged due to
      # buffer flushing. So we check if length is close to global_steps.
      self.assertGreaterEqual(
          len(
              rl_metric_logger.get_metric_history(
                  "global", metric_name, "train"
              )
          ),
          grpo_learner.rl_cluster.global_steps,
          msg=f"metric_name: {metric_name}",
      )
      self.assertLen(
          rl_metric_logger.get_metric_history("global", metric_name, "eval"),
          10,
          msg=f"metric_name: {metric_name}",
      )

    metric_logger = grpo_learner.rl_cluster.actor_trainer.metrics_logger
    for metric_name in ["loss", "kl"]:
      self.assertLen(
          metric_logger.get_metric_history("actor", metric_name, "train"),
          grpo_learner.rl_cluster.actor_trainer.train_steps,
          msg=f"metric_name: {metric_name}",
      )
      self.assertLen(
          metric_logger.get_metric_history("actor", metric_name, "eval"),
          10,
          msg=f"metric_name: {metric_name}",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="on_policy",
          offpolicy_steps=0,
      ),
      dict(
          testcase_name="off_policy_step_1",
          offpolicy_steps=1,
      ),
      dict(
          testcase_name="off_policy_step_2",
          offpolicy_steps=2,
      ),
  )
  def test_on_off_policy_training(self, offpolicy_steps):
    vocab = test_common.MockVocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=2,
            max_steps=4,
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
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        loss_algo="grpo",
        off_policy_steps=offpolicy_steps,
    )
    grpo_learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=grpo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
    )
    train_ds = _dummy_dataset(MySource(repeat=4), batch_size=2)
    eval_ds = _dummy_dataset(batch_size=1)

    with mock.patch.object(rl_cluster, "generate", side_effect=_mock_generate):
      grpo_learner.train(train_ds, eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_variables, variables
    )

    self.assertEqual(
        grpo_learner.rl_cluster.global_steps,
        4,
    )

  @unittest.skip("b/461854722")
  def test_grpo_with_lora_model(self):
    # reshard through default device_put.
    split_index = self.device_count // 2
    mesh1 = Mesh(
        np.array(
            sorted(jax.devices(), key=lambda d: d.id)[:split_index]
        ).reshape(split_index, 1),
        ("fsdp", "tp"),
    )
    mesh2 = Mesh(
        np.array(
            sorted(jax.devices(), key=lambda d: d.id)[split_index:]
        ).reshape(1, split_index),
        ("fsdp", "tp"),
    )
    vocab = test_common.MockVocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    actor_model = test_common.get_lora_model(
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
        rollout_engine="vanilla",
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
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
    )

    grpo_learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=grpo_config,
        chat_parser=MockChatParser(),
    )
    self.assertTrue(grpo_learner.should_sync_weights)
    train_ds = _dummy_dataset(batch_size=2)
    with mock.patch.object(rl_cluster, "generate", side_effect=_mock_generate):
      grpo_learner.train(train_ds, None)

    base_params = nnx.state(
        rl_cluster.actor_trainer.model, filterlib.Not(nnx.LoRAParam)
    )
    lora_params = nnx.state(rl_cluster.actor_trainer.model, nnx.LoRAParam)
    lora_params_from_sampler = nnx.state(
        grpo_learner.rl_cluster.rollout.model(), nnx.LoRAParam
    )
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_lora_variables, lora_params
    )
    jax.tree.map_with_path(
        test_common.assert_close, lora_params_from_sampler, lora_params
    )
    jax.tree.map_with_path(
        test_common.assert_equal, original_base_params, base_params
    )


if __name__ == "__main__":
  absltest.main()
