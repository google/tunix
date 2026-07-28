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

"""Tests for agentic_ppo_learner."""

import asyncio
import functools
import os
import queue
import random
import shutil
import tempfile
import types
from typing import Any, AsyncIterable, Iterable
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
from tunix.rl import algo_core
from tunix.rl import common as rl_common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.agentic import agentic_ppo_learner
from tunix.rl.agentic.agents.agent_types import Action, Step
from tunix.rl.agentic.agents.base_agent import ConversationAgentBase
from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult
from tunix.rl.queue import data_queue as queue_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.tests import test_common
from tunix.utils import trajectory_logger
from typing_extensions import override

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
Mesh = sharding.Mesh
TrainingInputT = agentic_ppo_learner.TrainingInputT


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
    (
        "This is a very long sentence that will be used for testing clipped"
        " ratio and it contains many extra additional words to make sure it"
        " gets clipped properly by the 20 tokens budget."
    ),
]


def _mock_generate(
    prompts: list[str] | list[list[dict[str, str]]],
    apply_chat_template: bool = False,
    mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
    micro_batch_size: int | None = None,
    trace_tags: dict[str, Any] | None = None,
    output_logprobs: bool = True,
    tokenizer: Any | None = None,
    **kwargs,
) -> base_rollout.RolloutOutput:
  del apply_chat_template, mode, micro_batch_size, trace_tags
  assert tokenizer is not None
  if isinstance(prompts, str):
    prompts = [prompts]
  batch_size = len(prompts)
  text = [random.choice(_MOCK_RESPONSES) for _ in range(batch_size)]
  tokens = [tokenizer.encode(text_i) for text_i in text]
  logprobs = [-np.random.rand(len(tokens[i])) for i in range(batch_size)]
  prompt_tokens = []
  for p in prompts:
    if isinstance(p, str):
      prompt_tokens.append(tokenizer.encode(p))
    else:
      prompt_tokens.append(tokenizer.encode(" ".join(m["content"] for m in p)))
  max_p_len = max(len(pt) for pt in prompt_tokens)
  padded_prompts = np.array([
      np.pad(pt, (max(0, max_p_len - len(pt)), 0), constant_values=0)
      for pt in prompt_tokens
  ], dtype=np.int32)
  return base_rollout.RolloutOutput(
      text=text,
      tokens=tokens,
      left_padded_prompt_tokens=padded_prompts,
      logits=None,
      logprobs=logprobs if output_logprobs else None,
  )


def _mock_vocab():
  unique_words = {word for line in _MOCK_RESPONSES for word in line.split()}
  words = [
      "<pad>",
      "<s>",
      "</s>",
      "System:",
      "User:",
      "Assistant:",
      "Initial",
      "prompt.",
      "System",
      "Observation",
      "after",
      "step",
      "Steps",
      "Remaining:",
      "You",
      "have",
      "reached",
      "the",
      "maximum",
      "number",
      "of",
      "steps.",
      "1",
      "2",
      "3",
      "4",
      "5",
      "6",
      "7",
      "8",
      "9",
      "10",
  ]
  words.extend(sorted(unique_words))
  mapping_text_to_id = {word: i for i, word in enumerate(words)}
  vocab = test_common.MockVocab(mapping_text_to_id=mapping_text_to_id)
  return vocab


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
    del is_first_msg
    if not messages:
      return ""

    result = ""
    for message in messages:
      if message["role"] == "system":
        result += f"System: {message['content']}"
      elif message["role"] == "user":
        result += f" User: {message['content']}"
      elif message["role"] == "assistant":
        result += f" Assistant: {message['content']}"
      else:
        raise ValueError(f"Unsupported message role: {message['role']}")

    if add_generation_prompt:
      result += " " + self.assistant_token
    return result

  @property
  def assistant_token(self):
    return "Assistant: "

  def update_assistant_end_tokens(self, tokens):
    return tokens, 0


class _LearnerWithException(agentic_ppo_learner.PPOLearner):

  def _batch_to_train_example(self, batch_results, mode):
    raise ValueError("test exception in producer")


class AgenticPpoLearnerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    chex.set_n_cpu_devices(2)
    cls.device_count = jax.device_count()

  def setUp(self):
    super().setUp()
    random.seed(42)
    self.vocab = _mock_vocab()
    self.tokenizer = tokenizer_adapter.TokenizerAdapter(self.vocab)
    self._mock_generate = functools.partial(
        _mock_generate, tokenizer=self.tokenizer
    )

  def test_iterator(self):
    class _MockTrainer(agentic_ppo_learner.PPOLearner):

      def __init__(self, algo_config):
        self.algo_config = algo_config
        self.rl_cluster = mock.Mock()
        self.metric_fns = []
        self._process_in_consumer = False

      def _create_micro_batch_iterator(self, iterator, batch_size):
        for batch in iterator:
          for i in range(len(batch["prompts"])):
            yield jax.tree.map(lambda x, index=i: x[index : index + 1], batch)

      @override
      def _batch_to_train_example(self, batch_results, mode):
        del mode
        return [
            types.SimpleNamespace(
                prompt_ids=batch_results[1][0]["prompts"],
            )
        ]

      @override
      async def _orchestrator_producer(
          self,
          orchestrator,
          prompt_iterator: (
              Iterable[TrainingInputT] | AsyncIterable[TrainingInputT]
          ),
          num_generations: int = 1,
          collect_mode: str = "Token",
      ):
        i = 0
        if hasattr(prompt_iterator, "__aiter__"):
          async for example in prompt_iterator:
            group = [types.SimpleNamespace(pair_index=i)]
            yield group, [example]
            i += 1
        else:
          for example in prompt_iterator:
            group = [types.SimpleNamespace(pair_index=i)]
            yield group, [example]
            i += 1

    algo_config = agentic_ppo_learner.PPOConfig(
        num_iterations=2,
    )
    trainer = _MockTrainer(algo_config)

    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)
    dataset = _dummy_dataset(MySource(data=[i for i in range(2)]), batch_size=2)
    prompt_queue = queue.Queue()
    for item in iter(dataset):
      prompt_queue.put(item)
    prompt_queue.put(None)

    asyncio.run(trainer._producer(mock.Mock(), prompt_queue, train_data_queue))

    results = []
    while True:
      item = train_data_queue.get(block=True)
      if item is None:
        break
      results.append(item)

    prompt_ids = [r.prompt_ids[0] for r in results]
    self.assertEqual(prompt_ids, [0, 1])

  def test_ppo_config_validation(self):
    with self.assertRaisesRegex(
        ValueError, "`epsilon_c` must be greater than 1"
    ):
      agentic_ppo_learner.PPOConfig(epsilon_c=0.5)
    with self.assertRaisesRegex(
        ValueError, "Invalid KL method"
    ):
      agentic_ppo_learner.PPOConfig(kl_method="invalid")

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
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=100,
            max_steps=10,
            gradient_accumulation_steps=None,
            mini_batch_size=1,
            train_micro_batch_size=1,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=256,
            max_tokens_to_generate=10,
            return_logprobs=True,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    ppo_config = agentic_ppo_learner.PPOConfig(
        num_iterations=2,  # > 1
        max_response_length=10,
    )
    ppo_learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
    )

    train_ds = _dummy_dataset(
        MySource(data=["1", "2", "3", "4"], repeat=1), batch_size=1
    )

    with (
        mock.patch.object(
            ppo_learner,
            "_batch_to_train_example",
            wraps=ppo_learner._batch_to_train_example,
        ) as mock_b2te,
        mock.patch.object(
            rl_cluster, "update_actor", wraps=rl_cluster.update_actor
        ) as mock_update_actor,
        mock.patch.object(
            rl_cluster,
            "generate",
            side_effect=self._mock_generate,
        ),
    ):
      ppo_learner.train(train_ds)

      # 4 prompts, so _batch_to_train_example is called 4 times.
      self.assertEqual(mock_b2te.call_count, 4)
      # For each prompt, producer loops num_iterations=2 times.
      # Total examples = 4 * 2 = 8.
      # train_micro_batch_size=1 -> 8 updates for update_actor.
      self.assertEqual(mock_update_actor.call_count, 8)

  def test_compute_logps_micro_batch_size(self):
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
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=100,
            max_steps=10,
            mini_batch_size=2,
            train_micro_batch_size=2,
            compute_logps_micro_batch_size=2,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=256,
            max_tokens_to_generate=10,
            return_logprobs=True,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    ppo_config = agentic_ppo_learner.PPOConfig(
        num_iterations=2,
        max_response_length=10,
    )
    ppo_learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
    )

    train_ds = _dummy_dataset(
        MySource(data=["1", "2", "3", "4"], repeat=1), batch_size=4
    )

    with (
        mock.patch.object(
            ppo_learner,
            "_batch_to_train_example",
            wraps=ppo_learner._batch_to_train_example,
        ) as mock_b2te,
        mock.patch.object(
            rl_cluster, "update_actor", wraps=rl_cluster.update_actor
        ) as mock_update_actor,
        mock.patch.object(
            rl_cluster,
            "generate",
            side_effect=self._mock_generate,
        ),
        mock.patch.object(
            rl_cluster,
            "get_ref_per_token_logps",
            wraps=rl_cluster.get_ref_per_token_logps,
        ) as mock_get_ref,
    ):
      ppo_learner.train(train_ds)

      self.assertEqual(mock_b2te.call_count, 2)
      self.assertEqual(mock_get_ref.call_count, 2)

  def test_compute_logps_chunk_size(self):
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
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=100,
            max_steps=1,
            mini_batch_size=2,
            train_micro_batch_size=2,
            compute_logps_micro_batch_size=2,
            compute_logps_chunk_size=4,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=256,
            max_tokens_to_generate=10,
            return_logprobs=True,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    ppo_config = agentic_ppo_learner.PPOConfig(
        num_iterations=1,
        max_response_length=10,
    )
    ppo_learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
    )

    train_ds = _dummy_dataset(
        MySource(data=["1", "2"], repeat=1), batch_size=2
    )

    with (
        mock.patch.object(
            rl_common,
            "compute_per_token_logps",
            wraps=rl_common.compute_per_token_logps,
        ) as mock_compute_logps,
        mock.patch.object(
            rl_cluster,
            "generate",
            side_effect=self._mock_generate,
        ),
    ):
      ppo_learner.train(train_ds)
      self.assertGreater(mock_compute_logps.call_count, 0)
      chunk_sizes = [
          call.kwargs.get("chunk_size")
          for call in mock_compute_logps.call_args_list
      ]
      self.assertIn(4, chunk_sizes)

  def test_ppo_loss_fn(self):
    batch_size, seq_len = 2, 8
    prompt_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    completion_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    completion_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
    advantages = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    returns = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    old_values = jnp.zeros((batch_size, seq_len), dtype=jnp.float32)
    ref_per_token_logps = jnp.full(
        (batch_size, seq_len), -0.1, dtype=jnp.float32
    )

    train_example = agentic_ppo_learner.TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_ids > -1,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        returns=returns,
        old_values=old_values,
        old_per_token_logps=ref_per_token_logps,
    )

    class MockModel(nnx.Module):

      def __init__(self, *, rngs: nnx.Rngs):
        self.lm_head = 1

      def __call__(
          self, inputs, positions, cache, attention_mask, **kwargs
      ):
        del kwargs
        return (
            jnp.full(
                (*inputs.shape, 32),
                0.1,
                dtype=jnp.float32,
            ),
            None,
        )

    algo_config = agentic_ppo_learner.PPOConfig(
        beta=0.1,
        epsilon=0.2,
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
    self.assertIn("pg_clipfrac", aux)
  
  def test_ppo_loss_fn_respects_mask(self):
    seq_len = 8
    prompt_ids = jnp.asarray(
        [
            [1] * seq_len,
            [1] * seq_len,
        ],
        dtype=jnp.int32,
    )
    completion_ids = jnp.ones((2, seq_len), dtype=jnp.int32)
    # Mask out the second half of sequence 1
    completion_mask = jnp.asarray(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=jnp.bool_,
    )
    advantages = jnp.asarray(
        [
            [1.0] * seq_len,
            [1.0] * seq_len,
        ],
        dtype=jnp.float32,
    )
    returns = jnp.ones((2, seq_len), dtype=jnp.float32)
    old_values = jnp.zeros((2, seq_len), dtype=jnp.float32)
    old_per_token_logps = jnp.full((2, seq_len), -0.1, dtype=jnp.float32)
    ref_per_token_logps = jnp.full((2, seq_len), -0.1, dtype=jnp.float32)

    class MockModel(nnx.Module):

      def __init__(self, *, rngs: nnx.Rngs):
        self.lm_head = 1

      def __call__(self, inputs, positions, cache, attention_mask, **kwargs):
        del kwargs
        return (
            jnp.full(
                (*inputs.shape, 32),
                0.1,
                dtype=jnp.float32,
            ),
            None,
        )

    train_example = agentic_ppo_learner.TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_ids > -1,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        returns=returns,
        old_values=old_values,
        old_per_token_logps=old_per_token_logps,
    )

    config = agentic_ppo_learner.PPOConfig(
        beta=0.0,
        epsilon=0.2,
    )
    policy_loss_fn = function_registry.get_policy_loss_fn(config.policy_loss_fn)

    loss, aux = policy_loss_fn(
        model=MockModel(rngs=nnx.Rngs(0)),
        train_example=train_example,
        algo_config=config,
        pad_id=0,
        eos_id=2,
    )

    # Compute expected loss: per_token_logps is -log(32.0).
    # ratio = exp(-log(32.0) - (-0.1))
    # clipped ratio = clip(ratio, 0.8, 1.2) = 1.2
    # surrogate loss per unmasked token = -1.2
    # With completion_mask zeroing padded tokens, masked_mean averages only over active tokens.
    per_token_logps = jnp.full(completion_ids.shape, -np.log(32.0), dtype=jnp.float32)
    ratio = jnp.exp(per_token_logps - old_per_token_logps)
    pg_loss = -jnp.minimum(ratio * advantages, jnp.clip(ratio, 0.8, 1.2) * advantages)
    expected_loss = float(algo_core.masked_mean(pg_loss, completion_mask))

    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6, atol=1e-6)

  def test_process_results_extracts_assistant_text(self):
    class MockTraj:
      def __init__(self, index):
        self.traj = {
            "conversation_text": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user query"},
                {"role": "assistant", "content": f"msg {index}"},
            ],
            "conversation_tokens": np.array([1, 2, 3]),
            "conversation_masks": np.array([1, 1, 1]),
            "old_logprobs": None,
            "policy_version": 0,
            "trajectory_reward": 1.0,
            "prompt_tokens": np.array([4, 5]),
            "original_input": {"prompts": "hello"},
            "group_id": "group1",
        }

    trajectories = [MockTraj(0), MockTraj(1)]

    extracted_completions = []
    def mock_compute_rewards(prompts, completions, **kwargs):
      extracted_completions.extend(completions)
      return jnp.ones(len(completions), dtype=jnp.float32)

    vocab = _mock_vocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=100,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=32,
            max_tokens_to_generate=10,
            return_logprobs=True,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    ppo_config = agentic_ppo_learner.PPOConfig(
        beta=0.1,
        epsilon=0.2,
        max_response_length=10,
    )

    learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        chat_parser=MockChatParser(),
    )

    with mock.patch.object(learner, "_compute_rewards", side_effect=mock_compute_rewards):
      with mock.patch.object(
          learner.rl_cluster,
          "get_ref_per_token_logps",
          return_value=jnp.zeros((2, 10)),
          autospec=True,
      ):
        with mock.patch.object(
            learner.rl_cluster,
            "get_values",
            return_value=jnp.zeros((2, 11)),
            autospec=True,
        ):
          learner._process_results(trajectories)

    self.assertEqual(extracted_completions, ["msg 0", "msg 1"])

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
      value_model = utils.create_critic_model(model)

      mesh = pxla.thread_resources.env.physical_mesh
      cluster_config = rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: mesh,
              rl_cluster_lib.Role.CRITIC: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine="vanilla",
          offload_to_cpu=False,
          training_config=rl_cluster_lib.RLTrainingConfig(
              actor_optimizer=optax.sgd(1e-3),
              critic_optimizer=optax.sgd(1e-3),
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
              max_prompt_length=32,
              max_tokens_to_generate=10,
              return_logprobs=True,
              kv_cache_size=256,
              temperature=0.5,
          ),
      )
      rl_cluster = rl_cluster_lib.RLCluster(
          actor=model,
          critic=value_model,
          reference=ref_model,
          tokenizer=tokenizer,
          cluster_config=cluster_config,
      )

      ppo_config = agentic_ppo_learner.PPOConfig(
          num_iterations=1,
          max_response_length=10,
      )
      ppo_learner = agentic_ppo_learner.PPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=reward_fn_1,
          algo_config=ppo_config,
          chat_parser=MockChatParser(),
      )
      return ppo_learner

    train_ds = [
        {"prompts": [str(i)], "answer": [str(i)], "question": [str(i)]}
        for i in range(4)
    ]

    ppo_learner = create_learner(ckpt_dir, max_steps=10)
    self.assertEqual(ppo_learner.rl_cluster.global_steps, 0)
    ppo_learner.train(train_ds[0:1])
    self.assertEqual(ppo_learner.rl_cluster.global_steps, 1)

    ppo_learner2 = create_learner(ckpt_dir, max_steps=3)
    self.assertEqual(ppo_learner2.rl_cluster.global_steps, 1)

    ppo_learner2.train(train_ds)
    self.assertEqual(ppo_learner2.rl_cluster.global_steps, 3)

  def test_micro_batch_training(
      self,
  ):
    def reward_fn(prompts, **kwargs):
      del kwargs
      return [1.0] * len(prompts)

    def create_learner(
        mini_batch_size,
        train_micro_batch_size,
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
      value_model = utils.create_critic_model(model)

      mesh = pxla.thread_resources.env.physical_mesh
      cluster_config = rl_cluster_lib.ClusterConfig(
          role_to_mesh={
              rl_cluster_lib.Role.ACTOR: mesh,
              rl_cluster_lib.Role.CRITIC: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine="vanilla",
          offload_to_cpu=False,
          training_config=rl_cluster_lib.RLTrainingConfig(
              actor_optimizer=optax.sgd(1e-3),
              critic_optimizer=optax.sgd(1e-3),
              eval_every_n_steps=10,
              max_steps=20,
              mini_batch_size=mini_batch_size,
              train_micro_batch_size=train_micro_batch_size,
          ),
          rollout_config=base_rollout.RolloutConfig(
              max_prompt_length=32,
              max_tokens_to_generate=10,
              return_logprobs=True,
              kv_cache_size=256,
              temperature=0.5,
          ),
      )
      rl_cluster = rl_cluster_lib.RLCluster(
          actor=model,
          critic=value_model,
          reference=ref_model,
          tokenizer=tokenizer,
          cluster_config=cluster_config,
      )

      ppo_config = agentic_ppo_learner.PPOConfig(
          num_iterations=1,
          max_response_length=10,
      )
      ppo_learner = agentic_ppo_learner.PPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=reward_fn,
          algo_config=ppo_config,
          chat_parser=MockChatParser(),
      )
      return ppo_learner

    train_ds = [{
        "prompts": [str(i) for i in range(4)],
        "answer": [str(i) for i in range(4)],
        "question": [str(i) for i in range(4)],
    }]

    ppo_learner_base = create_learner(
        mini_batch_size=None,
        train_micro_batch_size=None,
    )
    ppo_learner_base.train(train_ds)

    ppo_learner_micro = create_learner(
        mini_batch_size=4,
        train_micro_batch_size=2,
    )
    ppo_learner_micro.train(train_ds)

    self.assertEqual(
        ppo_learner_base.rl_cluster.global_steps,
        ppo_learner_micro.rl_cluster.global_steps,
    )
    self.assertEqual(ppo_learner_base.rl_cluster.global_steps, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="single_reward_fn",
          reward_fns=reward_fn_1,
          use_old_logprobs=False,
      ),
      dict(
          testcase_name="multiple_reward_fns",
          reward_fns=[
              reward_fn_1,
              reward_fn_2,
          ],
          use_old_logprobs=True,
      ),
  )
  def test_ppo_learner(self, reward_fns, use_old_logprobs):
    vocab = _mock_vocab()
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
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=2,
            max_steps=20,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=256,
            max_tokens_to_generate=20,
            return_logprobs=True,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    rl_cluster.with_external_metrics_logger(print)

    ppo_config = agentic_ppo_learner.PPOConfig(
        num_iterations=1,
        max_response_length=20,
    )
    ppo_learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=ppo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
    )
    self.assertFalse(ppo_learner.should_sync_weights)
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    eval_ds = _dummy_dataset(batch_size=1)

    with mock.patch.object(
        rl_cluster,
        "generate",
        side_effect=functools.partial(
            self._mock_generate,
            output_logprobs=use_old_logprobs,
        ),
    ):
      ppo_learner.train(train_ds, eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_variables, variables
    )

    self.assertEqual(
        ppo_learner.rl_cluster.global_steps,
        20,
    )

  def test_put_prompts_to_queue(self):
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
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=512,
            return_logprobs=True,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    ppo_config = agentic_ppo_learner.PPOConfig(max_response_length=512)
    ppo_learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        chat_parser=MockChatParser(),
    )
    ppo_learner._full_batch_size = 2

    prompt_queue = queue.Queue()
    batch1 = {"prompts": ["prompt1", "prompt2"]}
    ppo_learner._put_prompts_to_queue(prompt_queue, batch1)
    self.assertEqual(prompt_queue.get_nowait(), batch1)

    batch2 = {"prompts": ["prompt3"]}
    ppo_learner._put_prompts_to_queue(prompt_queue, batch2)
    self.assertIsNone(prompt_queue.get_nowait())

  def test_trajectory_logging(self):
    log_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, log_dir)
    vocab = _mock_vocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=2,
            max_steps=1,
            gradient_accumulation_steps=None,
            metrics_logging_options=metrics_logger.MetricsLoggerOptions(
                log_dir=log_dir
            ),
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=256,
            max_tokens_to_generate=10,
            return_logprobs=True,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    ppo_config = agentic_ppo_learner.PPOConfig(
        num_iterations=1,
        max_response_length=10,
    )
    ppo_learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
    )
    train_ds = _dummy_dataset(MySource(data=["1"], repeat=1), batch_size=1)

    with (
        mock.patch.object(trajectory_logger, "log_item") as mock_log_item,
        mock.patch.object(
            rl_cluster, "generate", side_effect=self._mock_generate
        ),
    ):
      ppo_learner.train(train_ds)
      if ppo_learner._trajectory_logger:
        ppo_learner._trajectory_logger.stop()
      self.assertEqual(ppo_learner.rl_cluster.global_steps, 1)
      self.assertEqual(mock_log_item.call_count, 1)
  
  def test_ppo_with_lora_model(self):
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
    vocab = _mock_vocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    actor_model = test_common.get_lora_model(
        ref_model,
        mesh=mesh1,
    )
    value_model = utils.create_critic_model(actor_model)

    original_base_params = jax.tree.map(
        jnp.copy, nnx.state(actor_model, filterlib.Not(nnx.LoRAParam))
    )
    original_lora_variables = jax.tree.map(
        jnp.copy, nnx.state(actor_model, nnx.LoRAParam)
    )

    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh1,
            rl_cluster_lib.Role.CRITIC: mesh1,
            rl_cluster_lib.Role.REFERENCE: mesh1,
            rl_cluster_lib.Role.ROLLOUT: mesh2,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=10,
            max_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=256,
            max_tokens_to_generate=10,
            return_logprobs=True,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=actor_model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    ppo_config = agentic_ppo_learner.PPOConfig(
        num_iterations=1,
        max_response_length=10,
    )

    ppo_learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        chat_parser=MockChatParser(),
    )
    self.assertTrue(ppo_learner.should_sync_weights)
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    with mock.patch.object(
        rl_cluster, "generate", side_effect=self._mock_generate
    ):
      ppo_learner.train(train_ds, None)

    base_params = nnx.state(
        rl_cluster.actor_trainer.model, filterlib.Not(nnx.LoRAParam)
    )
    lora_params = nnx.state(rl_cluster.actor_trainer.model, nnx.LoRAParam)
    lora_params_from_sampler = nnx.state(
        ppo_learner.rl_cluster.rollout.model(), nnx.LoRAParam
    )
    # 1. Base weights remain unchanged
    jax.tree.map_with_path(
        test_common.assert_equal, original_base_params, base_params
    )
    # 2. LoRA weights are updated during training
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_lora_variables, lora_params
    )
    # 3. Updated LoRA weights synced across meshes to rollout sampler
    jax.tree.map_with_path(
        test_common.assert_close, lora_params_from_sampler, lora_params
    )

  def test_customized_agent_env(self):
    class MockEnv(BaseTaskEnv):

      def __init__(self, entry: dict[str, str], max_steps: int, **kwargs):
        self.entry = entry
        super().__init__(max_steps=max_steps, **kwargs)

      def _initial_observation(self) -> Any:
        return "Initial prompt."

      def _step_impl(self, action: Any) -> EnvStepResult:
        done = self.step_count >= self.max_steps
        reward = 1.0 if not done else 0.0
        return EnvStepResult(
            observation=f"Observation after step {self.step_count}",
            reward=reward,
            done=done,
            info={"max_steps": self.max_steps},
        )

    class MockAgent(ConversationAgentBase):

      def __init__(self, system_prompt: str):
        super().__init__(system_prompt=system_prompt)
        self.step = 0

      def _observation_to_messages(self, observation, reward, done, info):
        max_steps = info.get("max_steps", None)
        if max_steps is not None:
          remaining_steps = max_steps - self.step - 1
          if remaining_steps > 0:
            observation += f" Steps Remaining: {remaining_steps}"
          else:
            observation += " You have reached the maximum number of steps."
        self._messages.append({"role": "user", "content": observation})
        step = self.get_current_step()
        if step:
          step.observation = observation

      def update_from_model(self, response, **kwargs):
        step = Step(model_response=response, action=f"Model action: {response}")
        self._trajectory.steps.append(step)

        self._messages.append({"role": "assistant", "content": response})
        self.step += 1
        return Action(action=step.action)

    vocab = _mock_vocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=2,
            max_steps=20,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=128,
            max_prompt_length=32,
            return_logprobs=True,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    rl_cluster.with_external_metrics_logger(print)

    ppo_config = agentic_ppo_learner.PPOConfig(
        num_iterations=1,
        max_response_length=128,
        max_concurrency=1,
    )
    ppo_learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
        agent_class=MockAgent,
        agent_kwargs={"system_prompt": "System prompt."},
        env_class=MockEnv,
        env_kwargs={"max_steps": 3},
    )

    agents, envs = [], []
    original_fn = ppo_learner._create_agent_env_pair

    def _patch_create_agent_env_pair(single_example, group_id, pair_index):
      agent, env = original_fn(single_example, group_id, pair_index)
      agents.append(agent)
      envs.append(env)
      return agent, env

    original_process_results = ppo_learner._process_results
    processed_results = []

    def _patch_process_results(
        trajectories,
        mode,
        expected_step,
    ):
      res = original_process_results(trajectories, mode, expected_step)
      processed_results.append(res)
      return res

    ppo_learner._create_agent_env_pair = _patch_create_agent_env_pair
    ppo_learner._process_results = _patch_process_results

    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    eval_ds = _dummy_dataset(batch_size=1)

    with mock.patch.object(
        rl_cluster, "generate", side_effect=self._mock_generate
    ):
      ppo_learner.train(train_ds, eval_ds)

    traj = agents[0].trajectory
    target_mask = []
    for step in traj.steps:
      target_mask.extend([1] * (len(step.model_response.split())))
      target_mask.extend([0] * (len(step.observation.split()) + 2))
    target_mask.extend(
        [0] * (ppo_config.max_response_length - len(target_mask))
    )
    target_mask = target_mask[: ppo_config.max_response_length]

    res = processed_results[0][0]
    pass_1 = np.array_equal(res.completion_mask[0], np.array(target_mask))
    pass_2 = np.array_equal(res.completion_mask[1], np.array(target_mask))
    self.assertTrue(pass_1 or pass_2)

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
      value_model = utils.create_critic_model(model)

      mesh = pxla.thread_resources.env.physical_mesh
      if ckpt_dir:
        checkpointing_options = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
        )
      else:
        checkpointing_options = None
      training_config = rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-3),
          critic_optimizer=optax.sgd(1e-3),
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
              rl_cluster_lib.Role.CRITIC: mesh,
              rl_cluster_lib.Role.REFERENCE: mesh,
              rl_cluster_lib.Role.ROLLOUT: mesh,
          },
          rollout_engine="vanilla",
          offload_to_cpu=False,
          training_config=training_config,
          rollout_config=base_rollout.RolloutConfig(
              max_prompt_length=32,
              max_tokens_to_generate=10,
              return_logprobs=True,
              kv_cache_size=256,
              temperature=0.5,
          ),
      )
      rl_cluster = rl_cluster_lib.RLCluster(
          actor=model,
          critic=value_model,
          reference=ref_model,
          tokenizer=tokenizer,
          cluster_config=cluster_config,
      )

      ppo_config = agentic_ppo_learner.PPOConfig(
          num_iterations=1,
          max_response_length=10,
      )
      ppo_learner = agentic_ppo_learner.PPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=reward_fn,
          algo_config=ppo_config,
          chat_parser=MockChatParser(),
      )
      return ppo_learner, model, value_model

    train_ds = [
        {"prompts": [str(i)], "answer": [str(i)], "question": [str(i)]}
        for i in range(2)
    ]

    # 1. Train in one go
    ppo_learner_full, model_full, critic_full = create_learner(ckpt_dir=None, max_steps=2)
    with mock.patch.object(
        ppo_learner_full.rl_cluster, "generate", side_effect=self._mock_generate
    ):
      ppo_learner_full.train(train_ds)
    self.assertEqual(ppo_learner_full.rl_cluster.global_steps, 2)

    # 2. Train interrupted
    ppo_learner_interrupt, _, _ = create_learner(ckpt_dir=ckpt_dir, max_steps=1)
    with mock.patch.object(
        ppo_learner_interrupt.rl_cluster, "generate", side_effect=self._mock_generate
    ):
      ppo_learner_interrupt.train(train_ds)
    self.assertEqual(ppo_learner_interrupt.rl_cluster.global_steps, 1)

    # 3. Resume training
    ppo_learner_resume, model_resume, critic_resume = create_learner(
        ckpt_dir=ckpt_dir, max_steps=2
    )
    self.assertEqual(ppo_learner_resume.rl_cluster.global_steps, 1)
    with mock.patch.object(
        ppo_learner_resume.rl_cluster, "generate", side_effect=self._mock_generate
    ):
      ppo_learner_resume.train(train_ds)
    self.assertEqual(ppo_learner_resume.rl_cluster.global_steps, 2)

    # 4. Compare actor and critic weights
    actor_params1 = nnx.state(model_full, nnx.Param)
    actor_params2 = nnx.state(model_resume, nnx.Param)
    jax.tree.map_with_path(test_common.assert_close, actor_params1, actor_params2)

    critic_params1 = nnx.state(critic_full, nnx.Param)
    critic_params2 = nnx.state(critic_resume, nnx.Param)
    jax.tree.map_with_path(test_common.assert_close, critic_params1, critic_params2)

  @parameterized.named_parameters(
      dict(
          testcase_name="use_rollout_logps_true",
          use_rollout_logps=True,
          return_logprobs=True,
          expect_get_actor_logps=False,
      ),
      dict(
          testcase_name="use_rollout_logps_false",
          use_rollout_logps=False,
          return_logprobs=False,
          expect_get_actor_logps=True,
      ),
  )
  def test_use_rollout_logps(
      self, use_rollout_logps, return_logprobs, expect_get_actor_logps
  ):
    vocab = _mock_vocab()
    tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=10,
            max_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=32,
            max_tokens_to_generate=10,
            return_logprobs=return_logprobs,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    ppo_config = agentic_ppo_learner.PPOConfig(
        beta=0.0,
        max_response_length=10,
        num_iterations=1,
        use_rollout_logps=use_rollout_logps,
    )
    learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        chat_parser=MockChatParser(),
    )

    class MockTraj:

      def __init__(self, index):
        self.traj = {
            "conversation_text": [
                {"role": "assistant", "content": f"msg {index}"}
            ],
            "conversation_tokens": np.array([1, 2, 3]),
            "conversation_masks": np.array([1, 1, 1]),
            "old_logprobs": (
                np.full(3, 1.0, dtype=np.float32) if return_logprobs else None
            ),
            "policy_version": 0,
            "trajectory_reward": 1.0,
            "prompt_tokens": np.array([4, 5]),
            "original_input": {"prompts": "hello"},
            "group_id": "test_group",
        }

    trajectories = [MockTraj(0), MockTraj(1)]

    with mock.patch.object(
        rl_cluster,
        "get_actor_per_token_logps",
        return_value=jnp.full((2, 10), -1.0),
        autospec=True,
    ) as mock_get_actor_logps:
      with mock.patch.object(
          rl_cluster,
          "get_values",
          return_value=jnp.zeros((2, 11)),
          autospec=True,
      ):
        results = learner._process_results(trajectories, expected_step=1)
        self.assertLen(results, 1)
        train_example = results[0]

        if expect_get_actor_logps:
          mock_get_actor_logps.assert_called_once()
          self.assertIsNotNone(train_example.old_per_token_logps)
          np.testing.assert_allclose(
              train_example.old_per_token_logps, jnp.full((2, 10), -1.0)
          )
        else:
          mock_get_actor_logps.assert_not_called()
          if return_logprobs:
            self.assertIsNotNone(train_example.old_per_token_logps)
            np.testing.assert_allclose(
                train_example.old_per_token_logps,
                np.array([[1.0] * 3 + [0.0] * 7] * 2),
            )
          else:
            self.assertIsNone(train_example.old_per_token_logps)

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
    vocab = _mock_vocab()
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
    value_model = utils.create_critic_model(model)

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.CRITIC: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            critic_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=2,
            max_steps=4,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=256,
            max_tokens_to_generate=10,
            return_logprobs=True,
            kv_cache_size=1024,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        critic=value_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    ppo_config = agentic_ppo_learner.PPOConfig(
        num_iterations=1,
        off_policy_steps=offpolicy_steps,
        max_response_length=10,
    )
    ppo_learner = agentic_ppo_learner.PPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=ppo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
    )
    train_ds = _dummy_dataset(MySource(repeat=4), batch_size=2)
    eval_ds = _dummy_dataset(batch_size=1)

    with mock.patch.object(
        rl_cluster, "generate", side_effect=self._mock_generate
    ):
      ppo_learner.train(train_ds, eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_variables, variables
    )

    self.assertEqual(
        ppo_learner.rl_cluster.global_steps,
        4,
    )


if __name__ == "__main__":
  absltest.main()
