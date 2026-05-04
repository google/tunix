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
from tunix.rl import common as rl_common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
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
  batch_size = len(prompts)
  text = [random.choice(_MOCK_RESPONSES) for _ in range(batch_size)]
  tokens = [tokenizer.encode(text_i) for text_i in text]
  logprobs = [-np.random.rand(len(tokens[i])) for i in range(batch_size)]
  return base_rollout.RolloutOutput(
      text=text,
      tokens=tokens,
      left_padded_prompt_tokens=np.ones((batch_size, 8), dtype=np.int32),
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


class _LearnerWithException(agentic_grpo_learner.GRPOLearner):

  def _batch_to_train_example(self, batch_results, mode):
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
    self.vocab = _mock_vocab()
    self.tokenizer = tokenizer_adapter.TokenizerAdapter(self.vocab)
    self._mock_generate = functools.partial(
        _mock_generate, tokenizer=self.tokenizer
    )

  def test_iterator(self):
    class _MockTrainer(agentic_grpo_learner.GRPOLearner):

      def __init__(self, algo_config):
        self.algo_config = algo_config
        self.rl_cluster = mock.Mock()
        self.metric_fns = []

      def _create_micro_batch_iterator(self, iterator, batch_size):
        # The dataset batch size is 2, and we want to test micro-batching
        # of size 1, as consumed by _orchestrator_producer.
        for batch in iterator:
          for i in range(len(batch["prompts"])):
            yield jax.tree.map(lambda x, index=i: x[index : index + 1], batch)

      @override
      def _batch_to_train_example(self, batch_results, mode):
        del mode
        examples = []
        for _ in range(self.algo_config.num_generations):
          examples.append(
              types.SimpleNamespace(
                  prompt_ids=batch_results[1][0]["prompts"],
              )
          )
        return examples

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
            group = [
                types.SimpleNamespace(
                    pair_index=i * self.algo_config.num_generations + j
                )
                for j in range(self.algo_config.num_generations)
            ]
            yield group, [example]
            i += 1
        else:
          for example in prompt_iterator:
            group = [
                types.SimpleNamespace(
                    pair_index=i * self.algo_config.num_generations + j
                )
                for j in range(self.algo_config.num_generations)
            ]
            yield group, [example]
            i += 1

    algo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
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
            max_prompt_length=256,
            max_tokens_to_generate=10,
            return_logprobs=True,
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
        max_response_length=10,
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
        mock.patch.object(
            rl_cluster,
            "generate",
            side_effect=self._mock_generate,
        ),
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
    algo_config.temperature = 1.0
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

  @parameterized.named_parameters(
      dict(testcase_name="unmasked", apply_masking=False),
      dict(testcase_name="masked", apply_masking=True),
  )
  def test_grpo_loss_fn_respects_mask(self, apply_masking):
    seq_len = 8
    prompt_ids = jnp.asarray(
        [
            [1] * seq_len,
            [1] * seq_len,
            [2] * seq_len,
            [2] * seq_len,
        ],
        dtype=jnp.int32,
    )
    completion_ids = jnp.ones((4, seq_len), dtype=jnp.int32)
    completion_mask = jnp.ones((4, seq_len), dtype=jnp.bool_)
    # Two prompts with two generations each.
    # Prompt 1 has a non-degenerate advantage group; prompt 2 is degenerate
    # (which would be filtered in _process_results with degenerate_group_masking=True).
    advantages = jnp.asarray([-1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    ref_per_token_logps = jnp.asarray(
        [
            [-0.1] * seq_len,
            [-0.1] * seq_len,
            [-1.1] * seq_len,
            [-1.1] * seq_len,
        ],
        dtype=jnp.float32,
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

    if apply_masking:
      # Masked example (simulating what _process_results would do)
      final_completion_mask = completion_mask.at[2:].set(0)
    else:
      # Unmasked example
      final_completion_mask = completion_mask

    train_example = agentic_grpo_learner.TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_ids > -1,
        completion_ids=completion_ids,
        completion_mask=final_completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=None,
    )

    config = agentic_grpo_learner.GRPOConfig(
        beta=0.1,
        epsilon=0.2,
        num_generations=2,
        loss_algo="grpo",
        loss_agg_mode="token-mean",
    )
    config.temperature = 0.5

    policy_loss_fn = function_registry.get_policy_loss_fn(config.policy_loss_fn)

    model = MockModel(rngs=nnx.Rngs(0))
    loss, _ = policy_loss_fn(
        model=model,
        train_example=train_example,
        algo_config=config,
        pad_id=0,
        eos_id=2,
    )

    # Expected values calculation:
    # old_per_token_logps=None makes the importance ratio 1, so the policy
    # term is simply -advantages. The mock model emits the same logit (0.1)
    # for all 32 vocabulary entries, so softmax is uniform with probability
    # 1/32 for every token and the selected per-token log-probability is
    # log(1 / 32) = -log(32). We then derive the KL penalty from the full
    # ref_per_token_logps tensor so each prompt group can have different KL.
    per_token_logps = jnp.full(
        completion_ids.shape,
        -np.log(32.0),
        dtype=jnp.float32,
    )
    per_token_kl = rl_common.compute_kl_divergence(
        per_token_logps,
        ref_per_token_logps,
        config.kl_loss_mode,
    )
    per_sequence_loss = -advantages + config.beta * per_token_kl[:, 0]

    if apply_masking:
      expected_loss = float(jnp.mean(per_sequence_loss[:2]))
    else:
      expected_loss = float(jnp.mean(per_sequence_loss))

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
    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
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
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    grpo_config = agentic_grpo_learner.GRPOConfig(
        beta=0.1,
        epsilon=0.2,
        num_generations=2,
        loss_algo="grpo",
        max_response_length=10,
    )

    learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=None,
        algo_config=grpo_config,
        chat_parser=MockChatParser(),
    )

    with mock.patch.object(learner, "_compute_rewards", side_effect=mock_compute_rewards):
      with mock.patch.object(
          learner.rl_cluster,
          "get_ref_per_token_logps",
          return_value=jnp.zeros((2, 10)),
          autospec=True,
      ):
        learner._process_results(trajectories)

    self.assertEqual(extracted_completions, ["msg 0", "msg 1"])

  @parameterized.named_parameters(
      dict(testcase_name="masking_disabled", masking=False),
      dict(testcase_name="masking_enabled", masking=True),
  )
  def test_process_results_masks_zero_advantage_group(self, masking):
    class MockTraj:

      def __init__(
          self,
          index,
          group_id,
          reward,
          has_assistant_message=True,
          old_logprobs=None,
      ):
        self.traj = {
            "conversation_text": [],
            "conversation_tokens": np.array([1, 2, 3]),
            "conversation_masks": np.array([1, 1, 1]),
            "old_logprobs": old_logprobs,
            "policy_version": 0,
            "trajectory_reward": reward,
            "prompt_tokens": np.array([4, 5]),
            "original_input": {"prompts": "hello"},
            "group_id": group_id,
        }
        self.traj["conversation_text"].append(
            {"role": "user", "content": "user message"}
        )
        if has_assistant_message:
          self.traj["conversation_text"].append(
              {"role": "assistant", "content": f"msg {index}"}
          )

    # Group 1: non-degenerate (different rewards)
    group1 = [MockTraj(0, "group1", -1.0), MockTraj(1, "group1", 1.0)]
    # Group 2: degenerate (same rewards)
    group2 = [MockTraj(2, "group2", 0.0), MockTraj(3, "group2", 0.0)]

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
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    grpo_config = agentic_grpo_learner.GRPOConfig(
        beta=0.1,
        epsilon=0.2,
        num_generations=2,
        loss_algo="grpo",
        degenerate_group_masking=masking,
        max_response_length=10,
    )

    learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=None,
        algo_config=grpo_config,
        chat_parser=MockChatParser(),
    )

    with mock.patch.object(
        learner.rl_cluster,
        "get_ref_per_token_logps",
        return_value=jnp.zeros((2, 10)),
        autospec=True,
    ) as mock_get_ref:
      [res_group1] = learner._process_results(group1)
      [res_group2] = learner._process_results(group2)

      # Group 1 should always be intact as it's non-degenerate
      self.assertTrue(jnp.any(res_group1.completion_mask > 0))

      # Group 2 should be masked based on the 'masking' parameter
      if masking:
        # Masking enabled: degenerate group should be masked out
        self.assertFalse(jnp.any(res_group2.completion_mask > 0))
      else:
        # Masking disabled: degenerate group should remain intact
        self.assertTrue(jnp.any(res_group2.completion_mask > 0))

      # Test group with missing assistant message
      group3 = [
          MockTraj(4, "group3", 0.0, has_assistant_message=False),
          MockTraj(5, "group3", 0.0),
      ]
      [res_group3] = learner._process_results(group3)
      if masking:
        self.assertFalse(jnp.any(res_group3.completion_mask > 0))
      else:
        self.assertTrue(jnp.any(res_group3.completion_mask > 0))

      # Test group with partially missing old_logprobs
      group4 = [
          MockTraj(6, "group4", 0.0, old_logprobs=np.array([-0.1, -0.2, -0.3])),
          MockTraj(7, "group4", 0.0, old_logprobs=None),
      ]
      [res_group4] = learner._process_results(group4)
      self.assertTrue(jnp.any(res_group4.old_per_token_logps == 0.0))

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
              max_prompt_length=32,
              max_tokens_to_generate=10,
              return_logprobs=True,
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
          max_response_length=10,
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
    def reward_fn(prompts, **kwargs):
      del kwargs
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
              max_prompt_length=32,
              max_tokens_to_generate=10,
              return_logprobs=True,
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
          max_response_length=10,
      )
      grpo_learner = agentic_grpo_learner.GRPOLearner(
          rl_cluster=rl_cluster,
          reward_fns=reward_fn,
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
              max_prompt_length=32,
              max_tokens_to_generate=10,
              return_logprobs=True,
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
          max_response_length=10,
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

  @parameterized.named_parameters(
      dict(
          testcase_name="default_beta_zero",
          beta=0.0,
          force_compute_kl=False,
          expect_ref_logps=False,
      ),
      dict(
          testcase_name="force_kl_beta_zero",
          beta=0.0,
          force_compute_kl=True,
          expect_ref_logps=True,
      ),
      dict(
          testcase_name="beta_non_zero",
          beta=0.1,
          force_compute_kl=False,
          expect_ref_logps=True,
      ),
  )
  def test_force_compute_kl(self, beta, force_compute_kl, expect_ref_logps):
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
            max_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=32,
            max_tokens_to_generate=10,
            return_logprobs=True,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    grpo_config = agentic_grpo_learner.GRPOConfig(
        beta=beta,
        force_compute_kl=force_compute_kl,
        max_response_length=10,
        num_generations=2,
        num_iterations=1,
    )
    learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=grpo_config,
        chat_parser=MockChatParser(),
    )

    # Mock trajectories to pass into _process_results
    class MockTraj:

      def __init__(self, index):
        self.traj = {
            "conversation_text": [
                {"role": "assistant", "content": f"msg {index}"}
            ],
            "conversation_tokens": np.array([1, 2, 3]),
            "conversation_masks": np.array([1, 1, 1]),
            "old_logprobs": None,
            "policy_version": 0,
            "trajectory_reward": 1.0,
            "prompt_tokens": np.array([4, 5]),
            "original_input": {"prompts": "hello"},
            "group_id": "test_group",
        }

    trajectories = [MockTraj(0), MockTraj(1)]

    with mock.patch.object(
        rl_cluster,
        "get_ref_per_token_logps",
        return_value=jnp.zeros((2, 10)),
        autospec=True,
    ) as mock_get_ref:
      results = learner._process_results(trajectories, expected_step=1)
      self.assertLen(results, 1)
      train_example = results[0]

      if expect_ref_logps:
        mock_get_ref.assert_called_once()
        self.assertIsNotNone(train_example.ref_per_token_logps)
      else:
        mock_get_ref.assert_not_called()
        self.assertIsNone(train_example.ref_per_token_logps)

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
            max_prompt_length=32,
            max_tokens_to_generate=10,
            return_logprobs=True,
            kv_cache_size=256,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    grpo_config = agentic_grpo_learner.GRPOConfig(max_response_length=10)
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
          use_old_logprobs=False,
      ),
      dict(
          testcase_name="multiple_reward_fns",
          reward_fns=[
              reward_fn_1,
              reward_fn_2,
          ],
          loss_algo="grpo",
          use_old_logprobs=True,
      ),
      dict(
          testcase_name="single_reward_fn_gspo",
          reward_fns=reward_fn_1,
          loss_algo="gspo-token",
          use_old_logprobs=True,
      ),
  )
  def test_grpo_learner(self, reward_fns, loss_algo, use_old_logprobs):
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
            max_prompt_length=256,
            max_tokens_to_generate=20,
            return_logprobs=True,
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
        max_response_length=20,
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

    with mock.patch.object(
        rl_cluster,
        "generate",
        side_effect=functools.partial(
            self._mock_generate,
            output_logprobs=use_old_logprobs,
        ),
    ):
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
        "generation/prompts/mean_length",
        "generation/prompts/max_length",
        "generation/prompts/min_length",
        "generation/completions/mean_length",
        "generation/completions/max_length",
        "generation/completions/min_length",
        "generation/completions/clip_ratio",
        "perf/global_step_time",
        "global/test_metric",
    ]:
      if metric_name == "rewards/reward_fn_2" and not isinstance(
          reward_fns, list
      ):
        continue
      # We log metrics per step, and sometimes one extra step is logged due to
      # buffer flushing. So we check if length is close to global_steps.
      prefix, metric_name = metric_name.split("/", maxsplit=1)
      self.assertGreaterEqual(
          len(
              rl_metric_logger.get_metric_history(prefix, metric_name, "train")
          ),
          grpo_learner.rl_cluster.global_steps,
          msg=f"metric_name: {metric_name}",
      )

      if metric_name != "global_step_time":
        self.assertLen(
            rl_metric_logger.get_metric_history(prefix, metric_name, "eval"),
            10,
            msg=f"metric_name: {metric_name}",
        )
    clip_ratio_history = rl_metric_logger.get_metric_history(
        "generation", "completions/clip_ratio", "train"
    )
    # self.assertGreater(np.sum(clip_ratio_history), 0)

    metric_logger = grpo_learner.rl_cluster.actor_trainer.metrics_logger
    for metric_name in ["loss", "kl", "entropy", "pg_clipfrac"]:
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
    self.assertLen(
        metric_logger.get_metric_history("actor", "grad_norm", "train"),
        grpo_learner.rl_cluster.actor_trainer.train_steps,
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
            max_prompt_length=256,
            max_tokens_to_generate=10,
            return_logprobs=True,
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
        max_response_length=10,
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

    with mock.patch.object(
        rl_cluster, "generate", side_effect=self._mock_generate
    ):
      grpo_learner.train(train_ds, eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_variables, variables
    )

    self.assertEqual(
        grpo_learner.rl_cluster.global_steps,
        4,
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

    mesh = pxla.thread_resources.env.physical_mesh
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=10,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=512,
            return_logprobs=True,
        ),
    )
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    grpo_config = agentic_grpo_learner.GRPOConfig(max_response_length=512)
    grpo_learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=grpo_config,
        chat_parser=MockChatParser(),
    )
    grpo_learner._full_batch_size = 2

    # Test with matching batch size
    prompt_queue = queue.Queue()
    batch1 = {"prompts": ["prompt1", "prompt2"]}
    grpo_learner._put_prompts_to_queue(prompt_queue, batch1)
    self.assertEqual(prompt_queue.get_nowait(), batch1)

    # Test with non-matching batch size
    batch2 = {"prompts": ["prompt3"]}
    grpo_learner._put_prompts_to_queue(prompt_queue, batch2)
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
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        loss_algo="grpo",
        max_response_length=10,
    )
    grpo_learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=grpo_config,
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
      grpo_learner.train(train_ds)
      if grpo_learner._trajectory_logger:
        grpo_learner._trajectory_logger.stop()
      self.assertEqual(grpo_learner.rl_cluster.global_steps, 1)
      self.assertEqual(mock_log_item.call_count, 1)

      logged_items = mock_log_item.call_args_list[0][0][1]
      self.assertLen(logged_items, grpo_config.num_generations)

      for traj in logged_items:
        self.assertIn("conversation_text", traj)
        conversation = traj["conversation_text"]
        assistant_msgs = [m for m in conversation if m["role"] == "assistant"]
        self.assertNotEmpty(assistant_msgs)
        self.assertIn(assistant_msgs[0]["content"], _MOCK_RESPONSES)
        self.assertEqual(traj.get("policy_version"), 0)

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
            max_prompt_length=256,
            max_tokens_to_generate=10,
            return_logprobs=True,
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
        max_response_length=10,
    )

    grpo_learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=grpo_config,
        chat_parser=MockChatParser(),
    )
    self.assertTrue(grpo_learner.should_sync_weights)
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    with mock.patch.object(
        rl_cluster, "generate", side_effect=self._mock_generate
    ):
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
            max_tokens_to_generate=128,
            max_prompt_length=32,
            return_logprobs=True,
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
        loss_algo="grpo",
        max_response_length=128,
        max_concurrency=1,  # so the output is deterministic.
    )
    grpo_learner = agentic_grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn_1,
        algo_config=grpo_config,
        metric_fns=[lambda **kwargs: {"test_metric": (1.0, np.mean)}],
        chat_parser=MockChatParser(),
        agent_class=MockAgent,
        agent_kwargs={"system_prompt": "System prompt."},
        env_class=MockEnv,
        env_kwargs={"max_steps": 3},
    )

    agents, envs = [], []

    original_fn = grpo_learner._create_agent_env_pair

    def _patch_create_agent_env_pair(single_example, group_id, pair_index):
      agent, env = original_fn(single_example, group_id, pair_index)
      agents.append(agent)
      envs.append(env)
      return agent, env

    original_process_results = grpo_learner._process_results
    processed_results = []

    def _patch_process_results(
        trajectories,
        mode,
        expected_step,
    ):
      res = original_process_results(trajectories, mode, expected_step)
      processed_results.append(res)
      return res

    grpo_learner._create_agent_env_pair = _patch_create_agent_env_pair
    grpo_learner._process_results = _patch_process_results

    self.assertFalse(grpo_learner.should_sync_weights)
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    eval_ds = _dummy_dataset(batch_size=1)

    with mock.patch.object(
        rl_cluster, "generate", side_effect=self._mock_generate
    ):
      grpo_learner.train(train_ds, eval_ds)

    traj = agents[0].trajectory

    target_mask = []
    for step in traj.steps:
      target_mask.extend([1] * (len(step.model_response.split())))
      # + 2 for user and assistant role token from MockChatParser
      target_mask.extend([0] * (len(step.observation.split()) + 2))
    target_mask.extend(
        [0] * (grpo_config.max_response_length - len(target_mask))
    )
    target_mask = target_mask[: grpo_config.max_response_length]

    res = processed_results[0][0]
    # Since rollout is async and two generations will be executed concurrently,
    # the order of the results is not guaranteed.
    pass_1 = np.array_equal(res.completion_mask[0], np.array(target_mask))
    pass_2 = np.array_equal(res.completion_mask[1], np.array(target_mask))
    self.assertTrue(pass_1 or pass_2)
    decoded_prompt = tokenizer.decode(np.array(res.prompt_ids[0]).tolist())
    decoded_completion = tokenizer.decode(
        np.array(res.completion_ids[0]).tolist()
    )
    self.assertEqual(decoded_prompt.count("Assistant:"), 1)
    self.assertEqual(
        decoded_completion.count("Assistant:"), 2
    )  # 3 turns but terminal env obs does not append generation msg

  def test_compute_rloo_advantages(self):
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    advantages = agentic_grpo_learner.compute_rloo_advantages(
        rewards, num_generations=3
    )
    expected_value = jnp.array([-1.5, 0.0, 1.5, -1.5, 0.0, 1.5])
    np.testing.assert_allclose(advantages, expected_value)

  def test_compute_rloo_advantages_low_generations(self):
    rewards = jnp.array([1.0, 2.0])
    advantages = agentic_grpo_learner.compute_rloo_advantages(
        rewards, num_generations=1
    )
    np.testing.assert_allclose(advantages, jnp.zeros_like(rewards))


if __name__ == "__main__":
  absltest.main()
