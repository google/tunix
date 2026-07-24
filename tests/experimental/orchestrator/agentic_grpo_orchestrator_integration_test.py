# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end integration test for AgenticGRPOOrchestrator.

Runs the orchestrator's inherited `train()` loop against a tiny toy model, with
the trainer routed through a real in-process trainer-worker handle and every
other seam falling back to the in-process cluster. This exercises the whole reuse
path -- rollout, grouping, reference scoring, advantage math, gradient
accumulation, weight bookkeeping -- and asserts that training actually happens
through the handle.
"""

import os
from unittest import mock

from absl.testing import absltest
import chex
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import optax
from tunix.experimental.orchestrator import agentic_grpo_orchestrator
from tunix.experimental.orchestrator import inprocess_workers
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"


def _reward_fn(prompts, completions, **kwargs):
  del prompts, kwargs
  # Distinct rewards -> non-degenerate group advantages -> real gradients.
  return [float(i) for i in range(len(completions))]


class _MockChatParser:
  """Minimal chat parser that flattens messages to text."""

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


class AgenticGrpoOrchestratorIntegrationTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    chex.set_n_cpu_devices(2)

  def setUp(self):
    super().setUp()
    self.vocab = test_common.MockVocab()
    self.tokenizer = tokenizer_adapter.TokenizerAdapter(self.vocab)

  def _build_cluster(self, max_steps):
    model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = test_common.ToyTransformer(
        config=test_common.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
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
            eval_every_n_steps=100,  # skip eval
            max_steps=max_steps,
            mini_batch_size=1,
            train_micro_batch_size=1,
            rollout_micro_batch_size=1,
            compute_logps_micro_batch_size=1,
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
        tokenizer=self.tokenizer,
        cluster_config=cluster_config,
    )
    return rl_cluster, model

  def test_orchestrator_trains_through_inprocess_trainer_handle(self):
    max_steps = 2
    rl_cluster, model = self._build_cluster(max_steps)
    original_params = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        max_response_length=10,
    )
    handle = inprocess_workers.InProcessTrainerWorker(rl_cluster)
    orchestrator = agentic_grpo_orchestrator.AgenticGRPOOrchestrator(
        trainer_worker=handle,
        rl_cluster=rl_cluster,
        reward_fns=_reward_fn,
        algo_config=grpo_config,
        chat_parser=_MockChatParser(),
    )

    train_ds = [
        {"prompts": [str(i)], "answer": [str(i)], "question": [str(i)]}
        for i in range(4)
    ]

    with mock.patch.object(
        handle, "train", wraps=handle.train
    ) as spy_train:
      orchestrator.train(train_ds)

    # Training was driven through the injected handle, not the base path.
    self.assertGreater(spy_train.call_count, 0)
    # The reused loop advanced the step counter to max_steps.
    self.assertEqual(rl_cluster.global_steps, max_steps)
    # Actor weights actually moved (gradients flowed through the handle).
    updated_params = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_params, updated_params
    )

  def test_inprocess_handle_scores_actor_logps(self):
    # The handle also satisfies the actor-scoring seam contract used by the
    # orchestrator's _actor_per_token_logps override.
    rl_cluster, _ = self._build_cluster(max_steps=1)
    handle = inprocess_workers.InProcessTrainerWorker(rl_cluster)
    prompt_ids = jnp.ones((2, 4), dtype=jnp.int32)
    completion_ids = jnp.ones((2, 6), dtype=jnp.int32)

    with mock.patch.object(
        rl_cluster,
        "get_actor_per_token_logps",
        return_value=jnp.zeros((2, 6)),
        autospec=True,
    ) as mock_get_actor:
      out = handle.per_token_logps(prompt_ids, completion_ids, pad_id=0, eos_id=2)

    self.assertEqual(out.shape, (2, 6))
    mock_get_actor.assert_called_once()
    _, kwargs = mock_get_actor.call_args
    self.assertEqual(kwargs["pad_id"], 0)
    self.assertEqual(kwargs["eos_id"], 2)
    self.assertEqual(kwargs["micro_batch_size"], 1)


if __name__ == "__main__":
  absltest.main()
