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

"""End-to-end test for the agentic GRPO learner refactored onto the orchestrator.

Runs OrchestratedAgenticGRPOLearner (which reuses the agentic loop but expresses
its postprocess via the orchestrator primitives + algorithm adapter) on a toy
model through the worker-backed OrchestratorRLCluster, and asserts it trains.
"""

import os

from absl.testing import absltest
import chex
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import optax
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import orchestrated_agentic_learner
from tunix.experimental.orchestrator import orchestrator_rl_cluster
from tunix.experimental.orchestrator import rl_orchestrator
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"


def _reward_fn(prompts, completions, **kwargs):
  del prompts, kwargs
  return [float(i) for i in range(len(completions))]


class _MockChatParser:

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


class OrchestratedAgenticGrpoLearnerTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    try:
      chex.set_n_cpu_devices(2)
    except RuntimeError:
      # Another test in this process already initialized JAX; reuse whatever
      # device count it established rather than failing collection.
      pass

  def setUp(self):
    super().setUp()
    self.vocab = test_common.MockVocab()
    self.tokenizer = tokenizer_adapter.TokenizerAdapter(self.vocab)

  def _build_orchestrator(self, max_steps, grpo_config):
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
            actor_optimizer=optax.sgd(1e-2),
            eval_every_n_steps=100,
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
    base = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=self.tokenizer,
        cluster_config=cluster_config,
    )
    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(base)
    orch = rl_orchestrator.RLOrchestrator(
        cluster, algorithm_adapter.GRPOAdapter(grpo_config)
    )
    return orch, model

  def test_refactored_learner_trains_via_postprocess_group(self):
    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        beta=0.0,
        max_response_length=10,
    )
    orch, model = self._build_orchestrator(max_steps=2, grpo_config=grpo_config)
    original_params = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    learner = orchestrated_agentic_learner.OrchestratedAgenticGRPOLearner(
        orchestrator=orch,
        reward_fns=_reward_fn,
        chat_parser=_MockChatParser(),
    )

    train_ds = [
        {"prompts": [str(i)], "answer": [str(i)], "question": [str(i)]}
        for i in range(4)
    ]
    learner.train(train_ds)

    self.assertEqual(orch.cluster.global_steps, 2)
    updated_params = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_params, updated_params
    )


if __name__ == "__main__":
  absltest.main()
