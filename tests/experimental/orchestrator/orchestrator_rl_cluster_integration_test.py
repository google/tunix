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

"""End-to-end test: the unchanged agentic learner runs on OrchestratorRLCluster.

The whole point of the cluster-swap approach is that an existing `GRPOLearner`
built against the in-process cluster works verbatim when handed an
`OrchestratorRLCluster` instead -- generation, training, scoring, and weight sync
route through the orchestrator's primitives with no learner changes. These tests
run the real agentic loop on a toy model both with a trainer worker handle and in
pure-delegation mode.
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
from tunix.experimental.orchestrator import inprocess_workers
from tunix.experimental.orchestrator import orchestrator_rl_cluster
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


class OrchestratorRlClusterIntegrationTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    chex.set_n_cpu_devices(2)

  def setUp(self):
    super().setUp()
    self.vocab = test_common.MockVocab()
    self.tokenizer = tokenizer_adapter.TokenizerAdapter(self.vocab)

  def _build_base_cluster(self, max_steps):
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
    base = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=self.tokenizer,
        cluster_config=cluster_config,
    )
    return base, model

  def _make_learner(self, cluster):
    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        max_response_length=10,
    )
    # NOTE: the learner class is used unmodified; only the cluster is swapped.
    return agentic_grpo_learner.GRPOLearner(
        rl_cluster=cluster,
        reward_fns=_reward_fn,
        algo_config=grpo_config,
        chat_parser=_MockChatParser(),
    )

  def _train_ds(self):
    return [
        {"prompts": [str(i)], "answer": [str(i)], "question": [str(i)]}
        for i in range(4)
    ]

  def test_learner_trains_through_orchestrator_cluster_with_handle(self):
    max_steps = 2
    base, model = self._build_base_cluster(max_steps)
    original_params = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    handle = inprocess_workers.InProcessTrainerWorker(base)
    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, trainer_worker=handle
    )
    learner = self._make_learner(cluster)

    with mock.patch.object(handle, "train", wraps=handle.train) as spy_train:
      learner.train(self._train_ds())

    # Training was driven through the trainer handle via the cluster primitive.
    self.assertGreater(spy_train.call_count, 0)
    self.assertEqual(base.global_steps, max_steps)
    updated_params = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_params, updated_params
    )

  def test_learner_trains_through_orchestrator_cluster_pure_delegation(self):
    # With no handles, OrchestratorRLCluster is a drop-in for the base cluster:
    # the unchanged learner trains exactly as it would in-process.
    max_steps = 2
    base, model = self._build_base_cluster(max_steps)
    original_params = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(base)
    learner = self._make_learner(cluster)

    learner.train(self._train_ds())

    self.assertEqual(base.global_steps, max_steps)
    updated_params = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_params, updated_params
    )


if __name__ == "__main__":
  absltest.main()
