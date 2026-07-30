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

"""The full agentic GRPO loop, running over the orchestrator's worker fleet."""

import os

from absl.testing import absltest
import chex
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import optax
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import agentic_runner
from tunix.experimental.orchestrator import worker_fleet
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

MAX_RESPONSE_LENGTH = 10


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


class AgenticRunnerTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    try:
      chex.set_n_cpu_devices(2)
    except RuntimeError:
      pass

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
            max_tokens_to_generate=MAX_RESPONSE_LENGTH,
            return_logprobs=True,
            kv_cache_size=256,
            temperature=0.5,
        ),
    )
    cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=self.tokenizer,
        cluster_config=cluster_config,
    )
    return cluster, model

  def _runner(self, cluster, max_steps):
    del max_steps
    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        beta=0.0,
        max_response_length=MAX_RESPONSE_LENGTH,
    )
    return agentic_runner.AgenticGRPORunner(
        cluster=cluster,
        algo_config=grpo_config,
        reward_fns=_reward_fn,
        chat_parser=_MockChatParser(),
    )

  def _train_ds(self, n=4):
    return [
        {"prompts": [str(i)], "answer": [str(i)], "question": [str(i)]}
        for i in range(n)
    ]

  def test_full_agentic_loop_trains_through_the_worker_fleet(self):
    max_steps = 2
    cluster, model = self._build_cluster(max_steps)
    before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    runner = self._runner(cluster, max_steps)
    runner.bring_up()

    # The control plane sees a healthy fleet before any training happens.
    health = runner.poll_health()
    self.assertLen(health, 3)
    for report in health.values():
      self.assertEqual(report.state, datatypes.WorkerState.READY)

    runner.train(self._train_ds())

    # The real agentic loop ran: steps advanced and the actor moved.
    self.assertEqual(runner.global_steps, max_steps)
    after = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(test_common.assert_not_equal, before, after)

    runner.shutdown()
    for report in runner.poll_health().values():
      self.assertEqual(report.state, datatypes.WorkerState.STOPPED)

  def test_runner_drives_the_handles_not_the_cluster_directly(self):
    cluster, _ = self._build_cluster(max_steps=1)
    runner = self._runner(cluster, max_steps=1)
    # Every compute primitive is routed to a fleet handle.
    routed = runner.orchestrator.cluster
    self.assertIs(routed._trainer_worker, runner.fleet.trainer)
    self.assertIs(routed._rollout_worker, runner.fleet.rollout)
    self.assertIs(routed._inference_worker, runner.fleet.inference)
    self.assertIs(routed._weight_sync, runner.fleet.weight_sync)

  def test_accepts_an_externally_built_fleet(self):
    cluster, _ = self._build_cluster(max_steps=1)
    fleet = worker_fleet.WorkerFleet.in_process(cluster)
    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        beta=0.0,
        max_response_length=MAX_RESPONSE_LENGTH,
    )
    runner = agentic_runner.AgenticGRPORunner(
        cluster=cluster,
        algo_config=grpo_config,
        reward_fns=_reward_fn,
        chat_parser=_MockChatParser(),
        fleet=fleet,
    )
    # An RPC-backed fleet would swap in exactly here, with no other change.
    self.assertIs(runner.fleet, fleet)


if __name__ == "__main__":
  absltest.main()
