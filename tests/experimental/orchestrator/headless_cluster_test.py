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

"""An orchestrator that owns no models can still be built against.

The point of these is negative as much as positive: the loops must construct,
and the process must contain no model parameters while they do.
"""

import os
from typing import Any

from absl.testing import absltest
import numpy as np
import optax
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import headless_cluster
from tunix.experimental.orchestrator import hosted_rollout_worker
from tunix.experimental.orchestrator import orchestrator_rl_cluster
from tunix.experimental.orchestrator import rl_orchestrator
from tunix.experimental.orchestrator import simple_grpo_loop
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

_MAX_RESPONSE_LENGTH = 10


def _reward_fn(prompts, completions, **kwargs):
  del prompts, kwargs
  return [float(i) for i in range(len(completions))]


class _MockChatParser:

  def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
    del is_first_msg
    result = ""
    for message in messages:
      result += f" {message['role']}: {message['content']}"
    if add_generation_prompt:
      result += " " + self.assistant_token
    return result

  @property
  def assistant_token(self):
    return "Assistant: "

  def update_assistant_end_tokens(self, tokens):
    return tokens, 0


class _Output:

  def __init__(self, prompts):
    self.text = [f"completion {p}" for p in prompts]
    self.tokens = [np.array([3, 4], dtype=np.int32) for _ in prompts]
    self.logprobs = [np.array([-0.1, -0.2], dtype=np.float32) for _ in prompts]
    self.left_padded_prompt_tokens = np.array(
        [[0, 1] for _ in prompts], dtype=np.int32
    )
    self.logits = None


class _Engine:

  def generate(self, prompts, *args, **kwargs):
    del args, kwargs
    return _Output(prompts)


class HeadlessClusterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = test_common.MockVocab()
    self.tokenizer = tokenizer_adapter.TokenizerAdapter(self.vocab)

  def _config(self):
    return rl_cluster_lib.ClusterConfig(
        role_to_mesh={},
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-2),
            eval_every_n_steps=100,
            mini_batch_size=2,
            train_micro_batch_size=2,
            compute_logps_micro_batch_size=2,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=32,
            max_tokens_to_generate=_MAX_RESPONSE_LENGTH,
            return_logprobs=True,
            kv_cache_size=256,
            temperature=0.5,
        ),
    )

  def _headless(self):
    return headless_cluster.HeadlessCluster(
        cluster_config=self._config(),
        tokenizer=self.tokenizer,
        pad_id=0,
        eos_id=2,
    )

  def test_serves_the_configuration_reads_without_a_model(self):
    cluster = self._headless()

    self.assertEqual(cluster.rollout.pad_id(), 0)
    self.assertEqual(cluster.rollout.eos_id(), 2)
    self.assertEqual(
        cluster.get_rollout_config(rl_cluster_lib.Mode.TRAIN).temperature, 0.5
    )
    self.assertIsNotNone(cluster.tokenizer)
    self.assertEqual(cluster.global_steps, 0)
    cluster.global_steps += 1
    self.assertEqual(cluster.global_steps, 1)

  def test_reports_that_no_weights_are_shared(self):
    """Across processes they cannot be; answering None says so honestly."""
    cluster = self._headless()

    self.assertIsNone(cluster.actor_trainer.model)
    self.assertIsNone(cluster.rollout.model())

  def test_accepts_the_trainer_wiring_the_learner_installs(self):
    cluster = self._headless()

    cluster.actor_trainer.with_loss_fn(lambda *a, **k: None, has_aux=True)
    cluster.actor_trainer.with_gen_model_input_fn(lambda x: {"x": x})
    cluster.actor_trainer.is_managed_externally = True

    self.assertIsNotNone(cluster.actor_trainer.loss_fn)
    self.assertIsNotNone(cluster.actor_trainer.gen_model_input_fn)
    self.assertTrue(cluster.actor_trainer.is_managed_externally)

  def test_every_compute_primitive_refuses_rather_than_falling_back(self):
    cluster = self._headless()

    for call in (
        lambda: cluster.generate(["p"]),
        lambda: cluster.update_actor([], None, False),
        lambda: cluster.get_ref_per_token_logps(),
        lambda: cluster.get_actor_per_token_logps(),
        lambda: cluster.sync_weights(),
    ):
      with self.assertRaises(headless_cluster.HeadlessClusterError):
        call()

  def test_a_missing_handle_is_an_error_not_a_silent_detour(self):
    """With models present this would quietly run locally instead."""
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(self._headless())

    with self.assertRaises(headless_cluster.HeadlessClusterError):
      routed.generate(["p"])

  def test_an_attached_handle_serves_the_primitive(self):
    worker = hosted_rollout_worker.HostedRolloutWorker(_Engine())
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(
        self._headless(), rollout_worker=_BatchedAdapter(worker)
    )

    self.assertIsNotNone(routed.generate(["p"]))

  def test_metrics_reach_the_logger(self):
    recorded = []
    cluster = headless_cluster.HeadlessCluster(
        cluster_config=self._config(),
        tokenizer=self.tokenizer,
        pad_id=0,
        eos_id=2,
        metrics_logger=_Logger(recorded),
    )

    cluster.buffer_metrics({"loss": 1.0})

    self.assertEqual(recorded, [{"loss": 1.0}])

  def test_the_simple_loop_constructs_against_it(self):
    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(self._headless())
    orch = rl_orchestrator.RLOrchestrator(
        cluster,
        algorithm_adapter.GRPOAdapter(
            agentic_grpo_learner.GRPOConfig(
                num_generations=2,
                num_iterations=1,
                beta=0.0,
                max_response_length=_MAX_RESPONSE_LENGTH,
            )
        ),
    )

    loop = simple_grpo_loop.SimpleGRPOLoop(
        orch,
        reward_fn=_reward_fn,
        tokenizer=self.tokenizer,
        num_generations=2,
        max_prompt_length=32,
        max_response_length=_MAX_RESPONSE_LENGTH,
        pad_id=0,
    )

    self.assertIsNotNone(loop)
    # Constructing it installed the loss on the headless trainer view.
    self.assertIsNotNone(cluster.actor_trainer.loss_fn)

  def test_building_it_allocates_no_arrays(self):
    """The whole point: this process holds configuration, not weights."""
    import jax  # pylint: disable=g-import-not-at-top

    jax.live_arrays()  # Settle anything a previous test left pending.
    before = len(jax.live_arrays())

    cluster = self._headless()
    orch = rl_orchestrator.RLOrchestrator(
        orchestrator_rl_cluster.OrchestratorRLCluster(cluster),
        algorithm_adapter.GRPOAdapter(
            agentic_grpo_learner.GRPOConfig(
                num_generations=2,
                num_iterations=1,
                beta=0.0,
                max_response_length=_MAX_RESPONSE_LENGTH,
            )
        ),
    )
    simple_grpo_loop.SimpleGRPOLoop(
        orch,
        reward_fn=_reward_fn,
        tokenizer=self.tokenizer,
        num_generations=2,
        max_prompt_length=32,
        max_response_length=_MAX_RESPONSE_LENGTH,
        pad_id=0,
    )

    self.assertEqual(len(jax.live_arrays()), before)


class _Logger:

  def __init__(self, sink):
    self._sink = sink

  def buffer_metrics(self, metrics, mode=None):
    del mode
    self._sink.append(metrics)


class _BatchedAdapter:
  """Presents a per-trajectory worker through the whole-batch handle verb."""

  def __init__(self, worker: Any):
    self._worker = worker

  def generate(self, prompts, *args, **kwargs):
    del args, kwargs
    return _Output(prompts)


if __name__ == "__main__":
  absltest.main()
