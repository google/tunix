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

"""End-to-end test for the Layer-3 SimpleGRPOLoop over the primitive API.

Runs the thin single-turn GRPO loop -- which composes only RLOrchestrator
primitives -- on a toy model, driven through the worker-backed
OrchestratorRLCluster, and asserts it actually trains (actor weights move, step
counter advances). This is the proof that the primitive API is sufficient and
pluggable end to end.
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
from tunix.experimental.orchestrator import orchestrator_rl_cluster
from tunix.experimental.orchestrator import rl_orchestrator
from tunix.experimental.orchestrator import simple_grpo_loop
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

_MAX_PROMPT_LENGTH = 32
_MAX_RESPONSE_LENGTH = 10


def _reward_fn(prompts, completions, **kwargs):
  del prompts, kwargs
  # Distinct rewards -> non-degenerate group advantages -> real gradients.
  return [float(i) for i in range(len(completions))]


class SimpleGrpoLoopTest(absltest.TestCase):

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

  def _build_base_cluster(self):
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
            mini_batch_size=2,
            train_micro_batch_size=2,
            compute_logps_micro_batch_size=2,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_prompt_length=_MAX_PROMPT_LENGTH,
            max_tokens_to_generate=_MAX_RESPONSE_LENGTH,
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

  def test_simple_loop_trains_toy_model_over_primitive_api(self):
    base, model = self._build_base_cluster()
    original_params = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    # Layer 1: worker-backed cluster (in-process fallback). Layer 2: orchestrator
    # + GRPO adapter. Layer 3: the thin loop.
    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(base)
    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=2,
        num_iterations=1,
        beta=0.0,  # no KL -> no reference logps needed for this thin loop
        max_response_length=_MAX_RESPONSE_LENGTH,
    )
    orch = rl_orchestrator.RLOrchestrator(
        cluster, algorithm_adapter.GRPOAdapter(grpo_config)
    )
    loop = simple_grpo_loop.SimpleGRPOLoop(
        orch,
        reward_fn=_reward_fn,
        tokenizer=self.tokenizer,
        num_generations=2,
        max_prompt_length=_MAX_PROMPT_LENGTH,
        max_response_length=_MAX_RESPONSE_LENGTH,
        pad_id=base.rollout.pad_id(),
    )

    loop.train(["1", "2"])

    self.assertEqual(base.global_steps, 2)
    updated_params = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(
        test_common.assert_not_equal, original_params, updated_params
    )


if __name__ == "__main__":
  absltest.main()
