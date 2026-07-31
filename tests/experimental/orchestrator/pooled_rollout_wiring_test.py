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

"""Training over a pool of rollout workers, single-turn and version-pinned.

This is the first configuration where rollouts for one training step are
produced by more than one worker. It is single-turn only, and weight syncing
is refused rather than pinned by convention, because installing new weights
across several workers needs a protocol that confirms each one took them.

It is not evidence about the multi-turn agentic path, which generates
per turn rather than per trajectory.
"""

import asyncio
import contextlib
import itertools
import os
from typing import Any

from absl.testing import absltest
import chex
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np
import optax
import portpicker
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import hosted_rollout_worker
from tunix.experimental.orchestrator import orchestrator_rl_cluster
from tunix.experimental.orchestrator import rl_orchestrator
from tunix.experimental.orchestrator import rollout_pool
from tunix.experimental.orchestrator import simple_grpo_loop
from tunix.experimental.orchestrator import worker_fleet
from tunix.experimental.worker import remote_execution as remote_lib
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

_MAX_PROMPT_LENGTH = 32
_MAX_RESPONSE_LENGTH = 10
_NUM_GENERATIONS = 2

# Distinguishes one sampled completion from the next, across all workers.
_VARIANTS = itertools.count()


def _reward_fn(prompts, completions, **kwargs):
  del prompts, kwargs
  return [float(i) for i in range(len(completions))]


def _some_param_moved(before, after) -> bool:
  """Whether the update reached the model at all.

  Stub engines return the same tokens every time, so parameters no token
  touched stay put; that a specific one moved is the existing single-worker
  loop test's business, not this one's.
  """
  return any(
      not np.allclose(np.asarray(b), np.asarray(a))
      for b, a in zip(
          jax.tree.leaves(before), jax.tree.leaves(after)
      )
  )


class _StubOutput:

  def __init__(self, prompts, worker_id: str, variant: int = 0):
    self.text = [f"{worker_id} says {p}" for p in prompts]
    # Members of a group sample different completions; identical ones would
    # make the group-relative gradients cancel exactly.
    self.tokens = [
        np.array([3 + variant % 4, 4, 5], dtype=np.int32) for _ in prompts
    ]
    self.logprobs = [
        np.array([-0.1, -0.2, -0.3], dtype=np.float32) for _ in prompts
    ]
    self.left_padded_prompt_tokens = np.array(
        [[0, 1, 2] for _ in prompts], dtype=np.int32
    )
    self.logits = None


class _StubEngine:
  """Stands in for a rollout engine that owns a model."""

  def __init__(self, worker_id: str, fail_always: bool = False):
    self.worker_id = worker_id
    self.calls = 0
    self._fail_always = fail_always

  def generate(self, prompts, *args, **kwargs):
    del args, kwargs
    self.calls += 1
    if self._fail_always:
      raise RuntimeError(f"{self.worker_id} is broken")
    # Shared across workers: two members of one group land on different
    # workers, and they must not sample the same completion.
    return _StubOutput(prompts, self.worker_id, variant=next(_VARIANTS))


@contextlib.asynccontextmanager
async def _served(worker: Any):
  port = portpicker.pick_unused_port()
  server = remote_lib.GrpcRemoteExecutionServer(worker)
  await server.start_serving_async(port=port)
  handle = remote_lib.GrpcRemoteActorHandle(
      target_address=f"grpc://localhost:{port}", rpc_timeout_s=30.0
  )
  try:
    yield handle
  finally:
    await handle.close()
    await server.stop_serving()


class PooledRolloutWiringTest(absltest.TestCase):

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

  def _build_cluster(self):
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

  def _build_loop(self, base, pool):
    orch = rl_orchestrator.RLOrchestrator(
        orchestrator_rl_cluster.OrchestratorRLCluster(base),
        algorithm_adapter.GRPOAdapter(
            agentic_grpo_learner.GRPOConfig(
                num_generations=_NUM_GENERATIONS,
                num_iterations=1,
                beta=0.0,
                max_response_length=_MAX_RESPONSE_LENGTH,
            )
        ),
    )
    return simple_grpo_loop.SimpleGRPOLoop(
        orch,
        reward_fn=_reward_fn,
        tokenizer=self.tokenizer,
        num_generations=_NUM_GENERATIONS,
        max_prompt_length=_MAX_PROMPT_LENGTH,
        max_response_length=_MAX_RESPONSE_LENGTH,
        pad_id=base.rollout.pad_id(),
        rollout_pool=pool,
    )

  def test_trains_over_two_in_process_pooled_workers(self):
    base, model = self._build_cluster()
    engines = [_StubEngine("w0"), _StubEngine("w1")]
    pool = rollout_pool.PooledRolloutWorker.from_workers(
        [
            hosted_rollout_worker.HostedRolloutWorker(
                engine, worker_id=engine.worker_id
            )
            for engine in engines
        ],
        max_concurrency=1,
    )
    original_params = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    self._build_loop(base, pool).train(["1", "2"])

    self.assertEqual(base.global_steps, 2)
    self.assertTrue(
        _some_param_moved(original_params, nnx.state(model, nnx.Param))
    )
    # Both workers contributed to the rollouts that produced those updates.
    self.assertTrue(all(engine.calls > 0 for engine in engines))

  def test_trains_over_two_pooled_workers_across_grpc(self):
    async def _handles(stack, engines):
      return [
          await stack.enter_async_context(
              _served(
                  hosted_rollout_worker.HostedRolloutWorker(
                      engine, worker_id=engine.worker_id
                  )
              )
          )
          for engine in engines
      ]

    async def _run():
      base, model = self._build_cluster()
      engines = [_StubEngine("w0"), _StubEngine("w1")]
      original_params = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

      async with contextlib.AsyncExitStack() as stack:
        handles = await _handles(stack, engines)
        pool = rollout_pool.PooledRolloutWorker(handles, max_concurrency=1)
        loop = self._build_loop(base, pool)
        # The loop is synchronous and drives the pool itself, so it runs off
        # this event loop.
        await asyncio.to_thread(loop.train, ["1", "2"])

      self.assertEqual(base.global_steps, 2)
      self.assertTrue(
          _some_param_moved(original_params, nnx.state(model, nnx.Param))
      )
      self.assertTrue(all(engine.calls > 0 for engine in engines))

    asyncio.run(_run())

  def test_a_broken_worker_skips_the_step_instead_of_killing_the_run(self):
    """A failed rollout is data: the group is dropped and the run continues."""
    base, _ = self._build_cluster()
    pool = rollout_pool.PooledRolloutWorker.from_workers(
        [
            hosted_rollout_worker.HostedRolloutWorker(
                _StubEngine("healthy")
            ),
            hosted_rollout_worker.HostedRolloutWorker(
                _StubEngine("broken", fail_always=True), worker_id="broken"
            ),
        ],
        max_concurrency=1,
    )

    self._build_loop(base, pool).train(["1", "2"])

    # Every group had a member on the broken worker, so none of them trained,
    # and nothing raised.
    self.assertEqual(base.global_steps, 0)

  def test_weight_sync_across_a_pool_is_refused(self):
    base, _ = self._build_cluster()
    pool = rollout_pool.PooledRolloutWorker.from_workers(
        [hosted_rollout_worker.HostedRolloutWorker(_StubEngine("w0"))],
        max_concurrency=1,
    )
    orch = rl_orchestrator.RLOrchestrator(
        orchestrator_rl_cluster.OrchestratorRLCluster(base),
        algorithm_adapter.GRPOAdapter(
            agentic_grpo_learner.GRPOConfig(
                num_generations=_NUM_GENERATIONS,
                num_iterations=1,
                beta=0.0,
                max_response_length=_MAX_RESPONSE_LENGTH,
            )
        ),
    )

    with self.assertRaises(ValueError):
      simple_grpo_loop.SimpleGRPOLoop(
          orch,
          reward_fn=_reward_fn,
          tokenizer=self.tokenizer,
          num_generations=_NUM_GENERATIONS,
          max_prompt_length=_MAX_PROMPT_LENGTH,
          max_response_length=_MAX_RESPONSE_LENGTH,
          pad_id=base.rollout.pad_id(),
          rollout_pool=pool,
          sync_weights=True,
      )

  def test_pooling_is_off_unless_asked_for(self):
    """The whole-batch path stays the default; pooling is opt-in."""
    base, _ = self._build_cluster()
    orch = rl_orchestrator.RLOrchestrator(
        orchestrator_rl_cluster.OrchestratorRLCluster(base),
        algorithm_adapter.GRPOAdapter(
            agentic_grpo_learner.GRPOConfig(
                num_generations=_NUM_GENERATIONS,
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
        num_generations=_NUM_GENERATIONS,
        max_prompt_length=_MAX_PROMPT_LENGTH,
        max_response_length=_MAX_RESPONSE_LENGTH,
        pad_id=base.rollout.pad_id(),
    )

    loop.train(["1"])

    self.assertEqual(base.global_steps, 1)


class FleetPoolGuardTest(absltest.TestCase):
  """A whole-batch worker must not be pooled by accident."""

  def test_rejects_a_whole_batch_worker(self):
    class _WholeBatchWorker:

      def generate(self, prompts, *args, **kwargs):
        del prompts, args, kwargs
        return "batched output"

    fleet = worker_fleet.WorkerFleet(rollout=[_WholeBatchWorker()])

    with self.assertRaises(TypeError):
      _ = fleet.rollout_pool

  def test_accepts_a_per_trajectory_worker(self):
    fleet = worker_fleet.WorkerFleet(
        rollout=[
            hosted_rollout_worker.HostedRolloutWorker(_StubEngine("w0")),
            hosted_rollout_worker.HostedRolloutWorker(
                _StubEngine("w1"), worker_id="w1"
            ),
        ]
    )

    self.assertIsNotNone(fleet.rollout_pool)


if __name__ == "__main__":
  absltest.main()
