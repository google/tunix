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

"""A CPU-only RL run with real workers, every call crossing localhost gRPC.

This is the smallest complete picture of the distributed shape:

  * one toy model on CPU, so it runs anywhere;
  * a trainer, a rollout and an inference worker, each hosted by its own
    `GrpcRemoteExecutionServer` on its own localhost port and served from its own
    thread;
  * an orchestrator that holds only RPC handles -- every primitive it invokes
    (generate, train_step, reference scoring) is a real gRPC round trip;
  * the control plane (registry, lifecycle, health) driving those same handles.

The only difference from a multi-host run is where the ports live. Swap the
in-process handles behind the servers for separately-launched processes and the
orchestrator code is unchanged.

Run it directly:

    python -m tunix.experimental.orchestrator.examples.localhost_rl_demo
"""

import asyncio
import threading
import time
from typing import Any, Optional

from absl import logging
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import optax
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import inprocess_workers
from tunix.experimental.orchestrator import rl_orchestrator
from tunix.experimental.orchestrator import rpc_workers
from tunix.experimental.orchestrator import simple_grpo_loop
from tunix.experimental.orchestrator import worker_fleet
from tunix.experimental.worker import remote_execution
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common

MAX_PROMPT_LENGTH = 32
MAX_RESPONSE_LENGTH = 10
NUM_GENERATIONS = 2


def reward_fn(prompts, completions, **kwargs):
  """Toy reward: distinct values so the group advantages are non-degenerate."""
  del prompts, kwargs
  return [float(i) for i in range(len(completions))]


class MockChatParser:
  """Minimal chat parser that flattens messages to a single string."""

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


def build_toy_cluster() -> tuple[Any, Any]:
  """Builds a CPU `RLCluster` around a toy transformer.

  Returns:
    (cluster, actor_model). The model is returned so callers can observe that
    training actually moved the weights.
  """
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
          actor_optimizer=optax.sgd(1e-2),
          eval_every_n_steps=100,
          mini_batch_size=NUM_GENERATIONS,
          train_micro_batch_size=NUM_GENERATIONS,
          compute_logps_micro_batch_size=NUM_GENERATIONS,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_prompt_length=MAX_PROMPT_LENGTH,
          max_tokens_to_generate=MAX_RESPONSE_LENGTH,
          return_logprobs=True,
          kv_cache_size=256,
          temperature=0.5,
      ),
  )
  cluster = rl_cluster_lib.RLCluster(
      actor=model,
      reference=ref_model,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )
  return cluster, model


def serve_worker(
    worker: Any, port: int
) -> tuple[remote_execution.GrpcRemoteExecutionServer, threading.Thread]:
  """Hosts `worker` on a localhost gRPC port, served from its own thread.

  `start_serving` is blocking and owns an event loop, so it goes on a daemon
  thread; that keeps the caller's thread free to drive the (synchronous) training
  loop without starving the server.
  """
  server = remote_execution.GrpcRemoteExecutionServer(worker)
  thread = threading.Thread(
      target=server.start_serving, args=(port,), daemon=True
  )
  thread.start()
  return server, thread


def wait_until_ready(
    handle: Any, timeout_s: float = 30.0, poll_s: float = 0.1
) -> None:
  """Blocks until the worker behind `handle` answers a heartbeat.

  A remote handle reports an unreachable worker as an ERROR heartbeat rather
  than raising (that is what the health monitor wants), so readiness is decided
  on the reported state, not on the absence of an exception.
  """
  deadline = time.monotonic() + timeout_s
  last_state = None
  while time.monotonic() < deadline:
    report = handle.heartbeat()
    last_state = report.state
    if report.state != datatypes.WorkerState.ERROR:
      return
    time.sleep(poll_s)
  raise TimeoutError(
      f"worker never became reachable (last heartbeat: {last_state})"
  )


def run_demo(prompts: Optional[list[str]] = None) -> dict[str, Any]:
  """Runs a short GRPO run with every worker call going over localhost gRPC.

  Args:
    prompts: Prompts to train on. Defaults to two toy prompts.

  Returns:
    A summary dict with the ports used, the health reports, and whether the
    actor weights changed.
  """
  import portpicker  # pylint: disable=g-import-not-at-top

  prompts = prompts if prompts is not None else ["1", "2"]
  cluster, model = build_toy_cluster()
  before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

  # --- Worker side: each role hosted on its own localhost port --------------
  hosted = {
      "trainer": inprocess_workers.InProcessTrainerWorker(cluster),
      "rollout": inprocess_workers.InProcessRolloutWorker(cluster),
      "inference": inprocess_workers.InProcessInferenceWorker(cluster),
  }
  ports = {role: portpicker.pick_unused_port() for role in hosted}
  servers = {
      role: serve_worker(worker, ports[role])
      for role, worker in hosted.items()
  }
  logging.info("serving workers on localhost ports: %s", ports)

  try:
    # --- Orchestrator side: only RPC handles, no direct worker references ---
    fleet = worker_fleet.WorkerFleet(
        trainer=rpc_workers.RemoteTrainerWorker.from_address(
            f"grpc://localhost:{ports['trainer']}", worker_id="trainer"
        ),
        rollout=rpc_workers.RemoteRolloutWorker.from_address(
            f"grpc://localhost:{ports['rollout']}", worker_id="rollout"
        ),
        inference=rpc_workers.RemoteInferenceWorker.from_address(
            f"grpc://localhost:{ports['inference']}", worker_id="inference"
        ),
        # Weight sync is local here: the toy actor and rollout share weights.
        weight_sync=inprocess_workers.InProcessWeightSync(cluster),
    )
    for handle in (fleet.trainer, fleet.rollout, fleet.inference):
      wait_until_ready(handle)

    # Control plane: bring the fleet up and confirm it is healthy.
    fleet.bring_up()
    health = fleet.poll_health()
    logging.info("fleet health: %s", {k: v.state for k, v in health.items()})

    # Data plane: an orchestrator whose primitives are all remote calls.
    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=NUM_GENERATIONS,
        num_iterations=1,
        beta=0.0,
        max_response_length=MAX_RESPONSE_LENGTH,
    )
    orchestrator = rl_orchestrator.RLOrchestrator(
        fleet.build_cluster(cluster),
        algorithm_adapter.GRPOAdapter(grpo_config),
    )
    loop = simple_grpo_loop.SimpleGRPOLoop(
        orchestrator,
        reward_fn=reward_fn,
        tokenizer=cluster.tokenizer,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_response_length=MAX_RESPONSE_LENGTH,
        pad_id=cluster.rollout.pad_id(),
    )
    loop.train(prompts)

    after = nnx.state(model, nnx.Param)
    changed = any(
        not jnp.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        )
    )
    fleet.shutdown()
    return {
        "ports": ports,
        "health": {k: str(v.state) for k, v in health.items()},
        "global_steps": cluster.global_steps,
        "weights_changed": changed,
    }
  finally:
    for server, _ in servers.values():
      loop = server.serve_loop
      if loop is not None:
        # Ask each server to stop on the loop it is serving from; the threads
        # are daemons, so a best-effort shutdown is enough.
        asyncio.run_coroutine_threadsafe(server.stop_serving(), loop)


def main() -> None:
  summary = run_demo()
  print("localhost RL demo finished")
  print(f"  worker ports    : {summary['ports']}")
  print(f"  fleet health    : {summary['health']}")
  print(f"  global steps    : {summary['global_steps']}")
  print(f"  weights changed : {summary['weights_changed']}")


if __name__ == "__main__":
  main()
