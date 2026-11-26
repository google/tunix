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

"""Torus-rank-ordered worker entrypoint for single-slice ICI distributed RL.

On GKE TPU slices, Kubernetes `POD_INDEX` does not match physical TPU torus
coordinates (`jax.process_index()`). Running this entrypoint on every pod in a
shared ICI JobSet initializes the JAX distributed runtime first, then assigns:
  - `jax.process_index() < cluster_config.trainer_replicas` -> `trainer.main`
    (contiguous low-rank sub-torus, always including `primary_host = 0`)
  - `jax.process_index() >= cluster_config.trainer_replicas` -> `rollout.main`
    (contiguous high-rank sub-torus)

Once the node's `main` returns, every process waits at an exit barrier, calls
`jax.distributed.shutdown()`, and exits without interpreter teardown. The coordination service runs in
distributed task 0, which can be a rollout process and exits as soon as the
shutdown barrier passes, so no process may call it after the barrier.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import logging
import os
import sys
import threading
from typing import NoReturn

import jax
from orbax.checkpoint import multihost as ocp_multihost
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.utils import subworld_checkpoint


# Slack on top of the trainer's async checkpoint deadline for the skew between
# the orchestrator's stop RPCs; equals JAX's default shutdown barrier timeout.
_SHUTDOWN_BARRIER_SLACK_S = 300


def _shutdown_barrier_timeout_s(hp: resolve_config.DistHyperParameters) -> int:
  """Returns the exit and JAX shutdown barrier timeout of the slice world.

  Every process waits at the exit barrier, then in `jax.distributed.shutdown()`,
  until all processes reach it. Rollout processes reach it as soon as they are
  stopped, while trainer processes first wait for their last async checkpoint
  save, which may take up to `checkpointing_options.timeout_secs`. A shorter
  barrier aborts the whole slice after a successful run. The exit barrier is
  needed because JAX passes `shutdown_timeout_seconds` only to the
  coordination service: each process still gives up waiting in
  `jax.distributed.shutdown()` after XLA's default of 5 minutes, which aborts
  the slice.

  Args:
    hp: Validated distributed RL configuration.

  Raises:
    ValueError: If `checkpointing_options.timeout_secs` is not a positive int.
  """
  options = hp.config_mapping("rl_training_config")["checkpointing_options"]
  timeout_s = options["timeout_secs"]
  if not isinstance(timeout_s, int) or timeout_s <= 0:
    raise ValueError(
        "ici_slice_worker requires a positive int"
        " rl_training_config.checkpointing_options.timeout_secs to size the"
        f" JAX shutdown barrier, got {timeout_s!r}."
    )
  return timeout_s + _SHUTDOWN_BARRIER_SLACK_S


def _validate_slice_config(
    hp: resolve_config.DistHyperParameters,
) -> tuple[int, int]:
  """Validates shared-slice ICI invariants and returns replica counts."""
  if (
      hp.cluster_targets.weight_sync_mode
      != common_datatypes.WeightSyncMode.ICI
  ):
    raise ValueError(
        "ici_slice_worker requires weight_sync_config.mode='ici', got"
        f" {hp.cluster_targets.weight_sync_mode!r}."
    )
  if hp.cluster_targets.trainer != hp.cluster_targets.rollout:
    raise ValueError(
        "ici_slice_worker requires matching trainer and rollout execution"
        f" targets, got trainer={hp.cluster_targets.trainer!r} vs"
        f" rollout={hp.cluster_targets.rollout!r}."
    )
  cluster = hp.config_mapping("cluster_config")
  return cluster["trainer_replicas"], cluster["rollout_replicas"]


def dispatch_slice_worker(
    argv: Sequence[str],
    context: context_lib.ProcessContext | None = None,
) -> None:
  """Initializes JAX, then dispatches to trainer or rollout by torus rank.

  Args:
    argv: Config overrides passed through to `resolve_config.initialize` and the
      selected node's `main`.
    context: Runtime process context providing JAX init and discovery.

  Raises:
    ValueError: If config invariants fail or `jax.process_count()` does not
      equal `trainer_replicas + rollout_replicas`.
  """
  process_context = node_utils.require_discovery_context(context)
  argv_list = list(argv)
  hp = resolve_config.initialize(argv_list)
  trainer_replicas, rollout_replicas = _validate_slice_config(hp)

  process_context.prepare_env(
      hp.cluster_targets.trainer,
      weight_sync_mode=hp.cluster_targets.weight_sync_mode,
  )
  # The context initializes JAX with the default 300 s shutdown barrier, so
  # initialize here first; `initialize_runtime` then skips JAX initialization.
  barrier_timeout_s = _shutdown_barrier_timeout_s(hp)
  jax.distributed.initialize(shutdown_timeout_seconds=barrier_timeout_s)
  # Collective over the whole world: trainers checkpoint on a subset of
  # processes, which needs Orbax to agree on distributed task ids.
  subworld_checkpoint.enable_distributed_process_ids()
  process_context.initialize_runtime(hp.cluster_targets.trainer)
  rank = int(jax.process_index())
  world_size = int(jax.process_count())
  expected_world = trainer_replicas + rollout_replicas
  if world_size != expected_world:
    raise ValueError(
        f"Shared ICI slice world size ({world_size}) must equal"
        f" trainer_replicas ({trainer_replicas}) +"
        f" rollout_replicas ({rollout_replicas}) = {expected_world}."
    )

  # Lazily import the selected node module after rank dispatch so trainer pods
  # do not trigger rollout-only import-time side effects (e.g. vLLM env vars).
  if rank < trainer_replicas:
    role, replica, total = "trainer", rank, trainer_replicas
    set_env, clear_env = (
        node_utils.TRAINER_REPLICA_ENV,
        node_utils.ROLLOUT_REPLICA_ENV,
    )
  else:
    role, replica, total = "rollout", rank - trainer_replicas, rollout_replicas
    set_env, clear_env = (
        node_utils.ROLLOUT_REPLICA_ENV,
        node_utils.TRAINER_REPLICA_ENV,
    )
  os.environ[set_env] = str(replica)
  os.environ.pop(clear_env, None)
  logging.info(
      "Dispatching JAX rank %d/%d (devices=%s) as %s replica %d/%d",
      rank,
      world_size,
      [d.id for d in jax.local_devices()],
      role.upper(),
      replica,
      total,
  )
  node_mod = importlib.import_module(f"tunix.dist_rl.launch.nodes.{role}")
  node_mod.main(argv_list, process_context)
  # The node has closed its worker, so trainers have finished their checkpoint
  # saves and no Orbax or JAX collective is left to run.
  logging.info("%s replica %d: waiting for the slice to exit.", role, replica)
  ocp_multihost.sync_global_processes(
      "tunix_ici_slice_worker_exit",
      timeout=barrier_timeout_s,
      processes=set(range(world_size)),
  )
  logging.info("%s replica %d: shutting down JAX.", role, replica)
  jax.distributed.shutdown()


def _exit_after_shutdown() -> NoReturn:
  """Exits with code 0, skipping the interpreter teardown.

  The teardown destroys the JAX backend client, which may still call the
  coordination service after `jax.distributed.shutdown()`. That service has
  exited with distributed task 0 by then, and on GKE a call to an exited pod
  hangs until its deadline (minutes) instead of failing fast.
  """
  logging.info("JAX is shut down; exiting.")
  logging.shutdown()
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(0)


def main(
    argv: Sequence[str],
    context: context_lib.ProcessContext | None = None,
) -> None:
  """Entrypoint invoked by `tunix.dist_rl.distributed.runtime.main`.

  Args:
    argv: Config overrides, see `dispatch_slice_worker`.
    context: Runtime process context providing JAX init and discovery.

  Raises:
    RuntimeError: If not called on the main thread, i.e. next to other
      processes of one `runtime.main` invocation, which the exit would kill.
  """
  if threading.current_thread() is not threading.main_thread():
    raise RuntimeError(
        "ici_slice_worker must be the only --process of its runtime.main"
        " invocation: it exits the OS process once JAX is shut down."
    )
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [IciSliceWorker] %(message)s",
      force=True,
  )
  dispatch_slice_worker(argv, context=context)
  _exit_after_shutdown()


if __name__ == "__main__":
  main(sys.argv[1:])
