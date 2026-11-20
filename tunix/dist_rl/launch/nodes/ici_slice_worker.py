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
"""

from __future__ import annotations

import logging
import os
import sys

import jax
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import rollout
from tunix.dist_rl.launch.nodes import trainer


def main(
    argv: list[str], context: context_lib.ProcessContext | None = None
) -> None:
  """Initializes JAX, then dispatches to trainer or rollout by torus rank."""
  process_context = node_utils.require_discovery_context(context)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [IciSliceWorker] %(message)s",
      force=True,
  )
  hp = resolve_config.initialize(argv)
  cluster = hp.config_mapping("cluster_config")
  trainer_replicas = int(cluster.get("trainer_replicas", 1))
  rollout_replicas = int(cluster["rollout_replicas"])

  process_context.initialize_runtime(hp.cluster_targets.trainer)
  rank = int(jax.process_index())
  world_size = int(jax.process_count())
  expected_world = trainer_replicas + rollout_replicas
  if world_size != expected_world:
    raise ValueError(
        f"Shared ICI slice world size ({world_size}) must equal "
        f"trainer_replicas ({trainer_replicas}) + "
        f"rollout_replicas ({rollout_replicas}) = {expected_world}."
    )

  if rank < trainer_replicas:
    os.environ[node_utils.TRAINER_REPLICA_ENV] = str(rank)
    os.environ.pop(node_utils.ROLLOUT_REPLICA_ENV, None)
    logging.info(
        "Dispatching JAX rank %d/%d (devices=%s) as TRAINER replica %d/%d",
        rank,
        world_size,
        [d.id for d in jax.local_devices()],
        rank,
        trainer_replicas,
    )
    trainer.main(argv, process_context)
  else:
    rollout_replica = rank - trainer_replicas
    os.environ[node_utils.ROLLOUT_REPLICA_ENV] = str(rollout_replica)
    os.environ.pop(node_utils.TRAINER_REPLICA_ENV, None)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("NUM_PRECOMPILE_WORKERS", "8")
    os.environ.setdefault("SLICE_ROPE_CACHE", "1")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1800")
    logging.info(
        "Dispatching JAX rank %d/%d (devices=%s) as ROLLOUT replica %d/%d",
        rank,
        world_size,
        [d.id for d in jax.local_devices()],
        rollout_replica,
        rollout_replicas,
    )
    rollout.main(argv, process_context)


if __name__ == "__main__":
  main(sys.argv[1:])
