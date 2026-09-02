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

"""Simple Cluster Orchestrator Example Notebook / Script (V2 Architecture)."""

from typing import Any
from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.worker import abstract_worker


class SimulatedRolloutWorker(abstract_worker.Worker):
  """Simulated RolloutWorker generating synthetic token responses."""

  def __init__(self, worker_id: str):
    self.worker_id = worker_id

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=frozenset([datatypes.Role.ROLLOUT]),
    )

  def initialize(self) -> datatypes.Response:
    return datatypes.Response()

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    del dummy_data
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    return datatypes.Response()

  def generate(self, prompts: Any, **kwargs: Any) -> list[datatypes.RolloutResponse]:
    del kwargs
    logging.info("[%s] Generating rollouts for %d prompt(s)", self.worker_id, len(prompts))
    responses = []
    for idx, _ in enumerate(prompts):
      responses.append(
          datatypes.RolloutResponse(
              request_id=f"req_{idx}",
              status="COMPLETED",
              env_reward=1.0,
              prompt_tokens=np.array([10, 11], dtype=np.int32),
              segments=[
                  datatypes.TokenSegment(
                      source="assistant",
                      tokens=np.array([20, 21], dtype=np.int32),
                      loss_mask=np.array([1, 1], dtype=np.int32),
                  )
              ],
              metadata={"prompt_id": f"prompt_{idx}"},
          )
      )
    return responses

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=datatypes.WorkerState.READY)


class SimulatedTrainerWorker(abstract_worker.Worker):
  """Simulated TrainerWorker executing gradient updates and weight sync staging."""

  def __init__(self, worker_id: str, role: datatypes.Role):
    self.worker_id = worker_id
    self._role = role

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=frozenset([self._role]),
    )

  def initialize(self) -> datatypes.Response:
    return datatypes.Response()

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    del dummy_data
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    return datatypes.Response()

  def fwd_bwd(self, batch: Any, accumulate_gradients: bool = False, apply_optimizer: bool = True, skip_jit: bool = False) -> dict[str, float]:
    del batch, accumulate_gradients, apply_optimizer, skip_jit
    logging.info("[%s] Executing gradient update (fwd_bwd)", self.worker_id)
    return {"loss": 0.25, "grad_norm": 1.2}

  def prepare_weight_sync(self) -> datatypes.WeightSyncMetadata:
    logging.info("[%s] Staging weights for sync", self.worker_id)
    meta = datatypes.WeightSyncMetadata(
        new_policy_version=1,
        transfer_mode="p2p",
        source_endpoints=["trainer:50051"],
    )
    return meta

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=datatypes.WorkerState.READY)


def main():
  orch = orchestrator.ClusterOrchestrator()

  rollout_0 = SimulatedRolloutWorker("rollout_0")
  rollout_1 = SimulatedRolloutWorker("rollout_1")
  actor_trainer = SimulatedTrainerWorker("actor_0", datatypes.Role.ACTOR)

  orch.register_worker(rollout_0)
  orch.register_worker(rollout_1)
  orch.register_worker(actor_trainer)

  print(f"Registered roles in cluster: {orch.registry.roles()}")

  algo = algorithm_adapter.GRPOAdapter(group_size=2, mini_batch_size=1, max_packed_len=32)
  assembler = batch_assembly.SequencePackedBatchAssembler(
      group_size=2, mini_batch_size=1, max_packed_len=32
  )

  train_dataset = [
      ["Solve 2 + 2", "Solve 3 * 4"],
      ["Solve 10 / 2", "Solve 7 - 5"],
  ]

  print("Executing Tier 1 Managed Run via ClusterOrchestrator...")
  orch.run(
      algo=algo,
      dataset=train_dataset,
      reward_fns=[lambda x: 1.0],
      assembler=assembler,
      max_steps=2,
  )
  print("Execution completed successfully!")


if __name__ == "__main__":
  main()
