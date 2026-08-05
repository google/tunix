# %%
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

# %% [markdown]
# # Simple Cluster Orchestrator Example
#
# This notebook demonstrates how to set up and execute a distributed RL workflow
# using the 5-layer `tunix.experimental.orchestrator` architecture:
# 1.  **Layer 5 (`ClusterOrchestrator`)**: Manages WorkerRegistry, LifecycleDriver,
#     and HealthMonitor.
# 2.  **Layer 1 (`WorkerRegistry` / Workers)**: Registers distributed or remote
#     worker handles (`RolloutWorker`, `TrainerWorker`, `InferenceWorker`).
# 3.  **Layer 2 (`DistributedRLEngine`)**: Automatically constructed from registered
#     worker role groups.
# 4.  **Layer 3 (`RLDriver`)**: Composes the engine with GRPO math, reward functions,
#     and tokenizer.
# 5.  **Layer 4 (`RLProgram`)**: Coordinates the iterative RL training loop
#     (`generate` -> `process_results` -> `train_step` -> `sync_weights`) and
#     lifecycle callbacks.

# %%
"""Simple Cluster Orchestrator Example Notebook."""

from typing import Any
from unittest import mock
from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.orchestrator import rl_driver
from tunix.experimental.orchestrator import rl_program

# %% [markdown]
# ## 1. Define Simulated / Remote Workers
#
# In a production distributed deployment, these worker objects correspond to remote
# gRPC service clients (`ActorHandle`) over `RolloutWorker`, `TrainerWorker`, and
# `InferenceWorker` services. Here we define lightweight mocks to simulate
# end-to-end execution.


# %%
class SimulatedRolloutWorker:

  def __init__(self, worker_id: str):
    self.worker_id = worker_id

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=[datatypes.Role.ROLLOUT],
        state="READY",
    )

  def generate(self, prompts, **kwargs):
    logging.info(
        "[%s] Generating rollouts for %d prompt(s)",
        self.worker_id,
        len(prompts),
    )
    return [{"prompt": p, "completion": f"Response to {p}"} for p in prompts]

  def heartbeat(self):
    return datatypes.HealthReport(worker_id=self.worker_id, status="HEALTHY")


class SimulatedTrainerWorker:

  def __init__(self, worker_id: str, role: datatypes.Role):
    self.worker_id = worker_id
    self._role = role

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=[self._role],
        state="READY",
    )

  def fwd_bwd(self, batch, skip_jit=False):
    logging.info("[%s] Executing gradient update (fwd_bwd)", self.worker_id)
    return {"loss": 0.25, "grad_norm": 1.2}

  def prepare_weight_sync(self):
    logging.info("[%s] Staging weights for sync", self.worker_id)
    return datatypes.WeightSyncMetadata(
        new_policy_version=1,
        transfer_mode="p2p",
        source_endpoints=["trainer:50051"],
        sharding_topology={"mesh": [2, 2]},
    )

  def heartbeat(self):
    return datatypes.HealthReport(worker_id=self.worker_id, status="HEALTHY")


class SimulatedInferenceWorker:

  def __init__(self, worker_id: str):
    self.worker_id = worker_id

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=[datatypes.Role.REFERENCE],
        state="READY",
    )

  def compute_logprobs(self, req: datatypes.LogprobsRequest):
    logging.info("[%s] Computing reference logprobs", self.worker_id)
    return datatypes.LogprobsResponse(
        per_token_logps=np.zeros((len(req.prompt_tokens), 16), dtype=np.float32)
    )

  def heartbeat(self):
    return datatypes.HealthReport(worker_id=self.worker_id, status="HEALTHY")


# %% [markdown]
# ## 2. Initialize Orchestrator and Register Workers (Layer 5 & Layer 1)
#
# Instantiate `ClusterOrchestrator` and register the workers. The orchestrator
# indexes workers by role (`ROLLOUT`, `ACTOR`, `CRITIC`, `REFERENCE`).

# %%
orch = orchestrator.ClusterOrchestrator()

rollout_0 = SimulatedRolloutWorker("rollout_0")
rollout_1 = SimulatedRolloutWorker("rollout_1")
actor_trainer = SimulatedTrainerWorker("actor_0", datatypes.Role.ACTOR)
critic_trainer = SimulatedTrainerWorker("critic_0", datatypes.Role.CRITIC)
ref_worker = SimulatedInferenceWorker("ref_0")

orch.register_worker(rollout_0)
orch.register_worker(rollout_1)
orch.register_worker(actor_trainer)
orch.register_worker(critic_trainer)
orch.register_worker(ref_worker)

print(f"Registered roles in cluster: {orch.registry.roles()}")

# %% [markdown]
# ## 3. Construct Distributed RL Engine (Layer 2)
#
# Construct a `DistributedRLEngine` from the registered role groups in the cluster.

# %%
engine = orch.create_engine()
print(
    f"Created DistributedRLEngine with {len(engine._rollout_workers)} rollout"
    " replica(s)"
)

# %% [markdown]
# ## 4. Build RL Driver and Program (Layer 3 & Layer 4)
#
# - **RLDriver** combines `DistributedRLEngine` with algorithm configurations and
#   reward functions.
# - **RLProgram** wraps the driver and coordinates the iterative RL training loop
#   with step-lifecycle callbacks (`on_step_begin`, `on_step_end`).

# %%
# Minimal algorithm config and mock tokenizer for demonstration
algo_config = mock.MagicMock()
algo_config.reward_manager = "agentic-sequence-level"
tokenizer = mock.MagicMock()


def simple_reward_fn(prompts, completions, **kwargs):
  return [1.0 for _ in completions]


driver = rl_driver.RLDriver(
    rl_engine=engine,
    algo_config=algo_config,
    reward_fns=[simple_reward_fn],
    tokenizer=tokenizer,
)


def on_step_end(step: int, result: Any):
  print(f"=== Completed Step {step} | Result: {result} ===")


program = rl_program.RLProgram(
    driver=driver,
    on_step_end=on_step_end,
)

# %% [markdown]
# ## 5. Execute Program via ClusterOrchestrator
#
# Invoke `orch.run_program(program, train_dataset, ...)` to execute the RL workflow
# across the distributed cluster.

# %%
# Synthetic training dataset yielding batches of prompts
train_dataset = [
    ["Solve 2 + 2", "Solve 3 * 4"],
    ["Solve 10 / 2", "Solve 7 - 5"],
    ["Solve 8 + 9", "Solve 6 * 6"],
]

print("Starting RL Program execution via ClusterOrchestrator...")
orch.run_program(
    program=program,
    train_dataset=train_dataset,
    num_steps=3,
    bring_up=False,  # Set True for full lifecycle initialize/compile/start
)
print("Execution completed successfully!")
