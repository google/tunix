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

"""Runs the full agentic GRPO loop on the orchestrator's worker fleet.

`SimpleGRPOLoop` proved the primitive API is sufficient, but it is deliberately
minimal: one completion per generation, synchronous, no episodes. This module is
the other end of that scale -- the real agentic loop, with multi-turn episodes,
the async producer/consumer, grouping, chunking, gradient accumulation, eval
cadence and weight-sync boundaries -- running over the same primitives.

It gets there by *reuse*, not reimplementation:

    AgenticRLLearner.train()          the loop, episode machinery, queues
    OrchestratedAgenticGRPOLearner    postprocess expressed on the primitives
    WorkerFleet                       registry + lifecycle + health over handles
    OrchestratorRLCluster             routes each primitive to its handle

so this class is only the assembly: it stands up the fleet, points an
orchestrator at it, hands that to the agentic learner, and runs. Whether the
handles are in-process or RPC-backed is a construction detail the loop never
sees.

    runner = AgenticGRPORunner(
        cluster=cluster, algo_config=grpo_config, reward_fns=reward_fn
    )
    runner.bring_up()
    runner.train(train_ds)
    runner.shutdown()
"""

from typing import Any, Optional

from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import orchestrated_agentic_learner
from tunix.experimental.orchestrator import rl_orchestrator
from tunix.experimental.orchestrator import worker_fleet as worker_fleet_lib


class AgenticGRPORunner:
  """Assembles the fleet, the orchestrator, and the agentic learner."""

  def __init__(
      self,
      *,
      cluster: Any,
      algo_config: Any,
      reward_fns: Any = None,
      chat_parser: Any = None,
      metric_fns: Any = None,
      fleet: Optional[worker_fleet_lib.WorkerFleet] = None,
      **learner_kwargs,
  ):
    """Wires the stack.

    Args:
      cluster: The backing `RLCluster`. It supplies the surface the orchestrator
        does not route (config, tokenizer, metrics, step counter) and, for an
        in-process fleet, backs the handles too.
      algo_config: `GRPOConfig` for the run; also drives the algorithm adapter.
      reward_fns: Reward function(s) for the learner.
      chat_parser: Optional chat parser.
      metric_fns: Optional metric functions.
      fleet: Worker fleet to drive. Defaults to an in-process fleet over
        `cluster`; pass an RPC-backed fleet to distribute the run.
      **learner_kwargs: Forwarded to the learner (agent_class, env_class, ...).
    """
    self._cluster = cluster
    self._fleet = fleet or worker_fleet_lib.WorkerFleet.in_process(cluster)
    self._orchestrator = rl_orchestrator.RLOrchestrator(
        self._fleet.build_cluster(cluster),
        algorithm_adapter.GRPOAdapter(algo_config),
    )
    self._learner = (
        orchestrated_agentic_learner.OrchestratedAgenticGRPOLearner(
            orchestrator=self._orchestrator,
            reward_fns=reward_fns,
            metric_fns=metric_fns,
            chat_parser=chat_parser,
            **learner_kwargs,
        )
    )

  # --- Control plane --------------------------------------------------------

  def bring_up(self, dummy_data: Any = None) -> None:
    """Brings the worker fleet up (initialize -> compile -> start)."""
    self._fleet.bring_up(dummy_data)

  def poll_health(self) -> dict[str, Any]:
    return self._fleet.poll_health()

  def shutdown(self) -> None:
    self._fleet.shutdown()

  # --- Run ------------------------------------------------------------------

  def train(self, train_ds: Any, eval_ds: Any = None) -> None:
    """Runs the full agentic loop over `train_ds`."""
    self._learner.train(train_ds, eval_ds)

  # --- Accessors ------------------------------------------------------------

  @property
  def fleet(self) -> worker_fleet_lib.WorkerFleet:
    return self._fleet

  @property
  def orchestrator(self) -> rl_orchestrator.RLOrchestrator:
    return self._orchestrator

  @property
  def learner(self) -> Any:
    return self._learner

  @property
  def global_steps(self) -> int:
    return self._cluster.global_steps
