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

"""The agentic GRPO learner refactored onto the orchestrator API.

This shows how the existing agentic learner maps onto the new stack:

  * It is constructed from an `RLOrchestrator` (Layer 2) rather than a raw
    cluster; the orchestrator carries the cluster (Layer 1, in-process or
    worker-backed) and the `AlgorithmAdapter` (Layer 2 algorithm hooks).
  * The async episode machinery and the producer/consumer loop are REUSED
    verbatim from `GRPOLearner` (they are plumbing, not cluster/algorithm logic)
    -- they drive the cluster through the orchestrator's cluster, so a worker-
    backed cluster distributes them with no change.
  * The postprocess (`_process_results`, the most cluster-heavy, algorithm-
    specific method) is re-expressed on the new API: it delegates to the
    adapter's `postprocess_group`, which scores via the orchestrator primitives
    and computes advantages via the shared estimator.

The remaining cluster interactions the reused loop performs (generate, train,
sync, metrics) already route through the orchestrator's cluster. As the loop is
further promoted, those top-level calls move onto `RLOrchestrator` primitives too
(see `SimpleGRPOLoop` for the fully-primitive form).
"""

from typing import Any

from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import rl_orchestrator as rl_orchestrator_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner


class OrchestratedAgenticGRPOLearner(agentic_grpo_learner.GRPOLearner):
  """Agentic GRPO learner built on an `RLOrchestrator`."""

  def __init__(
      self,
      *,
      orchestrator: rl_orchestrator_lib.RLOrchestrator,
      reward_fns: Any = None,
      metric_fns: Any = None,
      chat_parser: Any = None,
      **kwargs,
  ):
    """Initializes the learner from an orchestrator.

    Args:
      orchestrator: The `RLOrchestrator` (cluster + algorithm adapter). Its
        cluster backs the reused loop; its adapter drives the postprocess.
      reward_fns: Reward function(s), as for `GRPOLearner`.
      metric_fns: Not supported; see Raises.
      chat_parser: Optional chat parser.
      **kwargs: Forwarded to `GRPOLearner.__init__` (agent/env classes, etc.).

    Raises:
      algorithm_adapter.UnsupportedConfigError: If `metric_fns` are passed. The
        base learner invokes them from `_process_results`, which this class
        overrides, so they would be accepted and never called.
    """
    if metric_fns:
      raise algorithm_adapter.UnsupportedConfigError(
          "metric_fns are not supported by this learner: the base learner"
          " invokes them at the end of its postprocess, which this class"
          " replaces, so they would never run. Drop them or use the agentic"
          " GRPO learner."
      )
    self._orchestrator = orchestrator
    super().__init__(
        rl_cluster=orchestrator.cluster,
        reward_fns=reward_fns,
        algo_config=orchestrator.algorithm.algo_config,
        metric_fns=metric_fns,
        chat_parser=chat_parser,
        **kwargs,
    )

  @property
  def orchestrator(self) -> rl_orchestrator_lib.RLOrchestrator:
    return self._orchestrator

  def _process_results(
      self,
      trajectories: Any,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      expected_step: int | None = None,
  ):
    """Postprocess expressed on the new API (adapter + orchestrator primitives)."""
    return self._orchestrator.algorithm.postprocess_group(
        self._orchestrator,
        trajectories,
        compute_rewards=self._compute_rewards,
        mode=mode,
        expected_step=expected_step,
    )
