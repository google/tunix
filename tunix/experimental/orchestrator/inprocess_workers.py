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

"""In-process worker handles backed by an RLCluster.

These are the concrete handles `OrchestratorRLCluster` routes its compute
primitives to when everything runs in one process. Each satisfies two contracts
at once:

  * the *handle* contract the orchestrator calls (`train`, `generate`,
    `per_token_logps`, ...), which a remote RPC handle also satisfies; and
  * the `Worker` ABC, so the control plane (`WorkerRegistry`,
    `LifecycleDriver`, `HealthMonitor`) can register, bring up, and monitor them
    exactly as it would remote workers.

Lifecycle calls are near no-ops here -- the backing `RLCluster` already built
its models and engines -- but implementing them keeps a single control-plane
code path for in-process and distributed runs.
"""

from typing import Any, Mapping, Optional

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker

WorkerState = datatypes.WorkerState


class _InProcessWorker(abstract_worker.Worker):
  """Base for handles that delegate to an in-process `RLCluster`."""

  def __init__(self, rl_cluster: Any, *, worker_id: str, role: str):
    self._rl_cluster = rl_cluster
    self._info = datatypes.WorkerInfo(
        worker_id=worker_id, roles=frozenset({role})
    )

  @property
  def rl_cluster(self) -> Any:
    return self._rl_cluster

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=self.state)

  def initialize(self) -> datatypes.Response:
    # The cluster constructed its models/engines already; nothing to allocate.
    self.state = WorkerState.INITIALIZING
    return datatypes.Response()

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    del dummy_data  # Warmup is driven by the cluster's own first step.
    self.state = WorkerState.READY
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    self.state = WorkerState.READY
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    return datatypes.Response()


class InProcessTrainerWorker(_InProcessWorker):
  """Trainer handle: runs the actor trainer, and the critic trainer on request.

  Handle contract:
      train(chunks, eval_ds, skip_jit) -> None
      train_critic(chunks, eval_ds, skip_jit) -> None
      per_token_logps(prompt_ids, completion_ids, pad_id, eos_id) -> array

  The two trainer passes are separate verbs because the caller drives them
  separately: an algorithm that has a critic asks for both, one per step.
  """

  def __init__(self, rl_cluster: Any, *, worker_id: str = "trainer"):
    super().__init__(rl_cluster, worker_id=worker_id, role="trainer")

  def train(self, chunks: Any, eval_ds: Any, skip_jit: bool) -> None:
    """Runs one actor trainer pass over the micro-batch."""
    self._rl_cluster.update_actor(chunks, eval_ds, skip_jit)

  def train_critic(self, chunks: Any, eval_ds: Any, skip_jit: bool) -> None:
    """Runs one critic trainer pass over the micro-batch."""
    self._rl_cluster.update_critic(chunks, eval_ds, skip_jit)

  def per_token_logps(
      self, prompt_ids: Any, completion_ids: Any, pad_id: int, eos_id: int
  ) -> Any:
    """Actor-model per-token logprobs over a padded group."""
    return self._rl_cluster.get_actor_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=self._rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
    )


class InProcessRolloutWorker(_InProcessWorker):
  """Rollout handle: performs one generation (the LLM forward).

  Handle contract:
      generate(prompts, apply_chat_template, mode, micro_batch_size,
               trace_tags, max_generation_steps) -> RolloutOutput
  """

  def __init__(self, rl_cluster: Any, *, worker_id: str = "rollout"):
    super().__init__(rl_cluster, worker_id=worker_id, role="rollout")

  def generate(
      self,
      prompts: Any,
      apply_chat_template: bool = False,
      mode: Any = None,
      micro_batch_size: Optional[int] = None,
      trace_tags: Optional[Mapping[str, Any]] = None,
      max_generation_steps: Optional[int] = None,
  ) -> Any:
    """Generates completions for `prompts`."""
    return self._rl_cluster.generate(
        prompts,
        apply_chat_template,
        mode,
        micro_batch_size,
        trace_tags,
        max_generation_steps,
    )


class InProcessInferenceWorker(_InProcessWorker):
  """Inference handle: scores tokens under the frozen reference model.

  Handle contract:
      per_token_logps(prompt_ids, completion_ids, pad_id, eos_id) -> array
  """

  def __init__(self, rl_cluster: Any, *, worker_id: str = "inference"):
    super().__init__(rl_cluster, worker_id=worker_id, role="inference")

  def per_token_logps(
      self, prompt_ids: Any, completion_ids: Any, pad_id: int, eos_id: int
  ) -> Any:
    """Reference-model per-token logprobs over a padded group."""
    return self._rl_cluster.get_ref_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=self._rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
    )


class InProcessWeightSync:
  """Weight-sync handle: publishes trainer weights to the rollout replicas.

  Not a `Worker`: weight sync is an action performed across the trainer and
  rollout workers, not a resource the control plane owns.

  Handle contract:
      sync() -> None
  """

  def __init__(self, rl_cluster: Any):
    self._rl_cluster = rl_cluster

  def sync(self) -> None:
    self._rl_cluster.sync_weights()
