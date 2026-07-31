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

import traceback as traceback_lib
from typing import Any, Mapping, Optional

import numpy as np

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
  """Trainer handle: runs the actor (and optional critic) trainer.

  Handle contract:
      train(chunks, eval_ds, skip_jit) -> None
      per_token_logps(prompt_ids, completion_ids, pad_id, eos_id) -> array
  """

  def __init__(self, rl_cluster: Any, *, worker_id: str = "trainer"):
    super().__init__(rl_cluster, worker_id=worker_id, role="trainer")

  def train(self, chunks: Any, eval_ds: Any, skip_jit: bool) -> None:
    """Runs one actor (and optional critic) trainer pass over the micro-batch."""
    self._rl_cluster.update_actor(chunks, eval_ds, skip_jit)
    if hasattr(self._rl_cluster, "critic_trainer"):
      self._rl_cluster.update_critic(chunks, eval_ds, skip_jit)

  def configure_loss(self, spec: Any) -> None:
    """Builds the loss from its description and installs it locally."""
    spec.install_on(self._rl_cluster.actor_trainer)

  def drain_metrics(self) -> dict[str, float]:
    """Nothing to hand back: this trainer writes to the shared logger itself."""
    return {}

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
      compute_logps(LogprobsRequest) -> LogprobsResponse

  The request carries the temperature explicitly. In one process that looks
  redundant, since this handle could read it from the same config the sampler
  did -- which is exactly why it was omitted before. Across processes there is
  no shared config to read, and a score taken at a different temperature than
  the tokens were sampled at is silently biased, so the value travels with the
  request.
  """

  def __init__(self, rl_cluster: Any, *, worker_id: str = "inference"):
    super().__init__(rl_cluster, worker_id=worker_id, role="inference")

  def compute_logps(
      self, request: datatypes.LogprobsRequest
  ) -> datatypes.LogprobsResponse:
    """Scores a padded group under the reference model.

    Args:
      request: Tokens, the temperature to score at, and the model to use.

    Returns:
      The per-token log-probabilities, or a response carrying the failure.
    """
    try:
      if request.model_role != "reference":
        raise NotImplementedError(
            "This handle hosts the frozen reference model only; got"
            f" model_role={request.model_role!r}."
        )
      if request.temperature <= 1e-5:
        raise ValueError("Temperature must be strictly positive.")
      logps = self._score(request)
      return datatypes.LogprobsResponse(
          request_id=request.request_id,
          per_token_logps=np.asarray(logps, dtype=np.float32),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      return datatypes.LogprobsResponse(
          request_id=request.request_id,
          per_token_logps=np.zeros((0, 0), dtype=np.float32),
          error=datatypes.ErrorInfo(
              error_type=type(e).__name__,
              message=str(e),
              traceback=traceback_lib.format_exc(),
          ),
      )

  def _score(self, request: datatypes.LogprobsRequest) -> Any:
    return self._rl_cluster.get_ref_per_token_logps(
        prompt_tokens=request.prompt_tokens,
        completion_tokens=request.completion_tokens,
        pad_id=self._rl_cluster.rollout.pad_id(),
        eos_id=self._rl_cluster.rollout.eos_id(),
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


class AliasedInProcessReplica:
  """Stands in for a rollout replica when they all share one cluster.

  A sync round visits replicas one at a time so a partial round is visible.
  In a single process that shape is degenerate: every handle points at the
  same cluster and the same weights, so installing on one has already
  installed on all of them. Running the round anyway keeps one code path for
  both topologies, and this makes the degenerate case honest rather than
  accidental.

  Two things it deliberately does. It drives the cluster's sync **at most once
  per version**, because that call also advances the cluster's step counter --
  visiting N aliased replicas would advance it N times for one round of
  weights. And it reports the version it was asked for, because in this
  topology reaching it is what installing means.
  """

  def __init__(self, rl_cluster: Any, *, worker_id: str = "rollout"):
    self._rl_cluster = rl_cluster
    self._worker_id = worker_id
    self._installed_version = 0

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"rollout"})
    )

  @property
  def policy_version(self) -> int:
    return self._installed_version

  def prepare_weight_sync(self, metadata: Any) -> datatypes.Response:
    del metadata  # Generation is synchronous here; nothing to fence.
    return datatypes.Response()

  def sync_weights(self, metadata: Any) -> int:
    version = getattr(metadata, "policy_version", None)
    if version is None:
      version = self._installed_version + 1
    if version != self._installed_version:
      self._rl_cluster.sync_weights()
      self._installed_version = int(version)
    return self._installed_version
