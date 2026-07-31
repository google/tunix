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

"""RPC-backed worker handles.

The remote twins of `inprocess_workers`: they satisfy the same two contracts --
the handle methods `OrchestratorRLCluster` calls, and the `Worker` ABC the
control plane manages -- but forward every call over an `ActorHandle` to a
`RemoteExecutionServer` hosting the real worker.

Because both sides implement the same contracts, moving a role off-process is a
construction-time choice:

    trainer = InProcessTrainerWorker(cluster)                    # same process
    trainer = RemoteTrainerWorker.from_address("grpc://host:1")  # over RPC

and nothing in the orchestrator or the learner loop changes.
"""

from typing import Any, Mapping, Optional

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution

WorkerState = datatypes.WorkerState

# A blocking call is held open for as long as the work takes, and these verbs
# differ by orders of magnitude: a training step or a whole-batch generation
# can run for minutes on a real model, while a heartbeat that does not answer
# promptly is the signal the control plane is waiting for. One deadline cannot
# serve both -- too short kills real work, too long makes a hung worker look
# merely busy.
DEFAULT_METHOD_TIMEOUTS_S: Mapping[str, Optional[float]] = {
    "train": 1800.0,
    "train_critic": 1800.0,
    "generate": 1800.0,
    "per_token_logps": 600.0,
    "sync": 600.0,
    "heartbeat": 10.0,
    "info": 10.0,
}


class _RemoteWorker(abstract_worker.Worker):
  """Base for handles that forward to a worker behind an `ActorHandle`."""

  _role: str = ""

  def __init__(
      self,
      actor_handle: remote_execution.ActorHandle,
      *,
      worker_id: str,
      role: Optional[str] = None,
  ):
    self._actor = actor_handle
    self._info = datatypes.WorkerInfo(
        worker_id=worker_id, roles=frozenset({role or self._role})
    )

  @classmethod
  def from_address(
      cls,
      target_address: str,
      *,
      worker_id: Optional[str] = None,
      rpc_timeout_s: Optional[float] = remote_execution.RPC_TIMEOUT_S,
      method_timeouts_s: Optional[
          Mapping[str, Optional[float]]
      ] = None,
  ):
    """Builds a handle for the worker served at `target_address`.

    Args:
      target_address: URI of the served worker.
      worker_id: Identifier reported to the control plane.
      rpc_timeout_s: Deadline for verbs with no per-method entry.
      method_timeouts_s: Per-method deadlines; defaults to
        `DEFAULT_METHOD_TIMEOUTS_S`, which keeps heartbeats short while
        allowing training and generation to run long. Pass a mapping to
        override, or `{}` to put every verb on `rpc_timeout_s`.
    """
    timeouts = (
        DEFAULT_METHOD_TIMEOUTS_S
        if method_timeouts_s is None
        else method_timeouts_s
    )
    return cls(
        remote_execution.ActorHandle.from_address(
            target_address,
            rpc_timeout_s=rpc_timeout_s,
            method_timeouts_s=timeouts,
        ),
        worker_id=worker_id or target_address,
    )

  @property
  def actor(self) -> remote_execution.ActorHandle:
    return self._actor

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  def heartbeat(self) -> datatypes.HealthReport:
    """Asks the remote worker for its state; unreachable counts as ERROR."""
    try:
      return self._actor.submit("heartbeat")
    except Exception as e:  # pylint: disable=broad-exception-caught
      return datatypes.HealthReport(
          state=WorkerState.ERROR, last_error=str(e)
      )

  # State transitions mirror the lifecycle the control plane drives:
  # PENDING -> INITIALIZING -> READY, then STOPPED. (COMPILING is only
  # reachable from READY, so a first compile lands directly on READY.)

  def initialize(self) -> datatypes.Response:
    response = self._actor.submit("initialize")
    self.state = WorkerState.INITIALIZING
    return response

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    response = self._actor.submit("compile", dummy_data)
    self.state = WorkerState.READY
    return response

  def start(self) -> datatypes.Response:
    response = self._actor.submit("start")
    self.state = WorkerState.READY
    return response

  def stop(self) -> datatypes.Response:
    response = self._actor.submit("stop")
    self.state = WorkerState.STOPPED
    return response


class RemoteTrainerWorker(_RemoteWorker):
  """Trainer handle served over RPC."""

  _role = "trainer"

  def train(self, chunks: Any, eval_ds: Any, skip_jit: bool) -> None:
    self._actor.submit("train", chunks, eval_ds, skip_jit)

  def per_token_logps(
      self, prompt_ids: Any, completion_ids: Any, pad_id: int, eos_id: int
  ) -> Any:
    return self._actor.submit(
        "per_token_logps", prompt_ids, completion_ids, pad_id, eos_id
    )


class RemoteRolloutWorker(_RemoteWorker):
  """Rollout handle served over RPC."""

  _role = "rollout"

  def generate(
      self,
      prompts: Any,
      apply_chat_template: bool = False,
      mode: Any = None,
      micro_batch_size: Optional[int] = None,
      trace_tags: Optional[Mapping[str, Any]] = None,
      max_generation_steps: Optional[int] = None,
  ) -> Any:
    return self._actor.submit(
        "generate",
        prompts,
        apply_chat_template,
        mode,
        micro_batch_size,
        trace_tags,
        max_generation_steps,
    )


class RemoteInferenceWorker(_RemoteWorker):
  """Reference-scoring handle served over RPC."""

  _role = "inference"

  def compute_logps(
      self, request: datatypes.LogprobsRequest
  ) -> datatypes.LogprobsResponse:
    """Forwards the scoring request unchanged; failures come back in band."""
    return self._actor.submit("compute_logps", request)


class RemoteWeightSync:
  """Weight-sync handle served over RPC (an action, not a managed Worker)."""

  def __init__(self, actor_handle: remote_execution.ActorHandle):
    self._actor = actor_handle

  @classmethod
  def from_address(cls, target_address: str) -> "RemoteWeightSync":
    return cls(remote_execution.ActorHandle.from_address(target_address))

  def sync(self) -> None:
    self._actor.submit("sync")
