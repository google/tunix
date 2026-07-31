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

"""A rollout worker that answers one trajectory per request.

Two rollout shapes exist in this tree. The cluster's generation takes a whole
prompt batch and returns one bundled output, which is convenient in a single
process and useless for spreading work: the caller waits on the slowest prompt
and extra workers idle. The pooled data plane instead speaks one request per
trajectory, which is what can be balanced, retried, and reported on
individually.

This worker bridges them. It accepts per-trajectory requests, drives whatever
batched generation engine it was given, and returns a wire-safe response per
request with the identifiers the caller needs to reassemble a batch: the
request id it was asked under, and the policy version the tokens were drawn
from.

It is deliberately single-turn -- one prompt in, one completion out. Where
multi-turn episodes should run, worker-side or orchestrator-side, is an open
architectural question, and prejudging it here would bake an answer into the
wire contract.
"""

from __future__ import annotations

import traceback as traceback_lib
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.worker import rollout_worker

WorkerState = datatypes.WorkerState

RequestOrBatch = Union[
    datatypes.RolloutRequest, Sequence[datatypes.RolloutRequest]
]

FAILED_STATUS = "FAILED"
SUCCESS_STATUS = "SUCCEEDED"


class HostedRolloutWorker(rollout_worker.RolloutWorker):
  """Serves the per-trajectory rollout contract from a batched engine."""

  def __init__(
      self,
      engine: Any,
      *,
      worker_id: str = "rollout",
      policy_version: int = 0,
      generate_fn: Optional[Callable[..., Any]] = None,
      install_weights_fn: Optional[Callable[[Any], None]] = None,
  ):
    """Initializes the worker.

    Args:
      engine: Something exposing the cluster's batched
        `generate(prompts, ...) -> RolloutOutput`.
      worker_id: Identifier reported to the control plane.
      policy_version: Version stamped on responses, advanced by weight sync.
      generate_fn: Overrides how generation is invoked; defaults to
        `engine.generate`.
      install_weights_fn: Fetches and installs the weights a sync round points
        at, given the round's metadata. Defaults to the engine's own
        `update_weights` if it has one; otherwise the worker only adopts the
        version number, which is enough to exercise the protocol but means the
        weights themselves did not move. That is a real transport's job.
    """
    super().__init__(worker_id=worker_id)
    self._engine = engine
    self._policy_version = policy_version
    self._generate_fn = generate_fn or engine.generate
    self._install_weights_fn = install_weights_fn or getattr(
        engine, "update_weights", None
    )

  @property
  def policy_version(self) -> int:
    return self._policy_version

  # --- Data plane -----------------------------------------------------------

  async def generate(
      self,
      requests: RequestOrBatch,
      on_complete: Optional[
          Callable[[datatypes.RolloutResponse], None]
      ] = None,
  ) -> Union[datatypes.RolloutResponse, Sequence[datatypes.RolloutResponse]]:
    """Generates one completion per request.

    Args:
      requests: A single request or a batch of them.
      on_complete: Invoked with each response as it is produced.

    Returns:
      One response per request, in request order. A request that could not be
      generated comes back with its error attached rather than missing, so the
      caller can account for every request it made.
    """
    is_single = isinstance(requests, datatypes.RolloutRequest)
    batch = [requests] if is_single else list(requests)

    responses = []
    for request in batch:
      response = self._generate_one(request)
      responses.append(response)
      if on_complete is not None:
        on_complete(response)
    return responses[0] if is_single else responses

  def _generate_one(
      self, request: datatypes.RolloutRequest
  ) -> datatypes.RolloutResponse:
    """Runs one prompt through the engine and packages the result."""
    try:
      output = self._generate_fn([_prompt_text(request.prompt)])
      return self._to_response(request, output)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return datatypes.RolloutResponse(
          request_id=request.request_id,
          status=FAILED_STATUS,
          policy_version=self._policy_version,
          error=datatypes.ErrorInfo(
              error_type=type(e).__name__,
              message=str(e),
              traceback=traceback_lib.format_exc(),
          ),
      )

  def _to_response(
      self, request: datatypes.RolloutRequest, output: Any
  ) -> datatypes.RolloutResponse:
    """Converts one row of a batched output into a wire-safe response."""
    tokens = np.asarray(output.tokens[0], dtype=np.int32)
    logps = None
    if getattr(output, "logprobs", None):
      logps = np.asarray(output.logprobs[0], dtype=np.float32)
    prompt_tokens = np.asarray(
        output.left_padded_prompt_tokens[0], dtype=np.int32
    )
    text = output.text[0] if getattr(output, "text", None) else ""

    return datatypes.RolloutResponse(
        request_id=request.request_id,
        status=SUCCESS_STATUS,
        prompt_tokens=prompt_tokens,
        segments=[
            datatypes.TokenSegment(
                source="assistant",
                tokens=tokens,
                loss_mask=np.ones_like(tokens),
                logps=logps,
            )
        ],
        policy_version=self._policy_version,
        # Saves the caller a detokenize round trip it would otherwise do
        # against a tokenizer it may not share with this worker.
        metadata={"completion_text": text},
    )

  # --- Weight sync ----------------------------------------------------------

  def prepare_weight_sync(self, metadata: Any) -> datatypes.Response:
    """Fences generation ahead of a weight update.

    Generation here is synchronous, so nothing is in flight to drain by the
    time this is answered.
    """
    del metadata
    return datatypes.Response()

  def sync_weights(self, metadata: Any) -> int:
    """Installs the weights a round points at and reports the version reached.

    The version is adopted only after the install succeeds, so a worker never
    claims to be running weights it failed to load -- the round would then
    record it as synced while it generated from the old ones.

    Args:
      metadata: The round's request, carrying the target version and wherever
        the weights can be fetched from.

    Returns:
      The version now in effect.
    """
    if self._install_weights_fn is not None:
      self._install_weights_fn(metadata)
    version = getattr(metadata, "policy_version", None)
    self._policy_version = (
        int(version) if version is not None else self._policy_version + 1
    )
    return self._policy_version

  # --- Control plane --------------------------------------------------------

  def initialize(self) -> datatypes.Response:
    self.state = WorkerState.INITIALIZING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    del dummy_data  # The engine warms itself on the first generation.
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    self.state = WorkerState.READY
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    return datatypes.Response()

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id, roles=frozenset({"rollout"})
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state=self.state, policy_version=self._policy_version
    )


def _prompt_text(prompt: Any) -> str:
  """Extracts the text to generate from, however the request carried it."""
  if isinstance(prompt, str):
    return prompt
  if isinstance(prompt, dict):
    value = prompt.get("prompts", "")
    if isinstance(value, (list, tuple)) and value:
      return str(value[0])
    return str(value)
  return str(prompt)
