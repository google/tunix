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

"""Common data types and DTOs for the Tunix Orchestrator and Workers.

This module centralizes type aliases and dataclasses used for routing data and
commands between the orchestrator and distributed RL workers.

The dataclasses here intentionally start from a small field set. Additional
fields are added as concrete consumers require them, rather than up front.
"""

import dataclasses
from typing import Any

import numpy as np

from tunix.rl import common
from tunix.rl.agentic.agents import agent_types


@dataclasses.dataclass(kw_only=True)
class SamplingParams:
  """Engine-neutral sampling configuration for a generation request.

  Attributes:
    max_tokens: Maximum number of tokens to generate.
    temperature: Softmax temperature applied while sampling. Kept explicit so it
      can be carried through to any later log-probability scoring, ensuring the
      sampling distribution and the scoring distribution match.
  """

  max_tokens: int
  temperature: float = 1.0


@dataclasses.dataclass(kw_only=True)
class TrajectoryRequest:
  """Request to generate a trajectory from a given prompt.

  Attributes:
    request_id: Unique identifier for this request, echoed back on the
      corresponding result so callers can correlate and de-duplicate responses.
    prompt_id: Identifier of the source prompt (e.g. a dataset row). Provenance
      only; not unique across requests that reuse the same prompt.
    prompt_text: The prompt to generate from.
    sampling_params: Sampling configuration for the generation.
    max_turns: Maximum number of interaction turns for a multi-turn episode.
  """

  request_id: str
  prompt_id: str
  prompt_text: str
  sampling_params: SamplingParams
  max_turns: int = 10


@dataclasses.dataclass(kw_only=True)
class ErrorInfo:
  """Structured description of a failed request, carried in-band on a result.

  Attributes:
    error_type: Short classifier for the failure (e.g. an exception class name).
    message: Human-readable failure description.
    retryable: Whether re-issuing the request could plausibly succeed.
    traceback: Optional captured traceback, for diagnostics.
  """

  error_type: str
  message: str
  retryable: bool = False
  traceback: str = ""


@dataclasses.dataclass(kw_only=True)
class TokenSegment:
  """One contiguous span of the conversation token stream.

  Attributes:
    source: Origin of the span, e.g. "assistant" (model-emitted) or "env".
    tokens: int32 token ids for this span.
    loss_mask: int32 mask, 1 where the token is model-emitted (trainable).
    logprobs: float32 per-token log-probabilities under the sampling
      distribution, or None for spans the model did not emit (e.g. env tokens).
  """

  source: str
  tokens: np.ndarray
  loss_mask: np.ndarray
  logprobs: np.ndarray | None = None


@dataclasses.dataclass(kw_only=True)
class TrajectoryResult:
  """Serializable result of a generation request.

  This is the wire-facing counterpart to TrajectoryRequest (and to the
  worker-internal Trajectory below): it carries only primitives and numpy
  arrays, so it can cross a process boundary. A failed request is reported as a
  result with `error` set and a non-success `status`, never as a dropped
  response.

  Attributes:
    request_id: Echoes the originating request, for correlation/de-duplication.
    prompt_id: Echoes the source prompt id.
    status: Terminal status name (e.g. a rollout trajectory status, or
      "CANCELLED").
    prompt_tokens: int32 prompt token ids, unpadded, as tokenized by the worker.
    segments: Ordered conversation spans; concatenated they form the stream.
    env_reward: Scalar environment reward for the trajectory.
    policy_version: Weight version used to generate the trajectory.
    error: Failure details when the request did not succeed, else None.
  """

  request_id: str
  prompt_id: str
  status: str
  prompt_tokens: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(0, dtype=np.int32)
  )
  segments: list[TokenSegment] = dataclasses.field(default_factory=list)
  env_reward: float = 0.0
  policy_version: int = 0
  error: ErrorInfo | None = None


@dataclasses.dataclass(kw_only=True)
class StepReceipt:
  """Result of one fwd_bwd (gradient-accumulation) micro-step.

  Attributes:
    accum_id: Identifies the accumulation group this micro-step belongs to.
    micro_index: Position of this micro-batch within its accumulation group.
    applied: Whether an optimizer update was applied. False for fwd_bwd, which
      only accumulates gradients.
    micro_loss: The normalized loss for this micro-batch.
    denominator: The normalization denominator for this micro-batch (e.g. token
      or sequence count) so the caller can rescale and accumulate correctly.
  """

  accum_id: str
  micro_index: int
  applied: bool = False
  micro_loss: float = 0.0
  denominator: float = 0.0


@dataclasses.dataclass(kw_only=True)
class UpdateResult:
  """Result of applying accumulated gradients as one optimizer update.

  Attributes:
    step: The optimizer step count after this update.
    applied: Whether the update was applied (False for a no-op / duplicate).
    grad_norm: Global gradient norm for the update, when computed.
  """

  step: int
  applied: bool = True
  grad_norm: float | None = None


@dataclasses.dataclass(kw_only=True)
class TrainerPayload:
  """Generic trainer micro-batch payload (numpy wire).

  The base carries only what generic machinery must read to stay
  algorithm-agnostic: a loss mask (so gradient accumulation can derive a
  per-micro-batch denominator) and an optional packing layout. Algorithm-
  specific tensors live on subclasses (e.g. TrainExampleV1 for RL) and are
  reached by the trainer's gen_model_input_fn, not by the generic loop. Users
  subclass this to carry their own fields.

  Attributes:
    loss_mask: int32 [B, T], 1 where the position contributes to the loss.
    schema_version: DTO schema version (evolution is additive).
    segment_ids: Optional int32 [B, T] packing segment ids.
    segment_positions: Optional int32 [B, T] within-segment positions.
  """

  loss_mask: np.ndarray
  schema_version: int = 1
  segment_ids: np.ndarray | None = None
  segment_positions: np.ndarray | None = None


@dataclasses.dataclass(kw_only=True)
class TrainExampleV1(TrainerPayload):
  """RL training example: the numpy wire twin of the on-device TrainExample.

  The inherited `loss_mask` is the completion loss mask. Additional RL fields
  (ref/old per-token logps, per-row policy_version, temperature,
  sampler_is_weights) are added additively as consumers require them.

  Attributes:
    prompt_ids: int32 [B, P] left-padded prompt tokens.
    prompt_mask: int32 [B, P] prompt mask.
    completion_ids: int32 [B, C] right-padded completion tokens.
    advantages: float32 [B] or [B, C] advantages.
  """

  prompt_ids: np.ndarray
  prompt_mask: np.ndarray
  completion_ids: np.ndarray
  advantages: np.ndarray


@dataclasses.dataclass(kw_only=True)
class LogprobsRequest:
  """Request to score per-token log-probabilities under a frozen model.

  Attributes:
    request_id: Unique id for this request; echoed on the result.
    prompt_tokens: int32 [B, P], LEFT-padded.
    completion_tokens: int32 [B, C], RIGHT-padded; the result aligns to these
      completion columns.
    temperature: Softmax temperature to score under. Mandatory: it must match
      the temperature the tokens were sampled at, or the log-probs are biased.
    model_role: Which hosted model to score against (v1: "reference").
    micro_batch_size: Optional worker-internal micro-batch size; None scores the
      whole batch in one pass.
  """

  request_id: str
  prompt_tokens: np.ndarray
  completion_tokens: np.ndarray
  temperature: float
  model_role: str = "reference"
  micro_batch_size: int | None = None


@dataclasses.dataclass(kw_only=True)
class LogprobsResult:
  """Per-token log-probabilities for a LogprobsRequest.

  Attributes:
    request_id: Echoes the originating request.
    per_token_logps: float32 [B, C], aligned to the request's completion columns.
    model_version: Version of the scoring weights (constant for a frozen model).
  """

  request_id: str
  per_token_logps: np.ndarray
  model_version: int = 0


@dataclasses.dataclass(kw_only=True)
class ScoreRequest:
  """Request to score scalar rewards/values under a hosted model.

  Attributes:
    request_id: Unique id for this request; echoed on the result.
    prompt_tokens: int32 [B, P], LEFT-padded.
    completion_tokens: int32 [B, C], RIGHT-padded.
    model_role: Which hosted model to score against (e.g. "reward").
    micro_batch_size: Optional worker-internal micro-batch size; None scores the
      whole batch in one pass.
  """

  request_id: str
  prompt_tokens: np.ndarray
  completion_tokens: np.ndarray
  model_role: str = "reward"
  micro_batch_size: int | None = None


@dataclasses.dataclass(kw_only=True)
class ScoreResult:
  """Scalar scores for a ScoreRequest.

  Attributes:
    request_id: Echoes the originating request.
    scores: float32 [B], one scalar per row.
    model_version: Version of the scoring weights (constant for a frozen model).
  """

  request_id: str
  scores: np.ndarray
  model_version: int = 0


@dataclasses.dataclass(kw_only=True)
class ShapeConfig:
  """Shape hints a worker uses to synthesize its own warmup dummies for compile().

  Attributes:
    max_prompt_length: Maximum (left-padded) prompt length.
    max_response_tokens: Maximum completion length.
    micro_batch_sizes: Micro-batch sizes to warm up (compile) for.
  """

  max_prompt_length: int
  max_response_tokens: int
  micro_batch_sizes: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class HealthReport:
  """Liveness/status snapshot a worker reports to the orchestrator.

  Attributes:
    state: Coarse worker state, e.g. "READY", "COMPILING", "BUSY", "DRAINING",
      "SYNCING", "STOPPED".
    inflight: Number of in-flight units of work.
    queue_depth: Depth of the worker's internal work queue.
    policy_version: Installed weight/policy version.
    last_error: Most recent error string, if any.
    heartbeat_unix_s: Unix timestamp of this report.
  """

  state: str
  inflight: int = 0
  queue_depth: int = 0
  policy_version: int = 0
  last_error: str | None = None
  heartbeat_unix_s: float = 0.0


@dataclasses.dataclass(kw_only=True)
class WorkerInfo:
  """Static description of a worker, for registration and scheduling.

  Attributes:
    worker_id: Unique id of the worker.
    roles: Logical roles this worker serves (e.g. {"trainer", "inference"}); a
      fused worker joins every matching group.
    capabilities: Declared capability flags (e.g. "segment_attention").
    placement_group: Colocation group id, or None if not colocated.
    resources: Resource/topology description (e.g. process_count, mesh, hbm_gb).
  """

  worker_id: str
  roles: frozenset[str] = frozenset()
  capabilities: frozenset[str] = frozenset()
  placement_group: str | None = None
  resources: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class WeightSyncSpec:
  """Orchestrator -> trainer: what to stage for a weight sync.

  Attributes:
    version: Weight version being staged (assigned by the orchestrator).
    method: Transport method, e.g. "in_process", "checkpoint", "p2p".
    param_filter: Which params to stage, "full" or "lora".
  """

  version: int
  method: str = "in_process"
  param_filter: str = "full"


@dataclasses.dataclass(kw_only=True)
class WeightSyncMetadata:
  """Trainer -> replicas: how to fetch staged weights.

  Attributes:
    version: Weight version being fetched (echoes the spec).
    method: Transport method used to stage.
    locator: How to reach the weights (checkpoint path, (address, uuid), or a
      live in-process handle). This is the one wire carve-out that may hold a
      non-serializable handle, and only under the in-process transport.
  """

  version: int
  method: str
  locator: Any = None


@dataclasses.dataclass(kw_only=True)
class CompletedPage:
  """A page of completed results read from a rollout ticket's cursor channel.

  Attributes:
    results: The completed TrajectoryResults in this page (sequence-ordered).
    last_seq: Sequence number of the last result; the caller passes it back as
      `after_seq` to read the next page.
    done: Whether the ticket has produced all of its results.
  """

  results: list[TrajectoryResult] = dataclasses.field(default_factory=list)
  last_seq: int = 0
  done: bool = False


def validate_wire_safe(obj: object) -> None:
  """Raise TypeError if a wire payload holds a device array.

  Wire payloads cross process boundaries via cloudpickle and must carry numpy
  arrays, not device arrays (e.g. ``jax.Array``): a sharded device array cannot
  be reconstructed in a process that does not share its source mesh. This walks
  nested dataclasses, lists/tuples, and dict values, raising on the first
  array-like value that is not a ``numpy.ndarray``.
  """
  stack = [obj]
  while stack:
    value = stack.pop()
    if isinstance(value, np.ndarray):
      continue
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
      stack.extend(getattr(value, f.name) for f in dataclasses.fields(value))
    elif isinstance(value, dict):
      stack.extend(value.values())
    elif isinstance(value, (list, tuple)):
      stack.extend(value)
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
      raise TypeError(
          "wire payload contains a non-numpy array of type "
          f"{type(value).__module__}.{type(value).__qualname__}; convert device"
          " arrays (e.g. jax.Array) to numpy before serialization"
      )


# Worker-internal episode representation produced during rollout. This is the
# in-process type used inside a rollout worker; it is not a serialization format
# and is not meant to cross a process boundary. A dedicated serializable result
# type will be introduced separately when the wire path needs one.
Trajectory = agent_types.Trajectory


# On-device training example consumed by the trainer. This holds device-resident,
# potentially sharded arrays, so it cannot be serialized directly as a
# cross-process payload; a host/numpy-backed wire equivalent will be introduced
# separately when needed.
TrainExample = common.TrainExample
