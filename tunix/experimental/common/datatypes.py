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

This module centralizes type aliases and dataclasses used for:
1) Routing data and commands between Orchestrator and workers.
2) Defining common data structures used by Orchestrator and workers.
"""

import dataclasses
import enum
import os
import time
from typing import Any, Dict
import uuid
import flax
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.common import datatypes as common_datatypes
from tunix.rl.agentic.agents import agent_types

##### Worker-internal datatypes #####

# Worker-internal episode representation produced during rollout.
Trajectory = agent_types.Trajectory
Step = agent_types.Step
TrajectoryStatus = agent_types.TrajectoryStatus
TrajectoryItem = agent_types.TrajectoryItem
format_traj_id = agent_types.format_traj_id
Role = common_datatypes.Role

# Marks a router-replay slot the trainer must not replay, so the model falls
# back to its own gate there. Matches what MaxText's replay path expects.
UNSET_ROUTED_EXPERT = -1


def _pack_uint10(values: np.ndarray) -> np.ndarray:
  """Packs `[-1, 1022]` int16 values into 10-bit unsigned integers (5B / 4 vals)."""
  if values.size == 0:
    return np.zeros((0,), dtype=np.uint8)
  min_v = int(values.min())
  max_v = int(values.max())
  if min_v < UNSET_ROUTED_EXPERT or max_v > 1022:
    raise ValueError(
        "uint10 CompactRoutedExperts requires expert ids in [-1, 1022]; got"
        f" range [{min_v}, {max_v}]."
    )
  shifted = (values.astype(np.uint16, copy=False) + np.uint16(1)).ravel()
  rem = (-shifted.size) % 4
  if rem:
    shifted = np.pad(shifted, (0, rem), mode="constant", constant_values=0)
  quads = shifted.reshape(-1, 4)
  v0 = quads[:, 0]
  v1 = quads[:, 1]
  v2 = quads[:, 2]
  v3 = quads[:, 3]
  out = np.empty((quads.shape[0], 5), dtype=np.uint8)
  out[:, 0] = v0 & 0xFF
  out[:, 1] = ((v0 >> 8) & 0x03) | ((v1 & 0x3F) << 2)
  out[:, 2] = ((v1 >> 6) & 0x0F) | ((v2 & 0x0F) << 4)
  out[:, 3] = ((v2 >> 4) & 0x3F) | ((v3 & 0x03) << 6)
  out[:, 4] = (v3 >> 2) & 0xFF
  return out.ravel()


def _unpack_uint10(
    packed: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
  """Unpacks 10-bit packed bytes back into a `shape` int16 routing array."""
  total = shape[0] * shape[1] * shape[2]
  if total == 0:
    return np.zeros(shape, dtype=np.int16)
  groups = packed.reshape(-1, 5)
  b0 = groups[:, 0].astype(np.int16)
  b1 = groups[:, 1].astype(np.int16)
  b2 = groups[:, 2].astype(np.int16)
  b3 = groups[:, 3].astype(np.int16)
  b4 = groups[:, 4].astype(np.int16)
  quads = np.empty((groups.shape[0], 4), dtype=np.int16)
  quads[:, 0] = (b0 | ((b1 & 0x03) << 8)) - 1
  quads[:, 1] = ((b1 >> 2) | ((b2 & 0x0F) << 6)) - 1
  quads[:, 2] = ((b2 >> 4) | ((b3 & 0x3F) << 4)) - 1
  quads[:, 3] = ((b3 >> 6) | (b4 << 2)) - 1
  return quads.ravel()[:total].reshape(shape)


@dataclasses.dataclass(frozen=True, init=False)
class CompactRoutedExperts:
  """A batch's router-replay routing, stored only at its real-token positions.

  A batched `[B, T, num_layers, top_k]` int16 routing tensor is mostly padding
  for long-context microbatches, and even for densely sequence-packed batches
  each expert ID (-1 or 0..E-1) uses 16 bits while E <= 512 needs only 10 bits
  and unrouted prompt tokens carry all `-1`s. This keeps only the routing rows
  at `valid` positions (optionally stripping unrouted `-1` rows and bit-packing
  into `uint10` or `uint8` on the wire) and rebuilds the dense `[B, T,
  num_layers, top_k]` int16 tensor on the receiving side with `materialize()`.

  Attributes:
    values: `[N, num_layers, top_k]` int16 routing of the valid positions, in
      row-major (`valid.nonzero()`) order.
    valid: `[B, T]` bool mask of the positions `values` covers; `N ==
      valid.sum()`. Every other position materializes as `UNSET_ROUTED_EXPERT`.
    packed_values: Optional 1D `uint8` bit-packed representation when
      `encoding` is `"uint10"` or `"uint8"`.
    routed_shape: `(num_layers, top_k)` trailing dimensions.
    encoding: Wire encoding (`"int16"`, `"uint10"`, or `"uint8"`).
  """

  _values: np.ndarray | None
  valid: np.ndarray
  packed_values: np.ndarray | None = None
  routed_shape: tuple[int, int] = (0, 0)
  encoding: str = "int16"

  def __init__(
      self,
      *,
      values: np.ndarray | None = None,
      valid: np.ndarray,
      packed_values: np.ndarray | None = None,
      routed_shape: tuple[int, int] | None = None,
      encoding: str = "int16",
  ):
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_:
      raise TypeError(
          "CompactRoutedExperts.valid must be a bool numpy array; got"
          f" {type(valid).__name__} of dtype"
          f" {getattr(valid, 'dtype', None)}."
      )
    if valid.ndim != 2:
      raise ValueError(
          f"CompactRoutedExperts.valid must be [B, T]; got {valid.shape}."
      )
    num_valid = int(np.count_nonzero(valid))
    if encoding == "int16":
      if not isinstance(values, np.ndarray) or values.dtype != np.int16:
        raise TypeError(
            "CompactRoutedExperts.values must be an int16 numpy array; got"
            f" {type(values).__name__} of dtype"
            f" {getattr(values, 'dtype', None)}."
        )
      if values.ndim != 3:
        raise ValueError(
            "CompactRoutedExperts.values must be [N, num_layers, top_k]; got"
            f" shape {values.shape}."
        )
      if values.shape[0] != num_valid:
        raise ValueError(
            f"CompactRoutedExperts has {values.shape[0]} routing rows but"
            f" valid marks {num_valid} positions."
        )
      inferred_shape = (int(values.shape[1]), int(values.shape[2]))
      object.__setattr__(self, "_values", values)
      object.__setattr__(self, "packed_values", None)
      object.__setattr__(self, "routed_shape", routed_shape or inferred_shape)
    elif encoding in ("uint10", "uint8"):
      if (
          not isinstance(packed_values, np.ndarray)
          or packed_values.dtype != np.uint8
          or packed_values.ndim != 1
      ):
        raise TypeError(
            "CompactRoutedExperts.packed_values must be a 1D uint8 numpy array"
            f" when encoding={encoding!r}; got {type(packed_values).__name__}."
        )
      if routed_shape is None or len(routed_shape) != 2:
        raise ValueError(
            "CompactRoutedExperts requires routed_shape=(num_layers, top_k)"
            f" when encoding={encoding!r}; got {routed_shape!r}."
        )
      total_elems = num_valid * int(routed_shape[0]) * int(routed_shape[1])
      expected_bytes = (
          ((total_elems + 3) // 4) * 5 if encoding == "uint10" else total_elems
      )
      if packed_values.size != expected_bytes:
        raise ValueError(
            f"CompactRoutedExperts packed_values has {packed_values.size}"
            f" bytes but expected {expected_bytes} for {num_valid} valid"
            f" rows of shape {routed_shape} ({encoding})."
        )
      object.__setattr__(self, "_values", None)
      object.__setattr__(self, "packed_values", packed_values)
      object.__setattr__(
          self, "routed_shape", (int(routed_shape[0]), int(routed_shape[1]))
      )
    else:
      raise ValueError(
          f"Unsupported CompactRoutedExperts encoding: {encoding!r}."
      )
    object.__setattr__(self, "valid", valid)
    object.__setattr__(self, "encoding", encoding)

  @classmethod
  def pack(
      cls,
      values: np.ndarray,
      valid: np.ndarray,
      *,
      mode: str = "uint10",
      strip_unrouted: bool = True,
  ) -> "CompactRoutedExperts":
    """Packs `[N, num_layers, top_k]` routing with optional unrouted stripping and bit-packing."""
    # Validate inputs using standard int16 constructor rules first.
    base = cls(values=values, valid=valid)
    vals = base.values
    val_mask = base.valid
    routed_shape = base.routed_shape
    if strip_unrouted and vals.shape[0] > 0:
      routed_rows = np.any(vals != UNSET_ROUTED_EXPERT, axis=(1, 2))
      if not np.all(routed_rows):
        new_valid = np.zeros_like(val_mask)
        new_valid[val_mask] = routed_rows
        val_mask = new_valid
        vals = vals[routed_rows]
    norm_mode = mode.strip().lower()
    if norm_mode == "auto":
      max_v = int(vals.max()) if vals.size > 0 else -1
      if max_v <= 254:
        norm_mode = "uint8"
      elif max_v <= 1022:
        norm_mode = "uint10"
      else:
        norm_mode = "int16"
    if norm_mode == "int16":
      return cls(values=vals, valid=val_mask)
    if norm_mode == "uint8":
      if vals.size > 0:
        min_v = int(vals.min())
        max_v = int(vals.max())
        if min_v < UNSET_ROUTED_EXPERT or max_v > 254:
          raise ValueError(
              "uint8 CompactRoutedExperts requires expert ids in [-1, 254];"
              f" got range [{min_v}, {max_v}]."
          )
      packed = (vals.astype(np.int16, copy=False) + 1).astype(np.uint8).ravel()
      return cls(
          valid=val_mask,
          packed_values=packed,
          routed_shape=routed_shape,
          encoding="uint8",
      )
    if norm_mode == "uint10":
      packed = _pack_uint10(vals)
      return cls(
          valid=val_mask,
          packed_values=packed,
          routed_shape=routed_shape,
          encoding="uint10",
      )
    raise ValueError(f"Unknown CompactRoutedExperts pack mode: {mode!r}.")

  @classmethod
  def from_values(
      cls, values: np.ndarray, valid: np.ndarray
  ) -> "CompactRoutedExperts":
    """Constructs `CompactRoutedExperts` honoring `TUNIX_COMPACT_ROUTED_EXPERTS_MODE`."""
    env_mode = os.environ.get("TUNIX_COMPACT_ROUTED_EXPERTS_MODE", "").strip().lower()
    if not env_mode or env_mode in ("0", "false", "off", "int16"):
      return cls(values=values, valid=valid)
    return cls.pack(values, valid, mode=env_mode, strip_unrouted=True)

  @property
  def values(self) -> np.ndarray:
    """Returns the `[N, num_layers, top_k]` int16 routing values for `valid`."""
    if self._values is not None:
      return self._values
    num_valid = int(np.count_nonzero(self.valid))
    num_layers, top_k = self.routed_shape
    target_shape = (num_valid, num_layers, top_k)
    assert self.packed_values is not None
    if self.encoding == "uint8":
      if self.packed_values.size == 0:
        return np.zeros(target_shape, dtype=np.int16)
      return (self.packed_values.astype(np.int16) - 1).reshape(target_shape)
    if self.encoding == "uint10":
      return _unpack_uint10(self.packed_values, target_shape)
    raise ValueError(f"Unsupported encoding: {self.encoding!r}")

  @property
  def shape(self) -> tuple[int, int, int, int]:
    """Shape of the dense `[B, T, num_layers, top_k]` tensor it represents."""
    b, t = self.valid.shape
    num_layers, top_k = self.routed_shape
    return (b, t, num_layers, top_k)

  @property
  def dtype(self) -> np.dtype:
    return np.dtype(np.int16)

  @property
  def nbytes(self) -> int:
    payload_bytes = (
        self.packed_values.nbytes
        if self.packed_values is not None
        else (self._values.nbytes if self._values is not None else 0)
    )
    return payload_bytes + self.valid.nbytes

  def materialize(self) -> np.ndarray:
    """Returns the dense `[B, T, num_layers, top_k]` int16 routing tensor."""
    dense = np.full(self.shape, UNSET_ROUTED_EXPERT, dtype=np.int16)
    dense[self.valid] = self.values
    return dense


# Re-export assistant_text from canonical agent_types module.
assistant_text = agent_types.assistant_text


##### Common DTOs (Data Transfer Objects) #####


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


def _generate_request_id() -> str:
  return f"req_{uuid.uuid4().hex[:12]}"


@dataclasses.dataclass(kw_only=True)
class Request:
  """Standard base for generic RPC requests.

  Attributes:
    request_id: Unique identifier for this request, echoed back on the
      corresponding response so callers can correlate responses.
    metadata: Optional free-form data attached to the request.
  """

  request_id: str = dataclasses.field(default_factory=_generate_request_id)
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class Response:
  """Standard response for generic RPC requests.

  Attributes:
    request_id: Echoes the originating request_id for correlation.
    error: Structured failure details when the operation failed, else None.
    metadata: Optional free-form data attached to the response.
  """

  request_id: str = ""
  error: ErrorInfo | None = None
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class WorkerState(str, enum.Enum):
  """Worker lifecycle states.

  Attributes:
    PENDING: Worker is created but not yet initialized.
    INITIALIZING: Worker is currently allocating resources and running setup.
    COMPILING: Worker is compiling models or graphs for execution.
    READY: Worker is fully initialized and ready to accept requests.
    SYNCING: Worker is synchronizing model weights or policies.
    DRAINING: Worker is gracefully shutting down and finishing pending requests.
    STOPPED: Worker is stopped and no longer accepting requests.
    ERROR: Worker encountered an unrecoverable error.
  """

  PENDING = "PENDING"
  INITIALIZING = "INITIALIZING"
  COMPILING = "COMPILING"
  READY = "READY"
  SYNCING = "SYNCING"
  DRAINING = "DRAINING"
  STOPPED = "STOPPED"
  ERROR = "ERROR"

  def can_transition_to(self, new_state: "WorkerState") -> bool:
    """Checks if the transition to the new state is valid."""
    return new_state in _ALLOWED_TRANSITIONS.get(self, set())


_ALLOWED_TRANSITIONS: dict[WorkerState, set[WorkerState]] = {
    WorkerState.PENDING: {
        WorkerState.INITIALIZING,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.INITIALIZING: {
        WorkerState.READY,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.COMPILING: {
        WorkerState.READY,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.READY: {
        WorkerState.COMPILING,
        WorkerState.SYNCING,
        WorkerState.DRAINING,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.SYNCING: {
        WorkerState.READY,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.DRAINING: {WorkerState.STOPPED, WorkerState.ERROR},
    WorkerState.STOPPED: set(),
    WorkerState.ERROR: {WorkerState.STOPPED},
}


@dataclasses.dataclass(kw_only=True)
class HealthReport:
  """A snapshot of a worker's health and readiness state.

  Attributes:
    state: The current lifecycle state (e.g., WorkerState.READY).
    inflight: Number of active requests currently being processed.
    queue_depth: Number of pending requests queued by the worker.
    policy_version: The version of the weights currently loaded.
    last_error: A string summarizing the most recent error, if any.
    heartbeat_unix_s: The unix timestamp when this report was generated.
  """

  state: WorkerState
  inflight: int = 0
  queue_depth: int = 0
  policy_version: int = 0
  last_error: str | None = None
  heartbeat_unix_s: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(kw_only=True)
class WorkerInfo:
  """Static metadata describing a worker's identity and capabilities.

  Attributes:
    worker_id: The unique identifier for this worker.
    roles: The orchestrator roles this worker can serve (e.g., "trainer",
      "rollout").
    resources: Unstructured dictionary of hardware or configuration details
      (e.g., tokenizer_hash, fsdp_size) used during startup validation.
  """

  worker_id: str
  roles: frozenset[str] = frozenset()
  resources: dict[str, Any] = dataclasses.field(default_factory=dict)


##### Rollout DTOs #####


class NoHealthyRolloutWorkersError(RuntimeError, ValueError):
  """Raised when no active rollout workers remain available to serve or sync."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class RolloutFaultToleranceConfig:
  """Configuration for rollout worker fault tolerance (V1).

  Attributes:
    enabled: Whether rollout worker fault tolerance (eviction, retry, and
      dynamic rejoin) is enabled.
    evict_on_failure: Whether `PoolExecutionSession` automatically evicts a
      rollout worker from the pool when a dispatch or poll RPC fails.
    retry_on_worker_failure: Whether `PoolExecutionSession` automatically
      re-queues in-flight rollout tasks from a failed worker onto healthy
      workers.
    max_task_retries: Maximum number of retry attempts per rollout request_id
      before synthesizing a terminal FAILED placeholder trajectory.
    max_in_flight_per_worker: Optional default cap on concurrent in-flight
      rollouts dispatched to any single rollout worker.
    task_timeout_s: Optional per-task execution timeout in seconds once a
      rollout request is dispatched to a worker.
    max_zero_worker_wait_s: Maximum duration in seconds to wait when zero active
      rollout workers remain before raising `NoHealthyRolloutWorkersError`.
    retry_weight_sync_on_eviction: Whether `WeightSyncCoordinator.sync()`
      automatically evicts failed destination workers and retries the round once
      on surviving workers.
    recover_unknown_transfer_state: Whether `WeightSyncCoordinator.sync()`
      attempts recovery from `RoundState.UNKNOWN_TRANSFER_STATE` by evicting
      uncommitted workers, explicitly releasing source staging via
      `release_after_unknown_transfer()`, resetting poison, and retrying once.
      Defaults to False.
  """

  enabled: bool = True
  evict_on_failure: bool = True
  retry_on_worker_failure: bool = True
  max_task_retries: int = 3
  max_in_flight_per_worker: int | None = None
  task_timeout_s: float | None = None
  max_zero_worker_wait_s: float = 600.0
  retry_weight_sync_on_eviction: bool = True
  recover_unknown_transfer_state: bool = False

  def __post_init__(self):
    if self.max_task_retries < 0:
      raise ValueError("max_task_retries must be non-negative")
    if (
        self.max_in_flight_per_worker is not None
        and self.max_in_flight_per_worker <= 0
    ):
      raise ValueError("max_in_flight_per_worker must be positive")
    if self.task_timeout_s is not None and self.task_timeout_s <= 0:
      raise ValueError("task_timeout_s must be positive")
    if self.max_zero_worker_wait_s < 0:
      raise ValueError("max_zero_worker_wait_s must be non-negative")

  def with_overrides(
      self,
      *,
      max_in_flight_per_worker: int | None = None,
      task_timeout_s: float | None = None,
      max_zero_worker_wait_s: float | None = None,
  ) -> "RolloutFaultToleranceConfig":
    """Returns a copy where each non-None argument replaces the stored value.

    Legacy scalar constructor arguments (e.g. the engine/orchestrator
    `max_concurrent_rollouts_per_worker`) take precedence over this consolidated
    config; this is the single place that precedence is applied. The copy is
    validated by `__post_init__` exactly like a directly constructed config.
    """
    overrides = {
        "max_in_flight_per_worker": max_in_flight_per_worker,
        "task_timeout_s": task_timeout_s,
        "max_zero_worker_wait_s": max_zero_worker_wait_s,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return dataclasses.replace(self, **overrides) if overrides else self


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenerationArgs:
  """Typed per-turn generation arguments used by the orchestrator generate API.

  This holds per-turn sampling and generation configuration (as opposed to
  episode-level configuration like `max_turns` or `max_response_length` on
  `RolloutRequest`).

  Attributes:
    max_generation_steps: Maximum number of tokens to generate in a single turn.
    temperature: Sampling temperature for generation.
    top_p: Nucleus sampling probability threshold.
    top_k: Top-k sampling cutoff.
    seed: Random seed for reproducible generation.
    return_logprobs: Whether to return token log probabilities.
  """

  max_generation_steps: int | None = None
  temperature: float | None = None
  top_p: float | None = None
  top_k: int | None = None
  seed: int | None = None
  return_logprobs: bool | None = None
  return_routed_experts: bool | None = None

  def as_kwargs(self) -> dict[str, Any]:
    return {
        field.name: getattr(self, field.name)
        for field in dataclasses.fields(self)
        if getattr(self, field.name) is not None
    }


@dataclasses.dataclass(kw_only=True)
class RolloutRequest(Request):
  """Request to generate a rollout from a given prompt.

  Attributes:
    prompt: The prompt to generate from (e.g. formatted string, token array, or
      chat dictionary).
    prompt_id: Unique identifier for this prompt within a task or dataset.
    group_index: Optional index within a group for group-based algorithms (e.g.,
      GRPO). Defaults to 0 for ungrouped.
    generation_kwargs: Additional per-turn keyword arguments for generation
      (e.g. sampling parameters like max_generation_steps and temperature).
    max_turns: Maximum number of conversation turns for environment interaction.
    max_response_length: Optional cumulative response token budget across all
      turns of the rollout episode.
    target_policy_version: Policy model version identifier to use for rollout
      generation.
    exact_token_continuity: Whether to preserve exact token IDs across
      multi-turn rollout steps (Token-In-Token-Out). Defaults to True.
    priority: Sampler scheduling priority for every generation call of this
      rollout; lower values are served first. With
      `rollout_priority_scheduling` the RL program sets it to the prompt batch
      index, so under `max_staleness > 0` the oldest in-flight batch is served
      ahead of batches dispatched early; otherwise it stays 0.
  """

  prompt: Any = ""
  prompt_id: str = ""
  group_index: int = 0
  generation_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  max_turns: int = 10
  max_response_length: int | None = None
  target_policy_version: int = 0
  exact_token_continuity: bool = True
  priority: int = 0

  @property
  def traj_id(self) -> str:
    """Standardized trajectory identifier: traj_{prompt_id}_g{group_index}."""
    return format_traj_id(self.prompt_id, self.group_index)


@dataclasses.dataclass(kw_only=True)
class TokenSegment:
  """One contiguous span of the conversation token stream representing a single turn.

  Each segment corresponds to a single turn's response from either the assistant
  or the environment.

  Attributes:
    source: Origin of the span, e.g. "assistant" (model-emitted) or "env".
    tokens: Array of token ids for this span.
    loss_mask: Array of ints, 1 where the token is model-emitted (trainable).
    logps: Array of per-token log-probabilities under the sampling distribution,
      or None for spans the model did not emit (e.g. env tokens).
    routed_experts: `[len(tokens), num_layers, top_k]` MoE expert ids this span
      was routed through, so training can replay the routing the rollout
      actually used. None for dense models, spans the model did not emit, or
      when the sampler was not asked to capture routing.
  """

  source: str
  tokens: np.ndarray
  loss_mask: np.ndarray
  logps: np.ndarray | None = None
  routed_experts: np.ndarray | None = None

  def __post_init__(self):
    if self.loss_mask.shape != self.tokens.shape:
      raise ValueError(
          f"loss_mask shape {self.loss_mask.shape} != tokens shape"
          f" {self.tokens.shape}"
      )
    if self.logps is not None and self.logps.shape != self.tokens.shape:
      raise ValueError(
          f"logps shape {self.logps.shape} != tokens shape {self.tokens.shape}"
      )
    if self.routed_experts is not None:
      # The trailing axes are [num_layers, top_k] and are model-dependent, so
      # only the rank and the leading (per-token) axis are checked.
      if self.routed_experts.ndim != 3:
        raise ValueError(
            "routed_experts must be [length, num_layers, top_k]; got shape"
            f" {self.routed_experts.shape}"
        )
      if self.routed_experts.shape[0] != self.tokens.shape[0]:
        raise ValueError(
            f"routed_experts shape {self.routed_experts.shape} does not cover"
            f" tokens shape {self.tokens.shape} along axis 0"
        )


@dataclasses.dataclass(kw_only=True)
class RolloutResponse(Response):
  """Serializable result of a rollout generation request carrying a TrajectoryItem payload.

  Attributes:
    status: Terminal status name (e.g. "COMPLETED", "ERROR", "TIMEOUT",
      "CANCELLED").
    payload: TrajectoryItem carrying episode trajectory, token arrays, masks,
      and metadata.
  """

  status: str = "COMPLETED"
  payload: TrajectoryItem | None = None


##### Weight Sync DTOs #####


@dataclasses.dataclass(kw_only=True)
class WeightSyncRequest(Request):
  """Configuration and routing metadata for synchronizing policy model weights.

  Attributes:
    controller_id: Optional identifier for transport controllers (e.g., TPU
      Raiden).
    policy_version: Target policy version identifier of the weights to sync.
    weights: Optional source weights payload for non-Raiden / fallback sync.
    source_metadata: Optional transport/layout metadata describing source
      weights.
    extra_config: Optional backend-specific configuration parameters.
  """

  controller_id: str = ""
  policy_version: int = 0
  weights: Any = None
  source_metadata: Any = None
  extra_config: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class WeightSyncMetadata:
  """Metadata passed by Orchestrator during Phase 3 decentralized weight synchronization.

  Used to coordinate peer-to-peer (P2P) or broadcast weight transfers between
  TPU Trainer slices and RolloutWorker inference pods without routing large
  model weights through the central orchestrator.

  Attributes:
    worker_ips: List of target rollout worker IP addresses participating in
      sync.
    data_ports: Corresponding list of high-speed data transfer ports for each
      worker.
    ctrl_ports: List of control plane ports for coordination.
    mesh: Optional device mesh object for sharded tensor transfer.
    layout: Optional tensor memory sharding layout dimensions.
    new_policy_version: The updated policy weight version index being
      synchronized.
    transfer_mode: Transport mechanism option for decentralized weight sync: -
      "p2p" (default): Direct peer-to-peer transfer (e.g., direct RPC, Ray, or
      Raiden KV/weight streaming between source Trainer pods and target
      RolloutWorker pods using P2P DMA). - "broadcast" (Not implemented):
      Collective/tree broadcast (e.g., distributing weights across a pod mesh
      via collective communication like NCCL or Gloo over DMA links). - "fs"
      (Not implemented): Shared filesystem checkpoint transfer (where trainers
      save checkpoint weights to a shared filesystem like CNS/POSIX and rollout
      workers reload from disk).
    source_endpoints: Host/port endpoints of the source trainer pods serving the
      weights.
    sharding_topology: Optional device mesh sharding layout (e.g., {"mesh": [2,
      2]}).
  """

  worker_ips: list[str] = dataclasses.field(default_factory=list)
  data_ports: list[str] = dataclasses.field(default_factory=list)
  ctrl_ports: list[str] = dataclasses.field(default_factory=list)
  mesh: Any = None
  layout: list[int] = dataclasses.field(default_factory=list)
  new_policy_version: int = 0
  transfer_mode: str = "p2p"
  source_endpoints: list[str] = dataclasses.field(default_factory=list)
  sharding_topology: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    if self.transfer_mode in ("broadcast", "fs"):
      raise NotImplementedError(
          f"transfer_mode='{self.transfer_mode}' is not implemented yet."
          " Currently only 'p2p' is supported."
      )
    elif self.transfer_mode != "p2p":
      raise ValueError(f"Unknown transfer_mode: '{self.transfer_mode}'")


##### Training DTOs #####


@flax.struct.dataclass(frozen=True, kw_only=True)
class TrainerPayload:
  """Base abstract class for generic trainer payloads."""


@flax.struct.dataclass(frozen=True, kw_only=True)
class SFTTrainerPayload(TrainerPayload):
  """Supervised Fine-Tuning (SFT) trainer payload.

  Attributes:
    token_ids: [B, T] token IDs for a batched trainer payload. By default, each
      row is structured as left-padded prompt tokens concatenated with
      right-padded completion tokens.
    token_mask: [B, T] token mask to differentiate padding tokens from valid
      tokens.
    segment_ids: Optional [B, T] packing segment ids.
    segment_positions: Optional [B, T] position indices within each segment.
  """

  token_ids: ArrayLike
  token_mask: ArrayLike
  segment_ids: ArrayLike | None = None
  segment_positions: ArrayLike | None = None


# TODO(tunix-dev): Introduce PPOTrainerPayload to replace generic
# RLTrainerPayload when PPO specific fields are needed.
@flax.struct.dataclass(frozen=True, kw_only=True)
class RLTrainerPayload(TrainerPayload):
  """RL training payload.

  Attributes:
    advantages: [B] or [B, C] advantages.
    prompt_ids: Optional prompt token ids for GRPO-style losses. Unbatched
      payloads may carry 1D unpadded rows; batch assembly pads them to [B, P].
    prompt_mask: Optional [B, P] prompt mask.
    completion_ids: Optional completion token ids. Unbatched payloads may carry
      1D unpadded rows; batch assembly pads them to [B, C].
    completion_mask: Optional [B, C] completion/action mask.
    segment_ids: Optional [B, T] or [B, C] packing segment ids.
    segment_positions: Optional [B, T] or [B, C] position indices within each
      segment.
    ref_per_token_logps: Optional [B, C] reference model log-probabilities.
    old_per_token_logps: Optional [B, C] behavior policy log-probabilities.
    sampler_is_weights: Optional [B, C] importance sampling weights.
    routed_experts: Optional MoE expert ids captured during rollout. When set,
      a training engine that supports router replay forces these experts
      instead of re-running its own gate, so the training forward pass matches
      the routing the rollout actually used. `-1` marks a padded or unused
      slot. An unbatched payload carries a dense `[p + c, num_layers, top_k]`
      array; a batch assembler emits `CompactRoutedExperts` for the batched
      `[B, T, num_layers, top_k]` routing so the padding never crosses the RPC,
      and `TrainerWorker` materializes it before handing the payload to its
      trainer.
    returns: Optional [B, C] value baseline returns (for PPO / Critic).
    old_values: Optional [B, C] critic value estimates (for PPO / Critic).
    num_segments: Optional static upper bound on number of segments in packed
      rows.
    metadata: Extra payload metadata dictionary.
  """

  prompt_ids: ArrayLike
  prompt_mask: ArrayLike
  completion_ids: ArrayLike
  completion_mask: ArrayLike
  advantages: ArrayLike
  segment_ids: ArrayLike | None = None
  segment_positions: ArrayLike | None = None
  ref_per_token_logps: ArrayLike | None = None
  old_per_token_logps: ArrayLike | None = None
  sampler_is_weights: ArrayLike | None = None
  # The rollout engine's own per-token log-probabilities, and the collector's
  # verdict that a sequence was truncated by the response budget rather than
  # finished. Both are per-sequence quantities a sequence-gated GRPO loss
  # needs and that nothing else carries: `seq-mask-tis` and
  # `seq_logprob_error_threshold` compare the sampler against the trainer, and
  # `overlong_loss_masking` drops truncated sequences from the loss while
  # leaving them in the denominator. Without these two fields the loss silently
  # runs with those features disabled.
  rollout_per_token_logps: ArrayLike | None = None
  overlong: ArrayLike | None = None
  routed_experts: ArrayLike | CompactRoutedExperts | None = None
  returns: ArrayLike | None = None
  old_values: ArrayLike | None = None
  num_segments: int | None = flax.struct.field(default=None, pytree_node=False)
  metadata: dict[str, Any] = flax.struct.field(
      default_factory=dict, pytree_node=False
  )
  # TODO(tunix-dev): add ppo specific fields in PPORLTrainerPayload.


@dataclasses.dataclass(kw_only=True)
class TrainRequest(Request):
  """Request to execute training (forward/backward or eval) on a TrainerWorker.

  Attributes:
    payload: The TrainerPayload containing model inputs, masks, etc.
    target_policy_version: Version of the policy weights to train.
  """

  payload: TrainerPayload | None
  target_policy_version: int = 0


@dataclasses.dataclass(kw_only=True)
class LogprobsRequest(Request):
  """Request to score per-token log-probabilities under a frozen model.

  Attributes:
    prompt_tokens: [B, P] token ids, already LEFT-padded by the caller.
    completion_tokens: [B, C] token ids, already RIGHT-padded by the caller; the
      result aligns to these completion columns.
    temperature: Softmax temperature to score under. Mandatory: it must match
      the temperature the tokens were sampled at, or the log-probs are biased.
    model_role: Which hosted model to score against (v1: "reference").
    pad_id: Pad token id. Used by the trainer/actor scoring path; the reference
      path leaves it unset and relies on the worker's own id.
    eos_id: End-of-sequence token id, used by the trainer/actor scoring path.
    segment_ids: Optional packing segment ids (sequence packing); trainer path
      only.
    segment_positions: Optional packing local position indices (sequence
      packing); trainer path only.
    routed_experts: Optional `[B, T, num_layers, top_k]` MoE expert routing
      captured during rollout for router replay, dense or as the batch's
      `CompactRoutedExperts` (the scorer materializes it).
  """

  prompt_tokens: ArrayLike
  completion_tokens: ArrayLike
  temperature: float
  model_role: str = "reference"
  pad_id: int | None = None
  eos_id: int | None = None
  segment_ids: ArrayLike | None = None
  segment_positions: ArrayLike | None = None
  routed_experts: ArrayLike | CompactRoutedExperts | None = None


##### Inference DTOs #####


@dataclasses.dataclass(kw_only=True)
class LogprobsResponse(Response):
  """Per-token log-probabilities for a LogprobsRequest.

  Attributes:
    per_token_logps: [B, C], aligned to the request's completion columns.
    model_version: Version of the scoring weights (constant for a frozen model).
    error: Failure details when the request did not succeed, else None.
  """

  per_token_logps: np.ndarray
  model_version: int = 0


@dataclasses.dataclass(kw_only=True)
class ScoreRequest(Request):
  """Request to score scalar rewards/values under a hosted model.

  Attributes:
    prompt_tokens: [B, P] token ids, already LEFT-padded by the caller.
    completion_tokens: [B, C] token ids, already RIGHT-padded by the caller.
    model_role: Which hosted model to score against (e.g. "reward").
  """

  prompt_tokens: np.ndarray
  completion_tokens: np.ndarray
  model_role: str = "reward"


@dataclasses.dataclass(kw_only=True)
class ScoreResponse(Response):
  """Scalar scores for a ScoreRequest.

  Attributes:
    scores: [B], one scalar per row.
    model_version: Version of the scoring weights (constant for a frozen model).
    error: Failure details when the request did not succeed, else None.
  """

  scores: np.ndarray
  model_version: int = 0
