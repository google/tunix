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

"""OpenTelemetry metrics helpers and role instruments for dist_rl workers.

`WorkerMeter` is the role-agnostic part every worker uses the same way: the
optional `opentelemetry` import, the choice of `MeterProvider`, and stamping
`worker_id` on every data point. `TrainerWorkerOtelMetrics` and
`RolloutWorkerOtelMetrics` build on `WorkerMeter` to encapsulate each role's
instruments, boundary validators, and recording state so `trainer_worker.py`
and `rollout_worker.py` stay focused on worker lifecycle and RPC dispatch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
import contextlib
import math
import threading
import time
from typing import Any, TYPE_CHECKING
import weakref

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import lineage
from tunix.rl.agentic.agents import agent_types

if TYPE_CHECKING:
  from opentelemetry import metrics as otel_api_types  # pylint: disable=g-bad-import-order,g-import-not-at-top

try:
  from opentelemetry import metrics as otel_api  # pylint: disable=g-import-not-at-top
except ImportError:
  otel_api = None

# Metadata keys stamped by `BatchAssembler` onto `RLTrainerPayload.metadata`
# (and forwarded into `TrainRequest.metadata` by `DistributedRLEngine`) for
# `TrainerWorker` OpenTelemetry metrics.
VALID_TOKENS_METADATA_KEY = "valid_tokens"
CAPACITY_TOKENS_METADATA_KEY = "capacity_tokens"
MB_READY_TIME_S_METADATA_KEY = "mb_ready_time_s"

# Clock seams for deterministic unit tests without patching global `time` hooks.
now_s: Callable[[], float] = time.time
perf_counter_s: Callable[[], float] = time.perf_counter


class BoundInstrument:
  """A counter or histogram that stamps `worker_id` on every data point.

  Does nothing when `opentelemetry` is not installed, so role code records
  unconditionally instead of checking an enabled flag at every call site.
  """

  def __init__(self, emit: Callable[..., None] | None, worker_id: str):
    self._emit = emit
    self._worker_id = worker_id

  def record(self, value: int | float, **attributes: str) -> None:
    """Records `value` with `worker_id` plus the given attributes."""
    if self._emit is not None:
      self._emit(value, attributes={"worker_id": self._worker_id, **attributes})

  # Counters read better as `add`; the behavior is identical.
  add = record


class WorkerMeter:
  """Per-worker source of OpenTelemetry instruments, shared by all roles.

  Attributes:
    worker_id: Worker identifier stamped on every data point.
    provider: The resolved `MeterProvider`, or None when `opentelemetry` is not
      installed.
    meter: The underlying `Meter` for instruments that this class does not wrap
      (such as observable gauges), or None when `opentelemetry` is not
      installed.
  """

  def __init__(
      self,
      scope: str,
      worker_id: str,
      meter_provider: otel_api_types.MeterProvider | None = None,
  ):
    """Initializes the meter.

    Args:
      scope: Instrumentation scope name, for example `"tunix.trainer"`.
      worker_id: Worker identifier stamped on every data point.
      meter_provider: Optional `MeterProvider` override for testing; defaults to
        the global `MeterProvider` when None.
    """
    self.worker_id = worker_id
    self.provider: otel_api_types.MeterProvider | None = None
    self.meter: otel_api_types.Meter | None = None
    if otel_api is not None:
      self.provider = meter_provider or otel_api.get_meter_provider()
      self.meter = self.provider.get_meter(scope)

  @property
  def enabled(self) -> bool:
    """Whether instruments from this meter record anything."""
    return self.meter is not None

  def histogram(
      self, name: str, *, unit: str, description: str
  ) -> BoundInstrument:
    """Creates a histogram bound to this worker."""
    if self.meter is None:
      return BoundInstrument(None, self.worker_id)
    histogram = self.meter.create_histogram(
        name, unit=unit, description=description
    )
    return BoundInstrument(histogram.record, self.worker_id)

  def counter(
      self, name: str, *, unit: str, description: str
  ) -> BoundInstrument:
    """Creates a counter bound to this worker."""
    if self.meter is None:
      return BoundInstrument(None, self.worker_id)
    counter = self.meter.create_counter(
        name, unit=unit, description=description
    )
    return BoundInstrument(counter.add, self.worker_id)


def extract_payload_tokens(
    payload: datatypes.TrainerPayload,
) -> tuple[int, int] | None:
  """Extracts `(valid_tokens, padded_tokens)` stamped by `BatchAssembler`.

  Only `BatchAssembler`-stamped `RLTrainerPayload` batches carry precomputed
  token counts in `payload.metadata` so `TrainerWorker.fwd_bwd` runs in `O(1)`
  without scanning `[B, T]` masks; `SFTTrainerPayload` and unstamped legacy
  `RLTrainerPayload` batches return `None`.

  Args:
    payload: The trainer microbatch payload passed to `TrainerWorker.fwd_bwd`.

  Returns:
    `(valid_tokens, padded_tokens)` when both `VALID_TOKENS_METADATA_KEY` and
    `CAPACITY_TOKENS_METADATA_KEY` are present on `RLTrainerPayload.metadata`,
    or `None` when the payload is not an `RLTrainerPayload` or carries neither
    key.

  Raises:
    TypeError: If `payload.metadata` is not a `Mapping` or either token count is
      not an integer (`int` or `np.integer`, excluding `bool`).
    ValueError: If only one of the two keys is set, or if
      `not (0 <= valid_tokens <= capacity_tokens)`.
  """
  if not isinstance(payload, datatypes.RLTrainerPayload):
    return None
  if not isinstance(payload.metadata, Mapping):
    raise TypeError(
        "RLTrainerPayload.metadata must be a Mapping, got"
        f" {type(payload.metadata).__name__}."
    )
  raw_valid = payload.metadata.get(VALID_TOKENS_METADATA_KEY)
  raw_capacity = payload.metadata.get(CAPACITY_TOKENS_METADATA_KEY)
  if raw_valid is None and raw_capacity is None:
    return None
  if raw_valid is None or raw_capacity is None:
    raise ValueError(
        "payload.metadata must set both"
        f" {VALID_TOKENS_METADATA_KEY!r} and"
        f" {CAPACITY_TOKENS_METADATA_KEY!r} together, got"
        f" valid_tokens={raw_valid!r}, capacity_tokens={raw_capacity!r}."
    )
  if (
      isinstance(raw_valid, bool)
      or isinstance(raw_capacity, bool)
      or not isinstance(raw_valid, (int, np.integer))
      or not isinstance(raw_capacity, (int, np.integer))
  ):
    raise TypeError(
        f"{VALID_TOKENS_METADATA_KEY!r} and {CAPACITY_TOKENS_METADATA_KEY!r} in"
        " payload.metadata must be integers, got"
        f" valid_tokens={raw_valid!r}, capacity_tokens={raw_capacity!r}."
    )
  valid_tokens = int(raw_valid)
  capacity_tokens = int(raw_capacity)
  if not (0 <= valid_tokens <= capacity_tokens):
    raise ValueError(
        "Expected 0 <= valid_tokens <= capacity_tokens in payload.metadata,"
        f" got valid_tokens={valid_tokens},"
        f" capacity_tokens={capacity_tokens}."
    )
  return valid_tokens, capacity_tokens - valid_tokens


def extract_mb_ready_time_s(
    req_metadata: Mapping[str, Any] | None,
) -> float | None:
  """Validates and extracts `mb_ready_time_s` from `TrainRequest.metadata`.

  Args:
    req_metadata: Optional metadata mapping on `TrainRequest`.

  Returns:
    The validated non-negative finite `mb_ready_time_s` timestamp in seconds, or
    `None` if `req_metadata` is empty or does not contain
    `MB_READY_TIME_S_METADATA_KEY`.

  Raises:
    TypeError: If `mb_ready_time_s` is a boolean or not a real number (`int`,
      `float`, `np.integer`, or `np.floating`).
    ValueError: If `mb_ready_time_s` is `NaN`, infinite, or negative.
  """
  if not req_metadata:
    return None
  raw_ready_s = req_metadata.get(MB_READY_TIME_S_METADATA_KEY)
  if raw_ready_s is None:
    return None
  if isinstance(raw_ready_s, bool) or not isinstance(
      raw_ready_s, (int, float, np.integer, np.floating)
  ):
    raise TypeError(
        f"{MB_READY_TIME_S_METADATA_KEY!r} in request.metadata must be a real"
        f" number, got {raw_ready_s!r}."
    )
  mb_ready_s = float(raw_ready_s)
  if not math.isfinite(mb_ready_s) or mb_ready_s < 0.0:
    raise ValueError(
        f"{MB_READY_TIME_S_METADATA_KEY!r} in request.metadata must be a"
        f" non-negative finite number, got {raw_ready_s!r}."
    )
  return mb_ready_s


class TrainerWorkerOtelMetrics:
  """Encapsulates OpenTelemetry instruments and recording for TrainerWorker.

  Note on async JAX dispatch (TODO(b/579110728)): `fwd_bwd` records host-side
  dispatch duration so microbatch pipelining is not blocked by host
  synchronization, whereas `update` (`phase="optimizer_step"`, which includes
  `trainer.get_metrics()`), `prepare_weight_sync` (`direction="send"`), and
  `per_token_logps` synchronize the device on the host. Because
  `last_free_time_s` is anchored when each device-occupying RPC returns on the
  host, `tunix.trainer.starvation.duration{kind="trainer_idle"}` between back-to
  -back `fwd_bwd` microbatches measures host-side inter-RPC idle time and can
  overstate device idle time by up to the queued device execution duration.
  """

  def __init__(
      self,
      worker_id: str,
      meter_provider: otel_api_types.MeterProvider | None = None,
  ):
    self.last_free_time_s: float | None = None
    self._meter = WorkerMeter("tunix.trainer", worker_id, meter_provider)
    self._step_duration = self._meter.histogram(
        "tunix.trainer.step.duration",
        unit="s",
        description=(
            "Host-side execution or dispatch duration of trainer worker"
            " operations by phase."
        ),
    )
    self._tokens_processed = self._meter.counter(
        "tunix.trainer.tokens.processed",
        unit="{token}",
        description=(
            "Total valid and padded tokens processed by the trainer worker"
            " across BatchAssembler-stamped RLTrainerPayload microbatches."
        ),
    )
    self._weight_sync_duration = self._meter.histogram(
        "tunix.trainer.weight_sync.duration",
        unit="s",
        description=(
            "Duration of trainer weight sync staging and release (staging"
            " includes waiting on queued device work)."
        ),
    )
    self._starvation_duration = self._meter.histogram(
        "tunix.trainer.starvation.duration",
        unit="s",
        description=(
            "Host-side trainer idle time (now - last_free_time_s, anchored at"
            " host RPC completion) and data-fetch delay (max(0, now -"
            " max(last_free_time_s, mb_ready_time_s)), including post-assembly"
            " prep and RPC transfer across host wall clocks) before a"
            " microbatch."
        ),
    )

  def record_starvation(self, mb_ready_s: float | None) -> None:
    """Records `trainer_idle` and `data_fetch_delay` before a microbatch."""
    if not self._meter.enabled or self.last_free_time_s is None:
      return
    now_wall_s = now_s()
    idle_s = max(0.0, now_wall_s - self.last_free_time_s)
    self._starvation_duration.record(float(idle_s), kind="trainer_idle")
    if mb_ready_s is not None:
      fetch_delay_s = max(
          0.0, now_wall_s - max(self.last_free_time_s, mb_ready_s)
      )
      self._starvation_duration.record(
          float(fetch_delay_s), kind="data_fetch_delay"
      )

  @contextlib.contextmanager
  def track_rpc(
      self,
      *,
      phase: str | None = None,
      weight_sync_direction: str | None = None,
      token_counts: tuple[int, int] | None = None,
      arm: bool = False,
  ) -> Iterator[None]:
    """Times a worker RPC, updates `last_free_time_s`, and records metrics on success."""
    # Trigger: only `fwd_bwd` passes `arm=True`; `compile`,
    # `restore_checkpoint`, and weight-sync RPCs leave `arm=False`. Before the
    # first training microbatch arrives, startup `compile()` and initial weight
    # sync run followed by the first rollout generation wait. Keeping
    # `last_free_time_s` as `None` until the first `fwd_bwd` completes prevents
    # cold-start setup and initial rollout wait from polluting steady-state
    # `trainer_idle`, while refreshing the busy anchor on every device-occupying
    # RPC thereafter.
    timed = phase is not None or weight_sync_direction is not None
    t0 = perf_counter_s() if timed else 0.0
    try:
      yield
      elapsed_s = (perf_counter_s() - t0) if timed else 0.0
    finally:
      if arm or self.last_free_time_s is not None:
        self.last_free_time_s = now_s()
    if phase is not None:
      self._step_duration.record(float(elapsed_s), phase=phase)
    if weight_sync_direction is not None:
      self._weight_sync_duration.record(
          float(elapsed_s), direction=weight_sync_direction
      )
    if token_counts is not None:
      valid_tokens, padded_tokens = token_counts
      self._tokens_processed.add(valid_tokens, token_type="valid")
      self._tokens_processed.add(padded_tokens, token_type="padded")


_ERROR_STATUS_NAMES: frozenset[str] = frozenset(
    s.name for s in agent_types.INVALID_TRAJECTORY_STATUSES
)


def extract_lineage(
    metadata: Mapping[str, Any] | None,
) -> lineage.LineageContext | None:
  """Extracts `LineageContext` from `metadata` or raises `TypeError`."""
  if metadata is None:
    return None
  lineage_ctx = metadata.get("lineage")
  if lineage_ctx is None:
    return None
  if not isinstance(lineage_ctx, lineage.LineageContext):
    raise TypeError(
        "Expected LineageContext for metadata['lineage'], got"
        f" {type(lineage_ctx)!r}."
    )
  return lineage_ctx


def _require_mapping(
    traj: Mapping[str, Any], key: str
) -> Mapping[str, Any] | None:
  """Returns `traj[key]` when present and a `Mapping`, or raises `TypeError`."""
  val = traj.get(key)
  if val is None:
    return None
  if not isinstance(val, Mapping):
    raise TypeError(f"Expected Mapping for traj[{key!r}], got {type(val)!r}.")
  return val


def validate_trajectory_dict(traj: Any) -> tuple[Mapping[str, Any], str]:
  """Validates `TrajectoryItem.traj` mapping, status name, and timing dicts."""
  if not isinstance(traj, Mapping):
    raise TypeError(
        f"Expected Mapping for TrajectoryItem.traj, got {type(traj)!r}."
    )
  raw_status = traj["status"]
  if not isinstance(raw_status, str):
    raise TypeError(
        "Expected canonical str TrajectoryStatus name for traj['status'], got"
        f" {type(raw_status)!r}."
    )
  status = datatypes.TrajectoryStatus[raw_status].name
  _require_mapping(traj, "env_time")
  _require_mapping(traj, "reward_time")
  return traj, status


class RolloutWorkerOtelMetrics:
  """Encapsulates OpenTelemetry instruments and recording for RolloutWorker."""

  # OpenTelemetry's SDK deduplicates `create_observable_gauge` per MeterProvider
  # by returning the first instrument and silently dropping callbacks registered
  # by subsequent calls. While production pods run one worker per process,
  # multi-worker CPU harnesses share a single MeterProvider; keeping a weak
  # per-provider instance set lets one callback observe all live workers without
  # pinning providers or stopped workers in memory.
  _inflight_lock = threading.Lock()
  _inflight_by_provider: weakref.WeakKeyDictionary[
      Any, weakref.WeakSet[RolloutWorkerOtelMetrics]
  ] = weakref.WeakKeyDictionary()

  def __init__(
      self,
      worker_id: str,
      inflight_fn: Callable[[], int],
      meter_provider: otel_api_types.MeterProvider | None = None,
  ):
    self._worker_id = worker_id
    self._inflight_fn = inflight_fn
    self._sync_pause_start_s: float | None = None
    self._meter = WorkerMeter("tunix.worker", worker_id, meter_provider)
    # Per-trajectory recording short-circuits on this flag so a worker without
    # `opentelemetry` skips building attributes on the rollout hot path.
    self._enabled = self._meter.enabled
    self._traj_completed = self._meter.counter(
        "tunix.worker.trajectories.completed",
        unit="{trajectory}",
        description="Count of completed rollout trajectories.",
    )
    self._turns_total = self._meter.counter(
        "tunix.worker.turns.total",
        unit="{turn}",
        description="Total assistant turns generated across trajectories.",
    )
    self._prompt_tokens = self._meter.histogram(
        "tunix.worker.prompt_tokens",
        unit="{token}",
        description="Prompt token count per rollout trajectory.",
    )
    self._completion_tokens = self._meter.histogram(
        "tunix.worker.completion_tokens",
        unit="{token}",
        description="Completion token count per rollout trajectory.",
    )
    self._execution_sec = self._meter.histogram(
        "tunix.worker.execution_sec",
        unit="s",
        description="End-to-end episode execution time on the rollout worker.",
    )
    self._sampler_time_s = self._meter.histogram(
        "tunix.worker.sampler_time_s",
        unit="s",
        description=(
            "Cumulative model sampler inference latency per trajectory."
        ),
    )
    self._env_time_sec = self._meter.histogram(
        "tunix.worker.env_time_sec",
        unit="s",
        description="Environment interaction latency by operation.",
    )
    self._reward_time_sec = self._meter.histogram(
        "tunix.worker.reward_time_sec",
        unit="s",
        description="Final reward computation latency per trajectory.",
    )
    self._weight_sync_duration = self._meter.histogram(
        "tunix.worker.weight_sync.duration",
        unit="s",
        description=(
            "Rollout worker weight sync phase and total paused duration."
        ),
    )
    if self._meter.provider is not None and self._meter.meter is not None:
      self.register_inflight_gauge(
          self._meter.provider, self._meter.meter, self
      )

  @classmethod
  def register_inflight_gauge(
      cls,
      provider: otel_api_types.MeterProvider,
      meter: otel_api_types.Meter,
      instance: RolloutWorkerOtelMetrics,
  ) -> None:
    """Registers `instance` for `requests_inflight` observation on `provider`."""
    with cls._inflight_lock:
      instances = cls._inflight_by_provider.get(provider)
      if instances is not None:
        instances.add(instance)
        return
      instances = weakref.WeakSet([instance])
      cls._inflight_by_provider[provider] = instances

    def _observe_inflight(
        options: otel_api_types.CallbackOptions,
    ) -> Iterable[otel_api_types.Observation]:
      del options
      api = otel_api
      if api is None:  # Narrows module-global Optional for Pyrefly.
        return
      with cls._inflight_lock:
        live = list(instances)
      for inst in live:
        yield api.Observation(
            inst._inflight_fn(),  # pylint: disable=protected-access
            attributes={"worker_id": inst._worker_id},  # pylint: disable=protected-access
        )

    meter.create_observable_gauge(
        "tunix.worker.requests_inflight",
        callbacks=[_observe_inflight],
        unit="{request}",
        description=(
            "Number of active rollout episodes in flight on this worker."
        ),
    )

  def unregister_inflight(self) -> None:
    """Detaches this worker from `requests_inflight` observations on `stop()`."""
    if self._meter.provider is None:
      return
    with self._inflight_lock:
      instances = self._inflight_by_provider.get(self._meter.provider)
      if instances is not None:
        instances.discard(self)

  @staticmethod
  def resolve_task_name(metadata: Mapping[str, Any] | None) -> str:
    """Resolves semantic task_name from LineageContext."""
    lineage_ctx = extract_lineage(metadata)
    return lineage_ctx.task_name if lineage_ctx is not None else ""

  def record_trajectory_error(self, metadata: Mapping[str, Any] | None) -> None:
    """Records trajectory completion counter for a failed episode."""
    if not self._enabled:
      return
    self._traj_completed.add(
        1,
        task_name=self.resolve_task_name(metadata),
        status="ERROR",
        clipped="false",
        has_error="true",
    )

  def record_trajectory_item(self, item: datatypes.TrajectoryItem) -> None:
    """Records OpenTelemetry instruments from a completed TrajectoryItem."""
    if not self._enabled:
      return
    traj, resolved_status = validate_trajectory_dict(item.traj)
    metadata = item.metadata
    task_name = self.resolve_task_name(metadata)
    clipped_str = "true" if bool(metadata.get("clipped")) else "false"
    has_error_str = (
        "true" if resolved_status in _ERROR_STATUS_NAMES else "false"
    )

    self._traj_completed.add(
        1,
        task_name=task_name,
        status=resolved_status,
        clipped=clipped_str,
        has_error=has_error_str,
    )
    self.record_turns(traj, task_name=task_name)
    self.record_tokens(
        traj,
        task_name=task_name,
        status=resolved_status,
        clipped_str=clipped_str,
    )
    self.record_latencies(traj, task_name=task_name, status=resolved_status)

  def record_turns(self, traj: Mapping[str, Any], *, task_name: str) -> None:
    """Records assistant turn count counter from a trajectory."""
    if (turn_lens := traj.get("assistant_turn_lengths")) is not None:
      self._turns_total.add(len(turn_lens), task_name=task_name)

  def record_tokens(
      self,
      traj: Mapping[str, Any],
      *,
      task_name: str,
      status: str,
      clipped_str: str,
  ) -> None:
    """Records prompt and completion token histograms from a trajectory."""
    # Prefer `prompt_length` (matching
    # `payload_builder._extract_tokens_and_masks` and
    # `converter.compute_final_metrics`): `collector.py` stores placeholder
    # `prompt_tokens=[0]` with `prompt_length=0` when the sampler echoes no
    # prompt, while `TrajectoryCollectEngine._reset()` seeds `prompt_tokens=[]`
    # when tokenization never ran.
    prompt_len: int | None = None
    if (raw_prompt_len := traj.get("prompt_length")) is not None:
      prompt_len = int(raw_prompt_len)
    elif (raw_prompt_tokens := traj.get("prompt_tokens")) is not None and len(
        raw_prompt_tokens
    ) > 0:
      prompt_len = len(raw_prompt_tokens)
    if prompt_len is not None:
      self._prompt_tokens.record(prompt_len, task_name=task_name, status=status)

    if (turn_lens := traj.get("assistant_turn_lengths")) is not None:
      self._completion_tokens.record(
          int(sum(turn_lens)),
          task_name=task_name,
          status=status,
          clipped=clipped_str,
      )

  def _record_env_time(
      self,
      env_time: Mapping[str, Any],
      task_name: str,
  ) -> None:
    """Records `reset`, `step`, and non-zero `close` env latency points."""
    if not env_time:
      return
    self._env_time_sec.record(
        float(env_time["reset_latency"]), task_name=task_name, op="reset"
    )
    for step_val in env_time["step_latency"]:
      self._env_time_sec.record(float(step_val), task_name=task_name, op="step")
    # `TrajectoryCollectEngine._close()` seeds `close_latency=0.0` and leaves it
    # at `0.0` only when `env.close()` times out after 150s; every normal close
    # runs via `_run_with_timing` with `perf_counter` delta > 0.0.
    close_s = float(env_time["close_latency"])
    if close_s > 0.0:
      self._env_time_sec.record(close_s, task_name=task_name, op="close")

  def record_latencies(
      self,
      traj: Mapping[str, Any],
      *,
      task_name: str,
      status: str,
  ) -> None:
    """Records episode, sampler, environment, and reward latency histograms."""
    if (exec_s := traj.get("execution_time_s")) is not None:
      self._execution_sec.record(
          float(exec_s), task_name=task_name, status=status
      )
    if (sampler_s := traj.get("sampler_time_s")) is not None:
      self._sampler_time_s.record(float(sampler_s), task_name=task_name)
    if (env_time := _require_mapping(traj, "env_time")) is not None:
      self._record_env_time(env_time, task_name)
    # `TrajectoryCollectEngine._reset()` seeds `reward_latency=0.0` even when
    # `env.final_reward_fn is None` or `masked_out=True`; every executed
    # `final_reward_fn` (e.g. Harbor/SWE verification) runs via
    # `_run_with_timing` with `perf_counter` delta > 0.0.
    reward_time = _require_mapping(traj, "reward_time")
    if reward_time and (reward_s := float(reward_time["reward_latency"])) > 0.0:
      self._reward_time_sec.record(reward_s, task_name=task_name)

  def record_pre_weight_sync_start(self) -> float:
    """Marks the start of a weight-sync pause window and returns the timestamp."""
    t0 = perf_counter_s()
    if self._sync_pause_start_s is None:
      self._sync_pause_start_s = t0
    return t0

  def record_weight_sync_phase(self, phase: str, start_s: float) -> None:
    """Records duration of a weight-sync phase (`pre` or `h2d`)."""
    self._weight_sync_duration.record(
        float(perf_counter_s() - start_s), phase=phase
    )

  def record_post_weight_sync(self, start_s: float) -> None:
    """Records `post` and `total_paused` weight-sync durations."""
    end_s = perf_counter_s()
    self._weight_sync_duration.record(float(end_s - start_s), phase="post")
    if self._sync_pause_start_s is not None:
      self._weight_sync_duration.record(
          float(end_s - self._sync_pause_start_s), phase="total_paused"
      )
    self._sync_pause_start_s = None

  def record_abort_weight_sync(self) -> None:
    """Records `aborted_paused` weight-sync duration if a pause window was open."""
    end_s = perf_counter_s()
    if self._sync_pause_start_s is not None:
      self._weight_sync_duration.record(
          float(end_s - self._sync_pause_start_s), phase="aborted_paused"
      )
    self._sync_pause_start_s = None
