# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-level RolloutWorker abstractions (Service vs Client Driver)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import contextlib
import dataclasses
import inspect
import threading
import time
import types
from typing import Any, List, Optional, TYPE_CHECKING, Union
import weakref

from absl import logging
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import gcs_cache
from tunix.dist_rl.common import lineage
from tunix.dist_rl.common import registry
from tunix.dist_rl.common import timeouts
from tunix.dist_rl.rollout import manager as manager_lib
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.trajectory import store as trajectory_store_lib
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.worker import abstract_worker
from tunix.rl.agentic.agents import agent_types
from tunix.rl.rollout import base_rollout

if TYPE_CHECKING:
  from opentelemetry import metrics as otel_metrics_types  # pylint: disable=g-bad-import-order,g-import-not-at-top

try:
  from opentelemetry import metrics as otel_metrics  # pylint: disable=g-import-not-at-top
except ImportError:
  otel_metrics = None


@dataclasses.dataclass(kw_only=True)
class RolloutConfig(base_rollout.RolloutConfig):
  """Rollout configuration extending base RolloutConfig with worker options.

  Attributes:
    trajectory_store_config: Trajectory Store configuration for this worker
      process, or None to run without a store. See
      `store.TrajectoryStore.from_config`. The worker builds the store from this
      config and passes it to its RolloutManager, so rollout steps are written
      by this process. Must match what the orchestrator was given: for the file
      backend it is the shared root_dir and run_id that make these writes
      visible to the orchestrator's reads.
    drain_timeout_s: Soft drain window (seconds) in `pre_weight_sync` when
      `partial_rollout` is False before remaining stragglers are paused.
  """

  trajectory_store_config: Mapping[str, Any] | None = None
  drain_timeout_s: float = timeouts.DEFAULT_DRAIN_TIMEOUT_S

  def __post_init__(self) -> None:
    super().__post_init__()
    self.drain_timeout_s = timeouts.validate_positive_finite(
        self.drain_timeout_s, "drain_timeout_s"
    )


TrajectoryOrError = Union[
    trajectory_lib.Trajectory, trajectory_lib.TrajectoryError
]

WorkerState = datatypes.WorkerState

_ERROR_STATUS_NAMES: frozenset[str] = frozenset(
    s.name for s in agent_types.INVALID_TRAJECTORY_STATUSES
)


def _extract_lineage(
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


def _validate_trajectory_dict(traj: Any) -> tuple[Mapping[str, Any], str]:
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


class _RolloutWorkerOtelMetrics:
  """Encapsulates OpenTelemetry instruments and recording for RolloutWorker."""

  # OpenTelemetry's SDK deduplicates `create_observable_gauge` per MeterProvider
  # by returning the first instrument and silently dropping callbacks registered
  # by subsequent calls. While production pods run one worker per process,
  # multi-worker CPU harnesses share a single MeterProvider; keeping a weak
  # per-provider instance set lets one callback observe all live workers without
  # pinning providers or stopped workers in memory.
  _inflight_lock = threading.Lock()
  _inflight_by_provider: weakref.WeakKeyDictionary[
      Any, weakref.WeakSet[_RolloutWorkerOtelMetrics]
  ] = weakref.WeakKeyDictionary()

  def __init__(
      self,
      worker_id: str,
      inflight_fn: Callable[[], int],
      meter_provider: otel_metrics_types.MeterProvider | None = None,
  ):
    self._worker_id = worker_id
    self._inflight_fn = inflight_fn
    self._sync_pause_start_s: float | None = None
    self._provider: otel_metrics_types.MeterProvider | None = None
    if otel_metrics is None:
      self._enabled = False
      return
    self._enabled = True

    provider = meter_provider or otel_metrics.get_meter_provider()
    self._provider = provider
    meter = provider.get_meter("tunix.worker")
    self._traj_completed = meter.create_counter(
        "tunix.worker.trajectories.completed",
        unit="{trajectory}",
        description="Count of completed rollout trajectories.",
    )
    self._turns_total = meter.create_counter(
        "tunix.worker.turns.total",
        unit="{turn}",
        description="Total assistant turns generated across trajectories.",
    )
    self._prompt_tokens = meter.create_histogram(
        "tunix.worker.prompt_tokens",
        unit="{token}",
        description="Prompt token count per rollout trajectory.",
    )
    self._completion_tokens = meter.create_histogram(
        "tunix.worker.completion_tokens",
        unit="{token}",
        description="Completion token count per rollout trajectory.",
    )
    self._execution_sec = meter.create_histogram(
        "tunix.worker.execution_sec",
        unit="s",
        description="End-to-end episode execution time on the rollout worker.",
    )
    self._sampler_time_s = meter.create_histogram(
        "tunix.worker.sampler_time_s",
        unit="s",
        description=(
            "Cumulative model sampler inference latency per trajectory."
        ),
    )
    self._env_time_sec = meter.create_histogram(
        "tunix.worker.env_time_sec",
        unit="s",
        description="Environment interaction latency by operation.",
    )
    self._reward_time_sec = meter.create_histogram(
        "tunix.worker.reward_time_sec",
        unit="s",
        description="Final reward computation latency per trajectory.",
    )
    self._weight_sync_duration = meter.create_histogram(
        "tunix.worker.weight_sync.duration",
        unit="s",
        description=(
            "Rollout worker weight sync phase and total paused duration."
        ),
    )
    self.register_inflight_gauge(provider, meter, self)

  @classmethod
  def register_inflight_gauge(
      cls,
      provider: otel_metrics_types.MeterProvider,
      meter: otel_metrics_types.Meter,
      instance: _RolloutWorkerOtelMetrics,
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
        options: otel_metrics_types.CallbackOptions,
    ) -> Iterable[otel_metrics_types.Observation]:
      del options
      if otel_metrics is None:  # Narrows module-global Optional for Pyrefly.
        return
      with cls._inflight_lock:
        live = list(instances)
      for inst in live:
        yield otel_metrics.Observation(
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
    if self._provider is None:
      return
    with self._inflight_lock:
      instances = self._inflight_by_provider.get(self._provider)
      if instances is not None:
        instances.discard(self)

  @staticmethod
  def resolve_task_name(metadata: Mapping[str, Any] | None) -> str:
    """Resolves semantic task_name from LineageContext."""
    lineage_ctx = _extract_lineage(metadata)
    return lineage_ctx.task_name if lineage_ctx is not None else ""

  def record_trajectory_error(self, metadata: Mapping[str, Any] | None) -> None:
    """Records trajectory completion counter for a failed episode."""
    if not self._enabled:
      return
    task_name = self.resolve_task_name(metadata)
    self._traj_completed.add(
        1,
        attributes={
            "worker_id": self._worker_id,
            "task_name": task_name,
            "status": "ERROR",
            "clipped": "false",
            "has_error": "true",
        },
    )

  def record_trajectory_item(self, item: datatypes.TrajectoryItem) -> None:
    """Records OpenTelemetry instruments from a completed TrajectoryItem."""
    if not self._enabled:
      return
    traj, resolved_status = _validate_trajectory_dict(item.traj)
    metadata = item.metadata
    task_name = self.resolve_task_name(metadata)
    clipped_str = "true" if bool(metadata.get("clipped")) else "false"
    has_error_str = (
        "true" if resolved_status in _ERROR_STATUS_NAMES else "false"
    )

    self._traj_completed.add(
        1,
        attributes={
            "worker_id": self._worker_id,
            "task_name": task_name,
            "status": resolved_status,
            "clipped": clipped_str,
            "has_error": has_error_str,
        },
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
      self._turns_total.add(
          len(turn_lens),
          attributes={"worker_id": self._worker_id, "task_name": task_name},
      )

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
      self._prompt_tokens.record(
          prompt_len,
          attributes={
              "worker_id": self._worker_id,
              "task_name": task_name,
              "status": status,
          },
      )

    if (turn_lens := traj.get("assistant_turn_lengths")) is not None:
      self._completion_tokens.record(
          int(sum(turn_lens)),
          attributes={
              "worker_id": self._worker_id,
              "task_name": task_name,
              "status": status,
              "clipped": clipped_str,
          },
      )

  def _record_env_time(
      self,
      env_time: Mapping[str, Any],
      base_attrs: Mapping[str, str],
  ) -> None:
    """Records `reset`, `step`, and non-zero `close` env latency points."""
    if not env_time:
      return
    self._env_time_sec.record(
        float(env_time["reset_latency"]),
        attributes={**base_attrs, "op": "reset"},
    )
    for step_val in env_time["step_latency"]:
      self._env_time_sec.record(
          float(step_val), attributes={**base_attrs, "op": "step"}
      )
    # `TrajectoryCollectEngine._close()` seeds `close_latency=0.0` and leaves it
    # at `0.0` only when `env.close()` times out after 150s; every normal close
    # runs via `_run_with_timing` with `perf_counter` delta > 0.0.
    close_s = float(env_time["close_latency"])
    if close_s > 0.0:
      self._env_time_sec.record(
          close_s, attributes={**base_attrs, "op": "close"}
      )

  def record_latencies(
      self,
      traj: Mapping[str, Any],
      *,
      task_name: str,
      status: str,
  ) -> None:
    """Records episode, sampler, environment, and reward latency histograms."""
    base_attrs = {"worker_id": self._worker_id, "task_name": task_name}
    if (exec_s := traj.get("execution_time_s")) is not None:
      self._execution_sec.record(
          float(exec_s), attributes={**base_attrs, "status": status}
      )
    if (sampler_s := traj.get("sampler_time_s")) is not None:
      self._sampler_time_s.record(float(sampler_s), attributes=base_attrs)
    if (env_time := _require_mapping(traj, "env_time")) is not None:
      self._record_env_time(env_time, base_attrs)
    # `TrajectoryCollectEngine._reset()` seeds `reward_latency=0.0` even when
    # `env.final_reward_fn is None` or `masked_out=True`; every executed
    # `final_reward_fn` (e.g. Harbor/SWE verification) runs via
    # `_run_with_timing` with `perf_counter` delta > 0.0.
    reward_time = _require_mapping(traj, "reward_time")
    if reward_time and (reward_s := float(reward_time["reward_latency"])) > 0.0:
      self._reward_time_sec.record(reward_s, attributes=base_attrs)

  def record_pre_weight_sync_start(self) -> float:
    """Marks the start of a weight-sync pause window and returns the timestamp."""
    t0 = time.perf_counter()
    if self._sync_pause_start_s is None:
      self._sync_pause_start_s = t0
    return t0

  def record_weight_sync_phase(self, phase: str, start_s: float) -> None:
    """Records duration of a weight-sync phase (`pre` or `h2d`)."""
    if not self._enabled:
      return
    self._weight_sync_duration.record(
        float(time.perf_counter() - start_s),
        attributes={"worker_id": self._worker_id, "phase": phase},
    )

  def record_post_weight_sync(self, start_s: float) -> None:
    """Records `post` and `total_paused` weight-sync durations."""
    end_s = time.perf_counter()
    if self._enabled:
      self._weight_sync_duration.record(
          float(end_s - start_s),
          attributes={"worker_id": self._worker_id, "phase": "post"},
      )
      if self._sync_pause_start_s is not None:
        self._weight_sync_duration.record(
            float(end_s - self._sync_pause_start_s),
            attributes={
                "worker_id": self._worker_id,
                "phase": "total_paused",
            },
        )
    self._sync_pause_start_s = None

  def record_abort_weight_sync(self) -> None:
    """Records `aborted_paused` weight-sync duration if a pause window was open."""
    end_s = time.perf_counter()
    if self._enabled and self._sync_pause_start_s is not None:
      self._weight_sync_duration.record(
          float(end_s - self._sync_pause_start_s),
          attributes={
              "worker_id": self._worker_id,
              "phase": "aborted_paused",
          },
      )
    self._sync_pause_start_s = None


class RolloutWorker(abstract_worker.Worker):
  """Worker wrapper for rollout collection.

  Encapsulates RolloutManager and executes concurrent episode loops
  locally on its remote CPU host.
  """

  def __init__(
      self,
      worker_id: str,
      sampler: sampler_lib.Sampler,
      tokenizer: Any,
      chat_parser: Any,
      env_name: str,
      agent_name: str,
      config: Optional[RolloutConfig] = None,
      env_config: Mapping[str, Any] = types.MappingProxyType({}),
      agent_config: Mapping[str, Any] = types.MappingProxyType({}),
      max_concurrency: int = 64,
      jax_cache_config: Optional[gcs_cache.JaxCacheConfig] = None,
      otel_meter_provider: otel_metrics_types.MeterProvider | None = None,
  ):
    """Initializes the RolloutWorker.

    Args:
      worker_id: Unique identifier for this worker process.
      sampler: Pre-constructed Sampler instance.
      tokenizer: Tokenizer for prompt/response encoding.
      chat_parser: Chat parser for conversation templating.
      env_name: Name of the registered environment in `ENV_REGISTRY`.
      agent_name: Name of the registered agent in `AGENT_REGISTRY`.
      config: Optional RolloutConfig configuration options.
      env_config: Default environment constructor kwargs. A request's
        `metadata["env_config"]` entries are shallow-merged on top of these.
      agent_config: Default agent constructor kwargs. A request's
        `metadata["agent_config"]` replaces these entirely when present.
      max_concurrency: Maximum number of concurrent episodes.
      jax_cache_config: Optional JAX compilation cache configuration.
      otel_meter_provider: Optional OpenTelemetry MeterProvider override for
        testing; defaults to the global MeterProvider when None.
    """
    super().__init__()
    self.worker_id = worker_id
    self.config = config
    self._policy_version = 0
    self._state = datatypes.WorkerState.PENDING
    # Set when `stop()` failed to clean up; later stops re-raise it.
    self._stop_error: str | None = None
    self._init_lock = threading.Lock()
    # TODO(linchai): make the sampler the single owner of weight-sync round
    # state, then drop `_sync_round` and `_record_round`.
    self._sync_round = {"req_id": None, "uuid": 0, "phase": "idle"}
    self._jax_cache_syncer = gcs_cache.JaxCacheSyncer(
        worker_id, jax_cache_config
    )
    env_factory = registry.ENV_REGISTRY.get(env_name)
    agent_factory = registry.AGENT_REGISTRY.get(agent_name)
    # Built at most once per process: this __init__ runs exactly once per
    # RolloutWorker instance, so there is no separate guard against
    # constructing the store twice. See store.TrajectoryStore.from_config.
    self._trajectory_store = (
        trajectory_store_lib.TunixTrajectoryStore.from_config(
            config.trajectory_store_config if config is not None else None
        )
    )
    if self._trajectory_store is not None:
      # Several workers can share one log stream, and absl log lines carry no
      # process identity, so the worker_id is what attributes a reported
      # config to a process.
      logging.info(
          "[trajectory-store] worker %s built %s",
          worker_id,
          self._trajectory_store.to_redacted_config(),
      )
    drain_timeout_s = (
        config.drain_timeout_s
        if config is not None
        else timeouts.DEFAULT_DRAIN_TIMEOUT_S
    )
    self.manager = manager_lib.RolloutManager(
        config=config,
        sampler=sampler,
        env_factory=env_factory,
        agent_factory=agent_factory,
        env_config=env_config,
        agent_config=agent_config,
        max_concurrency=max_concurrency,
        tokenizer=tokenizer,
        chat_parser=chat_parser,
        drain_timeout_s=drain_timeout_s,
        trajectory_store=self._trajectory_store,
    )
    self._otel = _RolloutWorkerOtelMetrics(
        worker_id=worker_id,
        inflight_fn=lambda: len(self.manager._active_tasks),  # pylint: disable=protected-access
        meter_provider=otel_meter_provider,
    )

  @property
  def trajectory_store(
      self,
  ) -> trajectory_store_lib.TunixTrajectoryStore | None:
    return self._trajectory_store

  @property
  def sampler(self) -> sampler_lib.Sampler:
    return self.manager.sampler

  def get_worker_id(self) -> str:
    """Returns the unique worker ID."""
    return self.worker_id

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=frozenset({"rollout"}),
        resources={
            "sampler": type(self.sampler).__name__,
            "policy_version": self._policy_version,
            "trajectory_store_config": (
                self._trajectory_store.to_config()
                if self._trajectory_store is not None
                else None
            ),
            "drain_timeout_s": self.manager.drain_timeout_s,
            "partial_rollout": self.manager.partial_rollout,
        },
    )

  def _response(self, **metadata: Any) -> datatypes.Response:
    return datatypes.Response(
        metadata={
            "worker_id": self.worker_id,
            "state": self.state.value,
            "policy_version": self._policy_version,
            **metadata,
        }
    )

  def _ensure_initialized(self) -> None:
    if self.state == WorkerState.PENDING:
      self.initialize()

  def initialize(self) -> datatypes.Response:
    with self._init_lock:
      trajectory_store_config = (
          self._trajectory_store.to_config()
          if self._trajectory_store is not None
          else None
      )
      if self.state == WorkerState.READY:
        return self._response(
            initialized=True,
            ready=True,
            trajectory_store_config=trajectory_store_config,
            drain_timeout_s=self.manager.drain_timeout_s,
            partial_rollout=self.manager.partial_rollout,
        )
      self.state = WorkerState.INITIALIZING
      try:
        self.sampler.initialize()
        return self._response(
            trajectory_store_config=trajectory_store_config,
            drain_timeout_s=self.manager.drain_timeout_s,
            partial_rollout=self.manager.partial_rollout,
        )
      except Exception:
        self.state = WorkerState.ERROR
        raise
      finally:
        if self.state == WorkerState.INITIALIZING:
          self.state = WorkerState.READY

  def compile(self, dummy_data: Any) -> datatypes.Response:
    self._ensure_initialized()
    self.state = WorkerState.COMPILING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  async def start(self) -> datatypes.Response:
    self._ensure_initialized()
    try:
      await self.sampler.start()
      await self.manager.bind_weight_sync()
      self._jax_cache_syncer.sync(wait=False)
      return self._response(started=True)
    except Exception:
      self.state = WorkerState.ERROR
      raise

  def stop(self, *, abort: bool = False) -> datatypes.Response:
    """Stops the worker, cleaning up its resources at most once.

    If cleanup fails, the worker still ends up `STOPPED` so that teardown never
    re-runs a partially failed cleanup. The failure is kept instead of dropped:
    it is escalated to an abort for the `set_on_stop` callback and re-raised by
    every later `stop()` call, so local node teardown still exits non-zero.

    Args:
      abort: Whether the stop is part of an abnormal cluster teardown; see
        `Worker.stop`.

    Returns:
      A response with `stopped=True` and the requested `abort` flag, plus
      `already_stopped=True` if the worker was already stopped.

    Raises:
      RuntimeError: If an earlier `stop()` failed to clean up.
    """
    try:
      if self.state == WorkerState.STOPPED:
        if self._stop_error is not None:
          raise RuntimeError(
              f"RolloutWorker {self.worker_id!r} failed to stop cleanly"
              f" earlier: {self._stop_error}"
          )
        return self._response(
            stopped=True, already_stopped=True, abort=bool(abort)
        )
      self.state = WorkerState.STOPPED
      try:
        with contextlib.ExitStack() as stack:
          # Callbacks run in reverse order once `cancel_all` returns or raises,
          # so a failed stop still releases the store's background writer thread
          # and waits for the cache syncer instead of leaking them.
          stack.callback(self._otel.unregister_inflight)
          stack.callback(self._jax_cache_syncer.wait)
          if self._trajectory_store is not None:
            stack.callback(self._trajectory_store.close)
          self.manager.cancel_all()
      except Exception as exc:
        self._stop_error = str(exc)
        abort = True
        raise
      return self._response(stopped=True, abort=bool(abort))
    finally:
      self._notify_stopped(abort)

  def pause(self) -> datatypes.Response:
    self.manager.pause_all()
    return datatypes.Response()

  def resume(self) -> datatypes.Response:
    self.manager.resume_all()
    return datatypes.Response()

  def _infer_shapes(self) -> Any:
    return None

  def _compile_with_shapes(self, abstract_state: Any) -> None:
    pass

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state=self.state,
        policy_version=self._policy_version,
        inflight=len(self.manager._active_tasks),  # pylint: disable=protected-access
    )

  def get_target_state(self) -> Any:
    """Returns rollout-side target-state skeleton for trainer-side conversion."""
    self._ensure_initialized()
    return self.manager.get_target_state()

  def _stamp_worker_lineage(
      self,
      metadata: dict[str, Any] | None,
      *,
      component: str = "worker.rollout",
      execution_time_s: float | None = None,
  ) -> None:
    """Appends worker generation telemetry to the lineage context if present."""
    lineage_ctx = _extract_lineage(metadata)
    if lineage_ctx is None:
      return
    attributes: dict[str, Any] = {"worker_id": self.worker_id}
    if execution_time_s is not None:
      attributes["execution_time_s"] = float(execution_time_s)
    lineage_ctx.add_event(
        component=component,
        operation="generate",
        attributes=attributes,
    )

  def _to_rollout_response(
      self,
      item: manager_lib.TrajectoryOrError | datatypes.RolloutResponse,
      request_id: str = "",
      prompt_tokens: np.ndarray | None = None,
  ) -> datatypes.RolloutResponse:
    """Converts internal Trajectory or TrajectoryError to wire-safe RolloutResponse."""
    if isinstance(item, trajectory_lib.TrajectoryError):
      metadata = dict(item.metadata or {})
      metadata["prompt_id"] = item.prompt_id
      self._stamp_worker_lineage(metadata)
      self._otel.record_trajectory_error(metadata)
      return datatypes.RolloutResponse(
          request_id=request_id or item.trajectory_id or item.prompt_id,
          status="ERROR",
          error=datatypes.ErrorInfo(
              error_type="TrajectoryError",
              message=str(item.error_message),
          ),
          payload=None,
          metadata=metadata,
      )
    if isinstance(item, datatypes.RolloutResponse):
      return item
    if isinstance(item, datatypes.TrajectoryItem):
      traj, _ = _validate_trajectory_dict(item.traj)
      req_id = request_id or item.traj_id
      if (
          prompt_tokens is not None
          and traj.get("prompt_tokens") is None
          and item.metadata.get("prompt_tokens") is None
      ):
        item.metadata["prompt_tokens"] = prompt_tokens
      self._stamp_worker_lineage(
          item.metadata, execution_time_s=traj.get("execution_time_s")
      )
      self._otel.record_trajectory_item(item)
      return datatypes.RolloutResponse(
          request_id=req_id,
          status="COMPLETED",
          payload=item,
          metadata=item.metadata,
      )
    raise TypeError(
        f"Unsupported item type for RolloutResponse conversion: {type(item)}"
    )

  async def generate(
      self,
      requests: datatypes.RolloutRequest | Sequence[datatypes.RolloutRequest],
      on_complete: Optional[Callable[[datatypes.RolloutResponse], None]] = None,
  ) -> datatypes.RolloutResponse | List[datatypes.RolloutResponse]:
    """Coroutine method for single or batched generate requests."""
    self._ensure_initialized()
    req_list = (
        [requests]
        if isinstance(requests, datatypes.RolloutRequest)
        else requests
    )
    if (
        isinstance(req_list, (str, bytes))
        or not isinstance(req_list, Sequence)
        or not all(
            isinstance(req, datatypes.RolloutRequest) for req in req_list
        )
    ):
      raise TypeError(
          "generate requires `requests` to be a RolloutRequest or"
          " Sequence[RolloutRequest]."
      )
    for req in req_list:
      self._stamp_worker_lineage(req.metadata, component="worker.received")

    # Store `(item, response)` to pin `item` alive across a batched `generate()`
    # call: `TrajectoryError` sets `RolloutResponse.payload = None`, so without
    # pinning `item`, GC could reclaim it and reuse `id(item)` on a later item.
    converted: dict[
        int,
        tuple[
            manager_lib.TrajectoryOrError | datatypes.RolloutResponse,
            datatypes.RolloutResponse,
        ],
    ] = {}

    def _convert_once(
        item: manager_lib.TrajectoryOrError | datatypes.RolloutResponse,
    ) -> datatypes.RolloutResponse:
      item_key = id(item)
      if item_key not in converted:
        converted[item_key] = (item, self._to_rollout_response(item))
      return converted[item_key][1]

    cb = (
        (lambda item: on_complete(_convert_once(item)))
        if on_complete is not None
        else None
    )
    res = await self.manager.generate(requests, on_complete=cb)
    if isinstance(res, Sequence):
      responses = [_convert_once(r) for r in res]
      for resp in responses:
        self._jax_cache_syncer.sync_after_first_completed(resp.status)
      return responses
    resp = _convert_once(res)
    self._jax_cache_syncer.sync_after_first_completed(resp.status)
    return resp

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Quiesces the worker; it stays SYNCING until post or abort."""
    self._ensure_initialized()
    self.state = WorkerState.SYNCING
    t0 = self._otel.record_pre_weight_sync_start()
    self._record_round(sync_request, "idle")
    result = await self.manager.pre_weight_sync(sync_request, **kwargs)
    self._otel.record_weight_sync_phase("pre", t0)
    self._record_round(sync_request, "prepared")
    return result

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Materializes the received weights; the worker stays SYNCING."""
    self._ensure_initialized()
    self.state = WorkerState.SYNCING
    metadata = kwargs.pop("metadata", None)
    request = sync_request if sync_request is not None else metadata
    t0 = time.perf_counter()
    result = await self.manager.weight_sync(request, **kwargs)
    self._otel.record_weight_sync_phase("h2d", t0)
    if isinstance(result, int):
      self._policy_version = result
    else:
      version = getattr(request, "policy_version", None)
      if version is not None:
        self._policy_version = version
      else:
        self._policy_version += 1
    self._record_round(request, "h2d_done")
    return result

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Publishes the new weights and resumes serving."""
    t0 = time.perf_counter()
    result = await self.manager.post_weight_sync(sync_request, **kwargs)
    self.state = WorkerState.READY
    self._otel.record_post_weight_sync(t0)
    self._record_round(sync_request, "committed")
    return result

  async def bind_weight_sync(self, **kwargs) -> Any:
    """Binds the destination-side transport via the manager."""
    self._ensure_initialized()
    return await self.manager.bind_weight_sync(**kwargs)

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns the sampler's transport metadata via the manager."""
    self._ensure_initialized()
    return await self.manager.get_weight_sync_metadata(**kwargs)

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Discards the round and resumes serving the previous weights."""
    res = await self.manager.abort_weight_sync(sync_request, **kwargs)
    self.state = WorkerState.READY
    self._record_round(sync_request, "aborted")
    if self._sync_round["phase"] == "aborted":
      self._otel.record_abort_weight_sync()
    return res

  async def get_weight_sync_status(self, **kwargs) -> Any:
    """Returns this worker's view of the current weight sync round."""
    del kwargs
    # A sampler with an opened WorkerRoundTracker round is authoritative: it
    # knows when an abort was refused because the round had already committed.
    sampler = getattr(self.manager, "sampler", None)
    get_status = getattr(sampler, "get_weight_sync_status", None)
    if get_status is not None:
      report = get_status()
      if inspect.isawaitable(report):
        report = await report
      if report and report.get("req_id") is not None:
        return dict(report, policy_version=self._policy_version)
    return dict(self._sync_round, policy_version=self._policy_version)

  def _record_round(self, sync_request: Any, phase: str) -> None:
    extra = getattr(sync_request, "extra_config", None) or {}
    req_id = extra.get("req_id")
    key = (req_id, extra.get("uuid", 0)) if req_id is not None else None
    current = (self._sync_round["req_id"], self._sync_round["uuid"])
    if (
        phase == "aborted"
        and self._sync_round["phase"] == "committed"
        and (key is None or key == current or key[1] < current[1])
    ):
      # An abort for this or an older round does not roll back a commit.
      return
    if key is not None:
      self._sync_round["req_id"], self._sync_round["uuid"] = key
    self._sync_round["phase"] = phase
