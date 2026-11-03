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

import dataclasses
import inspect
import threading
import time
import types
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Union

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
from tunix.rl.rollout import base_rollout

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
      otel_meter_provider: Any = None,
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
    self._env_name = env_name
    self.config = config
    self._policy_version = 0
    self._state = datatypes.WorkerState.PENDING
    self._init_lock = threading.Lock()
    self._sync_pause_start_s: float | None = None
    meter: Any = None
    if otel_metrics is not None:
      provider = otel_meter_provider or otel_metrics.get_meter_provider()
      meter = provider.get_meter("tunix.worker")
      self._otel_traj_completed = meter.create_counter(
          "tunix.worker.trajectories.completed",
          unit="{trajectory}",
          description="Count of completed rollout trajectories.",
      )
      self._otel_turns_total = meter.create_counter(
          "tunix.worker.turns.total",
          unit="{turn}",
          description="Total assistant turns generated across trajectories.",
      )
      self._otel_prompt_tokens = meter.create_histogram(
          "tunix.worker.prompt_tokens",
          unit="{token}",
          description="Prompt token count per rollout trajectory.",
      )
      self._otel_completion_tokens = meter.create_histogram(
          "tunix.worker.completion_tokens",
          unit="{token}",
          description="Completion token count per rollout trajectory.",
      )
      self._otel_execution_sec = meter.create_histogram(
          "tunix.worker.execution_sec",
          unit="s",
          description=(
              "End-to-end episode execution time on the rollout worker."
          ),
      )
      self._otel_sampler_time_s = meter.create_histogram(
          "tunix.worker.sampler_time_s",
          unit="s",
          description=(
              "Cumulative model sampler inference latency per trajectory."
          ),
      )
      self._otel_env_time_sec = meter.create_histogram(
          "tunix.worker.env_time_sec",
          unit="s",
          description="Environment interaction latency by operation.",
      )
      self._otel_reward_time_sec = meter.create_histogram(
          "tunix.worker.reward_time_sec",
          unit="s",
          description="Final reward computation latency per trajectory.",
      )
      self._otel_weight_sync_duration = meter.create_histogram(
          "tunix.worker.weight_sync.duration",
          unit="s",
          description=(
              "Rollout worker weight sync phase and total paused duration."
          ),
      )
    else:
      self._otel_traj_completed = None
      self._otel_turns_total = None
      self._otel_prompt_tokens = None
      self._otel_completion_tokens = None
      self._otel_execution_sec = None
      self._otel_sampler_time_s = None
      self._otel_env_time_sec = None
      self._otel_reward_time_sec = None
      self._otel_weight_sync_duration = None
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
    if otel_metrics is not None and meter is not None:
      def _observe_inflight(options: Any) -> Iterable[Any]:
        del options
        yield otel_metrics.Observation(
            len(self.manager._active_tasks),  # pylint: disable=protected-access
            attributes={"worker_id": self.worker_id},
        )

      self._otel_requests_inflight = meter.create_observable_gauge(
          "tunix.worker.requests_inflight",
          callbacks=[_observe_inflight],
          unit="{request}",
          description=(
              "Number of active rollout episodes in flight on this worker."
          ),
      )
    else:
      self._otel_requests_inflight = None

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

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    try:
      self.manager.cancel_all()
    finally:
      try:
        # Runs even when cancel_all raises, so a failed stop releases the
        # store's background writer thread instead of leaking it.
        if self._trajectory_store is not None:
          self._trajectory_store.close()
      finally:
        self._jax_cache_syncer.wait()
    return datatypes.Response()

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

  def _resolve_task_name(self, metadata: Mapping[str, Any] | None) -> str:
    """Resolves semantic task_name from LineageContext or falls back to env_name."""
    if metadata is not None and "lineage" in metadata:
      lineage_ctx = metadata["lineage"]
      if (
          isinstance(lineage_ctx, lineage.LineageContext)
          and lineage_ctx.task_name
      ):
        return lineage_ctx.task_name
    return self._env_name

  def _stamp_worker_lineage(
      self,
      metadata: dict[str, Any] | None,
      *,
      component: str = "worker.rollout",
      execution_time_s: float | None = None,
  ) -> None:
    """Appends worker generation telemetry to the lineage context if present."""
    if metadata is None or "lineage" not in metadata:
      return
    lineage_ctx = metadata["lineage"]
    if isinstance(lineage_ctx, lineage.LineageContext):
      attributes: dict[str, Any] = {"worker_id": self.worker_id}
      if execution_time_s is not None:
        attributes["execution_time_s"] = float(execution_time_s)
      lineage_ctx.add_event(
          component=component,
          operation="generate",
          attributes=attributes,
      )

  def _record_trajectory_error_otel(
      self, metadata: Mapping[str, Any] | None
  ) -> None:
    """Records OpenTelemetry trajectory completion counter for a failed episode."""
    if self._otel_traj_completed is None:
      return
    task_name = self._resolve_task_name(metadata)
    self._otel_traj_completed.add(
        1,
        attributes={
            "worker_id": self.worker_id,
            "task_name": task_name,
            "status": "ERROR",
            "clipped": "false",
            "has_error": "true",
        },
    )

  def _record_trajectory_item_otel(
      self, item: datatypes.TrajectoryItem
  ) -> None:
    """Records direct OpenTelemetry instruments from a completed TrajectoryItem."""
    if self._otel_traj_completed is None:
      return
    traj = item.traj if isinstance(item.traj, Mapping) else {}
    metadata = item.metadata
    task_name = self._resolve_task_name(metadata)

    if "status" in traj and traj["status"]:
      raw_status = traj["status"]
    elif "status" in metadata and metadata["status"]:
      raw_status = metadata["status"]
    else:
      raw_status = "COMPLETED"
    status = (
        raw_status.name
        if isinstance(raw_status, datatypes.TrajectoryStatus)
        else str(raw_status)
    )
    clipped_bool = bool(metadata["clipped"]) if "clipped" in metadata else False
    clipped_str = "true" if clipped_bool else "false"
    has_error_str = "false" if status in ("SUCCEEDED", "COMPLETED") else "true"

    self._otel_traj_completed.add(
        1,
        attributes={
            "worker_id": self.worker_id,
            "task_name": task_name,
            "status": status,
            "clipped": clipped_str,
            "has_error": has_error_str,
        },
    )

    num_turns: int | None = None
    if (
        "assistant_turn_lengths" in traj
        and traj["assistant_turn_lengths"] is not None
    ):
      num_turns = len(traj["assistant_turn_lengths"])
    elif "conversation_text" in traj and isinstance(
        traj["conversation_text"], list
    ):
      num_turns = sum(
          1
          for msg in traj["conversation_text"]
          if isinstance(msg, Mapping)
          and "role" in msg
          and msg["role"] == "assistant"
      )
    if (
        num_turns is not None
        and num_turns > 0
        and self._otel_turns_total is not None
    ):
      self._otel_turns_total.add(
          num_turns,
          attributes={"worker_id": self.worker_id, "task_name": task_name},
      )

    prompt_len: int | None = None
    if "prompt_length" in traj and traj["prompt_length"] is not None:
      prompt_len = int(traj["prompt_length"])
    elif "prompt_tokens" in traj and traj["prompt_tokens"] is not None:
      prompt_len = len(traj["prompt_tokens"])
    elif "prompt_tokens" in metadata and metadata["prompt_tokens"] is not None:
      prompt_len = len(metadata["prompt_tokens"])
    if prompt_len is not None and self._otel_prompt_tokens is not None:
      self._otel_prompt_tokens.record(
          prompt_len,
          attributes={
              "worker_id": self.worker_id,
              "task_name": task_name,
              "status": status,
          },
      )

    completion_len: int | None = None
    if (
        "conversation_tokens" in traj
        and traj["conversation_tokens"] is not None
    ):
      completion_len = len(traj["conversation_tokens"])
    elif (
        "assistant_turn_lengths" in traj
        and traj["assistant_turn_lengths"] is not None
    ):
      completion_len = int(sum(traj["assistant_turn_lengths"]))
    elif (
        "completion_tokens" in metadata
        and metadata["completion_tokens"] is not None
    ):
      completion_len = len(metadata["completion_tokens"])
    if completion_len is not None and self._otel_completion_tokens is not None:
      self._otel_completion_tokens.record(
          completion_len,
          attributes={
              "worker_id": self.worker_id,
              "task_name": task_name,
              "status": status,
              "clipped": clipped_str,
          },
      )

    if (
        "execution_time_s" in traj
        and traj["execution_time_s"] is not None
        and self._otel_execution_sec is not None
    ):
      self._otel_execution_sec.record(
          float(traj["execution_time_s"]),
          attributes={
              "worker_id": self.worker_id,
              "task_name": task_name,
              "status": status,
          },
      )

    if (
        "sampler_time_s" in traj
        and traj["sampler_time_s"] is not None
        and self._otel_sampler_time_s is not None
    ):
      self._otel_sampler_time_s.record(
          float(traj["sampler_time_s"]),
          attributes={
              "worker_id": self.worker_id,
              "task_name": task_name,
          },
      )

    if (
        "env_time" in traj
        and isinstance(traj["env_time"], Mapping)
        and self._otel_env_time_sec is not None
    ):
      env_time = traj["env_time"]
      total_env_s = 0.0
      has_env_op = False
      if "reset_latency" in env_time and env_time["reset_latency"] is not None:
        reset_s = float(env_time["reset_latency"])
        self._otel_env_time_sec.record(
            reset_s,
            attributes={
                "worker_id": self.worker_id,
                "task_name": task_name,
                "op": "reset",
            },
        )
        total_env_s += reset_s
        has_env_op = True
      if "step_latency" in env_time and env_time["step_latency"] is not None:
        for step_val in env_time["step_latency"]:
          step_s = float(step_val)
          self._otel_env_time_sec.record(
              step_s,
              attributes={
                  "worker_id": self.worker_id,
                  "task_name": task_name,
                  "op": "step",
              },
          )
          total_env_s += step_s
          has_env_op = True
      if "close_latency" in env_time and env_time["close_latency"] is not None:
        close_s = float(env_time["close_latency"])
        self._otel_env_time_sec.record(
            close_s,
            attributes={
                "worker_id": self.worker_id,
                "task_name": task_name,
                "op": "close",
            },
        )
        total_env_s += close_s
        has_env_op = True
      if has_env_op:
        self._otel_env_time_sec.record(
            total_env_s,
            attributes={
                "worker_id": self.worker_id,
                "task_name": task_name,
                "op": "total",
            },
        )

    if (
        "reward_time" in traj
        and isinstance(traj["reward_time"], Mapping)
        and "reward_latency" in traj["reward_time"]
        and traj["reward_time"]["reward_latency"] is not None
        and self._otel_reward_time_sec is not None
    ):
      self._otel_reward_time_sec.record(
          float(traj["reward_time"]["reward_latency"]),
          attributes={
              "worker_id": self.worker_id,
              "task_name": task_name,
          },
      )

  def _to_rollout_response(
      self,
      item: Any,
      request_id: str = "",
      prompt_tokens: np.ndarray | None = None,
  ) -> datatypes.RolloutResponse:
    """Converts internal Trajectory or TrajectoryError to wire-safe RolloutResponse."""
    if isinstance(item, trajectory_lib.TrajectoryError):
      metadata = dict(item.metadata or {})
      metadata["prompt_id"] = item.prompt_id
      self._stamp_worker_lineage(metadata)
      self._record_trajectory_error_otel(metadata)
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
      req_id = request_id or item.traj_id
      if (
          prompt_tokens is not None
          and (
              not isinstance(item.traj, Mapping)
              or "prompt_tokens" not in item.traj
              or item.traj["prompt_tokens"] is None
          )
          and (
              "prompt_tokens" not in item.metadata
              or item.metadata["prompt_tokens"] is None
          )
      ):
        item.metadata["prompt_tokens"] = prompt_tokens
      exec_time_s = (
          item.traj["execution_time_s"]
          if isinstance(item.traj, Mapping) and "execution_time_s" in item.traj
          else None
      )
      self._stamp_worker_lineage(item.metadata, execution_time_s=exec_time_s)
      self._record_trajectory_item_otel(item)
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
    if isinstance(requests, datatypes.RolloutRequest):
      self._stamp_worker_lineage(requests.metadata, component="worker.received")
    elif isinstance(requests, Sequence) and not isinstance(
        requests, (str, bytes)
    ):
      if not all(isinstance(req, datatypes.RolloutRequest) for req in requests):
        raise TypeError(
            "generate requires `requests` to be a RolloutRequest or"
            " Sequence[RolloutRequest]."
        )
      for req in requests:
        self._stamp_worker_lineage(req.metadata, component="worker.received")
    else:
      raise TypeError(
          "generate requires `requests` to be a RolloutRequest or"
          " Sequence[RolloutRequest]."
      )

    converted: dict[int, datatypes.RolloutResponse] = {}

    def _convert_once(item: Any) -> datatypes.RolloutResponse:
      item_key = id(item)
      if item_key not in converted:
        converted[item_key] = self._to_rollout_response(item)
      return converted[item_key]

    cb = None
    if on_complete is not None:
      cb = lambda item: on_complete(_convert_once(item))
    res = await self.manager.generate(requests, on_complete=cb)
    if isinstance(res, (list, tuple)):
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
    t0 = time.perf_counter()
    self._sync_pause_start_s = t0
    self._record_round(sync_request, "idle")
    result = await self.manager.pre_weight_sync(sync_request, **kwargs)
    if self._otel_weight_sync_duration is not None:
      self._otel_weight_sync_duration.record(
          float(time.perf_counter() - t0),
          attributes={"worker_id": self.worker_id, "phase": "pre"},
      )
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
    if self._otel_weight_sync_duration is not None:
      self._otel_weight_sync_duration.record(
          float(time.perf_counter() - t0),
          attributes={"worker_id": self.worker_id, "phase": "h2d"},
      )
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
    end_s = time.perf_counter()
    self.state = WorkerState.READY
    if self._otel_weight_sync_duration is not None:
      self._otel_weight_sync_duration.record(
          float(end_s - t0),
          attributes={"worker_id": self.worker_id, "phase": "post"},
      )
      if self._sync_pause_start_s is not None:
        self._otel_weight_sync_duration.record(
            float(end_s - self._sync_pause_start_s),
            attributes={"worker_id": self.worker_id, "phase": "total_paused"},
        )
    self._sync_pause_start_s = None
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
    end_s = time.perf_counter()
    self.state = WorkerState.READY
    if (
        self._otel_weight_sync_duration is not None
        and self._sync_pause_start_s is not None
    ):
      self._otel_weight_sync_duration.record(
          float(end_s - self._sync_pause_start_s),
          attributes={"worker_id": self.worker_id, "phase": "aborted_paused"},
      )
    self._sync_pause_start_s = None
    self._record_round(sync_request, "aborted")
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
