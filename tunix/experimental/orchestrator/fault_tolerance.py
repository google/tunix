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

"""Rollout worker fault tolerance, capacity-gated retry queue, and rejoin coordinator."""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Awaitable, Callable, Mapping, Sequence
import dataclasses
import enum
import inspect
import logging
import threading
import time
from typing import Any
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import remote_execution


class NoHealthyRolloutWorkersError(RuntimeError):
  """Raised when 0 healthy rollout workers remain past `max_zero_worker_wait_s`."""


class WorkerHealthStatus(enum.Enum):
  """Health lifecycle status tracked per rollout worker."""

  HEALTHY = "HEALTHY"
  SUSPECT = "SUSPECT"  # Temporarily down (< max_consecutive_heartbeat_misses)
  FAILED = "FAILED"  # Evicted after >= max_consecutive_heartbeat_misses or RPC fault
  PENDING_REJOIN = "PENDING_REJOIN"  # Re-registered; waiting for weight sync


@dataclasses.dataclass
class RolloutFaultToleranceConfig:
  """Configuration for rollout worker fault tolerance and retry scheduling.

  Attributes:
    max_inflight_per_worker: Optional per-worker maximum active trajectory
      requests. When set (`> 0`), `drain_retry_queue` and `select_worker` never
      push more than `max_inflight_per_worker` concurrent trajectories to a
      single worker, holding excess retries in `_retry_queue` until healthy
      workers complete their active requests.
    heartbeat_interval_s: Interval (seconds) between background heartbeat polls.
    max_consecutive_heartbeat_misses: Number of consecutive failed heartbeats
      before a `SUSPECT` ("temporarily down") worker is declared `FAILED` and
      evicted from `RoutingActorPool`.
    max_zero_worker_wait_s: Maximum duration (seconds) to wait when 0 healthy
      rollout workers remain before raising `NoHealthyRolloutWorkersError`.
    max_trajectory_inflight_s: Per-trajectory timeout (seconds) on the
      orchestrator before a wedged trajectory is re-queued.
    max_trajectory_retries: Maximum dispatch attempts per `traj_id` before
      emitting a terminal `TrajectoryItem(status=FAILED)`.
    enable_background_heartbeat: Whether to automatically run the continuous
      background heartbeat loop when the manager starts.
  """

  max_inflight_per_worker: int | None = None
  heartbeat_interval_s: float = 5.0
  max_consecutive_heartbeat_misses: int = 3
  max_zero_worker_wait_s: float = 600.0
  max_trajectory_inflight_s: float = 720.0
  max_trajectory_retries: int = 3
  enable_background_heartbeat: bool = False


@dataclasses.dataclass
class InflightEntry:
  """Tracks a single in-flight trajectory attempt on a rollout worker."""

  traj_id: str
  worker_id: str
  attempt_id: int
  request: datatypes.RolloutRequest
  dispatched_at: float
  attempts: int = 1


class RolloutFaultToleranceManager:
  """Manages rollout worker health, in-flight trajectories, retries, and rejoin."""

  def __init__(
      self,
      rollout_pool: remote_execution.RoutingActorPool,
      *,
      config: RolloutFaultToleranceConfig | None = None,
      registry: worker_registry.WorkerRegistry | None = None,
      monitor: health_monitor.HealthMonitor | None = None,
      clock: Callable[[], float] = time.monotonic,
  ):
    self._rollout_pool = rollout_pool
    self.config = config or RolloutFaultToleranceConfig()
    self._registry = registry
    self._health_monitor = monitor
    self._clock = clock
    self._lock = threading.RLock()

    # Active in-flight trajectory ledger: traj_id -> InflightEntry
    self._inflight: dict[str, InflightEntry] = {}
    # Strictly monotonic dispatch counter per traj_id (used for RC-1 deduplication)
    self._attempt_counters: dict[str, int] = {}
    # Execution failure/timeout counter per traj_id (compared against max_trajectory_retries;
    # NOT incremented when a worker is evicted or when dispatch_task fails pre-execution)
    self._failure_counts: dict[str, int] = collections.defaultdict(int)

    # Capacity-gated retry queue (FIFO deque + deduplicating set of traj_ids)
    self._retry_queue: collections.deque[datatypes.RolloutRequest] = (
        collections.deque()
    )
    self._retry_set: set[str] = set()

    # Worker health status & failure counters
    self._worker_status: dict[str, WorkerHealthStatus] = {}
    self._consecutive_hb_misses: dict[str, int] = collections.defaultdict(int)
    self._worker_timeout_streaks: dict[str, int] = collections.defaultdict(int)
    self._pending_rejoin: dict[str, remote_execution.ActorHandle] = {}
    self._unhealthy_workers: dict[str, remote_execution.ActorHandle] = {}

    for actor in self._rollout_pool.actors:
      wid = self._worker_id_of(actor)
      self._worker_status[wid] = WorkerHealthStatus.HEALTHY

    self._zero_workers_since: float | None = (
        self._clock() if len(self._rollout_pool) == 0 else None
    )

    # Gate preventing dispatches while sync_weights() has closed admission (RC-3)
    self._weight_sync_idle = asyncio.Event()
    self._weight_sync_idle.set()
    self._weight_sync_idle_loop: asyncio.AbstractEventLoop | None = None

    self._heartbeat_task: asyncio.Task[None] | None = None

  @staticmethod
  def _worker_id_of(actor_or_id: str | Any) -> str:
    if isinstance(actor_or_id, str):
      return actor_or_id
    wid = getattr(actor_or_id, "worker_id", None)
    if isinstance(wid, str) and wid:
      return wid
    addr = getattr(actor_or_id, "target_address", None)
    if isinstance(addr, str) and addr:
      return addr
    return f"worker_{id(actor_or_id)}"

  @property
  def retry_queue_size(self) -> int:
    with self._lock:
      return len(self._retry_queue)

  @property
  def pending_rejoin_workers(self) -> dict[str, remote_execution.ActorHandle]:
    with self._lock:
      return dict(self._pending_rejoin)

  def worker_status(
      self, worker_or_id: str | remote_execution.ActorHandle
  ) -> WorkerHealthStatus:
    wid = self._worker_id_of(worker_or_id)
    with self._lock:
      return self._worker_status.get(wid, WorkerHealthStatus.HEALTHY)

  def inflight_count(
      self, worker_or_id: str | remote_execution.ActorHandle | None = None
  ) -> int:
    """Returns active in-flight trajectory count (total or for a specific worker)."""
    with self._lock:
      if worker_or_id is None:
        return len(self._inflight)
      wid = self._worker_id_of(worker_or_id)
      return sum(1 for e in self._inflight.values() if e.worker_id == wid)

  def _ensure_weight_sync_event(self) -> asyncio.Event:
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None
    if loop is not None and self._weight_sync_idle_loop is not loop:
      was_set = self._weight_sync_idle.is_set()
      self._weight_sync_idle = asyncio.Event()
      if was_set:
        self._weight_sync_idle.set()
      self._weight_sync_idle_loop = loop
    return self._weight_sync_idle

  def enter_weight_sync(self) -> None:
    """Closes the dispatch gate at the start of `sync_weights()` (RC-3)."""
    self._ensure_weight_sync_event().clear()

  def exit_weight_sync(self) -> None:
    """Re-opens the dispatch gate at the end of `sync_weights()` (RC-3)."""
    self._ensure_weight_sync_event().set()

  async def wait_for_weight_sync_idle(self) -> None:
    """Waits until no `sync_weights()` operation is actively in progress."""
    await self._ensure_weight_sync_event().wait()

  def check_zero_worker_timeout(self) -> None:
    """Raises `NoHealthyRolloutWorkersError` if 0 workers remain past deadline."""
    with self._lock:
      if len(self._rollout_pool) > 0:
        self._zero_workers_since = None
        return
      if self._zero_workers_since is None:
        self._zero_workers_since = self._clock()
      elapsed = self._clock() - self._zero_workers_since
      if elapsed >= self.config.max_zero_worker_wait_s:
        raise NoHealthyRolloutWorkersError(
            f"0 healthy rollout workers available for {elapsed:.1f}s "
            f"(exceeding max_zero_worker_wait_s={self.config.max_zero_worker_wait_s:.1f}s). "
            f"Pending retries={len(self._retry_queue)}, "
            f"pending rejoin={list(self._pending_rejoin.keys())}."
        )

  def select_worker(
      self,
      req: datatypes.RolloutRequest,
      *,
      extra_loads: Mapping[Any, int] | None = None,
  ) -> remote_execution.ActorHandle | None:
    """Selects an eligible rollout worker respecting `max_inflight_per_worker`, `SUSPECT` status, and HRW affinity."""
    with self._lock:
      actors = self._rollout_pool.actors
      if not actors:
        return None

      max_cap = self.config.max_inflight_per_worker
      has_cap = max_cap is not None and max_cap > 0
      eligible = (
          [a for a in actors if self.inflight_count(a) < max_cap]
          if has_cap
          else list(actors)
      )
      if not eligible:
        return None

      # Prefer HEALTHY workers over SUSPECT ("temporarily down") workers when
      # at least one HEALTHY worker has available capacity.
      healthy_eligible = [
          a
          for a in eligible
          if self.worker_status(a) != WorkerHealthStatus.SUSPECT
      ]
      candidates = healthy_eligible if healthy_eligible else eligible

      def _sort_key(a: remote_execution.ActorHandle) -> tuple[int, int, float]:
        inflight = self.inflight_count(a)
        batch_extra = extra_loads.get(a, 0) if extra_loads is not None else inflight
        if has_cap:
          return (
              inflight,
              batch_extra,
              -self._rollout_pool.hrw_score(a, req.traj_id),
          )
        return (
            batch_extra,
            inflight,
            -self._rollout_pool.hrw_score(a, req.traj_id),
        )

      return min(candidates, key=_sort_key)

  def record_dispatch(
      self,
      req: datatypes.RolloutRequest,
      worker: remote_execution.ActorHandle,
  ) -> InflightEntry:
    """Records `req` as in-flight on `worker` with a monotonically increasing attempt ID."""
    with self._lock:
      wid = self._worker_id_of(worker)
      self._zero_workers_since = None
      self._worker_status.setdefault(wid, WorkerHealthStatus.HEALTHY)
      attempt_id = self._attempt_counters.get(req.traj_id, 0) + 1
      self._attempt_counters[req.traj_id] = attempt_id
      failures_so_far = self._failure_counts.get(req.traj_id, 0)
      entry = InflightEntry(
          traj_id=req.traj_id,
          worker_id=wid,
          attempt_id=attempt_id,
          request=req,
          dispatched_at=self._clock(),
          attempts=failures_so_far + 1,
      )
      self._inflight[req.traj_id] = entry
      self._retry_set.discard(req.traj_id)
      return entry

  def enqueue_retry(
      self,
      req: datatypes.RolloutRequest,
      *,
      front: bool = False,
  ) -> bool:
    """Queues `req` into `_retry_queue` if not already queued."""
    with self._lock:
      if req.traj_id in self._retry_set:
        if front:
          try:
            self._retry_queue.remove(req)
            self._retry_queue.appendleft(req)
          except ValueError:
            pass
        return False
      self._retry_set.add(req.traj_id)
      if front:
        self._retry_queue.appendleft(req)
      else:
        self._retry_queue.append(req)
      return True

  def requeue_inflight_for_worker(
      self,
      worker_or_id: str | remote_execution.ActorHandle,
      *,
      front: bool = False,
  ) -> list[datatypes.RolloutRequest]:
    """Moves in-flight trajectories on `worker_or_id` to `_retry_queue` without evicting the worker (e.g., RC-3 AdmissionClosedError)."""
    with self._lock:
      wid = self._worker_id_of(worker_or_id)
      orphaned_entries = sorted(
          (e for e in self._inflight.values() if e.worker_id == wid),
          key=lambda e: e.dispatched_at,
          reverse=front,
      )
      requeued: list[datatypes.RolloutRequest] = []
      for entry in orphaned_entries:
        self._inflight.pop(entry.traj_id, None)
        self.enqueue_retry(entry.request, front=front)
        requeued.append(entry.request)
      return requeued

  def evict_rollout_worker(
      self,
      worker_or_id: str | remote_execution.ActorHandle,
      *,
      reason: str = "",
  ) -> list[datatypes.RolloutRequest]:
    """Evicts a failed rollout worker from the pool and moves its in-flight requests to `_retry_queue`."""
    with self._lock:
      wid = self._worker_id_of(worker_or_id)
      removed_handle = self._rollout_pool.remove_actor(worker_or_id)
      handle = removed_handle or (
          worker_or_id
          if isinstance(worker_or_id, remote_execution.ActorHandle)
          else None
      )
      if handle is not None:
        self._unhealthy_workers[wid] = handle
      self._worker_status[wid] = WorkerHealthStatus.FAILED

      if self._registry is not None:
        try:
          self._registry.unregister(wid)
        except Exception:  # pylint: disable=broad-exception-caught
          pass

      requeued = self.requeue_inflight_for_worker(wid, front=False)

      if len(self._rollout_pool) == 0 and self._zero_workers_since is None:
        self._zero_workers_since = self._clock()

      logging.warning(
          "Evicting dead rollout worker %r (reason=%s); queueing %d in-flight"
          " trajectories into _retry_queue (remaining_healthy=%d).",
          wid,
          reason or "unspecified",
          len(requeued),
          len(self._rollout_pool),
      )
      return requeued

  async def dispatch_to_worker(
      self,
      worker: remote_execution.ActorHandle,
      req: datatypes.RolloutRequest,
      *,
      dispatch_fn: (
          Callable[
              [remote_execution.ActorHandle, datatypes.RolloutRequest],
              Awaitable[Any] | Any,
          ]
          | None
      ) = None,
      extra_loads: collections.abc.MutableMapping[Any, int] | None = None,
  ) -> bool:
    """Records `req` in-flight on `worker` and invokes `dispatch_fn`, rolling back and evicting `worker` on RPC failure."""
    if dispatch_fn is None:
      dispatch_fn = lambda w, r: w.dispatch_task(
          method_name="generate", requests=[r]
      )
    with self._lock:
      self.record_dispatch(req, worker)
      if isinstance(extra_loads, collections.abc.MutableMapping):
        extra_loads[worker] = extra_loads.get(worker, 0) + 1

    try:
      res = dispatch_fn(worker, req)
      if inspect.isawaitable(res):
        await res
      return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Dispatch of trajectory %r to worker %r failed (%s); evicting worker"
          " and re-queueing at front.",
          req.traj_id,
          self._worker_id_of(worker),
          exc,
      )
      with self._lock:
        if isinstance(extra_loads, collections.abc.MutableMapping):
          extra_loads[worker] = max(0, extra_loads.get(worker, 0) - 1)
        self._inflight.pop(req.traj_id, None)
        self.evict_rollout_worker(worker, reason=str(exc))
        self.enqueue_retry(req, front=True)
      return False

  async def drain_retry_queue(
      self,
      dispatch_fn: (
          Callable[
              [remote_execution.ActorHandle, datatypes.RolloutRequest],
              Awaitable[Any] | Any,
          ]
          | None
      ) = None,
      *,
      extra_loads: collections.abc.MutableMapping[Any, int] | None = None,
  ) -> list[str]:
    """Drains `_retry_queue` into healthy workers up to `max_inflight_per_worker`."""
    await self.wait_for_weight_sync_idle()
    self.check_zero_worker_timeout()

    dispatched_ids: list[str] = []
    while True:
      with self._lock:
        if not self._retry_queue:
          break
        self.check_zero_worker_timeout()
        if len(self._rollout_pool) == 0:
          break
        req = self._retry_queue[0]
        worker = self.select_worker(req, extra_loads=extra_loads)
        if worker is None:
          # All healthy rollout workers are currently at max_inflight_per_worker.
          # Leave remaining retries in _retry_queue until a slot frees up.
          break
        self._retry_queue.popleft()
        self._retry_set.discard(req.traj_id)

      if await self.dispatch_to_worker(
          worker, req, dispatch_fn=dispatch_fn, extra_loads=extra_loads
      ):
        dispatched_ids.append(req.request_id)

    return dispatched_ids

  def _make_terminal_failed_item(
      self, req: datatypes.RolloutRequest, error_msg: str
  ) -> datatypes.TrajectoryItem:
    return datatypes.TrajectoryItem(
        prompt_id=str(req.prompt_id),
        group_index=req.group_index,
        traj={
            "status": datatypes.TrajectoryStatus.FAILED,
            "trajectory_reward": 0.0,
            "prompt_tokens": np.zeros(0, dtype=np.int32),
            "conversation_tokens": np.zeros(0, dtype=np.int32),
            "conversation_masks": np.zeros(0, dtype=np.float32),
            "old_logprobs": np.zeros(0, dtype=np.float32),
        },
        metadata={
            **(req.metadata or {}),
            "error": error_msg,
        },
    )

  def filter_and_complete_responses(
      self,
      worker: remote_execution.ActorHandle,
      items: Sequence[datatypes.TrajectoryItem],
  ) -> list[datatypes.TrajectoryItem]:
    """Filters zombie completions (RC-1) and re-queues transient failures."""
    wid = self._worker_id_of(worker)
    accepted: list[datatypes.TrajectoryItem] = []

    with self._lock:
      for item in items:
        traj_id = getattr(item, "traj_id", None) or datatypes.format_traj_id(
            item.prompt_id, item.group_index
        )
        if traj_id in self._attempt_counters:
          entry = self._inflight.get(traj_id)
          if entry is None or entry.worker_id != wid:
            logging.warning(
                "Dropping stale/duplicate completion for trajectory %r from"
                " worker %r (active_owner=%r).",
                traj_id,
                wid,
                entry.worker_id if entry else None,
            )
            continue

          status = (
              item.traj.get("status")
              if isinstance(item.traj, Mapping)
              else None
          )
          if status == datatypes.TrajectoryStatus.FAILED:
            self._inflight.pop(traj_id, None)
            self._failure_counts[traj_id] += 1
            failures = self._failure_counts[traj_id]
            if failures < self.config.max_trajectory_retries:
              logging.warning(
                  "Trajectory %r returned FAILED on worker %r (failure"
                  " %d/%d); queueing in _retry_queue instead of dropping.",
                  traj_id,
                  wid,
                  failures,
                  self.config.max_trajectory_retries,
              )
              self.enqueue_retry(entry.request)
              continue
            logging.error(
                "Trajectory %r exhausted all %d retries; emitting terminal"
                " FAILED TrajectoryItem.",
                traj_id,
                self.config.max_trajectory_retries,
            )
            accepted.append(item)
            continue

          self._inflight.pop(traj_id, None)
          self._worker_timeout_streaks[wid] = 0
          accepted.append(item)
        else:
          # Direct poll_rollouts() in unit tests without prior dispatch_rollouts
          accepted.append(item)

    return accepted

  def check_inflight_timeouts(self) -> list[datatypes.TrajectoryItem]:
    """Re-queues or fails trajectories exceeding `max_trajectory_inflight_s`."""
    now = self._clock()
    terminal_failures: list[datatypes.TrajectoryItem] = []

    with self._lock:
      timed_out = [
          e
          for e in list(self._inflight.values())
          if (now - e.dispatched_at) >= self.config.max_trajectory_inflight_s
      ]
      workers_to_evict: dict[str, int] = {}
      for entry in timed_out:
        if self._inflight.pop(entry.traj_id, None) is None:
          continue
        wid = entry.worker_id
        self._worker_timeout_streaks[wid] += 1
        self._failure_counts[entry.traj_id] += 1
        failures = self._failure_counts[entry.traj_id]

        if failures < self.config.max_trajectory_retries:
          logging.warning(
              "Trajectory %r timed out after %.1fs on worker %r (failure"
              " %d/%d); re-queueing.",
              entry.traj_id,
              now - entry.dispatched_at,
              wid,
              failures,
              self.config.max_trajectory_retries,
          )
          self.enqueue_retry(entry.request)
        else:
          logging.error(
              "Trajectory %r timed out after %.1fs on worker %r and exhausted"
              " all %d attempts; emitting terminal FAILED item.",
              entry.traj_id,
              now - entry.dispatched_at,
              wid,
              self.config.max_trajectory_retries,
          )
          terminal_failures.append(
              self._make_terminal_failed_item(
                  entry.request,
                  f"Timed out after {now - entry.dispatched_at:.1f}s",
              )
          )

        if self._worker_timeout_streaks[wid] >= 2:
          workers_to_evict[wid] = self._worker_timeout_streaks[wid]

      for wid, streak in workers_to_evict.items():
        self.evict_rollout_worker(
            wid,
            reason=f"{streak} consecutive timed-out trajectories",
        )

    return terminal_failures

  def _apply_heartbeat_outcome(
      self,
      wid: str,
      *,
      ok: bool,
      error_msg: str | None = None,
  ) -> None:
    """Updates worker status (`HEALTHY` <-> `SUSPECT` -> `FAILED`) on a heartbeat outcome."""
    with self._lock:
      if not ok:
        self._consecutive_hb_misses[wid] += 1
        misses = self._consecutive_hb_misses[wid]
        if misses >= self.config.max_consecutive_heartbeat_misses:
          if self._rollout_pool.get_actor(wid) is not None:
            self.evict_rollout_worker(
                wid,
                reason=(
                    f"Failed {misses} consecutive heartbeats"
                    f" (last_error={error_msg})"
                ),
            )
        else:
          self._worker_status[wid] = WorkerHealthStatus.SUSPECT
          logging.warning(
              "Rollout worker %r missed heartbeat (%d/%d); marking SUSPECT"
              " ('temporarily down') before declaring FAILED.",
              wid,
              misses,
              self.config.max_consecutive_heartbeat_misses,
          )
      else:
        if self._consecutive_hb_misses.get(wid, 0) > 0:
          logging.info(
              "Rollout worker %r recovered from SUSPECT to HEALTHY.", wid
          )
        self._consecutive_hb_misses[wid] = 0
        if self._worker_status.get(wid) == WorkerHealthStatus.SUSPECT:
          self._worker_status[wid] = WorkerHealthStatus.HEALTHY

  def poll_health_once(self) -> dict[str, datatypes.HealthReport]:
    """Runs one HealthMonitor poll with SUSPECT grace threshold before eviction."""
    if self._health_monitor is None:
      return {}

    reports = self._health_monitor.poll(isolate_errors=True)
    overdue_ids = {o.worker_id for o in self._health_monitor.overdue()}

    with self._lock:
      for wid, report in reports.items():
        if (
            wid not in self._worker_status
            and self._rollout_pool.get_actor(wid) is None
        ):
          continue
        failed = (
            report.state == datatypes.WorkerState.ERROR or wid in overdue_ids
        )
        self._apply_heartbeat_outcome(
            wid,
            ok=not failed,
            error_msg=report.last_error,
        )

    return reports

  async def run_heartbeat_once(self) -> dict[str, bool]:
    """Probes active and unhealthy rollout workers once and updates `SUSPECT`/`FAILED`/`PENDING_REJOIN` status."""
    outcomes: dict[str, bool] = {}
    if self._health_monitor is not None:
      loop = asyncio.get_running_loop()
      reports = await loop.run_in_executor(None, self.poll_health_once)
      outcomes.update({
          wid: r.state != datatypes.WorkerState.ERROR
          for wid, r in reports.items()
          if wid in self._worker_status
      })
    else:
      actors = self._rollout_pool.actors
      for actor in actors:
        wid = self._worker_id_of(actor)
        try:
          res = actor.asubmit("heartbeat")
          if inspect.isawaitable(res):
            await res
          self._apply_heartbeat_outcome(wid, ok=True)
          outcomes[wid] = True
        except Exception as exc:  # pylint: disable=broad-exception-caught
          self._apply_heartbeat_outcome(wid, ok=False, error_msg=str(exc))
          outcomes[wid] = False

    # Also probe evicted workers in _unhealthy_workers for pull-based reconnect
    with self._lock:
      unhealthy_snapshot = list(self._unhealthy_workers.items())
    for wid, handle in unhealthy_snapshot:
      try:
        res = handle.asubmit("heartbeat")
        if inspect.isawaitable(res):
          await res
        self.on_worker_rejoined(wid, handle)
        outcomes[wid] = True
      except Exception:  # pylint: disable=broad-exception-caught
        outcomes[wid] = False

    return outcomes

  async def _heartbeat_loop(self) -> None:
    while True:
      await asyncio.sleep(self.config.heartbeat_interval_s)
      try:
        await self.run_heartbeat_once()
        self.check_zero_worker_timeout()
      except NoHealthyRolloutWorkersError:
        raise
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning("Background heartbeat poll encountered error: %s", exc)

  def start_heartbeat_loop(self) -> asyncio.Task[None] | None:
    """Starts the continuous background heartbeat loop on the running event loop."""
    if self._heartbeat_task is not None and not self._heartbeat_task.done():
      return self._heartbeat_task
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      return None
    self._heartbeat_task = loop.create_task(self._heartbeat_loop())
    return self._heartbeat_task

  async def stop_heartbeat_loop(self) -> None:
    """Stops the background heartbeat loop if running."""
    if self._heartbeat_task is not None:
      self._heartbeat_task.cancel()
      try:
        await self._heartbeat_task
      except asyncio.CancelledError:
        pass
      self._heartbeat_task = None

  def register_rejoining_worker(
      self,
      handle: remote_execution.ActorHandle,
      *,
      worker_id: str | None = None,
      shim: Any = None,
  ) -> bool:
    """Stages a rejoined rollout worker in `_pending_rejoin` until the next weight sync."""
    wid = worker_id or self._worker_id_of(handle)
    return self.on_worker_rejoined(wid, handle, shim=shim)

  def on_worker_rejoined(
      self,
      worker_id: str,
      handle: remote_execution.ActorHandle,
      *,
      shim: Any = None,
  ) -> bool:
    """Stages a rejoined rollout worker in `_pending_rejoin` (Option 3 hybrid sync).

    Returns:
      True if 0 healthy rollout workers are currently active (signaling that an
      immediate catch-up weight sync should be triggered once the trainer is
      idle), or False if >= 1 healthy rollout workers remain active (so the
      rejoined worker waits in `_pending_rejoin` until the next end-of-step
      `sync_weights()` round).
    """
    with self._lock:
      setattr(handle, "worker_id", worker_id)
      self._pending_rejoin[worker_id] = handle
      self._unhealthy_workers.pop(worker_id, None)
      self._worker_status[worker_id] = WorkerHealthStatus.PENDING_REJOIN
      if self._registry is not None and shim is not None:
        self._registry.register(shim, override=True)
      needs_immediate_sync = len(self._rollout_pool) == 0
      logging.info(
          "Rollout worker %r rejoined (active_healthy=%d);"
          " needs_immediate_sync=%s.",
          worker_id,
          len(self._rollout_pool),
          needs_immediate_sync,
      )
      return needs_immediate_sync

  def on_weight_sync_completed(
      self,
      synced_workers: Mapping[Any, bool] | Sequence[Any] | None = None,
  ) -> list[str]:
    """Promotes rejoined workers from `_pending_rejoin` into `RoutingActorPool` (RC-2)."""
    promoted: list[str] = []
    with self._lock:
      synced_ids: set[str] | None = None
      if isinstance(synced_workers, Mapping):
        synced_ids = {
            self._worker_id_of(k) for k, v in synced_workers.items() if v
        }
      elif (
          isinstance(synced_workers, Sequence)
          and not isinstance(synced_workers, str)
          and len(synced_workers) > 0
      ):
        synced_ids = {
            self._worker_id_of(w)
            for w in synced_workers
            if getattr(w, "phase", "committed") == "committed"
            and not getattr(w, "error", "")
            and not getattr(w, "needs_restart", False)
        }

      for wid, handle in list(self._pending_rejoin.items()):
        was_synced = True if synced_ids is None else (wid in synced_ids)

        if was_synced:
          self._pending_rejoin.pop(wid, None)
          self._rollout_pool.upsert_actor(handle, worker_id=wid)
          self._worker_status[wid] = WorkerHealthStatus.HEALTHY
          self._consecutive_hb_misses[wid] = 0
          self._worker_timeout_streaks[wid] = 0
          self._zero_workers_since = None
          promoted.append(wid)
          logging.info(
              "Promoted rejoined rollout worker %r into active"
              " RoutingActorPool.",
              wid,
          )
    return promoted
