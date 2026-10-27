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

"""Lifecycle driver for bringing a fleet of workers up and down.

Sequences the worker lifecycle (`initialize` -> `compile` -> `start`) across an
entire registry, phase by phase, so every worker finishes one phase before the
next begins (e.g. all compiles happen before any worker starts serving). Also
drives background bring-up (`REGISTERED` -> `PENDING_WEIGHT_SYNC` or `ACTIVE`)
for late-joining rollout workers. Shutdown closes registration, stops background
bring-up, unregisters uninitialized workers, and runs best-effort `stop()`
across every remaining worker.
"""

from collections.abc import Callable, Sequence
import concurrent.futures
import threading
from typing import Any

from absl import logging
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import remote_execution

_INIT_METADATA_KEYS = (
    "trajectory_store_config",
    "drain_timeout_s",
    "partial_rollout",
)


class LifecycleError(RuntimeError):
  """Aggregated failures from a lifecycle phase across multiple workers."""

  def __init__(self, phase: str, failures: list[tuple[str, BaseException]]):
    self.phase = phase
    self.failures = failures
    detail = "; ".join(f"{wid}: {err!r}" for wid, err in failures)
    super().__init__(f"{phase} failed for {len(failures)} worker(s): {detail}")


class LifecycleDriver:
  """Drives a WorkerRegistry through the worker lifecycle phases."""

  def __init__(
      self,
      registry: worker_registry.WorkerRegistry,
      max_workers: int = 32,
      verify_fn: Callable[[Sequence[datatypes.WorkerInfo]], None] | None = None,
  ):
    self._registry = registry
    self._max_workers = max_workers
    self._verify_fn = verify_fn
    self._late_lock = threading.Lock()
    self._late_executor: concurrent.futures.ThreadPoolExecutor | None = None
    self._reconciler_thread: threading.Thread | None = None

  def _run_bring_up_phases(
      self,
      resolved_targets: Sequence[
          tuple[datatypes.WorkerInfo, remote_execution.ActorHandle]
      ],
      dummy_data: Any = None,
  ) -> None:
    """Runs initialize -> verify -> compile -> start, phase by phase."""

    def _init_worker(
        target: tuple[datatypes.WorkerInfo, remote_execution.ActorHandle],
    ) -> None:
      info, handle = target
      resp = handle.submit("initialize")
      metadata = {
          k: resp.metadata[k] for k in _INIT_METADATA_KEYS if k in resp.metadata
      }
      if not self._registry.update_resources(
          info.worker_id, metadata, expected_handle=handle
      ):
        raise RuntimeError(
            f"worker {info.worker_id!r} was unregistered or re-registered"
            " with a different handle during bring-up"
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._max_workers
    ) as pool:
      # TODO(noghabi): Refactor to allow successful workers to proceed to
      # compile/start and potentially retry failed ones, rather than fail-fast
      # on the first error.
      list(pool.map(_init_worker, resolved_targets))
      if self._verify_fn is not None:
        self._verify_fn([info for info, _ in resolved_targets])
      list(
          pool.map(
              lambda target: target[1].submit("compile", dummy_data),
              resolved_targets,
          )
      )
      list(
          pool.map(
              lambda target: target[1].submit("start"),
              resolved_targets,
          )
      )

  def bring_up(
      self,
      dummy_data: Any = None,
  ) -> None:
    """Runs initialize -> verify -> compile -> start across workers, phase by phase.

    Brings up all currently `REGISTERED` workers and promotes them to `ACTIVE`
    on completion. Each phase runs across all target workers before the next
    phase begins. If any worker raises an exception during a phase, the
    exception is re-raised after all workers have attempted the phase,
    preventing subsequent phases from running.

    Args:
      dummy_data: Dummy data each worker uses to synthesize warmup dummies.

    Raises:
      RuntimeError: If a target worker was unregistered or re-registered with a
        different handle during bring-up.
    """
    targets = self._registry.targets(
        statuses=(worker_registry.AdmissionStatus.REGISTERED,)
    )
    self._run_bring_up_phases(targets, dummy_data)
    for info, handle in targets:
      self._registry.set_status(
          info.worker_id,
          worker_registry.AdmissionStatus.ACTIVE,
          expected_handle=handle,
          expected_status=worker_registry.AdmissionStatus.REGISTERED,
      )

  def enable_late_registration(
      self,
      *,
      require_weight_sync: bool = True,
  ) -> None:
    """Starts the background reconciler for late-registering rollout workers.

    Only `Role.ROLLOUT` workers may register after initial cluster bring-up, and
    `RolloutWorker.compile` ignores `dummy_data`. If late-registering workers
    ever require warmup data, inject a per-role provider (e.g.
    `warmup_data_fn(info) -> Any`) rather than caching the initial bring-up
    `dummy_data`.

    Args:
      require_weight_sync: If True (default), late rollout workers transition to
        `PENDING_WEIGHT_SYNC` after bring-up and await weight sync promotion. If
        False (e.g. `weight_sync_mode in (None, "none")`), late rollout workers
        transition directly to `ACTIVE`.
    """
    if self._registry.is_closed:
      return
    with self._late_lock:
      if self._reconciler_thread is not None:
        return
      self._late_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=self._max_workers,
          thread_name_prefix="late-rollout-bringup",
      )
      target_status = (
          worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC
          if require_weight_sync
          else worker_registry.AdmissionStatus.ACTIVE
      )
      self._reconciler_thread = threading.Thread(
          target=self._reconcile_loop,
          args=(target_status,),
          name="late-rollout-reconciler",
          daemon=True,
      )
      self._reconciler_thread.start()

  def _reconcile_loop(
      self, target_status: worker_registry.AdmissionStatus
  ) -> None:
    """Watches `WorkerRegistry` for `REGISTERED` workers and brings them up."""
    rollout_roles = frozenset({datatypes.Role.ROLLOUT.value})
    # Handles already submitted to `_late_executor`. Owned by this thread only.
    # A dispatched worker stays `REGISTERED` until its bring-up finishes, so
    # pruning against the current `REGISTERED` snapshot drops exactly the
    # finished ones (promoted, unregistered, or displaced).
    dispatched: set[remote_execution.ActorHandle] = set()

    def _undispatched() -> list[
        tuple[datatypes.WorkerInfo, remote_execution.ActorHandle]
    ]:
      registered = self._registry.targets(
          statuses=(worker_registry.AdmissionStatus.REGISTERED,)
      )
      dispatched.intersection_update(handle for _, handle in registered)
      return [(i, h) for i, h in registered if h not in dispatched]

    while True:
      self._registry.wait_for_change(
          lambda: self._registry.is_closed or bool(_undispatched())
      )
      if self._registry.is_closed:
        return
      for info, handle in _undispatched():
        # TODO(linchai): Support late registration and background bring-up for
        # non-rollout roles (e.g. trainer/reference) once role-specific warmup
        # data providers and checkpoint recovery are wired up.
        if info.roles != rollout_roles:
          logging.warning(
              "Rejecting late registration for worker %r with non-rollout"
              " roles %s; only %r is permitted after initial bring-up.",
              info.worker_id,
              sorted(info.roles),
              datatypes.Role.ROLLOUT.value,
          )
          self._registry.unregister(info.worker_id, expected_handle=handle)
          continue
        with self._late_lock:
          # `shutdown()` cleared the executor; it unregisters all remaining
          # `REGISTERED` workers after joining this thread.
          if self._late_executor is None:
            return
          self._late_executor.submit(
              self._bring_up_late_worker, info, handle, target_status
          )
          dispatched.add(handle)

  def _bring_up_late_worker(
      self,
      info: datatypes.WorkerInfo,
      handle: remote_execution.ActorHandle,
      target_status: worker_registry.AdmissionStatus,
  ) -> None:
    """Brings up a late rollout worker and promotes it to `target_status`."""
    worker_id = info.worker_id
    if self._registry.is_closed:
      self._registry.unregister(worker_id, expected_handle=handle)
      return
    try:
      self._run_bring_up_phases([(info, handle)])
      if self._registry.set_status(
          worker_id,
          target_status,
          expected_handle=handle,
          expected_status=worker_registry.AdmissionStatus.REGISTERED,
      ):
        logging.info(
            "Late rollout worker %r finished bring-up and entered status %s.",
            worker_id,
            target_status.value,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Late rollout worker %r failed background bring-up; unregistering:"
          " %s",
          worker_id,
          exc,
      )
      self._registry.unregister(worker_id, expected_handle=handle)

  def shutdown(self) -> None:
    """Closes registration, stops late bring-up tasks, and stops all workers."""
    self._registry.close_registration()
    with self._late_lock:
      reconciler = self._reconciler_thread
      executor = self._late_executor
      self._late_executor = None
    if reconciler is not None and reconciler is not threading.current_thread():
      reconciler.join()
    # Registration is closed, so no new `REGISTERED` workers can appear. Close
    # their handles before waiting on the background executor so in-flight
    # initialize/compile/start RPCs abort immediately and queued futures that
    # get cancelled still have their handles closed.
    for info, handle in self._registry.targets(
        statuses=(worker_registry.AdmissionStatus.REGISTERED,)
    ):
      self._registry.unregister(info.worker_id, expected_handle=handle)
    if executor is not None:
      executor.shutdown(wait=True, cancel_futures=True)

    failures: list[tuple[str, BaseException]] = []
    targets = self._registry.targets(statuses=None)

    def _stop_worker(
        info: datatypes.WorkerInfo, handle: remote_execution.ActorHandle
    ) -> None:
      wid = info.worker_id
      try:
        current_handle = self._registry.handle(wid)
      except KeyError:
        current_handle = None
      if current_handle is not handle:
        logging.warning(
            "Worker %r unregistered concurrently, nothing to stop.", wid
        )
        return
      handle.submit("stop")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._max_workers
    ) as pool:
      futures = {
          pool.submit(_stop_worker, info, handle): info.worker_id
          for info, handle in targets
      }
      for future in concurrent.futures.as_completed(futures):
        wid = futures[future]
        try:
          future.result()
        except Exception as err:  # pylint: disable=broad-except
          failures.append((wid, err))
    if failures:
      raise LifecycleError("shutdown", failures)
