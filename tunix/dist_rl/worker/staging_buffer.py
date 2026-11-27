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

"""Bounded RolloutWorker staging buffer for MoE routed_experts arrays."""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Callable, Sequence
import dataclasses
import os
import pathlib
import secrets
import threading
import time
from typing import Literal
import weakref

from absl import logging
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import timeouts

_MIN_DEFAULT_STAGING_BYTES = 256 * 1024 * 1024  # 256 MiB
_MAX_DEFAULT_STAGING_BYTES = 32 * 1024 * 1024 * 1024  # 32 GiB
_DEFAULT_FALLBACK_STAGING_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_STALL_TIMEOUT_S = 2.0 * timeouts.DEFAULT_PRE_WEIGHT_SYNC_TIMEOUT_S

_CGROUP_LIMIT_PATHS: tuple[pathlib.Path, ...] = (
    pathlib.Path("/sys/fs/cgroup/memory.max"),
    pathlib.Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)

_ReservationState = Literal[
    "pending", "granted", "committed", "cancelled", "revoked"
]


def _read_cgroup_memory_limit() -> int | None:
  """Returns positive finite cgroup memory limit in bytes, or None."""
  for path in _CGROUP_LIMIT_PATHS:
    try:
      if not path.is_file():
        continue
      raw = path.read_text(encoding="utf-8").strip()
      if not raw or raw == "max":
        continue
      limit = int(raw)
    except (OSError, ValueError):
      continue
    # Ignore kernel sentinel values (~2^63 - 4096 on cgroup v1).
    if 0 < limit < (1 << 60):
      return limit
  return None


def default_staging_max_bytes() -> int:
  """Computes default per-worker staging cap (25% of available RAM, clamped)."""
  candidates: list[int] = []
  try:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
    if page_size > 0 and phys_pages > 0:
      candidates.append(page_size * phys_pages)
  except (AttributeError, OSError, ValueError):
    pass
  cgroup_limit = _read_cgroup_memory_limit()
  if cgroup_limit is not None:
    candidates.append(cgroup_limit)
  if not candidates:
    return _DEFAULT_FALLBACK_STAGING_BYTES
  quarter = min(candidates) // 4
  return max(
      _MIN_DEFAULT_STAGING_BYTES,
      min(_MAX_DEFAULT_STAGING_BYTES, quarter),
  )


def _require_non_bool_int(value: int, name: str, *, min_value: int = 0) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < min_value:
    raise ValueError(f"{name} must be an int >= {min_value}, got {value!r}.")
  return value


@dataclasses.dataclass(frozen=True)
class _StagedEntry:
  arr: np.ndarray
  nbytes: int
  staged_policy_version: int
  traj_id: str


@dataclasses.dataclass(eq=False)
class _ReservationSlot:
  state: _ReservationState
  clear_epoch: int
  task_attached: bool = False


class StagingReservation:
  """Single-use worst-case byte reservation inside `RolloutStagingBuffer`."""

  def __init__(
      self,
      buffer: RolloutStagingBuffer,
      nbytes: int,
      *,
      clear_epoch: int,
  ) -> None:
    self._buffer = buffer
    self.nbytes = nbytes
    self.clear_epoch = clear_epoch

  @property
  def is_revoked(self) -> bool:
    return self._buffer.is_reservation_revoked(self)

  def attach_to_task(self) -> bool:
    """Marks reservation owned by a tracked task; returns False if revoked/clearing."""
    return self._buffer.attach_reservation(self)

  def cancel(self) -> None:
    """Idempotently releases an uncommitted reservation."""
    self._buffer.cancel_reservation(self)


@dataclasses.dataclass(eq=False)
class _Waiter:
  nbytes: int
  loop: asyncio.AbstractEventLoop
  future: asyncio.Future[StagingReservation]
  reservation: StagingReservation
  slot: _ReservationSlot


class RolloutStagingBuffer:
  """Thread-safe, FIFO-admitted staging buffer for RolloutWorker routing arrays."""

  def __init__(
      self,
      worker_id: str,
      instance_id: str,
      *,
      max_bytes: int | None = None,
      stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
      incarnation: int | None = None,
  ) -> None:
    if not isinstance(worker_id, str) or not worker_id:
      raise ValueError("worker_id must be a non-empty str.")
    if not isinstance(instance_id, str) or not instance_id:
      raise ValueError("instance_id must be a non-empty str.")
    resolved_max = (
        default_staging_max_bytes()
        if max_bytes is None
        else _require_non_bool_int(max_bytes, "max_bytes", min_value=1)
    )
    self._worker_id = worker_id
    self._instance_id = instance_id
    self._max_bytes = resolved_max
    self._stall_timeout_s = timeouts.validate_positive_finite(
        stall_timeout_s, "stall_timeout_s"
    )
    if incarnation is None:
      self._incarnation = secrets.randbits(31)
    else:
      inc = _require_non_bool_int(incarnation, "incarnation", min_value=0)
      if inc >= (1 << 31):
        raise ValueError(f"incarnation must fit in 31 bits, got {inc:#x}.")
      self._incarnation = inc

    self._lock = threading.Lock()
    self._next_seq: int = 1
    self._used_bytes: int = 0
    self._reserved_bytes: int = 0
    self._current_policy_version: int = 0
    self._zombie_evictions_total: int = 0
    self._clear_epoch: int = 0
    self._is_clearing: bool = False
    self._closed: bool = False
    self._progress_seq: int = 0
    self._entries: dict[int, _StagedEntry] = {}
    self._waiters: collections.deque[_Waiter] = collections.deque()
    self._granted_reservations: set[StagingReservation] = set()
    self._slots: weakref.WeakKeyDictionary[
        StagingReservation, _ReservationSlot
    ] = weakref.WeakKeyDictionary()

  @property
  def instance_id(self) -> str:
    return self._instance_id

  @property
  def max_bytes(self) -> int:
    return self._max_bytes

  @property
  def used_bytes(self) -> int:
    with self._lock:
      return self._used_bytes

  @property
  def reserved_bytes(self) -> int:
    with self._lock:
      return self._reserved_bytes

  @property
  def zombie_evictions_total(self) -> int:
    with self._lock:
      return self._zombie_evictions_total

  @property
  def is_clearing(self) -> bool:
    with self._lock:
      return self._is_clearing

  def metrics(self) -> dict[str, int | float]:
    with self._lock:
      return {
          "staging/used_bytes": self._used_bytes,
          "staging/reserved_bytes": self._reserved_bytes,
          "staging/zombie_evictions_total": self._zombie_evictions_total,
      }

  def _deliver_grant(self, waiter: _Waiter) -> None:
    """Resolves `waiter.future` on its loop, or refunds/errors on cancel/clear."""
    if waiter.future.done():
      self._abort_waiter(waiter)
      return
    with self._lock:
      state = waiter.slot.state
    if state == "granted":
      waiter.future.set_result(waiter.reservation)
    elif state == "revoked":
      waiter.future.set_exception(
          RuntimeError("RolloutStagingBuffer is clearing staged entries.")
      )

  @staticmethod
  def _deliver_clear_error(waiter: _Waiter) -> None:
    if not waiter.future.done():
      waiter.future.set_exception(
          RuntimeError("RolloutStagingBuffer is clearing staged entries.")
      )

  def _wake_waiters(self, waiters: Sequence[_Waiter]) -> None:
    for waiter in waiters:
      waiter.loop.call_soon_threadsafe(self._deliver_grant, waiter)

  def _pop_grantable_waiters_locked(self) -> list[_Waiter]:
    """Grants queued waiters strictly in head-of-line FIFO arrival order."""
    if self._is_clearing:
      return []
    granted: list[_Waiter] = []
    while self._waiters:
      head = self._waiters[0]
      if (
          self._used_bytes + self._reserved_bytes + head.nbytes
          > self._max_bytes
      ):
        break
      self._waiters.popleft()
      head.slot.state = "granted"
      self._granted_reservations.add(head.reservation)
      self._reserved_bytes += head.nbytes
      granted.append(head)
    return granted

  def is_reservation_revoked(self, reservation: StagingReservation) -> bool:
    """Returns True if `reservation` was revoked by `begin_clear()`."""
    with self._lock:
      slot = self._slots.get(reservation)
      if slot is None:
        raise ValueError("StagingReservation belongs to a different buffer.")
      return slot.state == "revoked"

  def attach_reservation(self, reservation: StagingReservation) -> bool:
    """Marks `reservation` attached to a tracked task if still committable."""
    with self._lock:
      slot = self._slots.get(reservation)
      if slot is None:
        raise ValueError("StagingReservation belongs to a different buffer.")
      if (
          self._is_clearing
          or slot.clear_epoch != self._clear_epoch
          or slot.state != "granted"
      ):
        return False
      slot.task_attached = True
      return True

  def _refund_granted_locked(
      self, reservation: StagingReservation, slot: _ReservationSlot
  ) -> list[_Waiter]:
    """Refunds a granted reservation, marks it cancelled, and pops waiters."""
    self._granted_reservations.discard(reservation)
    self._reserved_bytes -= reservation.nbytes
    slot.state = "cancelled"
    self._progress_seq += 1
    return self._pop_grantable_waiters_locked()

  def cancel_reservation(self, reservation: StagingReservation) -> None:
    """Idempotently releases an uncommitted `reservation` and wakes waiters."""
    to_wake: list[_Waiter] = []
    with self._lock:
      slot = self._slots.get(reservation)
      if slot is None:
        raise ValueError("StagingReservation belongs to a different buffer.")
      if slot.state == "granted":
        to_wake = self._refund_granted_locked(reservation, slot)
    self._wake_waiters(to_wake)

  def _abort_waiter(self, waiter: _Waiter) -> None:
    """Removes a pending waiter or refunds an unclaimed granted waiter."""
    to_wake: list[_Waiter] = []
    with self._lock:
      slot = waiter.slot
      if slot.state == "pending":
        self._waiters.remove(waiter)
        slot.state = "cancelled"
        to_wake = self._pop_grantable_waiters_locked()
      elif slot.state == "granted":
        to_wake = self._refund_granted_locked(waiter.reservation, slot)
    self._wake_waiters(to_wake)

  async def _await_waiter_with_watchdog(
      self,
      waiter: _Waiter,
      is_paused_fn: Callable[[], bool] | None,
  ) -> StagingReservation:
    """Awaits `waiter.future` with a sync-frozen no-progress watchdog."""
    poll_s = min(1.0, self._stall_timeout_s / 4.0)
    stalled_s = 0.0
    with self._lock:
      last_seq = self._progress_seq
    try:
      while not waiter.future.done():
        t0 = time.monotonic()
        await asyncio.wait({waiter.future}, timeout=poll_s)
        if waiter.future.done():
          break
        with self._lock:
          cur_seq = self._progress_seq
        paused = bool(is_paused_fn()) if is_paused_fn is not None else False
        if cur_seq != last_seq or paused:
          stalled_s = 0.0
          last_seq = cur_seq
          continue
        stalled_s += time.monotonic() - t0
        if stalled_s >= self._stall_timeout_s:
          raise TimeoutError(
              f"RolloutStagingBuffer [{self._worker_id}] stalled for"
              f" {stalled_s:.2f}s waiting for {waiter.nbytes} bytes"
              f" (used={self.used_bytes}, reserved={self.reserved_bytes},"
              f" max={self._max_bytes})."
          )
      return waiter.future.result()
    except BaseException:
      self._abort_waiter(waiter)
      raise

  async def reserve(
      self,
      nbytes: int,
      *,
      is_paused_fn: Callable[[], bool] | None = None,
  ) -> StagingReservation:
    """Reserves `nbytes` in strict FIFO arrival order before rollout admission."""
    _require_non_bool_int(nbytes, "nbytes", min_value=0)
    if nbytes > self._max_bytes:
      raise ValueError(
          f"Requested staging reservation ({nbytes} bytes) exceeds worker"
          f" staging max_bytes ({self._max_bytes} bytes)."
      )
    loop = asyncio.get_running_loop()
    with self._lock:
      if self._closed or self._is_clearing:
        raise RuntimeError(
            f"RolloutStagingBuffer [{self._worker_id}] is clearing or closed."
        )
      epoch = self._clear_epoch
      if (
          not self._waiters
          and self._used_bytes + self._reserved_bytes + nbytes
          <= self._max_bytes
      ):
        res = StagingReservation(self, nbytes, clear_epoch=epoch)
        self._slots[res] = _ReservationSlot(state="granted", clear_epoch=epoch)
        self._granted_reservations.add(res)
        self._reserved_bytes += nbytes
        return res
      res = StagingReservation(self, nbytes, clear_epoch=epoch)
      slot = _ReservationSlot(state="pending", clear_epoch=epoch)
      self._slots[res] = slot
      waiter = _Waiter(
          nbytes=nbytes,
          loop=loop,
          future=loop.create_future(),
          reservation=res,
          slot=slot,
      )
      self._waiters.append(waiter)
    return await self._await_waiter_with_watchdog(waiter, is_paused_fn)

  def commit(
      self,
      reservation: StagingReservation,
      arr: np.ndarray,
      *,
      traj_id: str,
  ) -> datatypes.RoutedExpertsRef:
    """Commits an encoded routing array against `reservation` and returns a ref."""
    if not isinstance(arr, np.ndarray):
      raise TypeError(
          f"Expected np.ndarray for staged array, got {type(arr)!r}."
      )
    contiguous = np.ascontiguousarray(arr)
    contiguous.setflags(write=False)
    actual_bytes = int(contiguous.nbytes)

    err: Exception | None = None
    ref_id = 0
    to_wake: list[_Waiter] = []
    with self._lock:
      slot = self._slots.get(reservation)
      if slot is None:
        raise ValueError("StagingReservation belongs to a different buffer.")
      if (
          self._is_clearing
          or slot.clear_epoch != self._clear_epoch
          or slot.state != "granted"
      ):
        prior_state = slot.state
        if prior_state == "granted":
          to_wake = self._refund_granted_locked(reservation, slot)
        err = RuntimeError(
            f"Cannot commit reservation in state {prior_state!r}"
            f" (epoch={slot.clear_epoch}, current_epoch={self._clear_epoch},"
            f" clearing={self._is_clearing})."
        )
      else:
        self._granted_reservations.discard(reservation)
        self._reserved_bytes -= reservation.nbytes
        self._progress_seq += 1
        if actual_bytes > reservation.nbytes:
          slot.state = "cancelled"
          err = ValueError(
              f"Encoded routed_experts ({actual_bytes} bytes) exceeds reserved"
              f" worst_case_bytes ({reservation.nbytes} bytes) for {traj_id}."
          )
        elif self._next_seq >= (1 << 32):
          slot.state = "cancelled"
          err = OverflowError("RolloutStagingBuffer 32-bit sequence exhausted.")
        else:
          seq = self._next_seq
          self._next_seq += 1
          ref_id = (self._incarnation << 32) | seq
          slot.state = "committed"
          self._used_bytes += actual_bytes
          self._entries[ref_id] = _StagedEntry(
              arr=contiguous,
              nbytes=actual_bytes,
              staged_policy_version=self._current_policy_version,
              traj_id=traj_id,
          )
        to_wake = self._pop_grantable_waiters_locked()
    self._wake_waiters(to_wake)
    if err is not None:
      raise err
    return datatypes.RoutedExpertsRef(
        worker_id=self._worker_id,
        instance_id=self._instance_id,
        ref_id=ref_id,
    )

  def stage_trajectory_item(
      self,
      reservation: StagingReservation,
      trajectory: datatypes.TrajectoryItem,
      *,
      traj_id: str,
  ) -> None:
    """Commits `trajectory`'s `routed_experts` or cancels on invalid items."""
    if not isinstance(trajectory, datatypes.TrajectoryItem):
      reservation.cancel()
      raise TypeError(
          f"Expected TrajectoryItem, got {type(trajectory).__name__}."
      )
    if isinstance(trajectory.metadata, dict):
      trajectory.metadata.pop("routed_experts", None)
    if not trajectory.is_valid or not isinstance(trajectory.traj, dict):
      if isinstance(trajectory.traj, dict):
        trajectory.traj.pop("routed_experts", None)
      reservation.cancel()
      return
    routed = trajectory.traj.pop("routed_experts", None)
    if routed is None:
      reservation.cancel()
      raise ValueError(
          f"RolloutRequest(stage_routed_experts=True) for {traj_id} produced"
          " no routed_experts."
      )
    ref = self.commit(reservation, np.asarray(routed), traj_id=traj_id)
    try:
      trajectory.routed_experts_ref = ref
    except BaseException:
      self.release_staged([ref.ref_id])
      raise

  def fetch_staged(self, ref_ids: Sequence[int]) -> dict[int, np.ndarray]:
    """Returns read-only staged arrays for `ref_ids`, failing fast if missing."""
    validated_ids = [
        _require_non_bool_int(raw_id, "ref_id", min_value=0)
        for raw_id in ref_ids
    ]
    out: dict[int, np.ndarray] = {}
    with self._lock:
      for rid in validated_ids:
        if (rid >> 32) != self._incarnation:
          raise KeyError(
              f"Stale worker incarnation in ref_id={rid:#x} on"
              f" {self._worker_id!r} (current={self._incarnation:#x})."
          )
        entry = self._entries.get(rid)
        if entry is None:
          raise KeyError(
              f"Missing or already-released ref_id={rid:#x} on"
              f" {self._worker_id!r}."
          )
        out[rid] = entry.arr
    return out

  def release_staged(self, ref_ids: Sequence[int]) -> int:
    """Idempotently frees `ref_ids` and returns total bytes reclaimed."""
    validated_ids = [
        _require_non_bool_int(raw_id, "ref_id", min_value=0)
        for raw_id in ref_ids
    ]
    freed_bytes = 0
    to_wake: list[_Waiter] = []
    with self._lock:
      for rid in validated_ids:
        if (rid >> 32) != self._incarnation:
          continue
        entry = self._entries.pop(rid, None)
        if entry is not None:
          self._used_bytes -= entry.nbytes
          freed_bytes += entry.nbytes
      if freed_bytes > 0:
        self._progress_seq += 1
        to_wake = self._pop_grantable_waiters_locked()
    self._wake_waiters(to_wake)
    return freed_bytes

  def on_post_weight_sync(
      self,
      new_policy_version: int,
      *,
      min_retained_policy_version: int | None = None,
  ) -> int:
    """Updates policy version and evicts entries older than `min_retained_policy_version`."""
    if isinstance(new_policy_version, bool) or not isinstance(
        new_policy_version, int
    ):
      raise TypeError(
          "new_policy_version must be an int (not bool), got"
          f" {new_policy_version!r}."
      )
    if new_policy_version < 0:
      raise ValueError(
          f"new_policy_version must be >= 0, got {new_policy_version}."
      )
    if min_retained_policy_version is not None and (
        isinstance(min_retained_policy_version, bool)
        or not isinstance(min_retained_policy_version, int)
    ):
      raise TypeError(
          "min_retained_policy_version must be an int or None, got"
          f" {min_retained_policy_version!r}."
      )

    evicted: list[tuple[int, _StagedEntry]] = []
    to_wake: list[_Waiter] = []
    with self._lock:
      self._current_policy_version = new_policy_version
      self._progress_seq += 1
      if min_retained_policy_version is not None:
        stale_ids = [
            rid
            for rid, entry in self._entries.items()
            if entry.staged_policy_version < min_retained_policy_version
        ]
        for rid in stale_ids:
          entry = self._entries.pop(rid)
          self._used_bytes -= entry.nbytes
          self._zombie_evictions_total += 1
          evicted.append((rid, entry))
      if evicted:
        to_wake = self._pop_grantable_waiters_locked()

    for rid, entry in evicted:
      logging.warning(
          "[RolloutStagingBuffer %s] Evicted zombie staged routed_experts"
          " ref_id=%#x traj_id=%s staged_policy_version=%d <"
          " min_retained_policy_version=%s (%d bytes).",
          self._worker_id,
          rid,
          entry.traj_id,
          entry.staged_policy_version,
          min_retained_policy_version,
          entry.nbytes,
      )
    self._wake_waiters(to_wake)
    return len(evicted)

  def begin_clear(self) -> None:
    """Closes reservation gate, cancels waiters, and revokes pre-task grants."""
    cancelled_waiters: list[_Waiter] = []
    with self._lock:
      self._is_clearing = True
      self._clear_epoch += 1
      while self._waiters:
        waiter = self._waiters.popleft()
        waiter.slot.state = "revoked"
        cancelled_waiters.append(waiter)
      for res in list(self._granted_reservations):
        slot = self._slots.get(res)
        if slot is not None and not slot.task_attached:
          self._granted_reservations.discard(res)
          self._reserved_bytes -= res.nbytes
          slot.state = "revoked"
      self._progress_seq += 1
    for waiter in cancelled_waiters:
      waiter.loop.call_soon_threadsafe(self._deliver_clear_error, waiter)

  def abort_clear(self) -> None:
    """Reopens the reservation gate if `clear_staged()` is cancelled mid-drain."""
    with self._lock:
      self._entries.clear()
      self._used_bytes = 0
      self._is_clearing = self._closed
      self._progress_seq += 1

  def finish_clear(self) -> int:
    """Drops all staged entries after tasks drain and reopens the buffer."""
    with self._lock:
      if self._reserved_bytes != 0 or self._granted_reservations:
        raise RuntimeError(
            f"RolloutStagingBuffer [{self._worker_id}] finish_clear called"
            f" with active reservations: reserved_bytes={self._reserved_bytes},"
            f" count={len(self._granted_reservations)}."
        )
      cleared_count = len(self._entries)
      self._entries.clear()
      self._used_bytes = 0
      self._is_clearing = self._closed
      self._progress_seq += 1
      return cleared_count

  def close(self) -> None:
    """Closes the buffer, wakes blocked waiters, and drops staged entries."""
    self.begin_clear()
    with self._lock:
      self._closed = True
      self._entries.clear()
      self._used_bytes = 0
      self._progress_seq += 1
