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

"""Trace writer implementations and helper functions.

The Perfetto trace writer here writes long runs as a directory of immutable
shard files (``trace.shard_NNNN.binpb``) plus a single in-flight pending file
(``trace.shard_pending.binpb``):

* Every ``shard_steps`` committed steps observed by the writer, the writer
  seals one shard. A sealed shard is written exactly once via
  ``write_bytes()`` and never rewritten. This is compatible with
  immutable-object stores (GCS) and keeps the full trace history across long
  runs.
* The pending file is rewritten on every flush and contains everything since
  the last seal. It is what users see "live" while a run is in progress.
* Sealed shards' steps are dropped from ``Timeline`` memory immediately after
  sealing, so memory stays bounded to a few shards' worth of spans regardless
  of run length.

A ``trace.manifest.json`` companion file tracks which shards have been sealed
so far. Perfetto's TracePacket format is concatenable, so a complete trace can
be reassembled with::

    cat trace.shard_*.binpb trace.shard_pending.binpb > trace.binpb

or via ``python -m tunix.cli.perfetto_cat <trace_dir>``.

Lane assignment is streaming: each new span on a timeline is greedily placed
on the lowest-index lane whose last span ends at or before this span's begin
time, persisting the per-timeline lane busy-times across shards. Lane indices
(and the perfetto UUIDs derived from them) are therefore stable across shards
-- a span placed on lane 0 of timeline T in shard N stays on lane 0 of
timeline T in every subsequent shard, and the concatenated trace shows a
consistent track layout.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Mapping
import dataclasses
import itertools
import json
import os
import time
from typing import Any

from absl import logging
from etils import epath
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import timeline
from tunix.perf.experimental import timeline_utils

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent


Timeline = timeline.Timeline

_UUID_OFFSET = 100_000  # Offset for lane UUIDs.

_DEFAULT_SHARD_STEPS = 100
_SHARD_STEPS_ENV = "TUNIX_TRACE_SHARD_STEPS"

_SHARD_FILE_FMT = "trace.shard_{index:04d}.binpb"
_PENDING_FILE = "trace.shard_pending.binpb"
_MANIFEST_FILE = "trace.manifest.json"
_MANIFEST_VERSION = 1


def _resolve_shard_steps(shard_steps: int | None) -> int:
  """Resolves the effective shard size from arg + env var override.

  Args:
    shard_steps: The value requested by the caller, or ``None`` to fall back to
      the env var / default.

  Returns:
    A positive integer to use as the number of committed steps per sealed
    shard. The env var ``TUNIX_TRACE_SHARD_STEPS``, when set to a parseable
    positive integer, takes precedence over the caller-provided value to give
    operators a uniform override across all writers in a run.
  """
  env_val = os.environ.get(_SHARD_STEPS_ENV)
  if env_val is not None:
    try:
      parsed = int(env_val)
    except ValueError:
      logging.warning(
          "%s=%r is not a valid integer; ignoring.", _SHARD_STEPS_ENV, env_val
      )
    else:
      if parsed >= 1:
        return parsed
      logging.warning(
          "%s=%d is not >= 1; ignoring.", _SHARD_STEPS_ENV, parsed
      )
  if shard_steps is None:
    return _DEFAULT_SHARD_STEPS
  if shard_steps < 1:
    raise ValueError(
        f"shard_steps must be a positive integer, got {shard_steps!r}."
    )
  return shard_steps


def _create_span_name(name: str, tags: Mapping[str, Any]) -> str:
  """Creates a descriptive name for the span based on its tags."""
  parts = []
  if perf_constants.STEP in tags:
    parts.append(f"step={tags[perf_constants.STEP]}")

  if name == perf_constants.PEFT_TRAIN:
    if perf_constants.ROLE in tags:
      parts.append(f"role={tags[perf_constants.ROLE]}")
    if perf_constants.MINI_BATCH in tags:
      parts.append(f"mini_batch={tags[perf_constants.MINI_BATCH]}")
    if perf_constants.MICRO_BATCH in tags:
      parts.append(f"micro_batch={tags[perf_constants.MICRO_BATCH]}")

  if name in [
      perf_constants.ROLLOUT,
      perf_constants.REFERENCE_INFERENCE,
      perf_constants.OLD_ACTOR_INFERENCE,
      perf_constants.ADVANTAGE_COMPUTATION,
      perf_constants.ENVIRONMENT,
  ]:
    if perf_constants.GROUP_ID in tags:
      parts.append(f"group_id={tags[perf_constants.GROUP_ID]}")

  if name in [perf_constants.ROLLOUT, perf_constants.ENVIRONMENT]:
    if perf_constants.PAIR_INDEX in tags:
      parts.append(f"pair_index={tags[perf_constants.PAIR_INDEX]}")

  if name == perf_constants.QUEUE:
    if perf_constants.NAME in tags:
      parts.append(f"name={tags[perf_constants.NAME]}")

  if parts:
    return f"{name} ({', '.join(parts)})"
  return name


class TraceWriter(abc.ABC):
  """An abstract base class for writing traces."""

  @abc.abstractmethod
  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    """Writes timelines to the trace."""


class NoopTraceWriter(TraceWriter):
  """A no-op trace writer that does nothing."""

  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    del self, timelines  # Unused.


@dataclasses.dataclass
class TrackInfo:
  """Information about a parent track in the perfetto layout.

  Attributes:
    name: The display name of the track.
    uuid: The unique identifier for the track in Perfetto.
  """

  name: str
  uuid: int | None = None


class PerfettoTraceWriter(TraceWriter):
  """A writer for Perfetto trace events.

  Writes long runs as a directory of immutable sharded files plus an in-flight
  pending file. See the module docstring for the file layout and the rationale.

  Constructor parameters:

    trace_dir: Local path or remote URI (e.g. ``gs://...``). The directory is
      created if it does not exist. If creation fails the writer enters a
      no-op mode -- tracing is best-effort and never crashes the application.
    role_to_devices: Optional mapping from role names to the device IDs that
      handle that role. Used to label per-device tracks ("Actor Cluster",
      "Rollout Cluster", etc.).
    shard_steps: Committed steps per sealed shard. ``None`` defers to the
      ``TUNIX_TRACE_SHARD_STEPS`` env var, then to a built-in default. The
      env var, if valid, wins over an explicit caller value to provide a
      uniform per-run override for operators.
  """

  def __init__(
      self,
      trace_dir: str,
      role_to_devices: Mapping[str, Any] | None = None,
      *,
      shard_steps: int | None = None,
  ):
    self._shard_steps = _resolve_shard_steps(shard_steps)
    self._trace_dir_raw = trace_dir
    self._role_to_devices = (
        dict(role_to_devices) if role_to_devices is not None else {}
    )

    # Parent track grouping (e.g. "Host - Main threads", "Actor Cluster"),
    # populated lazily on first observation of each timeline.
    self._track_info: dict[str, TrackInfo] = {}
    self._timeline_tracks: dict[str, str] = {}
    self._timeline_uuids: dict[str, int] = {}

    # Per-timeline streaming lane assignment state. ``_lane_busy_until``
    # records, for each timeline, the end-time of the latest span placed on
    # each lane; it only grows in length as new lanes are needed. New spans
    # reuse the lowest-index lane that is free at their begin-time, so lane
    # indices are stable across seals.
    self._lane_busy_until: dict[str, list[float]] = {}
    self._lane_assignment: dict[str, dict[int, int]] = {}
    self._lane_descriptors_emitted: dict[str, int] = {}

    # Step accounting for shard sealing. ``_unsealed_step_count`` tracks how
    # many committed step boundaries have occurred since the most recent seal,
    # using the maximum delta across timelines as the synchronized count (all
    # timelines created at tracer init commit in lockstep, so this collapses
    # to the per-step delta in practice).
    self._observed_committed_count: dict[str, int] = {}
    self._unsealed_step_count = 0
    self._sealed_step_count = 0
    self._next_shard_index = 1
    self._sealed_shards: list[str] = []

    self._trace_dir: epath.Path | None = None
    self._pending_path: epath.Path | None = None
    self._manifest_path: epath.Path | None = None
    try:
      trace_dir_path = epath.Path(self._trace_dir_raw)
      trace_dir_path.mkdir(parents=True, exist_ok=True)
      self._trace_dir = trace_dir_path
      self._pending_path = trace_dir_path / _PENDING_FILE
      self._manifest_path = trace_dir_path / _MANIFEST_FILE
      logging.info(
          "Initializing perfetto trace writer at: %s (shard_steps=%d)",
          self._trace_dir,
          self._shard_steps,
      )
    except Exception:  # pylint: disable=broad-except
      # Catching broad exceptions to ensure that failures in trace
      # initialization (e.g., due to file system errors, permissions, etc.)
      # do not crash the application. Tracing is best-effort.
      logging.exception(
          "Failed to initialize perfetto trace writer in directory %r."
          " Skipping trace dumping for this run.",
          self._trace_dir_raw,
      )

  @property
  def shard_steps(self) -> int:
    """The number of committed steps per sealed shard."""
    return self._shard_steps

  @property
  def sealed_shards(self) -> list[str]:
    """File names (relative to the trace dir) of all sealed shards so far."""
    return list(self._sealed_shards)

  def _get_device_track_name(self, tl_id: str) -> str | None:
    """Gets a formatted track name for a device timeline.

    Args:
      tl_id: The timeline ID.

    Returns:
      A formatted track name. Returns None if the timeline is a host timeline
      or if the device is not found in the role_to_devices mapping.
    """
    if timeline_utils.is_host_timeline(tl_id):
      return None

    base_tl_id = (
        tl_id[:-6] if timeline_utils.is_queued_timeline(tl_id) else tl_id
    )

    cluster_roles = []

    for role, devices in self._role_to_devices.items():
      for device in devices:
        device_str = timeline_utils.generate_device_timeline_id(device)
        if device_str == base_tl_id and role not in cluster_roles:
          cluster_roles.append(role)

    if cluster_roles:
      camel_roles = [
          "".join(word.capitalize() for word in role.split("_"))
          for role in cluster_roles
      ]
      return f"{', '.join(camel_roles)} Cluster"
    return None

  def _safe_write_bytes(self, path: epath.Path, payload: bytes) -> bool:
    """Writes ``payload`` to ``path``, logging and swallowing failures.

    Returns True on success, False on failure. Tracing is best-effort and
    never raises into the caller.
    """
    try:
      path.write_bytes(payload)
      return True
    except Exception:  # pylint: disable=broad-except
      logging.exception("Failed to write trace bytes to %s", path)
      return False

  def _update_manifest(self) -> None:
    """Updates the on-disk manifest summarizing the trace directory layout."""
    if self._manifest_path is None:
      return
    payload = {
        "version": _MANIFEST_VERSION,
        "shard_steps": self._shard_steps,
        "sealed_shards": list(self._sealed_shards),
        "sealed_step_count": self._sealed_step_count,
        "pending_file": _PENDING_FILE,
    }
    try:
      self._manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          "Failed to write trace manifest at %s", self._manifest_path
      )

  def _init_tracks(self, timelines: Mapping[str, Timeline]) -> None:
    """Populates parent track info for any newly-seen timelines.

    This is idempotent: timelines already registered are skipped. Timelines
    that have never carried any non-empty committed step yet are also skipped
    so we don't allocate a track for a timeline that may turn out to be unused.
    """
    for tl_id in sorted(timelines):
      if tl_id in self._timeline_tracks:
        continue

      tl = timelines[tl_id]
      if not any(tl.committed_steps):
        continue

      if timeline_utils.is_host_timeline(tl_id):
        if timeline_utils.is_timeline_only_of_allowed_type(
            tl, [perf_constants.ROLLOUT], include_cur_step=False
        ):
          self._track_info["host_rollout"] = TrackInfo(
              name="Host - Rollout threads",
              uuid=_UUID_OFFSET + 1,
          )
          self._timeline_tracks[tl_id] = "host_rollout"
        else:
          self._track_info["host_main"] = TrackInfo(
              name="Host - Main threads",
              uuid=_UUID_OFFSET,
          )
          self._timeline_tracks[tl_id] = "host_main"
      else:
        track_name = self._get_device_track_name(tl_id)
        if track_name:
          if not self._track_info.get(track_name):
            self._track_info[track_name] = TrackInfo(
                name=track_name,
                uuid=_UUID_OFFSET + (2 + len(self._track_info)),
            )
          self._timeline_tracks[tl_id] = track_name
        else:
          logging.warning(
              "Failed to get track name for timeline ID: %s", tl_id
          )

  def _detect_and_track_commits(
      self, timelines: Mapping[str, Timeline]
  ) -> None:
    """Detects new commit_step boundaries since the previous flush.

    Per-call delta is computed per timeline and the maximum is taken as the
    synchronized step count, since the tracer commits all timelines together.
    Using the max (rather than min) makes the writer robust to timelines that
    are created or first observed mid-run -- they catch up over subsequent
    commits without blocking sealing of the older timelines.
    """
    max_delta = 0
    for tl_id, tl in timelines.items():
      actual = len(tl.committed_steps)
      observed = self._observed_committed_count.get(tl_id, 0)
      if actual > observed:
        max_delta = max(max_delta, actual - observed)
      # Always refresh; if a timeline drained (actual < observed) we lower it
      # so the next call accounts only for genuinely new commits.
      self._observed_committed_count[tl_id] = actual
    self._unsealed_step_count += max_delta

  def _update_lane_assignments(
      self, timelines: Mapping[str, Timeline]
  ) -> None:
    """Assigns lanes to any spans not yet seen by the writer.

    Iterates each timeline's currently-held committed spans in (begin, id)
    order. Spans already in ``_lane_assignment[tl_id]`` keep their lane;
    fresh spans pick the lowest-index lane whose ``_lane_busy_until`` is at
    or before ``span.begin``, otherwise a new lane is appended.
    """
    for tl_id, tl in timelines.items():
      if not any(tl.committed_steps):
        continue
      assignments = self._lane_assignment.setdefault(tl_id, {})
      busy = self._lane_busy_until.setdefault(tl_id, [])
      # Sort by (begin, id) for deterministic placement; matches the original
      # _assign_lanes behavior.
      all_spans = sorted(
          itertools.chain.from_iterable(
              step.values() for step in tl.committed_steps
          ),
          key=lambda s: (s.begin, s.id),
      )
      for s in all_spans:
        if s.id in assignments:
          continue
        placed = False
        for lane_idx, lane_end in enumerate(busy):
          if lane_end <= s.begin:
            busy[lane_idx] = s.end
            assignments[s.id] = lane_idx
            placed = True
            break
        if not placed:
          assignments[s.id] = len(busy)
          busy.append(s.end)

  def _emit_descriptors(
      self,
      builder: TraceProtoBuilder,
      timelines: Mapping[str, Timeline],
  ) -> None:
    """Emits all track descriptors for the current trace fragment.

    Each shard (sealed or pending) re-emits the full set of descriptors so it
    is viewable standalone. Perfetto tolerates duplicate descriptors by UUID
    when shards are concatenated, so this is safe.
    """
    for track_info in self._track_info.values():
      packet = builder.add_packet()
      packet.track_descriptor.uuid = track_info.uuid
      packet.track_descriptor.name = track_info.name

    for tl_id in sorted(timelines):
      tl = timelines[tl_id]
      if tl_id not in self._timeline_tracks and not any(tl.committed_steps):
        # Timeline never carried data; skip its descriptors.
        continue
      if tl_id not in self._timeline_uuids:
        self._timeline_uuids[tl_id] = _UUID_OFFSET * (
            len(self._timeline_uuids) + 2
        )
      tl_uuid = self._timeline_uuids[tl_id]
      packet = builder.add_packet()
      packet.track_descriptor.uuid = tl_uuid
      packet.track_descriptor.name = tl_id
      if tl_id in self._timeline_tracks:
        parent = self._track_info[self._timeline_tracks[tl_id]]
        packet.track_descriptor.parent_uuid = parent.uuid

      num_lanes = len(self._lane_busy_until.get(tl_id, []))
      if num_lanes > 1:
        for lane_idx in range(num_lanes):
          lane_uuid = tl_uuid + lane_idx + 1
          packet = builder.add_packet()
          packet.track_descriptor.uuid = lane_uuid
          packet.track_descriptor.parent_uuid = tl_uuid
          packet.track_descriptor.name = ""

  def _emit_events_for_steps(
      self,
      builder: TraceProtoBuilder,
      timelines: Mapping[str, Timeline],
      step_slice: slice,
  ) -> None:
    """Emits begin/end events for the given slice of each timeline's steps.

    Args:
      builder: The proto builder to write into.
      timelines: All timelines.
      step_slice: A slice applied to each timeline's ``committed_steps``.
        ``slice(None)`` emits everything currently held.
    """
    events = []
    for tl_id in sorted(timelines):
      tl = timelines[tl_id]
      tl_committed = tl.committed_steps
      if not any(tl_committed):
        continue
      tl_uuid = self._timeline_uuids.get(tl_id)
      if tl_uuid is None:
        continue
      assignments = self._lane_assignment.get(tl_id, {})
      num_lanes = len(self._lane_busy_until.get(tl_id, []))
      steps = tl_committed[step_slice]
      for step in steps:
        for s in step.values():
          lane_idx = assignments.get(s.id, 0)
          lane_uuid = tl_uuid if num_lanes <= 1 else (tl_uuid + lane_idx + 1)
          start_ns = int((s.begin - tl.born) * 1e9)
          events.append({
              "timestamp": start_ns,
              "type": TrackEvent.Type.TYPE_SLICE_BEGIN,
              "uuid": lane_uuid,
              "name": _create_span_name(s.name, s.tags),
          })
          if s.ended:
            end_ns = int((s.end - tl.born) * 1e9)
            events.append({
                "timestamp": end_ns,
                "type": TrackEvent.Type.TYPE_SLICE_END,
                "uuid": lane_uuid,
                "name": None,
            })

    # Perfetto requires strict timestamp ordering within a sequence; END
    # events break ties before BEGIN events so a zero-duration handoff doesn't
    # stretch the previous span.
    events.sort(
        key=lambda e: (
            e["timestamp"],
            0 if e["type"] == TrackEvent.Type.TYPE_SLICE_END else 1,
        )
    )

    for e in events:
      packet = builder.add_packet()
      packet.timestamp = e["timestamp"]
      packet.trusted_packet_sequence_id = 1
      packet.track_event.type = e["type"]
      packet.track_event.track_uuid = e["uuid"]
      if e["name"] is not None:
        packet.track_event.name = e["name"]

  def _build_trace_fragment(
      self,
      timelines: Mapping[str, Timeline],
      step_slice: slice,
  ) -> TraceProtoBuilder:
    """Builds a self-contained perfetto trace covering a slice of each timeline."""
    builder = TraceProtoBuilder()
    self._emit_descriptors(builder, timelines)
    self._emit_events_for_steps(builder, timelines, step_slice)
    return builder

  def _seal_one_shard(self, timelines: Mapping[str, Timeline]) -> None:
    """Seals the next shard from the first ``shard_steps`` of each timeline."""
    if self._trace_dir is None:
      # If the directory failed to initialize, still advance the bookkeeping
      # so we don't accumulate memory forever. Best-effort drop of steps.
      for tl_id, tl in timelines.items():
        tl.drop_oldest_committed_steps(self._shard_steps)
        self._observed_committed_count[tl_id] = len(tl.committed_steps)
      self._unsealed_step_count = max(
          0, self._unsealed_step_count - self._shard_steps
      )
      return

    shard_name = _SHARD_FILE_FMT.format(index=self._next_shard_index)
    shard_path = self._trace_dir / shard_name
    builder = self._build_trace_fragment(
        timelines, step_slice=slice(0, self._shard_steps)
    )
    ok = self._safe_write_bytes(shard_path, builder.serialize())
    if not ok:
      # Don't drop spans we couldn't write; let the next flush retry.
      return

    # Drop the sealed steps from each timeline's memory and from the lane
    # assignment cache so we don't hold them forever.
    for tl_id, tl in timelines.items():
      dropped = tl.drop_oldest_committed_steps(self._shard_steps)
      if dropped:
        assignments = self._lane_assignment.get(tl_id)
        if assignments is not None:
          for step in dropped:
            for sid in step:
              assignments.pop(sid, None)
      self._observed_committed_count[tl_id] = len(tl.committed_steps)

    self._sealed_shards.append(shard_name)
    self._next_shard_index += 1
    self._sealed_step_count += self._shard_steps
    self._unsealed_step_count -= self._shard_steps
    self._update_manifest()

  def _write_pending(self, timelines: Mapping[str, Timeline]) -> None:
    """Writes (or removes) the in-flight pending shard file."""
    if self._pending_path is None:
      return
    has_pending = any(any(tl.committed_steps) for tl in timelines.values())
    if not has_pending:
      # Nothing unsealed; remove any stale pending file so concatenating
      # ``trace.shard_*.binpb`` is a complete trace.
      try:
        if self._pending_path.exists():
          self._pending_path.unlink()
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "Failed to remove stale pending trace at %s", self._pending_path
        )
      return
    builder = self._build_trace_fragment(timelines, step_slice=slice(None))
    self._safe_write_bytes(self._pending_path, builder.serialize())

  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    """Writes timelines to the trace directory, sealing shards as needed."""
    if not timelines:
      return
    if (
        not any(any(tl.committed_steps) for tl in timelines.values())
        and self._sealed_step_count == 0
    ):
      return

    self._init_tracks(timelines)
    self._detect_and_track_commits(timelines)
    self._update_lane_assignments(timelines)

    # Seal as many full shards as we have data for. A long pause between
    # flushes could produce more than one shard's worth of unsealed data; we
    # drain it all rather than queueing seals behind future flushes.
    while self._unsealed_step_count >= self._shard_steps:
      sealed_index_before = self._next_shard_index
      self._seal_one_shard(timelines)
      if self._next_shard_index == sealed_index_before:
        # Seal did not advance (e.g. write failure or no trace dir). Avoid an
        # infinite loop.
        break

    self._write_pending(timelines)


# Backwards-compatible alias retained for callers that imported the
# internal helper directly. Equivalent to one-shot lane assignment over a
# bag of spans; the writer itself now uses the streaming assignment on
# ``PerfettoTraceWriter``.
def _assign_lanes(
    spans: Iterable[timeline.Span],
) -> tuple[Mapping[int, int], int]:
  """Assigns lanes to spans to handle overlaps (one-shot, no streaming state).

  Perfetto requires spans on the same track to be strictly non-overlapping.
  This helper assigns a lane index to each span such that spans in the same
  lane do not overlap. It is left here for callers that previously used it
  directly; the trace writer uses streaming lane assignment internally.

  Args:
    spans: An iterable of spans to assign to lanes.

  Returns:
    A tuple ``(lane_by_span_id, num_lanes)``.
  """
  sorted_spans = sorted(spans, key=lambda s: (s.begin, s.id))
  lanes_end_times: list[float] = []
  lane_by_span_id: dict[int, int] = {}

  for s in sorted_spans:
    placed = False
    for lane_idx, lane_end in enumerate(lanes_end_times):
      if lane_end <= s.begin:
        lanes_end_times[lane_idx] = s.end
        lane_by_span_id[s.id] = lane_idx
        placed = True
        break
    if not placed:
      lane_by_span_id[s.id] = len(lanes_end_times)
      lanes_end_times.append(s.end)

  return lane_by_span_id, len(lanes_end_times)


# Module-level sentinel used by tests that previously patched ``TraceProtoBuilder``.
# Kept as a public attribute so existing test mocks keep working.
__all__ = [
    "PerfettoTraceWriter",
    "NoopTraceWriter",
    "TrackInfo",
    "TraceWriter",
    "TraceProtoBuilder",
    "_create_span_name",
    "_assign_lanes",
    "TrackDescriptor",
    "TrackEvent",
]
