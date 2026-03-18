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

"""Utilities for working with timelines."""

from __future__ import annotations

from collections.abc import Sequence
import threading
from typing import Any

from absl import logging
import numpy as np
from tunix.perf.experimental import constants
from tunix.perf.experimental import timeline


JaxDevice = Any


# Utility functions for handling timeline IDs.


def generate_host_timeline_id() -> str:
  """Generates a string ID for a host timeline.

  Returns:
    A string ID formatted as 'host-<thread_id>'.
  """
  return f"host-{threading.get_ident()}"


def is_host_timeline(tl_id: str) -> bool:
  """Checks if the timeline ID corresponds to a host timeline.

  Args:
    tl_id: The timeline ID to check.

  Returns:
    True if the timeline ID starts with 'host-', False otherwise.
  """
  return tl_id.startswith("host-")


def generate_device_timeline_id(device_id: str | JaxDevice) -> str:
  """Generates a string ID for a device timeline.

  Args:
    device_id: A string ID or a JAX device object.

  Returns:
    A string representation of the device ID. For a JAX device object, it will
    be the platform name followed by the device ID, e.g., "tpu0".

  Raises:
    ValueError: If the input device_id type is not supported. Only string and
      JAX device objects (with platform and id attributes) are supported.
  """

  if isinstance(device_id, str):
    return device_id
  elif hasattr(device_id, "platform") and hasattr(device_id, "id"):
    # if it's a JAX device object, convert to string
    return f"{device_id.platform}{device_id.id}"
  else:
    raise ValueError(f"Unsupported id type: {type(device_id)}")


def generate_device_timeline_ids(
    devices: Sequence[str | JaxDevice] | np.ndarray | None,
) -> Sequence[str]:
  """Generates a list of string IDs for a list of devices.

  Args:
    devices: A sequence of devices, a numpy array of devices, or None.
      Devices can be represented as strings or JAX device objects.

  Returns:
    A list of string representations of the device IDs.
  """
  if devices is None:
    return []
  if isinstance(devices, np.ndarray):
    device_list = devices.flatten().tolist()
  else:
    device_list = devices
  return [generate_device_timeline_id(device) for device in device_list]


# Utility functions for handling timelines.


def is_timeline_only_of_allowed_type(
    tl: timeline.Timeline, allowed_span_names: Sequence[str]
) -> bool:
  """Checks if all spans in a timeline are of allowed types.

  Args:
    tl: The timeline to check.
    allowed_span_names: A sequence of allowed span names.

  Returns:
    True if the timeline has spans and all spans have a name in
    `allowed_span_names`, False otherwise.
  """
  if not tl.spans:
    return False
  return all(span.name in allowed_span_names for span in tl.spans.values())


def deepcopy_timeline(tl: timeline.Timeline) -> timeline.Timeline:
  """Creates a deep copy of the timeline and its spans.

  Args:
    tl: The timeline to copy.

  Returns:
    A new timeline with deeply copied spans.
  """
  # pylint: disable=protected-access
  with tl._lock:
    new_tl = tl.__class__(tl.id, tl.born)
    new_tl._last_span_id = tl._last_span_id
    new_tl._active_spans = list(tl._active_spans)
    for span_id, span in tl.spans.items():
      new_span = timeline.Span(
          name=span.name,
          begin=span.begin,
          id=span.id,
          parent_id=span.parent_id,
          tags=dict(span.tags) if span.tags else None,
      )
      new_span.end = span.end
      new_tl.spans[span_id] = new_span
    return new_tl


def add_idle_spans(
    tl: timeline.Timeline,
    use_snapshot: bool = True,
    ignore_active_spans: bool = True,
    idle_name: str = constants.IDLE,
) -> timeline.Timeline:
  """Creates a new timeline with 'idle' spans added in the gaps.

  From the time the timeline was created (`born`), until the maximum `end`
  time of any completed span in the timeline, this function finds gaps where
  no span is active and inserts non-overlapping 'idle' spans.

  Args:
    tl: The original timeline.
    use_snapshot: If True, the snapshot of the timeline is used (shallow copy of
      finished spans). Otherwise, a deep copy of the original timeline is used.
    ignore_active_spans: If True, spans with end=inf are ignored when computing
      gaps.

  Returns:
    A new timeline containing all original completed spans plus the idle spans.
  """
  new_tl = tl.snapshot() if use_snapshot else deepcopy_timeline(tl)

  if not new_tl.spans:
    new_tl._last_span_id = tl._last_span_id  # pylint: disable=protected-access
    return new_tl

  spans_to_process = list(new_tl.spans.values())
  if ignore_active_spans:
    spans_to_process = [s for s in spans_to_process if s.end != float("inf")]

  if not spans_to_process:
    new_tl._last_span_id = tl._last_span_id  # pylint: disable=protected-access
    return new_tl

  intervals = [[s.begin, s.end] for s in spans_to_process]
  intervals.sort(key=lambda x: x[0])

  merged = []
  for begin, end in intervals:
    if not merged:
      merged.append([begin, end])
    else:
      _, last_end = merged[-1]
      if begin <= last_end:
        merged[-1][1] = max(last_end, end)
      else:
        merged.append([begin, end])

  current_time = new_tl.born
  next_id = max(new_tl.spans.keys()) + 1

  for begin, end in merged:
    if begin > current_time:
      idle_span = timeline.Span(
          name=idle_name,
          begin=current_time,
          id=next_id,
      )
      idle_span.end = begin
      new_tl.spans[next_id] = idle_span
      next_id += 1
    current_time = max(current_time, end)

  new_tl._last_span_id = next_id - 1  # pylint: disable=protected-access
  return new_tl


def merge_overlapping_spans(tl: timeline.Timeline) -> timeline.Timeline:
  """Merges overlapping spans in a timeline into new spans.

  For any group of overlapping spans, they are merged into a single span.
  The start time of the merged span is the minimum start time of the group.
  The end time is the maximum end time of the group.
  The name is a comma-separated list of the names of the merged spans.
  The ID and parent ID of the merged span match the first span in the group
  (ordered by start time).
  The tags are the union of the tags of the group, with earlier spans in the
  group overriding earlier spans with the same tag key.

  Args:
    tl: The original timeline.

  Returns:
    A new timeline containing only the merged spans.
  """
  new_tl = tl.__class__(tl.id, tl.born)
  # pylint: disable=protected-access
  with tl._lock:
    new_tl._last_span_id = tl._last_span_id
    spans = list(tl.spans.values())

  if not spans:
    return new_tl

  sorted_spans = sorted(spans, key=lambda s: (s.begin, s.end))

  merged_groups = []
  current_group = []
  current_end = float("-inf")

  for s in sorted_spans:
    if not current_group:
      current_group.append(s)
      current_end = s.end
    else:
      if s.begin <= current_end:
        current_group.append(s)
        current_end = max(current_end, s.end)
      else:
        merged_groups.append(current_group)
        current_group = [s]
        current_end = s.end
  if current_group:
    merged_groups.append(current_group)

  for group in merged_groups:
    first_span = group[0]
    unique_names = sorted(list(set(s.name for s in group)))
    if len(unique_names) == 1:
      merged_name = unique_names[0]
    else:
      merged_name = ",".join(unique_names)
    merged_begin = min(s.begin for s in group)
    merged_end = max(s.end for s in group)

    merged_tags = {}
    for s in reversed(group):
      if s.tags:
        merged_tags.update(s.tags)

    merged_span = timeline.Span(
        name=merged_name,
        begin=merged_begin,
        id=first_span.id,
        parent_id=first_span.parent_id,
        tags=merged_tags if merged_tags else None,
    )
    merged_span.end = merged_end
    new_tl.spans[merged_span.id] = merged_span

  return new_tl


def flatten_overlapping_spans(
    tl: timeline.Timeline,
) -> tuple[timeline.Timeline, timeline.Timeline]:
  """Flattens overlapping spans into execution and queue timelines.

  If spans overlap, they are assumed to be executed sequentially. The first span
  executes immediately, and subsequent overlapping spans are queued. The
  execution start time of a queued span is adjusted to the end time of the
  previous span, and the difference is recorded as a new non-overlapping queue
  span.Spans that are fully contained within the queue time of
  preceding spans are currently dropped from the execution timeline.

  This generates two new timelines:
  1. An execution timeline containing non-overlapping execution spans.
  2. A queue timeline containing non-overlapping queue times.

  Args:
    tl: The original timeline.

  Returns:
    A tuple of two timelines: (exec_tl, queue_tl).
  """
  exec_tl = tl.__class__(f"{tl.id}_exec", tl.born)
  queue_tl = tl.__class__(f"{tl.id}_queue", tl.born)

  # pylint: disable=protected-access
  with tl._lock:
    spans = list(tl.spans.values())

  if not spans:
    return exec_tl, queue_tl

  sorted_spans = sorted(spans, key=lambda s: (s.begin, s.id))

  exec_tl._last_span_id = tl._last_span_id
  queue_tl._last_span_id = tl._last_span_id
  next_queue_id = (max(tl.spans.keys()) + 1) if tl.spans else 0

  current_end = float("-inf")
  last_queue_end = float("-inf")

  for s in sorted_spans:
    if s.begin < current_end:
      exec_begin = current_end
      exec_end = max(exec_begin, s.end)

      queue_begin = max(s.begin, last_queue_end)
      if queue_begin < exec_begin:
        tags = dict(s.tags) if s.tags else {}
        tags[constants.QUEUED_SPAN] = s.name
        queue_span = timeline.Span(
            name=f"{constants.QUEUE}",
            begin=queue_begin,
            id=next_queue_id,
            parent_id=s.parent_id,
            tags=tags,
        )
        queue_span.end = exec_begin
        queue_tl.spans[next_queue_id] = queue_span
        next_queue_id += 1
        last_queue_end = exec_begin
    else:
      exec_begin = s.begin
      exec_end = s.end

    # TODO - noghabi: Handle spans completely subsumed by queue time.
    # For now, we drop spans that would have zero or negative duration
    # in the execution timeline.
    if exec_begin < exec_end:
      exec_span = timeline.Span(
          name=s.name,
          begin=exec_begin,
          id=s.id,
          parent_id=s.parent_id,
          tags=dict(s.tags) if s.tags else None,
      )
      exec_span.end = exec_end
      exec_tl.spans[s.id] = exec_span

      current_end = exec_end
    else:
      # If the span was completely queued, the current_end doesn't change.
      pass

  return exec_tl, queue_tl


def filter_spans_by_name(
    tl: timeline.Timeline, span_names: Sequence[str] | str
) -> Sequence[timeline.Span]:
  """Filters spans in a timeline by a list of names.

  Args:
    tl: The timeline to filter.
    span_names: A Sequence of span names to include.

  Returns:
    A Sequence of spans whose names are in the specified list.
  """
  if isinstance(span_names, str):
    span_names = [span_names]
  return [s for s in tl.spans.values() if s.name in span_names]


def add_spans_to_timeline(
    target_tl: timeline.Timeline,
    spans_to_add: Sequence[timeline.Span],
    keep_both: bool = True,
) -> timeline.Timeline:
  """Adds a list of spans to a target timeline.

  Args:
    target_tl: The timeline to add spans to.
    spans_to_add: A Sequence of spans to add.
    keep_both: If True, resolves ID conflicts by assigning new IDs to the added
      spans.

  Returns:
    A new Timeline object containing the original spans and the new spans.
  """
  new_tl = timeline.Timeline(id=target_tl.id, born=target_tl.born)
  # pylint: disable=protected-access
  with target_tl._lock:
    new_tl._last_span_id = target_tl._last_span_id
    for s in target_tl.spans.values():
      new_tl.spans[s.id] = s

  if spans_to_add:
    new_tl._last_span_id = max(
        new_tl._last_span_id, max(s.id for s in spans_to_add)
    )

  for s in spans_to_add:
    if s.id in new_tl.spans:
      if keep_both:
        new_tl._last_span_id += 1
        new_id = new_tl._last_span_id
        logging.warning(
            "Span ID %s already exists in timeline %s. Assigning new ID %s.",
            s.id,
            target_tl.id,
            new_id,
        )
        new_span = timeline.Span(
            name=s.name,
            begin=s.begin,
            id=new_id,
            parent_id=s.parent_id,
            tags=dict(s.tags) if s.tags else None,
        )
        new_span.end = s.end
        new_tl.spans[new_id] = new_span
      else:
        logging.warning(
            "Span ID %s already exists in timeline %s. Overwriting.",
            s.id,
            target_tl.id,
        )
        new_tl.spans[s.id] = s
    else:
      new_tl.spans[s.id] = s

  return new_tl


def strip_tags_from_timeline(
    tl: timeline.Timeline, tags_to_strip: Sequence[str]
) -> None:
  """Removes specific tags from all spans in the given timeline in place.

  Args:
    tl: The timeline to modify.
    tags_to_strip: A sequence of tag keys to remove from the spans.
  """
  tags_to_strip_set = set(tags_to_strip)

  # pylint: disable=protected-access
  with tl._lock:
    for span in tl.spans.values():
      tags = span.tags
      if tags is not None:
        span.tags = {
            k: v for k, v in tags.items() if k not in tags_to_strip_set
        }
