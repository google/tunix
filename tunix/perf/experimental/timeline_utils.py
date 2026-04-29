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

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def generate_queued_timeline_id(tl_id: str) -> str:
  """Generates a string ID for a queued timeline based on a base timeline ID.

  Args:
    tl_id: The base timeline ID.

  Returns:
    The queued timeline ID.
  """
  return f"{tl_id}_queue"


def is_queued_timeline(tl_id: str) -> bool:
  """Checks if the timeline ID corresponds to a queued timeline.

  Args:
    tl_id: The timeline ID to check.

  Returns:
    True if the timeline ID ends with '_queue', False otherwise.
  """
  return tl_id.endswith("_queue")


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


# Utility functions for handling spans.


def is_timeline_only_of_allowed_type(
    tl: timeline.Timeline,
    allowed_span_names: Sequence[str],
    include_cur_step: bool = False,
) -> bool:
  """Checks if all spans in a timeline are of allowed types.

  Args:
    tl: The timeline to check.
    allowed_span_names: A sequence of allowed span names.
    include_cur_step: Whether to include the uncommitted cur_step spans.

  Returns:
    True if the timeline has spans and all spans have a name in
    `allowed_span_names`, False otherwise.
  """
  has_spans = False
  steps = tl.all_steps if include_cur_step else tl.committed_steps
  for step in steps:
    for span in step.values():
      has_spans = True
      if span.name not in allowed_span_names:
        return False
  return has_spans


def sequentialize_overlapping_spans(
    spans_dict: Mapping[int, timeline.Span],
) -> tuple[Mapping[int, timeline.Span], Mapping[int, timeline.Span]]:
  """Sequentializes overlapping spans into non-overlapping active and queue spans.

  If spans overlap, they are assumed to be executed sequentially. The first span
  is active immediately, and subsequent overlapping spans are queued.
    - For spans that the active duration is entirely contained within another
    span,
    the contained span is dropped from the active spans and only added to the
    queue spans.
    - If multiple spans overlap with a long-running span, their "queue" spans
    will also be sequentialized (i.e., no two queue spans will overlap).

  Args:
    spans_dict: A dictionary of active spans.

  Returns:
    A tuple of two dictionaries: (active_spans, queue_spans).
  """
  active_spans: dict[int, timeline.Span] = {}
  queue_spans: dict[int, timeline.Span] = {}

  if not spans_dict:
    return active_spans, queue_spans

  sorted_spans = sorted(spans_dict.values(), key=lambda s: (s.begin, s.id))

  current_end = float("-inf")
  last_queue_end = float("-inf")

  for s in sorted_spans:
    if s.begin < current_end:
      active_begin = current_end
      active_end = max(active_begin, s.end)

      queue_begin = max(s.begin, last_queue_end)
      if queue_begin < active_begin:
        queue_span = timeline.Span(
            name=constants.QUEUE,
            begin=queue_begin,
            id=s.id,
            parent_id=s.parent_id,
            tags={constants.NAME: s.name},
        )
        queue_span.end = active_begin
        queue_spans[s.id] = queue_span
        last_queue_end = active_begin
    else:
      active_begin = s.begin
      active_end = s.end

    if active_begin < active_end:
      active_span = timeline.Span(
          name=s.name,
          begin=active_begin,
          id=s.id,
          parent_id=s.parent_id,
          tags=dict(s.tags) if s.tags else None,
      )
      active_span.end = active_end
      active_spans[s.id] = active_span

      current_end = active_end
    else:
      logging.warning(
          "Span %r has zero or negative duration due to complete overlap with"
          " another span and will be dropped from active spans. Span: %r",
          s.name,
          s,
      )

  return active_spans, queue_spans
