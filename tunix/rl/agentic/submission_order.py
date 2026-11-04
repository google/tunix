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

"""Releases completed rollout groups to the trainer in submission order.

With several optimizer updates per global step (`mini_batch_size <
batch_size`), handing groups to the trainer in completion order puts the
short, fast groups in the first mini-batch and the long ones in the last. The
last mini-batch is the most off-policy one (furthest from the weights that
sampled it, most clipped), so long responses systematically get the weakest
gradient: a length bias, even though no group is ever dropped.

`SubmissionOrderEmitter` decides mini-batch membership by dataset position
instead: mini-batch `m` is the `m`-th block of `window_size` kept groups in
`group_id` order, where a group dropped by dynamic sampling gives its place to
the next id. Within a mini-batch the order does not matter (gradients are
summed before the optimizer update), so a member is released as soon as it
completes; the window moves on once its last member has been released.
"""

import bisect
import heapq
from typing import Generic, TypeVar

_T = TypeVar("_T")


class SubmissionOrderEmitter(Generic[_T]):
  """Reorders completed groups into submission-order mini-batch windows.

  Group ids must be consecutive integers from `first_group_id`, assigned in
  submission order, as the agentic learner numbers prompt groups.

  Attributes:
    window_size: Groups per mini-batch.
  """

  def __init__(self, window_size: int, first_group_id: int = 0):
    """Initializes the emitter.

    Args:
      window_size: Groups per mini-batch.
      first_group_id: Id of the first group of the stream.

    Raises:
      ValueError: If `window_size` is not positive.
    """
    if window_size <= 0:
      raise ValueError(f"window_size must be positive, got {window_size}")
    self.window_size = window_size
    # Every id below `_base` is resolved (released or dropped); resolved ids
    # above it wait, sorted, in `_resolved` until `_base` catches up.
    self._base = first_group_id
    self._resolved: list[int] = []
    # Completed, kept groups waiting for their window: group_id -> payload,
    # plus a min-heap of their ids.
    self._held: dict[int, _T] = {}
    self._held_ids: list[int] = []
    self._released_in_window = 0

  @property
  def held(self) -> int:
    """Number of completed groups waiting for their mini-batch window."""
    return len(self._held)

  def _rank(self, group_id: int) -> int:
    """Returns the number of unresolved ids below `group_id`."""
    return group_id - self._base - bisect.bisect_left(self._resolved, group_id)

  def _resolve(self, group_id: int) -> None:
    """Marks `group_id` as released or dropped."""
    bisect.insort(self._resolved, group_id)
    while self._resolved and self._resolved[0] == self._base:
      self._resolved.pop(0)
      self._base += 1

  def _is_resolved(self, group_id: int) -> bool:
    """Whether `group_id` was already released or dropped."""
    index = bisect.bisect_left(self._resolved, group_id)
    return group_id < self._base or (
        index < len(self._resolved) and self._resolved[index] == group_id
    )

  def _release(self) -> list[_T]:
    """Releases held groups of the open window, moving the window on."""
    out: list[_T] = []
    while self._held_ids:
      group_id = self._held_ids[0]
      # The smallest held id has the lowest rank; nothing else can release.
      if self._rank(group_id) >= self.window_size - self._released_in_window:
        break
      heapq.heappop(self._held_ids)
      out.append(self._held.pop(group_id))
      self._resolve(group_id)
      self._released_in_window = (
          self._released_in_window + 1
      ) % self.window_size
    return out

  def complete(self, group_id: int, payload: _T) -> list[_T]:
    """Registers a completed, kept group.

    Args:
      group_id: The group's id.
      payload: What to release for the group (e.g. its TrainExamples).

    Returns:
      The payloads that can be released now, in release order.

    Raises:
      ValueError: If the group was already completed or dropped.
    """
    if self._is_resolved(group_id) or group_id in self._held:
      raise ValueError(f"group {group_id} was already completed or dropped")
    self._held[group_id] = payload
    heapq.heappush(self._held_ids, group_id)
    return self._release()

  def drop(self, group_id: int) -> list[_T]:
    """Removes a dropped group from the order.

    Args:
      group_id: The dropped group's id.

    Returns:
      The payloads that the drop releases, in release order.

    Raises:
      ValueError: If the group was already completed or dropped.
    """
    if self._is_resolved(group_id) or group_id in self._held:
      raise ValueError(f"group {group_id} was already completed or dropped")
    self._resolve(group_id)
    return self._release()

  def flush(self) -> list[_T]:
    """Releases everything still held, in id order (end of the stream).

    Returns:
      The held payloads, in id order.
    """
    out = [self._held[group_id] for group_id in sorted(self._held)]
    for group_id in self._held:
      self._resolve(group_id)
    self._held.clear()
    self._held_ids.clear()
    self._released_in_window = 0
    return out
