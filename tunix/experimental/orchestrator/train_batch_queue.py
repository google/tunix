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

"""A bounded FIFO queue of assembled train batches (backpressure bound).

The second of the pipeline's nested bounds: it holds ready-to-train items and,
when full, `put` fails so the producer stops assembling. It is intentionally a
plain, single-threaded structure -- flow control is expressed by the boolean
`put` return, not by blocking -- and holds unique items only (mu-replay re-reads
retained batches rather than re-enqueueing them).
"""

import collections
from typing import Generic, TypeVar

_T = TypeVar("_T")


class QueueEmpty(Exception):
  """Raised by `get` when the queue is empty."""


class TrainBatchQueue(Generic[_T]):
  """A bounded, non-blocking FIFO queue."""

  def __init__(self, maxsize: int):
    if maxsize <= 0:
      raise ValueError(f"maxsize must be positive, got {maxsize}")
    self._maxsize = maxsize
    self._items: collections.deque[_T] = collections.deque()

  @property
  def maxsize(self) -> int:
    return self._maxsize

  def put(self, item: _T) -> bool:
    """Enqueues an item; returns False (without enqueuing) if the queue is full."""
    if self.is_full():
      return False
    self._items.append(item)
    return True

  def get(self) -> _T:
    """Dequeues the oldest item.

    Raises:
      QueueEmpty: If the queue is empty.
    """
    if not self._items:
      raise QueueEmpty("get from an empty TrainBatchQueue")
    return self._items.popleft()

  def try_get(self) -> _T | None:
    """Dequeues the oldest item, or returns None if the queue is empty."""
    if not self._items:
      return None
    return self._items.popleft()

  def is_full(self) -> bool:
    return len(self._items) >= self._maxsize

  def is_empty(self) -> bool:
    return not self._items

  def remaining(self) -> int:
    """Free slots before the queue is full."""
    return self._maxsize - len(self._items)

  def __len__(self) -> int:
    return len(self._items)
