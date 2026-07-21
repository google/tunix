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

"""Dispatch credits: the outermost backpressure bound on in-flight work.

A fixed pool of credits caps how much rollout work may be outstanding at once
(e.g. `(staleness + 1) x groups_per_step`, so `staleness = 0` is a synchronous
loop). A unit of work acquires credits when dispatched and releases them when it
reaches a terminal state -- trained, dropped, or failed -- so credits are never
leaked by drops. This is a plain non-blocking counter; the caller decides the
granularity (per group or per request) and when terminal release happens.
"""


class DispatchCredits:
  """A bounded, non-blocking credit counter."""

  def __init__(self, capacity: int):
    if capacity < 0:
      raise ValueError(f"capacity must be non-negative, got {capacity}")
    self._capacity = capacity
    self._in_use = 0

  @property
  def capacity(self) -> int:
    return self._capacity

  def in_use(self) -> int:
    return self._in_use

  def available(self) -> int:
    return self._capacity - self._in_use

  def try_acquire(self, n: int = 1) -> bool:
    """Acquires n credits if available; returns False without acquiring if not."""
    if n < 0:
      raise ValueError(f"cannot acquire a negative amount: {n}")
    if n > self.available():
      return False
    self._in_use += n
    return True

  def release(self, n: int = 1) -> None:
    """Returns n credits to the pool.

    Args:
      n: Number of credits to release.

    Raises:
      ValueError: If n is negative or exceeds the credits currently in use
        (which would signal a double-release / accounting bug).
    """
    if n < 0:
      raise ValueError(f"cannot release a negative amount: {n}")
    if n > self._in_use:
      raise ValueError(
          f"releasing {n} credits but only {self._in_use} are in use"
      )
    self._in_use -= n
