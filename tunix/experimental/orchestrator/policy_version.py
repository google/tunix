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

"""The orchestrator's own count of which weights are in play.

In a single process, "which weights" and "how many steps have been taken" are
the same question, and the cluster answers both with one counter it bumps
whenever weights are synced. Across several rollout workers they come apart:
some workers may be running weights the trainer has already moved past, a sync
round may reach only part of the fleet, and a step may take place with no sync
at all. A step count cannot express any of that.

So the orchestrator mints its own version. It is the identity a worker
acknowledges installing, the value stamped on the trajectories that worker
produces, and the thing a staleness check compares. It advances only when a
sync round is started, never as a side effect of training, and it is
deliberately not read from the cluster: the cluster's counter remains correct
for in-process runs and is simply not what the distributed path relies on.
"""

from __future__ import annotations

import itertools
import threading


class PolicyVersionMinter:
  """Hands out monotonically increasing policy versions.

  Thread-safe, because sync rounds may be started from a control path while
  training runs elsewhere.
  """

  def __init__(self, initial_version: int = 0):
    """Initializes the minter.

    Args:
      initial_version: The version in effect before any sync round. Resuming a
        run should pass the version its weights were last known to carry.

    Raises:
      ValueError: If `initial_version` is negative.
    """
    if initial_version < 0:
      raise ValueError(
          f"initial_version must be >= 0, got {initial_version}."
      )
    self._current = initial_version
    self._counter = itertools.count(initial_version + 1)
    self._lock = threading.Lock()

  @property
  def current(self) -> int:
    """The version most recently minted."""
    return self._current

  def mint(self) -> int:
    """Starts a new version and returns it.

    The version exists as soon as it is minted, before any worker has it.
    Whether a given worker is actually running it is a separate fact, tracked
    by whoever runs the sync round.
    """
    with self._lock:
      self._current = next(self._counter)
      return self._current
