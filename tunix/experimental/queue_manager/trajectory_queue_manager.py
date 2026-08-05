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

"""Specialized TrajectoryQueueManager for TrajectoryItem instances."""

from __future__ import annotations

from typing import Optional

from tunix.experimental.common import datatypes
from tunix.rl.agentic.queue_manager import group_queue_manager

TrajectoryItem = datatypes.TrajectoryItem
GroupFn = group_queue_manager.GroupFn[TrajectoryItem]
FilterFn = group_queue_manager.FilterFn[TrajectoryItem]


class TrajectoryQueueManager(
    group_queue_manager.GroupQueueManager[TrajectoryItem]
):
  """Specialized queue manager for TrajectoryItem instances.

  Inherits from `GroupQueueManager[TrajectoryItem]`. If no custom `group_fn` is
  provided, uses the default grouping function that groups items by
  `item.group_id` or `item.prompt_id` up to `group_size`.
  """

  def __init__(
      self,
      *,
      group_size: Optional[int] = None,
      group_fn: Optional[GroupFn] = None,
      filter_fn: Optional[FilterFn] = None,
  ):
    """Initializes TrajectoryQueueManager.

    Args:
      group_size: Optional target number of trajectories per ready group when
        using default grouping.
      group_fn: Optional custom grouping function. If None, `group_size` must be
        provided.
      filter_fn: Optional pluggable function to filter candidate groups.
    """
    super().__init__(
        group_size=group_size,
        group_fn=group_fn,
        filter_fn=filter_fn,
    )
