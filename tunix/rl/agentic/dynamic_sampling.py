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

"""DAPO dynamic sampling for the agentic RL learner.

`DynamicSampler` is the prompt feed of the learner when `group_filters` or
`max_in_flight_groups` is set. It admits prompts one at a time through a
`DispatchWindow` and keeps the accounting of the groups they produce. A group
that a filter of `group_filters` drops (e.g. all G rewards tie, so every
advantage is zero) is not trained on; the window then admits exactly one
replacement prompt, so every global step still trains on `batch_size` groups.
Global steps are counted by kept groups rather than by group id, since
replacements make the group ids run ahead of the trainer.

`on_global_step` and `close` may be called from any thread; every other method
runs on the learner's event loop. (The asyncio events created in the
constructor bind to the loop on first use.)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from typing import Any

from absl import logging
from jax import typing as jax_typing
from tunix.dist_rl.common import dispatch_window as dispatch_window_lib
from tunix.rl.agentic import group_filters as group_filters_lib

PromptBatch = dict[str, Any]


class DynamicSamplingStalledError(RuntimeError):
  """Too many consecutive groups were dropped by `group_filters`."""


class GroupDroppedError(Exception):
  """Raised while scoring a training group that `group_filters` drop.

  Raising as soon as the advantages are known lets the learner skip the
  reference / actor log-prob passes for a group that will not be trained on.

  Attributes:
    reason: The filter that dropped the group.
  """

  def __init__(self, reason: str):
    super().__init__(reason)
    self.reason = reason


class DynamicSampler:
  """Per-prompt admission and kept / dropped group accounting.

  Prompt `i` is admitted once fewer than `window_capacity` prompts are ahead of
  the prompts the trainer has consumed. Consumption advances by `batch_size`
  prompts per global step (`on_global_step`) and by one prompt per dropped
  group (`on_group_dropped`).
  """

  def __init__(
      self,
      *,
      batches: Iterator[PromptBatch],
      split: Callable[[PromptBatch], Iterable[PromptBatch]],
      loop: asyncio.AbstractEventLoop,
      group_filters: Sequence[str],
      max_consecutive_filtered_groups: int,
      off_policy_steps: int,
      full_batch_size: int,
      max_in_flight_groups: int | None,
      global_step_fn: Callable[[], int],
  ):
    """Initializes the sampler at the trainer's current global step.

    Args:
      batches: Full prompt batches, read on the event loop's executor.
      split: Splits a batch into single prompts.
      loop: The learner's event loop.
      group_filters: Names from `group_filters.GROUP_FILTERS`.
      max_consecutive_filtered_groups: Raise after this many groups in a row
        are dropped; -1 disables the guard.
      off_policy_steps: Weight syncs a group may lag behind the trainer.
      full_batch_size: Groups per global step.
      max_in_flight_groups: Admission cap; None admits up to
        `(off_policy_steps + 1) * full_batch_size` prompts ahead.
      global_step_fn: Returns the trainer's current global step.

    Raises:
      ValueError: If `max_in_flight_groups` cannot fill a batch or exceeds the
        staleness window.
    """
    staleness_window = (off_policy_steps + 1) * full_batch_size
    if max_in_flight_groups is not None and not (
        full_batch_size <= max_in_flight_groups <= staleness_window
    ):
      # Below the batch size a global step could never be assembled. Above
      # the staleness window, groups sampled more than `off_policy_steps`
      # syncs ago would be trained: the window bounds the count of groups in
      # flight, not their age.
      raise ValueError(
          f"max_in_flight_groups={max_in_flight_groups} must be between the"
          f" batch size {full_batch_size} and (off_policy_steps + 1) *"
          f" batch_size = {staleness_window}."
      )
    self._batches = batches
    self._split = split
    self._loop = loop
    self._filters = tuple(group_filters)
    self._max_consecutive_filtered = max_consecutive_filtered_groups
    self._full_batch_size = full_batch_size
    self._start_step = global_step_fn()
    self._window = dispatch_window_lib.DispatchWindow(
        max_staleness=off_policy_steps,
        next_batch_fn=lambda: global_step_fn() - self._start_step,
        full_batch_size=full_batch_size,
        max_in_flight_groups=max_in_flight_groups,
    )
    self._stopped = asyncio.Event()
    self._admitted = 0
    self._kept = 0
    self._dropped = 0
    self._consecutive_dropped = 0
    if self._start_step > 0:
      logging.warning(
          "Resuming at step %d with per-prompt admission: the dataset is"
          " fast-forwarded by whole batches, which does not account for"
          " prompts consumed by dropped groups in the previous run.",
          self._start_step,
      )

  @property
  def window_capacity(self) -> int:
    """Prompt groups that may be in flight ahead of the trainer."""
    return self._window.window_capacity

  @property
  def expected_step(self) -> int:
    """Global step that the next kept group will be trained in."""
    return self._start_step + self._kept // self._full_batch_size

  def check_group(
      self, group: Sequence[Any], advantages: jax_typing.ArrayLike
  ) -> None:
    """Raises `GroupDroppedError` if a filter rejects the scored group.

    Args:
      group: The group's trajectory items.
      advantages: The group's advantages.

    Raises:
      GroupDroppedError: With the name of the filter that rejected the group.
    """
    reason = group_filters_lib.drop_reason(self._filters, group, advantages)
    if reason is not None:
      raise GroupDroppedError(reason)

  def on_group_kept(self) -> None:
    """Counts a scored group that will be trained on."""
    self._kept += 1
    self._consecutive_dropped = 0

  def on_group_dropped(self, group: Sequence[Any], reason: str) -> None:
    """Counts a dropped group and admits one replacement prompt.

    Args:
      group: The dropped group's trajectory items.
      reason: The filter that dropped the group.

    Raises:
      DynamicSamplingStalledError: If `max_consecutive_filtered_groups` groups
        in a row were dropped.
    """
    self._dropped += 1
    self._consecutive_dropped += 1
    if 0 < self._max_consecutive_filtered <= self._consecutive_dropped:
      raise DynamicSamplingStalledError(
          f"{self._consecutive_dropped} consecutive groups dropped by"
          f" group_filters={list(self._filters)} (last reason: {reason})."
      )
    self._window.on_group_filtered(group)

  async def prompts(self) -> AsyncIterator[PromptBatch]:
    """Yields the dataset's prompts one at a time, each once admitted.

    Yields:
      Single prompts, until the dataset is exhausted or the sampler closed.
    """
    if self._stopped.is_set():
      return
    stopped = asyncio.ensure_future(self._stopped.wait())
    try:
      async for prompt in self._dataset_prompts():
        admit = asyncio.ensure_future(self._window.admit(self._admitted))
        await asyncio.wait(
            (admit, stopped), return_when=asyncio.FIRST_COMPLETED
        )
        if self._stopped.is_set():
          admit.cancel()
          return
        self._admitted += 1
        yield prompt
        if self._stopped.is_set():
          break
      if not self._stopped.is_set():
        logging.info(
            "The dataset ran out of prompts after %d admitted groups, %d of"
            " which were dropped by group_filters and replaced.",
            self._admitted,
            self._dropped,
        )
    finally:
      stopped.cancel()

  async def _dataset_prompts(self) -> AsyncIterator[PromptBatch]:
    """Yields single prompts without blocking the loop on dataset reads."""
    while not self._stopped.is_set():
      batch = await self._loop.run_in_executor(None, next, self._batches, None)
      if batch is None:
        return
      for prompt in self._split(batch):
        yield prompt

  def on_global_step(self, done: bool) -> None:
    """Opens the window for the next batch, or stops admitting prompts.

    Args:
      done: Whether training is over, so no further prompt is needed.
    """
    self._loop.call_soon_threadsafe(
        self._stopped.set if done else self._window.release
    )

  def close(self) -> None:
    """Stops admitting prompts so `prompts()` returns.

    The learner calls this once the producer has handed out every completed
    group, or when training is over.
    """
    self._loop.call_soon_threadsafe(self._stopped.set)
