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

"""Policy-staleness dispatch window for prompt batches."""

import asyncio
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Protocol
from absl import logging


class HasPromptId(Protocol):
  """Protocol for trajectory or prompt objects carrying `prompt_id`."""

  @property
  def prompt_id(self) -> Hashable:
    ...


def _group_prompt_idx(group: Sequence[Any]) -> int | None:
  """Reads the 0-based dataset `prompt_idx` stamped onto a trajectory group."""
  if not group:
    return None
  metadata = getattr(group[0], "metadata", None)
  if isinstance(metadata, Mapping) and "prompt_idx" in metadata:
    raw_idx = metadata["prompt_idx"]
    if raw_idx is not None and not isinstance(raw_idx, bool):
      try:
        idx = int(raw_idx)
      except (TypeError, ValueError):
        return None
      return idx if idx >= 0 else None
  return None


class DispatchWindow:
  """Bounds how many prompts ahead of training the dispatcher may run.

  - **Capacity bound**: `admit()` allows up to `window_capacity` prompts in
    flight at once (`(max_staleness + 1) * full_batch_size` by default, or
    `max_in_flight_groups` when set), parking when full until `release()` wakes
    it.
  - **In-flight accounting**:
    `in_flight_prompts = admitted_prompts - (next_batch - start_step) *
    full_batch_size - filtered_groups`, where `start_step` is anchored to
    `next_batch` on the first admitted prompt:
    - Each completed training step advances `next_batch` by 1, freeing
      `full_batch_size` slots.
    - Each dropped trajectory group (`on_group_filtered`) increments
      `filtered_groups` by 1, replenishing 1 slot immediately.
  - **Checkpoint resume**:
    - In-order (`out_of_order_resume=False`, or before any out-of-order settle):
      skips the first `next_batch * full_batch_size` dataset items via
      `dataset_skip_count`.
    - Out-of-order (`out_of_order_resume=True`): tracks a compact monotonic
      `prompt_idx` watermark `(min_unsettled_prompt_idx, settled_offsets)`. At
      any checkpoint every `prompt_idx < min_unsettled_prompt_idx` is already
      trained or dropped and `settled_offsets` holds the positive offsets
      `prompt_idx - min_unsettled_prompt_idx` settled above the watermark
      (`O(window_capacity)` absent multi-step stragglers). On resume,
      `dataset_skip_count` jumps straight to
      `min_unsettled_prompt_idx` and `admit()` skips only those bounded offsets,
      working cleanly across `.repeat()` epochs with constant metadata size.
  """

  def __init__(
      self,
      max_staleness: int,
      next_batch_fn: Callable[[], int],
      *,
      full_batch_size: int = 1,
      max_in_flight_groups: int | None = None,
      out_of_order_resume: bool = False,
  ):
    if max_staleness < 0:
      raise ValueError(
          f"max_staleness must be non-negative, got {max_staleness}."
      )
    if full_batch_size <= 0:
      raise ValueError(
          f"full_batch_size must be positive, got {full_batch_size}."
      )
    if max_in_flight_groups is not None:
      if isinstance(max_in_flight_groups, bool) or not isinstance(
          max_in_flight_groups, int
      ):
        raise TypeError(
            "max_in_flight_groups must be an int or None, got"
            f" {max_in_flight_groups!r}."
        )
      if max_in_flight_groups < full_batch_size:
        raise ValueError(
            "max_in_flight_groups must be at least full_batch_size"
            f" ({full_batch_size}), got {max_in_flight_groups}."
        )
    self.max_staleness = max_staleness
    self.full_batch_size = full_batch_size
    self.max_in_flight_groups = max_in_flight_groups
    self.out_of_order_resume = out_of_order_resume
    self._next_batch_fn = next_batch_fn
    self._release = asyncio.Event()
    self._filtered_groups = 0
    self._min_unsettled_idx = 0
    self._settled_above: set[int] = set()
    self._has_settled_any = False
    self._has_resume_state = False
    self._next_synthetic_idx = 0
    self._start_step: int | None = None
    self._admitted_prompts = 0

  @property
  def window_capacity(self) -> int:
    """Maximum prompt groups admitted but not yet trained or filtered."""
    if self.max_in_flight_groups is not None:
      return self.max_in_flight_groups
    return (self.max_staleness + 1) * self.full_batch_size

  @property
  def next_batch(self) -> int:
    """Index of the next prompt batch the trainer will consume."""
    return self._next_batch_fn()

  @property
  def min_unsettled_prompt_idx(self) -> int:
    """Smallest dataset `prompt_idx` not yet trained or filtered."""
    return self._min_unsettled_idx

  @property
  def settled_offsets(self) -> list[int]:
    """Sorted offsets `idx - min_unsettled_prompt_idx` settled out of order."""
    return sorted(idx - self._min_unsettled_idx for idx in self._settled_above)

  @property
  def has_resume_state(self) -> bool:
    """Whether out-of-order watermark state was restored from a checkpoint."""
    return self.out_of_order_resume and self._has_resume_state

  @property
  def dataset_skip_count(self) -> int:
    """Number of leading dataset items `iter_dataset_async` can skip."""
    if self.has_resume_state:
      return self._min_unsettled_idx
    return self.next_batch * self.full_batch_size

  @property
  def in_flight_prompts(self) -> int:
    """Net admitted prompts relative to steps advanced and groups filtered."""
    if self._start_step is None:
      return 0
    trained_since_start = (
        self.next_batch - self._start_step
    ) * self.full_batch_size
    return self._admitted_prompts - trained_since_start - self._filtered_groups

  @property
  def is_released(self) -> bool:
    """Whether a release signal is currently set."""
    return self._release.is_set()

  def release(self) -> None:
    """Wakes the dispatcher after `next_batch` advances."""
    self._release.set()

  @staticmethod
  def _advance_watermark(
      min_unsettled: int, settled_above: set[int], prompt_idx: int
  ) -> int:
    """Records `prompt_idx` into `(min_unsettled, settled_above)`."""
    if prompt_idx < min_unsettled:
      return min_unsettled
    if prompt_idx > min_unsettled:
      settled_above.add(prompt_idx)
      return min_unsettled
    min_unsettled += 1
    while min_unsettled in settled_above:
      settled_above.remove(min_unsettled)
      min_unsettled += 1
    return min_unsettled

  def _mark_settled(self, prompt_idx: int) -> None:
    """Records `prompt_idx` as settled and advances the contiguous watermark."""
    if prompt_idx < 0:
      return
    if not self._has_settled_any and not self._has_resume_state:
      self._min_unsettled_idx = max(
          self._min_unsettled_idx,
          (self._start_step or 0) * self.full_batch_size,
      )
    self._has_settled_any = True
    self._min_unsettled_idx = self._advance_watermark(
        self._min_unsettled_idx, self._settled_above, prompt_idx
    )

  def on_group_filtered(self, filtered_group: Sequence[HasPromptId]) -> None:
    """Replenishes capacity and marks the group's `prompt_idx` settled."""
    if (
        self.out_of_order_resume
        and (idx := _group_prompt_idx(filtered_group)) is not None
    ):
      self._mark_settled(idx)
    self._filtered_groups += 1
    self.release()

  def record_committed_groups(
      self, groups: Sequence[Sequence[HasPromptId]]
  ) -> None:
    """Marks `prompt_idx` of groups committed at the end of a training step."""
    if not self.out_of_order_resume:
      return
    for group in groups:
      if (idx := _group_prompt_idx(group)) is not None:
        self._mark_settled(idx)

  def _snapshot_watermark(
      self, pending_groups: Sequence[Sequence[HasPromptId]]
  ) -> tuple[int, list[int], bool]:
    """Computes `(min_unsettled, settled_offsets, has_any)` including `pending_groups`."""
    min_unsettled = self._min_unsettled_idx
    if not self._has_settled_any and not self._has_resume_state:
      min_unsettled = max(
          min_unsettled, (self._start_step or 0) * self.full_batch_size
      )
    settled = set(self._settled_above)
    pending_indices = [
        idx
        for group in pending_groups
        if (idx := _group_prompt_idx(group)) is not None
        and idx >= min_unsettled
    ]
    for idx in pending_indices:
      min_unsettled = self._advance_watermark(min_unsettled, settled, idx)
    offsets = sorted(idx - min_unsettled for idx in settled)
    return (
        min_unsettled,
        offsets,
        (self._has_settled_any or bool(pending_indices)),
    )

  def checkpoint_metadata(
      self, pending_groups: Sequence[Sequence[HasPromptId]] = ()
  ) -> dict[str, int | list[int]]:
    """Returns compact `(min_unsettled_prompt_idx, settled_offsets)` metadata."""
    if not self.out_of_order_resume:
      return {}
    min_unsettled, offsets, has_any = self._snapshot_watermark(pending_groups)
    is_pure_prefix = (
        self.max_staleness <= 0
        and self.window_capacity <= self.full_batch_size
        and self._filtered_groups == 0
        and not offsets
        and min_unsettled == (self.next_batch + 1) * self.full_batch_size
    )
    if not self._has_resume_state and (is_pure_prefix or not has_any):
      return {}
    return {
        "min_unsettled_prompt_idx": min_unsettled,
        "settled_offsets": offsets,
    }

  def restore_checkpoint_metadata(
      self, ckpt_meta: Mapping[str, object] | None
  ) -> bool:
    """Restores `(min_unsettled_prompt_idx, settled_offsets)` from metadata."""
    if not self.out_of_order_resume or not isinstance(ckpt_meta, Mapping):
      return False
    raw_min = ckpt_meta.get("min_unsettled_prompt_idx")
    if isinstance(raw_min, bool) or not isinstance(raw_min, int) or raw_min < 0:
      if {"committed_prompt_ids", "skipped_prompt_ids"} & ckpt_meta.keys():
        logging.warning(
            "Legacy prompt_id checkpoint metadata ignored; using prefix resume."
        )
      return False
    raw_offsets = ckpt_meta.get("settled_offsets", ())
    valid_seq = isinstance(raw_offsets, Sequence) and not isinstance(
        raw_offsets, (str, bytes)
    )
    offsets = {
        raw_min + item
        for item in (raw_offsets if valid_seq else ())
        if not isinstance(item, bool) and isinstance(item, int) and item > 0
    }
    self._min_unsettled_idx = (
        self._advance_watermark(raw_min - 1, offsets, raw_min - 1)
        if raw_min > 0
        else raw_min
    )
    while raw_min in offsets:
      offsets.remove(raw_min)
      raw_min += 1
    self._min_unsettled_idx = raw_min
    self._settled_above = offsets
    self._next_synthetic_idx = raw_min
    self._has_resume_state = True
    return True

  async def admit(
      self, prompt_id: Hashable, *, prompt_idx: int | None = None
  ) -> bool:
    """Waits for a free in-flight slot and claims it for `(prompt_idx, prompt_id)`.

    Args:
      prompt_id: Non-empty identifier of the candidate prompt.
      prompt_idx: Optional 0-based dataset index from `iter_dataset_async`. When
        omitted, indices are assigned sequentially starting at
        `min_unsettled_prompt_idx`.

    Returns:
      True once an in-flight slot is claimed, or False if `prompt_idx` was
      already settled before checkpoint resume.
    """
    if prompt_id in (None, ""):
      raise ValueError("prompt_id must be non-empty.")
    if not self.has_resume_state and self._start_step is None:
      self._start_step = self.next_batch
      if not self._has_settled_any and self._start_step > 0:
        self._min_unsettled_idx = self._start_step * self.full_batch_size
        self._next_synthetic_idx = max(
            self._next_synthetic_idx, self._min_unsettled_idx
        )
    if prompt_idx is None:
      idx = self._next_synthetic_idx
      self._next_synthetic_idx += 1
    else:
      if prompt_idx < 0:
        raise ValueError(f"prompt_idx must be non-negative, got {prompt_idx}.")
      idx = prompt_idx
      self._next_synthetic_idx = max(self._next_synthetic_idx, idx + 1)
    if self.has_resume_state and (
        idx < self._min_unsettled_idx or idx in self._settled_above
    ):
      return False
    if self._start_step is None:
      self._start_step = self.next_batch
    while self.in_flight_prompts >= self.window_capacity:
      # Wait first, clear second. The reverse loses a release that lands
      # between the test above and the clear, and nothing would set it again.
      await self._release.wait()
      self._release.clear()
    self._admitted_prompts += 1
    return True
