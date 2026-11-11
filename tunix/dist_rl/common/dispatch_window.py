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
from typing import Protocol


class HasPromptId(Protocol):
  """Protocol for trajectory or prompt objects carrying a `prompt_id`."""

  @property
  def prompt_id(self) -> Hashable:
    ...


class DispatchWindow:
  """Bounds how many prompts ahead of training the dispatcher may run.

  - **Capacity bound**: `admit()` allows up to `(max_staleness + 1) *
    full_batch_size` prompts in flight at once, parking when full until
    `release()` wakes it.
  - **In-flight accounting**:
    `in_flight_prompts = admitted_prompts - (next_batch - start_step) *
    full_batch_size - filtered_groups`, where `start_step` is anchored to
    `next_batch` on the first admitted prompt:
    - Each completed training step advances `next_batch` by 1, freeing
      `full_batch_size` slots.
    - Each dropped trajectory group (`on_group_filtered`) increments
      `filtered_groups` by 1, replenishing 1 slot immediately.
  - **Checkpoint resume**:
    - In-order (`out_of_order_resume=False`): skips the first `next_batch *
      full_batch_size` dataset items via `dataset_skip_count`.
    - Out-of-order (`out_of_order_resume=True`): records `committed_prompt_ids`
      and `skipped_prompt_ids` so `admit()` can skip already-handled prompts
      on resume.
  """

  def __init__(
      self,
      max_staleness: int,
      next_batch_fn: Callable[[], int],
      *,
      full_batch_size: int = 1,
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
    self.max_staleness = max_staleness
    self.full_batch_size = full_batch_size
    self.out_of_order_resume = out_of_order_resume
    self._next_batch_fn = next_batch_fn
    self._release = asyncio.Event()
    self._filtered_groups = 0
    self._committed_prompt_ids: set[str] = set()
    self._skipped_prompt_ids: set[str] = set()
    self._has_resume_state = False
    self._start_step: int | None = None
    self._admitted_prompts = 0

  @property
  def next_batch(self) -> int:
    """Index of the next prompt batch the trainer will consume."""
    return self._next_batch_fn()

  @property
  def committed_prompt_ids(self) -> set[str]:
    """Prompt IDs committed across completed training steps."""
    return self._committed_prompt_ids

  @property
  def skipped_prompt_ids(self) -> set[str]:
    """Prompt IDs dropped by upstream queue filtering."""
    return self._skipped_prompt_ids

  @property
  def uses_prompt_id_resume(self) -> bool:
    """Whether dataset iteration filters by prompt_id vs. prefix-skipping."""
    return self.out_of_order_resume and self._has_resume_state

  @property
  def dataset_skip_count(self) -> int:
    """Number of leading dataset items `iter_dataset_async` can skip."""
    if self.uses_prompt_id_resume:
      return 0
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

  def on_group_filtered(self, filtered_group: Sequence[HasPromptId]) -> None:
    """Replenishes capacity and records skipped prompt_id on group drop."""
    if filtered_group and filtered_group[0].prompt_id not in (None, ""):
      self._skipped_prompt_ids.add(str(filtered_group[0].prompt_id))
    self._filtered_groups += 1
    self.release()

  def record_committed_groups(
      self, groups: Sequence[Sequence[HasPromptId]]
  ) -> None:
    """Records prompt_ids of groups committed at the end of a training step."""
    if not self.out_of_order_resume:
      return
    for g in groups:
      if g and g[0].prompt_id not in (None, ""):
        self._committed_prompt_ids.add(str(g[0].prompt_id))

  def checkpoint_metadata(
      self, pending_groups: Sequence[Sequence[HasPromptId]] = ()
  ) -> dict[str, list[str]]:
    """Returns off-policy prompt_id sets for checkpoint metadata when active."""
    if not self.out_of_order_resume or (
        self.max_staleness <= 0
        and not self._skipped_prompt_ids
        and not self._has_resume_state
    ):
      return {}
    step_pids = {
        str(g[0].prompt_id)
        for g in pending_groups
        if g and g[0].prompt_id not in (None, "")
    }
    return {
        "committed_prompt_ids": sorted(self._committed_prompt_ids | step_pids),
        "skipped_prompt_ids": sorted(self._skipped_prompt_ids),
    }

  def restore_checkpoint_metadata(
      self, ckpt_meta: Mapping[str, object] | None
  ) -> bool:
    """Restores committed and skipped prompt_ids from checkpoint metadata."""
    if (
        not self.out_of_order_resume
        or not isinstance(ckpt_meta, Mapping)
        or (
            "committed_prompt_ids" not in ckpt_meta
            and "skipped_prompt_ids" not in ckpt_meta
        )
    ):
      return False
    self._has_resume_state = True

    def _extract_ids(key: str) -> set[str]:
      raw = ckpt_meta[key] if key in ckpt_meta else ()
      if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return {str(pid) for pid in raw}
      return set()

    self._committed_prompt_ids = _extract_ids("committed_prompt_ids")
    self._skipped_prompt_ids = _extract_ids("skipped_prompt_ids")
    return True

  async def admit(self, prompt_id: Hashable) -> bool:
    """Waits for a free in-flight slot and claims it for `prompt_id`.

    Args:
      prompt_id: Identifier of the candidate prompt.

    Returns:
      True once an in-flight slot is claimed for `prompt_id`, or False if
      `prompt_id` was already committed or skipped before checkpoint resume.
    """
    if prompt_id in (None, ""):
      raise ValueError("prompt_id must be non-empty.")
    if self.uses_prompt_id_resume:
      pid = str(prompt_id)
      if pid in self._committed_prompt_ids or pid in self._skipped_prompt_ids:
        return False
    if self._start_step is None:
      self._start_step = self.next_batch
    window_capacity = (self.max_staleness + 1) * self.full_batch_size
    while self.in_flight_prompts >= window_capacity:
      # Wait first, clear second. The reverse loses a release that lands
      # between the test above and the clear, and nothing would set it again.
      await self._release.wait()
      self._release.clear()
    self._admitted_prompts += 1
    return True
