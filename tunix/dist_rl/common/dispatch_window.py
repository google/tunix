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
import typing
from typing import Protocol


@typing.runtime_checkable
class HasPromptId(Protocol):
  """Protocol for trajectory or prompt objects carrying a `prompt_id`."""

  @property
  def prompt_id(self) -> Hashable:
    ...


class DispatchWindow:
  """Bounds how many prompt batches ahead of `next_batch` the dispatcher may run.

  The dispatcher may emit prompts for `batch_idx` while
  `batch_idx <= next_batch + max_staleness`. When `batch_idx` exceeds that
  bound, `wait_for(batch_idx)` parks until `release()` signals that
  `next_batch` has advanced.

  Also tracks filtered-group replenishment and off-policy `prompt_id`
  checkpoint/resume state (`committed_prompt_ids` and `skipped_prompt_ids`).
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
    self._resume_base_prompts: int | None = None
    self._dispatched_since_resume = 0

  @property
  def next_batch(self) -> int:
    """Index of the next prompt batch the trainer will consume."""
    filtered_batch_offset = (
        self._filtered_groups + self.full_batch_size - 1
    ) // self.full_batch_size
    return self._next_batch_fn() + filtered_batch_offset

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
    """Whether dataset iteration filters by prompt_id instead of prefix-skipping."""
    return self.out_of_order_resume and self._has_resume_state

  @property
  def dataset_skip_count(self) -> int:
    """Number of leading dataset items `iter_dataset_async` can skip directly."""
    if self.uses_prompt_id_resume:
      return 0
    return self.next_batch * self.full_batch_size

  @property
  def is_released(self) -> bool:
    """Whether a release signal is currently set."""
    return self._release.is_set()

  def release(self) -> None:
    """Wakes the dispatcher after `next_batch` advances."""
    self._release.set()

  def on_group_filtered(self, filtered_group: Sequence[HasPromptId]) -> None:
    """Replenishes dispatch capacity and records skipped prompt_id when a group is dropped."""
    if filtered_group and filtered_group[0].prompt_id:
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
      if g and g[0].prompt_id:
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
        str(g[0].prompt_id) for g in pending_groups if g and g[0].prompt_id
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
    self._committed_prompt_ids = set()
    if "committed_prompt_ids" in ckpt_meta:
      raw_committed = ckpt_meta["committed_prompt_ids"]
      if isinstance(raw_committed, Sequence) and not isinstance(
          raw_committed, (str, bytes)
      ):
        self._committed_prompt_ids = {str(pid) for pid in raw_committed}
    self._skipped_prompt_ids = set()
    if "skipped_prompt_ids" in ckpt_meta:
      raw_skipped = ckpt_meta["skipped_prompt_ids"]
      if isinstance(raw_skipped, Sequence) and not isinstance(
          raw_skipped, (str, bytes)
      ):
        self._skipped_prompt_ids = {str(pid) for pid in raw_skipped}
    return True

  async def wait_for(self, batch_idx: int) -> None:
    """Blocks until `batch_idx` is inside the policy-staleness window."""
    while batch_idx > self.next_batch + self.max_staleness:
      # Wait first, clear second. The reverse loses a release that lands
      # between the test above and the clear, and nothing would set it again.
      await self._release.wait()
      self._release.clear()

  async def admit(
      self,
      prompt_idx: int,
      prompt_item: Mapping[str, object] | HasPromptId | object = None,
  ) -> bool:
    """Waits until the prompt's batch is within the window, or returns False if already handled on resume."""
    if self.uses_prompt_id_resume:
      if isinstance(prompt_item, Mapping):
        raw_pid = (
            prompt_item["prompt_id"] if "prompt_id" in prompt_item else None
        )
        candidate_pid = str(raw_pid) if raw_pid else f"prompt_{prompt_idx}"
      elif isinstance(prompt_item, HasPromptId) and prompt_item.prompt_id:
        candidate_pid = str(prompt_item.prompt_id)
      else:
        candidate_pid = f"prompt_{prompt_idx}"
      if (
          candidate_pid in self._committed_prompt_ids
          or candidate_pid in self._skipped_prompt_ids
      ):
        return False
      if self._resume_base_prompts is None:
        self._resume_base_prompts = self.next_batch * self.full_batch_size
      dispatch_batch_idx = (
          self._resume_base_prompts + self._dispatched_since_resume
      ) // self.full_batch_size
      self._dispatched_since_resume += 1
    else:
      dispatch_batch_idx = prompt_idx // self.full_batch_size
    await self.wait_for(dispatch_batch_idx)
    return True
