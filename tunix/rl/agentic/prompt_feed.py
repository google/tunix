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

"""Prompt feeds of the agentic RL learner.

A prompt feed hands single prompts to the rollout producer and is told by the
trainer when a global step ends. `QueuePromptFeed` admits one batch of prompts
per global step; `dynamic_sampling.DynamicSampler` admits them one at a time.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
import contextlib
import queue
from typing import Any, Protocol

PromptBatch = dict[str, Any]


class PromptFeed(Protocol):
  """Source of the prompts that the rollout producer turns into groups."""

  def prompts(self) -> AsyncIterator[PromptBatch]:
    """Yields single prompts; consumed on the learner's event loop."""

  def on_global_step(self, done: bool) -> None:
    """Called by the trainer thread after each global step.

    Args:
      done: Whether training is over, so no further prompt is needed.
    """

  def close(self) -> None:
    """Ends `prompts()`; may be called from any thread, more than once."""


class QueuePromptFeed:
  """Feeds one batch of prompts per global step.

  `prefill_batches` batches are queued up front (`off_policy_steps + 1`), and
  one more after every global step, so prompts run at most that many batches
  ahead of the trainer.
  """

  def __init__(
      self,
      batches: Iterator[PromptBatch],
      *,
      put_batch: Callable[[queue.Queue[PromptBatch | None], PromptBatch], None],
      split: Callable[[PromptBatch], Iterable[PromptBatch]],
      prefill_batches: int,
      loading_span: Callable[
          [], contextlib.AbstractContextManager[Any]
      ] = contextlib.nullcontext,
  ):
    """Initializes the feed and queues the first batches.

    Args:
      batches: Full prompt batches.
      put_batch: Queues a batch, or None when it must not be trained on.
      split: Splits a batch into single prompts.
      prefill_batches: Batches to queue before training starts.
      loading_span: Returns a context manager that traces loading a batch.
    """
    self._batches = batches
    self._put_batch = put_batch
    self._split = split
    self._loading_span = loading_span
    self._queue: queue.Queue[PromptBatch | None] = queue.Queue()
    for _ in range(prefill_batches):
      batch = next(self._batches, None)
      if batch is None:
        self.close()
        break
      self._put_batch(self._queue, batch)

  async def prompts(self) -> AsyncIterator[PromptBatch]:
    """Yields the single prompts of the queued batches.

    Yields:
      One single-prompt batch at a time, until the feed is closed.
    """
    loop = asyncio.get_running_loop()
    while (
        batch := await loop.run_in_executor(None, self._queue.get)
    ) is not None:
      for prompt in self._split(batch):
        yield prompt

  def on_global_step(self, done: bool) -> None:
    """Queues the next batch of prompts, or closes the feed when done.

    Args:
      done: Whether training is over.
    """
    if done:
      self.close()
      return
    with self._loading_span():
      batch = next(self._batches, None)
    if batch is None:
      self.close()
    else:
      self._put_batch(self._queue, batch)

  def close(self) -> None:
    """Ends `prompts()` once the queued batches are consumed."""
    self._queue.put(None)
