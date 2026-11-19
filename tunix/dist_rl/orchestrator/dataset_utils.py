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

"""Asynchronous dataset iteration and prefetching utilities for RL programs."""

import asyncio
import collections
from collections.abc import AsyncIterator, Iterable, Iterator
import concurrent.futures
import contextlib
import itertools
import threading
from typing import TypeVar

_T = TypeVar("_T")

_DATASET_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="tunix-dataset",
)
_DATASET_END = object()


def total_prompt_groups(
    max_steps: int,
    batch_size: int,
    max_staleness: int = 0,
) -> int:
  """Returns total prompt groups to yield, including off-policy lookahead."""
  extra_steps = (
      max(max_staleness * 4, 6, max_steps // 4) if max_staleness > 0 else 0
  )
  return (max_steps + extra_steps) * batch_size


async def iter_dataset_async(
    dataset: Iterable[_T],
    *,
    skip_count: int = 0,
    prefetch_size: int = 1,
) -> AsyncIterator[tuple[int, _T]]:
  """Iterates over `dataset` on a background thread with bounded prefetching.

  Offloads `iter(dataset)`, checkpoint-resume prefix skipping (`skip_count`),
  and each `next()` call to `_DATASET_EXECUTOR` so blocking I/O or CPU work in
  the data loader never stalls the `asyncio` event loop, while prefetching up to
  `prefetch_size` items ahead to overlap dataset reads with rollout and
  training compute.

  In-memory `list` and `tuple` datasets use a zero-overhead `O(1)` slice
  fast-path directly on the event loop.

  Args:
    dataset: Synchronous prompt iterable or generator.
    skip_count: Number of leading items to skip (e.g. when resuming from a
      checkpoint).
    prefetch_size: Maximum number of `(prompt_idx, prompt_item)` pairs to buffer
      ahead of the consumer.

  Yields:
    `(prompt_idx, prompt_item)` tuples with 0-based `prompt_idx` matching
    `enumerate(dataset)`.
  """
  if isinstance(dataset, (list, tuple)):
    for prompt_idx in range(max(0, skip_count), len(dataset)):
      yield prompt_idx, dataset[prompt_idx]
    return

  loop = asyncio.get_running_loop()
  queue: asyncio.Queue[
      tuple[str, tuple[int, _T] | BaseException | None]
  ] = asyncio.Queue(maxsize=max(1, prefetch_size))
  state_lock = threading.Lock()
  in_next = False
  close_requested = False

  def _init_and_skip() -> tuple[Iterator[_T], Iterator[tuple[int, _T]]]:
    raw_it = iter(dataset)
    try:
      enum_it = enumerate(raw_it)
      if skip_count > 0:
        collections.deque(itertools.islice(enum_it, skip_count), maxlen=0)
      return raw_it, enum_it
    except BaseException:
      close_fn = getattr(raw_it, "close", None)
      if callable(close_fn):
        close_fn()
      raise

  def _next_item(
      raw_it: Iterator[_T], enum_it: Iterator[tuple[int, _T]]
  ) -> tuple[int, _T] | object:
    nonlocal in_next
    with state_lock:
      if close_requested:
        return _DATASET_END
      in_next = True
    try:
      return next(enum_it, _DATASET_END)
    finally:
      should_close_now = False
      with state_lock:
        in_next = False
        should_close_now = close_requested
      if should_close_now:
        close_fn = getattr(raw_it, "close", None)
        if callable(close_fn):
          close_fn()

  async def _producer() -> None:
    nonlocal close_requested
    raw_it: Iterator[_T] | None = None
    try:
      raw_it, enum_it = await loop.run_in_executor(
          _DATASET_EXECUTOR, _init_and_skip
      )
      while True:
        item = await loop.run_in_executor(
            _DATASET_EXECUTOR, _next_item, raw_it, enum_it
        )
        if item is _DATASET_END:
          await queue.put(("eof", None))
          return
        assert isinstance(item, tuple)
        await queue.put(("item", item))
    except asyncio.CancelledError:
      raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
      await queue.put(("error", exc))
    finally:
      if raw_it is not None:
        should_close_now = False
        with state_lock:
          close_requested = True
          should_close_now = not in_next
        if should_close_now:
          close_fn = getattr(raw_it, "close", None)
          if callable(close_fn):
            close_fn()

  producer_task = asyncio.create_task(_producer())
  try:
    while True:
      kind, payload = await queue.get()
      if kind == "eof":
        break
      if kind == "error":
        assert isinstance(payload, BaseException)
        raise payload
      assert isinstance(payload, tuple)
      yield payload
  finally:
    if not producer_task.done():
      producer_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await producer_task
