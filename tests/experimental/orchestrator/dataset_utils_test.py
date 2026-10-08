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

import asyncio
import time

from absl.testing import absltest
from tunix.experimental.orchestrator import dataset_utils


class DatasetUtilsTest(absltest.TestCase):
  """Tests for `dataset_utils.iter_dataset_async`."""

  def test_list_and_tuple_fast_path_with_skip_count(self):
    async def _run():
      list_items = [
          item
          async for item in dataset_utils.iter_dataset_async(
              ["p0", "p1", "p2", "p3"], skip_count=2, prefetch_size=2
          )
      ]
      tuple_items = [
          item
          async for item in dataset_utils.iter_dataset_async(
              ("a", "b", "c"), skip_count=0, prefetch_size=1
          )
      ]
      self.assertEqual(list_items, [(2, "p2"), (3, "p3")])
      self.assertEqual(tuple_items, [(0, "a"), (1, "b"), (2, "c")])

    asyncio.run(_run())

  def test_blocking_generator_dataset_does_not_freeze_event_loop(self):
    sleep_per_item_s = 0.025
    num_items = 4

    def _blocking_gen():
      for i in range(num_items):
        time.sleep(sleep_per_item_s)
        yield f"prompt_{i}"

    async def _run():
      heartbeat_gaps_s: list[float] = []
      stop_hb = False

      async def _heartbeat():
        last = time.perf_counter()
        while not stop_hb:
          await asyncio.sleep(0.002)
          now = time.perf_counter()
          heartbeat_gaps_s.append(now - last)
          last = now

      hb_task = asyncio.create_task(_heartbeat())
      try:
        collected: list[tuple[int, str]] = []
        async for idx, example in dataset_utils.iter_dataset_async(
            _blocking_gen(), skip_count=0, prefetch_size=2
        ):
          collected.append((idx, example))
      finally:
        stop_hb = True
        await hb_task

      self.assertEqual(
          collected,
          [(0, "prompt_0"), (1, "prompt_1"), (2, "prompt_2"), (3, "prompt_3")],
      )
      self.assertGreaterEqual(len(heartbeat_gaps_s), 10)
      self.assertLess(max(heartbeat_gaps_s), sleep_per_item_s * 1.8)

    asyncio.run(_run())

  def test_generator_dataset_skip_count_and_close_on_early_break(self):
    closed = False
    yielded_indices: list[int] = []

    def _infinite_gen():
      nonlocal closed
      try:
        i = 0
        while True:
          yielded_indices.append(i)
          yield {"prompt": f"p_{i}", "id": f"id_{i}"}
          i += 1
      finally:
        closed = True

    async def _run():
      collected: list[tuple[int, dict[str, str]]] = []
      async for idx, example in dataset_utils.iter_dataset_async(
          _infinite_gen(), skip_count=3, prefetch_size=2
      ):
        collected.append((idx, example))
        if len(collected) == 2:
          break

      self.assertEqual(
          collected,
          [
              (3, {"prompt": "p_3", "id": "id_3"}),
              (4, {"prompt": "p_4", "id": "id_4"}),
          ],
      )

    asyncio.run(_run())
    self.assertTrue(closed)

  def test_generator_dataset_exception_propagates_cleanly(self):
    closed = False

    def _failing_gen():
      nonlocal closed
      try:
        yield "prompt_0"
        raise RuntimeError("corrupt dataset shard")
      finally:
        closed = True

    async def _run():
      seen: list[tuple[int, str]] = []
      with self.assertRaisesRegex(RuntimeError, "corrupt dataset shard"):
        async for item in dataset_utils.iter_dataset_async(
            _failing_gen(), skip_count=0, prefetch_size=2
        ):
          seen.append(item)
      self.assertEqual(seen, [(0, "prompt_0")])

    asyncio.run(_run())
    self.assertTrue(closed)

  def test_total_prompt_groups(self):
    self.assertEqual(
        dataset_utils.total_prompt_groups(
            max_steps=2, batch_size=3, max_staleness=0
        ),
        6,
    )
    self.assertEqual(
        dataset_utils.total_prompt_groups(
            max_steps=2, batch_size=3, max_staleness=2
        ),
        30,
    )


if __name__ == "__main__":
  absltest.main()
