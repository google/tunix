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
import dataclasses
from absl.testing import absltest
from tunix.experimental.common import dispatch_window


@dataclasses.dataclass
class _FakeItem:
  prompt_id: str


class DispatchWindowTest(absltest.TestCase):

  def test_rejects_negative_max_staleness_or_non_positive_batch_size(self):
    with self.assertRaises(ValueError):
      dispatch_window.DispatchWindow(max_staleness=-1, next_batch_fn=lambda: 0)
    with self.assertRaises(ValueError):
      dispatch_window.DispatchWindow(
          max_staleness=0, next_batch_fn=lambda: 0, full_batch_size=0
      )

  def test_admits_batches_within_max_staleness_and_parks_beyond(self):
    async def _run():
      cursor = 0
      window = dispatch_window.DispatchWindow(
          max_staleness=1, next_batch_fn=lambda: cursor
      )
      admitted: list[int] = []

      async def _dispatch_all():
        for batch_idx in range(4):
          await window.wait_for(batch_idx)
          admitted.append(batch_idx)

      task = asyncio.create_task(_dispatch_all())
      await asyncio.sleep(0.01)
      # cursor=0, max_staleness=1 -> batches 0 and 1 admitted; batch 2 parked.
      self.assertEqual(admitted, [0, 1])

      cursor = 1
      window.release()
      await asyncio.sleep(0.01)
      self.assertEqual(admitted, [0, 1, 2])

      cursor = 2
      window.release()
      await asyncio.wait_for(task, timeout=1.0)
      self.assertEqual(admitted, [0, 1, 2, 3])

    asyncio.run(_run())

  def test_skipping_holes_opens_multiple_batches_on_single_release(self):
    async def _run():
      cursor = 0
      window = dispatch_window.DispatchWindow(
          max_staleness=0, next_batch_fn=lambda: cursor
      )
      admitted: list[int] = []

      async def _dispatch_all():
        for batch_idx in range(4):
          await window.wait_for(batch_idx)
          admitted.append(batch_idx)

      task = asyncio.create_task(_dispatch_all())
      await asyncio.sleep(0.01)
      self.assertEqual(admitted, [0])

      # Cursor jumps from 0 to 3 (batches 1 and 2 were empty holes).
      cursor = 3
      window.release()
      await asyncio.wait_for(task, timeout=1.0)
      self.assertEqual(admitted, [0, 1, 2, 3])

    asyncio.run(_run())

  def test_on_group_filtered_advances_next_batch_and_records_skipped_ids(self):
    step = 0
    window = dispatch_window.DispatchWindow(
        max_staleness=1,
        next_batch_fn=lambda: step,
        full_batch_size=2,
        out_of_order_resume=True,
    )
    self.assertEqual(window.next_batch, 0)
    window.on_group_filtered([_FakeItem("p1")])
    # ceil(1 / 2) = 1 -> next_batch shifts from 0 to 1.
    self.assertEqual(window.next_batch, 1)
    self.assertEqual(window.skipped_prompt_ids, {"p1"})
    self.assertTrue(window.is_released)

    window.on_group_filtered([_FakeItem("p2")])
    # ceil(2 / 2) = 1 -> still 1 until a 3rd group is filtered.
    self.assertEqual(window.next_batch, 1)
    window.on_group_filtered([_FakeItem("p3")])
    self.assertEqual(window.next_batch, 2)

  def test_checkpoint_metadata_and_out_of_order_resume_admit(self):
    async def _run():
      step = 1
      window = dispatch_window.DispatchWindow(
          max_staleness=0,
          next_batch_fn=lambda: step,
          full_batch_size=2,
          out_of_order_resume=True,
      )
      window.on_group_filtered([_FakeItem("prompt_1")])
      window.record_committed_groups([[_FakeItem("prompt_0")]])
      ckpt = window.checkpoint_metadata(
          pending_groups=[[_FakeItem("prompt_3")]]
      )
      self.assertEqual(
          ckpt,
          {
              "committed_prompt_ids": ["prompt_0", "prompt_3"],
              "skipped_prompt_ids": ["prompt_1"],
          },
      )

      resumed = dispatch_window.DispatchWindow(
          max_staleness=0,
          next_batch_fn=lambda: step,
          full_batch_size=2,
          out_of_order_resume=True,
      )
      self.assertTrue(resumed.restore_checkpoint_metadata(ckpt))
      self.assertEqual(resumed.dataset_skip_count, 0)

      admitted: list[int] = []

      async def _dispatch():
        for idx in range(6):
          if await resumed.admit(idx, {"prompt_id": f"prompt_{idx}"}):
            admitted.append(idx)

      task = asyncio.create_task(_dispatch())
      await asyncio.sleep(0.01)
      # Step 1 (max_staleness=0) admits 2 non-skipped prompts (prompt_2,
      # prompt_4) into batch_idx=1, then parks prompt_5 (batch_idx=2).
      self.assertEqual(admitted, [2, 4])

      step = 2
      resumed.release()
      await asyncio.wait_for(task, timeout=1.0)
      self.assertEqual(admitted, [2, 4, 5])

    asyncio.run(_run())

  def test_fresh_run_recording_commits_does_not_flip_uses_prompt_id_resume(
      self,
  ):
    async def _run():
      step = 0
      window = dispatch_window.DispatchWindow(
          max_staleness=1,
          next_batch_fn=lambda: step,
          full_batch_size=2,
          out_of_order_resume=True,
      )
      admitted: list[int] = []

      async def _dispatch():
        for idx in range(8):
          if await window.admit(idx, {"prompt_id": f"prompt_{idx}"}):
            admitted.append(idx)

      task = asyncio.create_task(_dispatch())
      await asyncio.sleep(0.01)
      # Step 0 with max_staleness=1 admits batches 0 and 1 (prompts 0..3).
      self.assertEqual(admitted, [0, 1, 2, 3])
      self.assertFalse(window.uses_prompt_id_resume)

      # Recording committed groups at Step 0 must not flip uses_prompt_id_resume
      # or admit 2 * max_staleness + 1 batches.
      window.record_committed_groups(
          [[_FakeItem("prompt_0")], [_FakeItem("prompt_1")]]
      )
      self.assertFalse(window.uses_prompt_id_resume)
      step = 1
      window.release()
      await asyncio.sleep(0.01)
      # Only batch 2 (prompts 4, 5) should be admitted; batch 3 (prompts 6, 7)
      # remains parked.
      self.assertEqual(admitted, [0, 1, 2, 3, 4, 5])

      step = 2
      window.release()
      await asyncio.wait_for(task, timeout=1.0)
      self.assertEqual(admitted, [0, 1, 2, 3, 4, 5, 6, 7])

    asyncio.run(_run())

  def test_in_order_resume_skips_checkpoint_metadata(self):
    window = dispatch_window.DispatchWindow(
        max_staleness=2,
        next_batch_fn=lambda: 1,
        full_batch_size=2,
        out_of_order_resume=False,
    )
    window.record_committed_groups([[_FakeItem("prompt_0")]])
    self.assertEqual(window.committed_prompt_ids, set())
    self.assertEqual(
        window.checkpoint_metadata(pending_groups=[[_FakeItem("prompt_1")]]),
        {},
    )
    self.assertFalse(
        window.restore_checkpoint_metadata(
            {"committed_prompt_ids": ["prompt_0"], "skipped_prompt_ids": []}
        )
    )
    self.assertFalse(window.uses_prompt_id_resume)
    self.assertEqual(window.dataset_skip_count, 2)


if __name__ == "__main__":
  absltest.main()
