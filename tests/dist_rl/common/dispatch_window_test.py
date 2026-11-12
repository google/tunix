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
from collections.abc import Mapping
import dataclasses
from typing import Any
from absl.testing import absltest
from tunix.dist_rl.common import dispatch_window


@dataclasses.dataclass
class _FakeItem:
  prompt_id: int | str
  metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _item(prompt_id: int | str, prompt_idx: int | None = None) -> _FakeItem:
  meta = {"prompt_idx": prompt_idx} if prompt_idx is not None else {}
  return _FakeItem(prompt_id=prompt_id, metadata=meta)


class DispatchWindowTest(absltest.TestCase):

  def test_rejects_negative_max_staleness_or_non_positive_batch_size(self):
    with self.assertRaises(ValueError):
      dispatch_window.DispatchWindow(max_staleness=-1, next_batch_fn=lambda: 0)
    with self.assertRaises(ValueError):
      dispatch_window.DispatchWindow(
          max_staleness=0, next_batch_fn=lambda: 0, full_batch_size=0
      )
    with self.assertRaisesRegex(ValueError, "at least full_batch_size"):
      dispatch_window.DispatchWindow(
          max_staleness=0,
          next_batch_fn=lambda: 0,
          full_batch_size=4,
          max_in_flight_groups=3,
      )
    window = dispatch_window.DispatchWindow(
        max_staleness=0, next_batch_fn=lambda: 0
    )
    with self.assertRaises(ValueError):
      asyncio.run(window.admit(""))
    with self.assertRaises(ValueError):
      asyncio.run(window.admit(None))
    with self.assertRaises(ValueError):
      asyncio.run(window.admit("p0", prompt_idx=-1))
    self.assertEqual(window.in_flight_prompts, 0)

  def test_max_in_flight_groups_overrides_window_capacity(self):
    async def _run():
      cursor = 0
      window = dispatch_window.DispatchWindow(
          max_staleness=0,
          next_batch_fn=lambda: cursor,
          full_batch_size=2,
          max_in_flight_groups=3,
      )
      self.assertEqual(window.window_capacity, 3)
      admitted: list[int] = []

      async def _dispatch_all():
        for idx in range(6):
          await window.admit(f"prompt_{idx}", prompt_idx=idx)
          admitted.append(idx)

      task = asyncio.create_task(_dispatch_all())
      await asyncio.sleep(0.01)
      self.assertEqual(admitted, [0, 1, 2])

      window.on_group_filtered([_item("prompt_0", prompt_idx=0)])
      await asyncio.sleep(0.01)
      self.assertEqual(admitted, [0, 1, 2, 3])

      cursor = 1
      window.release()
      await asyncio.wait_for(task, timeout=1.0)
      self.assertEqual(admitted, [0, 1, 2, 3, 4, 5])
      self.assertEqual(window.in_flight_prompts, 3)

    asyncio.run(_run())

  def test_admits_batches_within_max_staleness_and_parks_beyond(self):
    async def _run():
      cursor = 0
      window = dispatch_window.DispatchWindow(
          max_staleness=1, next_batch_fn=lambda: cursor
      )
      admitted: list[int] = []

      async def _dispatch_all():
        for batch_idx in range(4):
          await window.admit(f"prompt_{batch_idx}")
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
          await window.admit(f"prompt_{batch_idx}")
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

  def test_on_group_filtered_advances_watermark_and_releases(self):
    step = 0
    window = dispatch_window.DispatchWindow(
        max_staleness=1,
        next_batch_fn=lambda: step,
        full_batch_size=2,
        out_of_order_resume=True,
    )
    self.assertEqual(window.next_batch, 0)
    window.on_group_filtered([_item("p1", prompt_idx=1)])
    self.assertEqual(window.min_unsettled_prompt_idx, 0)
    self.assertEqual(window.settled_offsets, [1])
    self.assertTrue(window.is_released)

    window.on_group_filtered([])
    window.on_group_filtered([_item("")])
    window.on_group_filtered([_item("p0", prompt_idx=0)])
    self.assertEqual(window.min_unsettled_prompt_idx, 2)
    self.assertEqual(window.settled_offsets, [])

  def test_checkpoint_metadata_and_out_of_order_resume_admit(self):
    async def _run():
      step = 1
      window = dispatch_window.DispatchWindow(
          max_staleness=0,
          next_batch_fn=lambda: step,
          full_batch_size=2,
          out_of_order_resume=True,
      )
      self.assertEqual(window.checkpoint_metadata(), {})
      window.on_group_filtered([_item("prompt_1", prompt_idx=1)])
      window.record_committed_groups(
          [[], [_item("")], [_item(0, prompt_idx=0)]]
      )
      ckpt = window.checkpoint_metadata(
          pending_groups=[[], [_item("")], [_item("prompt_3", prompt_idx=3)]]
      )
      self.assertEqual(
          ckpt,
          {
              "min_unsettled_prompt_idx": 2,
              "settled_offsets": [1],
          },
      )

      # Construct at cursor=0 and advance cursor=1 before first admit, matching
      # StandardRLProgram.__init__ -> _resume_from_checkpoint.
      cursor = 0
      resumed = dispatch_window.DispatchWindow(
          max_staleness=0,
          next_batch_fn=lambda: cursor,
          full_batch_size=2,
          out_of_order_resume=True,
      )
      cursor = 1
      self.assertEqual(resumed.in_flight_prompts, 0)
      self.assertFalse(resumed.restore_checkpoint_metadata(None))
      self.assertFalse(resumed.restore_checkpoint_metadata({}))
      self.assertFalse(
          resumed.restore_checkpoint_metadata(
              {"min_unsettled_prompt_idx": "bad", "settled_offsets": 123}
          )
      )
      self.assertTrue(resumed.restore_checkpoint_metadata(ckpt))
      self.assertEqual(resumed.dataset_skip_count, 2)

      admitted: list[int] = []

      async def _dispatch():
        for idx in range(resumed.dataset_skip_count, 6):
          pid: int | str = 0 if idx == 0 else f"prompt_{idx}"
          if await resumed.admit(pid, prompt_idx=idx):
            admitted.append(idx)

      task = asyncio.create_task(_dispatch())
      await asyncio.sleep(0.01)
      # Prompts 0, 1 are below `dataset_skip_count=2`; prompt_3 (offset +1) is
      # skipped; step 1 (max_staleness=0) admits 2 non-skipped prompts (2, 4),
      # then parks prompt_5.
      self.assertEqual(admitted, [2, 4])
      self.assertEqual(resumed.in_flight_prompts, 2)

      cursor = 2
      resumed.release()
      await asyncio.wait_for(task, timeout=1.0)
      self.assertEqual(admitted, [2, 4, 5])
      self.assertEqual(resumed.in_flight_prompts, 1)

    asyncio.run(_run())

  def test_repeat_dataset_stable_prompt_ids_work_across_epochs_after_resume(
      self,
  ):
    """Stable `prompt_id`s on `.repeat()` datasets must not spin-lock Epoch 2."""

    async def _run():
      step = 0
      window = dispatch_window.DispatchWindow(
          max_staleness=1,
          next_batch_fn=lambda: step,
          full_batch_size=2,
          out_of_order_resume=True,
      )
      # Epoch 1 has 2 prompts (`q_0`, `q_1`). Drop `q_0` at idx=0, commit `q_1`
      # at idx=1 and `q_0` (Epoch 2!) at idx=2, leaving idx=3 in flight.
      for idx in range(4):
        self.assertTrue(await window.admit(f"q_{idx % 2}", prompt_idx=idx))
      window.on_group_filtered([_item("q_0", prompt_idx=0)])
      ckpt = window.checkpoint_metadata(
          pending_groups=[
              [_item("q_1", prompt_idx=1)],
              [_item("q_0", prompt_idx=2)],
          ]
      )
      self.assertEqual(
          ckpt,
          {"min_unsettled_prompt_idx": 3, "settled_offsets": []},
      )

      cursor = 1
      resumed = dispatch_window.DispatchWindow(
          max_staleness=1,
          next_batch_fn=lambda: cursor,
          full_batch_size=2,
          out_of_order_resume=True,
      )
      self.assertTrue(resumed.restore_checkpoint_metadata(ckpt))
      self.assertEqual(resumed.dataset_skip_count, 3)

      # Epoch 2+ yields the same `q_0`, `q_1` prompt_ids at higher `prompt_idx`
      # and must be admitted without spinning or permanently blacklisting `q_0`.
      admitted = []
      for idx in range(resumed.dataset_skip_count, 7):
        if await resumed.admit(f"q_{idx % 2}", prompt_idx=idx):
          admitted.append((idx, f"q_{idx % 2}"))
      self.assertEqual(
          admitted,
          [(3, "q_1"), (4, "q_0"), (5, "q_1"), (6, "q_0")],
      )

    asyncio.run(_run())

  def test_prefix_resume_followed_by_first_drop_preserves_prefix_watermark(
      self,
  ):
    async def _run():
      # Run 1 wrote an on-policy checkpoint with no watermark metadata at step=2
      # (prompts 0..3 trained). Run 2 resumes by prefix (`dataset_skip_count=4`)
      # and then drops prompt_4, switching to watermark checkpoints.
      step = 0
      window = dispatch_window.DispatchWindow(
          max_staleness=0,
          next_batch_fn=lambda: step,
          full_batch_size=2,
          out_of_order_resume=True,
      )
      step = 2
      self.assertFalse(window.restore_checkpoint_metadata({}))
      self.assertEqual(window.dataset_skip_count, 4)

      for idx in (4, 5):
        self.assertTrue(await window.admit(f"p_{idx}", prompt_idx=idx))
      window.on_group_filtered([_item("p_4", prompt_idx=4)])
      self.assertTrue(await window.admit("p_6", prompt_idx=6))
      ckpt = window.checkpoint_metadata(
          pending_groups=[
              [_item("p_5", prompt_idx=5)],
              [_item("p_6", prompt_idx=6)],
          ]
      )
      # Must advance all the way to 7 (including Run 1's prefix 0..3), NOT 0!
      self.assertEqual(
          ckpt,
          {"min_unsettled_prompt_idx": 7, "settled_offsets": []},
      )

    asyncio.run(_run())

  def test_in_order_prefix_resume_anchors_start_step_on_first_admit(self):
    async def _run():
      cursor = 0
      window = dispatch_window.DispatchWindow(
          max_staleness=1,
          next_batch_fn=lambda: cursor,
          full_batch_size=2,
          out_of_order_resume=False,
      )
      cursor = 2
      self.assertEqual(window.dataset_skip_count, 4)

      admitted: list[int] = []

      async def _dispatch():
        for idx in range(4, 10):
          if await window.admit(f"prompt_{idx}", prompt_idx=idx):
            admitted.append(idx)

      task = asyncio.create_task(_dispatch())
      await asyncio.sleep(0.01)
      self.assertEqual(admitted, [4, 5, 6, 7])
      self.assertEqual(window.in_flight_prompts, 4)

      cursor = 3
      window.release()
      await asyncio.wait_for(task, timeout=1.0)
      self.assertEqual(admitted, [4, 5, 6, 7, 8, 9])
      self.assertEqual(window.in_flight_prompts, 4)

    asyncio.run(_run())

  def test_fresh_run_recording_commits_does_not_flip_has_resume_state(
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
          if await window.admit(f"prompt_{idx}", prompt_idx=idx):
            admitted.append(idx)

      task = asyncio.create_task(_dispatch())
      await asyncio.sleep(0.01)
      self.assertEqual(admitted, [0, 1, 2, 3])
      self.assertFalse(window.has_resume_state)

      window.record_committed_groups(
          [[_item("prompt_0", 0)], [_item("prompt_1", 1)]]
      )
      self.assertFalse(window.has_resume_state)
      step = 1
      window.release()
      await asyncio.sleep(0.01)
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
    window.record_committed_groups([[_item("prompt_0", 0)]])
    self.assertEqual(
        window.checkpoint_metadata(pending_groups=[[_item("prompt_1", 1)]]),
        {},
    )
    self.assertFalse(
        window.restore_checkpoint_metadata(
            {"min_unsettled_prompt_idx": 1, "settled_offsets": []}
        )
    )
    self.assertFalse(window.has_resume_state)
    self.assertEqual(window.dataset_skip_count, 2)

  def test_admit_replenishes_one_prompt_per_filtered_group(self):
    async def _run():
      step = 0
      window = dispatch_window.DispatchWindow(
          max_staleness=1,
          next_batch_fn=lambda: step,
          full_batch_size=4,
          out_of_order_resume=True,
      )
      admitted: list[int] = []

      async def _dispatch():
        for idx in range(12):
          if await window.admit(f"prompt_{idx}", prompt_idx=idx):
            admitted.append(idx)

      task = asyncio.create_task(_dispatch())
      await asyncio.sleep(0.01)
      self.assertEqual(admitted, list(range(8)))
      self.assertEqual(window.in_flight_prompts, 8)

      window.on_group_filtered([_item("prompt_0", 0)])
      await asyncio.sleep(0.01)
      self.assertEqual(admitted, list(range(9)))
      self.assertEqual(window.in_flight_prompts, 8)

      window.on_group_filtered([_item("prompt_1", 1)])
      await asyncio.sleep(0.01)
      self.assertEqual(admitted, list(range(10)))
      self.assertEqual(window.in_flight_prompts, 8)

      step = 1
      window.release()
      await asyncio.wait_for(task, timeout=1.0)
      self.assertEqual(admitted, list(range(12)))
      self.assertEqual(window.in_flight_prompts, 6)

    asyncio.run(_run())

  def test_incomplete_batch_eof_writes_watermark_and_prefix_resume_synthetic_idx(
      self,
  ):
    # Sub-batch EOF: `full_batch_size=4`, only 2 prompts settled at `step=0`.
    # `min_unsettled=2 != (0 + 1) * 4`, so `checkpoint_metadata` MUST emit
    # `min_unsettled_prompt_idx=2` instead of `{}`!
    window = dispatch_window.DispatchWindow(
        max_staleness=0,
        next_batch_fn=lambda: 0,
        full_batch_size=4,
        out_of_order_resume=True,
    )
    ckpt = window.checkpoint_metadata(
        pending_groups=[[_item("p0", 0)], [_item("p1", 1)]]
    )
    self.assertEqual(
        ckpt, {"min_unsettled_prompt_idx": 2, "settled_offsets": []}
    )

    # Synthetic `prompt_idx` (`prompt_idx=None`) on a prefix-resumed window
    # (`step=2`, `full_batch_size=2`) starts at `4`, not `0`.
    step = 0
    resumed = dispatch_window.DispatchWindow(
        max_staleness=0,
        next_batch_fn=lambda: step,
        full_batch_size=2,
        out_of_order_resume=True,
    )
    step = 2
    self.assertTrue(asyncio.run(resumed.admit("p4")))
    resumed.on_group_filtered([_item("p4", 4)])
    self.assertEqual(resumed.min_unsettled_prompt_idx, 5)


if __name__ == "__main__":
  absltest.main()
