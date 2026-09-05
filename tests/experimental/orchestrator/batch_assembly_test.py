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

"""Unit tests for Universal BatchAssembler (SequencePacked, GRPO, & Padded)."""

import dataclasses

from absl.testing import absltest
import jax
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import batch_assembly


class HelperFunctionsTest(absltest.TestCase):

  def test_left_pad_shorter_array(self):
    out, mask = batch_assembly._left_pad(
        np.array([1, 2, 3]), length=5, pad_id=0
    )
    np.testing.assert_array_equal(out, [0, 0, 1, 2, 3])
    np.testing.assert_array_equal(mask, [0.0, 0.0, 1.0, 1.0, 1.0])

  def test_left_pad_longer_array(self):
    out, mask = batch_assembly._left_pad(
        np.array([1, 2, 3, 4, 5]), length=3, pad_id=0
    )
    np.testing.assert_array_equal(out, [3, 4, 5])
    np.testing.assert_array_equal(mask, [1.0, 1.0, 1.0])

  def test_left_pad_empty_array(self):
    out, mask = batch_assembly._left_pad(
        np.array([], dtype=np.int32), length=4, pad_id=0
    )
    np.testing.assert_array_equal(out, [0, 0, 0, 0])
    np.testing.assert_array_equal(mask, [0.0, 0.0, 0.0, 0.0])

  def test_right_pad_shorter_array(self):
    out, mask = batch_assembly._right_pad(
        np.array([1, 2]), length=4, pad_value=0, dtype=np.int32
    )
    np.testing.assert_array_equal(out, [1, 2, 0, 0])
    np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0, 0.0])

  def test_right_pad_longer_array(self):
    out, mask = batch_assembly._right_pad(
        np.array([1, 2, 3, 4]), length=2, pad_value=0, dtype=np.int32
    )
    np.testing.assert_array_equal(out, [1, 2])
    np.testing.assert_array_equal(mask, [1.0, 1.0])

  def test_right_pad_empty_array(self):
    out, mask = batch_assembly._right_pad(
        np.array([], dtype=np.int32), length=3, pad_value=0, dtype=np.int32
    )
    np.testing.assert_array_equal(out, [0, 0, 0])
    np.testing.assert_array_equal(mask, [0.0, 0.0, 0.0])

  def test_completion_aligned_pads_shorter_values(self):
    result = batch_assembly._completion_aligned(
        values=np.array([1.0, 2.0], dtype=np.float32),
        completion_len=4,
        max_response_length=6,
    )
    np.testing.assert_allclose(result, [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])

  def test_completion_aligned_handles_none(self):
    result = batch_assembly._completion_aligned(
        values=None,
        completion_len=3,
        max_response_length=5,
        fill_value=2.0,
    )
    np.testing.assert_allclose(result, [2.0, 2.0, 2.0, 0.0, 0.0])

  def test_completion_aligned_scalar_broadcast(self):
    result = batch_assembly._completion_aligned(
        values=1.5,
        completion_len=3,
        max_response_length=5,
    )
    np.testing.assert_allclose(result, [1.5, 1.5, 1.5, 0.0, 0.0])

  def test_completion_aligned_slices_full_sequence(self):
    values = np.array([10.0, 20.0, 1.0, 2.0, 3.0], dtype=np.float32)
    result = batch_assembly._completion_aligned(
        values=values,
        completion_len=3,
        max_response_length=5,
        prompt_len=2,
    )
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 0.0, 0.0])

  def test_as_1d_with_scalar(self):
    out = batch_assembly._as_1d(42, np.int32)
    np.testing.assert_array_equal(out, [42])
    self.assertEqual(out.shape, (1,))
    self.assertEqual(out.dtype, np.int32)

  def test_as_1d_with_multidimensional_array(self):
    out = batch_assembly._as_1d([[1, 2], [3, 4]], np.float32)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])
    self.assertEqual(out.shape, (4,))
    self.assertEqual(out.dtype, np.float32)

  def test_require_unbatched_valid(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([3, 4], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.ones(2, dtype=np.float32),
    )
    # Should not raise.
    batch_assembly._require_unbatched(payload)

    # Optional fields set to None should also pass.
    payload_optional_none = datatypes.RLTrainerPayload(
        prompt_ids=None,
        prompt_mask=None,
        completion_ids=np.array([3, 4], dtype=np.int32),
        completion_mask=None,
        advantages=np.ones(2, dtype=np.float32),
    )
    batch_assembly._require_unbatched(payload_optional_none)

  def test_require_unbatched_missing_completion_ids_raises(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=None,
        completion_mask=None,
        advantages=np.ones(2, dtype=np.float32),
    )
    with self.assertRaisesRegex(
        ValueError,
        "RLTrainerPayload.completion_ids is required for sequence packing",
    ):
      batch_assembly._require_unbatched(payload)

  def test_require_unbatched_batched_rank_raises(self):
    for field_name in (
        "prompt_ids",
        "prompt_mask",
        "completion_ids",
        "completion_mask",
    ):
      kwargs = {
          "prompt_ids": np.array([1, 2], dtype=np.int32),
          "prompt_mask": np.ones(2, dtype=np.float32),
          "completion_ids": np.array([3, 4], dtype=np.int32),
          "completion_mask": np.ones(2, dtype=np.float32),
          "advantages": np.ones(2, dtype=np.float32),
      }
      kwargs[field_name] = np.zeros((2, 2))
      payload = datatypes.RLTrainerPayload(**kwargs)
      with self.assertRaisesRegex(
          ValueError,
          f"RLTrainerPayload.{field_name} has rank 2; sequence packing takes"
          " UNBATCHED payloads",
      ):
        batch_assembly._require_unbatched(payload)


class WithRefPerTokenLogpsTest(absltest.TestCase):

  def _make_payload(self, b=2, p=3, c=4):
    return datatypes.RLTrainerPayload(
        prompt_ids=np.ones((b, p), dtype=np.int32),
        prompt_mask=np.ones((b, p), dtype=np.float32),
        completion_ids=np.ones((b, c), dtype=np.int32),
        completion_mask=np.ones((b, c), dtype=np.float32),
        advantages=np.ones((b, c), dtype=np.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
    )

  def test_success_with_ndarray(self):
    batch = self._make_payload(b=2, p=3, c=4)
    ref_logps = np.full((2, 4), -0.5, dtype=np.float32)
    updated = batch_assembly.with_ref_per_token_logps(batch, ref_logps)

    self.assertIsInstance(updated, datatypes.RLTrainerPayload)
    self.assertIsNotNone(updated.ref_per_token_logps)
    self.assertEqual(updated.ref_per_token_logps.shape, (2, 4))
    np.testing.assert_allclose(updated.ref_per_token_logps, ref_logps)
    np.testing.assert_array_equal(updated.prompt_ids, batch.prompt_ids)
    np.testing.assert_array_equal(updated.completion_ids, batch.completion_ids)

  def test_success_with_logprobs_response(self):
    batch = self._make_payload(b=2, p=3, c=4)
    resp = datatypes.LogprobsResponse(
        per_token_logps=np.full((2, 4), -0.8, dtype=np.float32)
    )
    updated = batch_assembly.with_ref_per_token_logps(batch, resp)

    self.assertIsInstance(updated, datatypes.RLTrainerPayload)
    self.assertIsNotNone(updated.ref_per_token_logps)
    self.assertEqual(updated.ref_per_token_logps.shape, (2, 4))
    np.testing.assert_allclose(
        updated.ref_per_token_logps, resp.per_token_logps
    )

  def test_error_in_logprobs_response_raises_runtime_error(self):
    batch = self._make_payload(b=2, p=3, c=4)
    resp = datatypes.LogprobsResponse(
        per_token_logps=None,
        error=datatypes.ErrorInfo(
            error_type="InferenceError", message="inference worker failed"
        ),
    )
    with self.assertRaisesRegex(RuntimeError, "inference worker failed"):
      batch_assembly.with_ref_per_token_logps(batch, resp)

  def test_rejects_unsupported_type(self):
    with self.assertRaisesRegex(TypeError, "expects a padded RLTrainerPayload"):
      batch_assembly.with_ref_per_token_logps(
          {"raw": "batch"}, np.zeros((2, 2))
      )

  def test_mismatched_shape_raises_value_error(self):
    batch = self._make_payload(b=2, p=3, c=4)
    bad_shape_logps = np.zeros((2, 3), dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        "Reference logps shape must match padded completion_ids shape",
    ):
      batch_assembly.with_ref_per_token_logps(batch, bad_shape_logps)


class SequencePackedBatchAssemblerTest(absltest.TestCase):

  def _make_assembler(self, max_packed_len=16, **kwargs):
    defaults = dict(
        batch_size=1,
        group_size=2,
        mini_batch_size=2,
        max_packed_len=max_packed_len,
    )
    defaults.update(kwargs)
    return batch_assembly.SequencePackedBatchAssembler(**defaults)

  def _drain(self, assembler, items):
    """Fully packs `items` via the streaming contract (feed then flush)."""
    return [
        batch.payload
        for batch in (assembler.feed(items) + assembler.flush())
    ]

  def test_empty_input_returns_empty_list(self):
    assembler = self._make_assembler(max_packed_len=16)
    self.assertEmpty(self._drain(assembler, []))

  def test_sequence_packed_assembler_with_trainer_payload(self):
    payload1 = _make_payload(2, 2, advantage=1.5)
    payload2 = _make_payload(3, 1, advantage=-0.5)

    assembler = self._make_assembler(max_packed_len=16)
    payloads = self._drain(assembler, [payload1, payload2])

    self.assertLen(payloads, 1)
    payload = payloads[0]
    self.assertEqual(payload.completion_ids.shape, (1, 16))
    self.assertEqual(payload.completion_mask.shape, (1, 16))
    self.assertEqual(payload.prompt_ids.shape, (1, 0))
    self.assertEqual(payload.prompt_mask.shape, (1, 0))
    self.assertEqual(payload.segment_ids.shape, (1, 16))
    self.assertEqual(payload.segment_positions.shape, (1, 16))
    self.assertEqual(payload.advantages.shape, (1, 16))

    # Check segment boundaries
    seg_ids = payload.segment_ids[0]
    self.assertTrue(np.all(seg_ids[:4] == 1))
    self.assertTrue(np.all(seg_ids[4:8] == 2))
    self.assertTrue(np.all(seg_ids[8:] == 0))

    np.testing.assert_array_equal(
        payload.segment_positions[0][:8], [0, 1, 2, 3, 0, 1, 2, 3]
    )
    np.testing.assert_array_equal(payload.segment_ids[0] > 0, [1] * 8 + [0] * 8)
    real_prompts = (payload.segment_ids[0] > 0) & (
        payload.completion_mask[0] == 0
    )
    np.testing.assert_array_equal(
        real_prompts, [1, 1, 0, 0, 1, 1, 1, 0] + [0] * 8
    )
    np.testing.assert_array_equal(
        payload.completion_mask[0], [0, 0, 1, 1, 0, 0, 0, 1] + [0] * 8
    )
    np.testing.assert_allclose(
        payload.advantages[0], [0, 0, 1.5, 1.5, 0, 0, 0, -0.5] + [0] * 8
    )

  def test_sequence_packed_assembler_rejects_partial_optional_fields(self):
    items = [
        _make_payload(2, 2),
        _make_payload(2, 2, ref_logps=np.full(2, -0.3, dtype=np.float32)),
    ]
    assembler = self._make_assembler(max_packed_len=16)

    with self.assertRaisesRegex(ValueError, "ref_per_token_logps"):
      self._drain(assembler, items)

  def test_streaming_feed_buffers_and_packs_across_groups(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        group_size=1,
        mini_batch_size=3,
        max_packed_len=10,
    )
    p1 = dataclasses.replace(_make_payload(2, 2), metadata={"traj_id": "t1"})
    p2 = dataclasses.replace(_make_payload(2, 3), metadata={"traj_id": "t2"})
    p3 = dataclasses.replace(_make_payload(1, 2), metadata={"traj_id": "t3"})

    # Feed 1 (4 tokens < 10): buffers without emitting.
    self.assertEmpty(assembler.feed([p1]))

    # Feed 2 (4 + 5 = 9 tokens < 10): still held back, cannot fill a chunk yet.
    self.assertEmpty(assembler.feed([p2]))

    # Feed 3 reaches total_step_rollouts=3 -> drains the whole buffer (12
    # tokens) into a full chunk plus a final remainder chunk.
    batches3 = assembler.feed([p3])
    self.assertLen(batches3, 2)
    self.assertFalse(batches3[0].is_final_batch)
    self.assertEqual(set(batches3[0].trajectory_ids), {"t1", "t2"})
    self.assertEqual(
        set(batches3[0].payload.metadata["trajectory_ids"]), {"t1", "t2"}
    )
    self.assertTrue(batches3[1].is_final_batch)
    self.assertEqual(batches3[1].trajectory_ids, ("t3",))

  def test_streaming_feed_emits_full_chunk_mid_step(self):
    # batch_size=1, max_packed_len=6 -> a full chunk holds 6 tokens.
    # mini_batch_size is large so the step boundary is never reached here.
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        group_size=1,
        mini_batch_size=10,
        max_packed_len=6,
    )
    p1 = dataclasses.replace(_make_payload(2, 2), metadata={"traj_id": "t1"})
    p2 = dataclasses.replace(_make_payload(2, 2), metadata={"traj_id": "t2"})

    # 4 tokens < 6: buffered.
    self.assertEmpty(assembler.feed([p1]))

    # 4 + 4 = 8 >= 6: emits one full chunk mid-step (not final), holding the
    # remainder back for the next feed.
    mid = assembler.feed([p2])
    self.assertLen(mid, 1)
    self.assertFalse(mid[0].is_final_batch)
    self.assertEqual(mid[0].trajectory_ids, ("t1",))

    # The held-back remainder is drained (and marked final) on flush.
    flushed = assembler.flush()
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].trajectory_ids, ("t2",))

  def test_streaming_flush_and_reset(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        group_size=2,
        mini_batch_size=2,
        max_packed_len=16,
    )
    p1 = dataclasses.replace(_make_payload(2, 2), metadata={"traj_id": "t1"})
    batches = assembler.feed([p1])
    self.assertEmpty(batches)

    # Explicit flush should emit buffered items and mark final
    flushed = assembler.flush()
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].trajectory_ids, ("t1",))

    # Reset clears state
    assembler.reset()
    self.assertEmpty(assembler.flush())

  def _make_streaming_payload(
      self,
      prompt_length: int = 0,
      completion_length: int = 4,
      val: int = 1,
      prompt_id: str = "",
      group_index: int = 0,
  ) -> datatypes.RLTrainerPayload:
    metadata = {}
    if prompt_id:
      metadata = {"traj_id": datatypes.format_traj_id(prompt_id, group_index)}
    return datatypes.RLTrainerPayload(
        prompt_ids=np.full(prompt_length, val, dtype=np.int32),
        prompt_mask=np.ones(prompt_length, dtype=np.float32),
        completion_ids=np.full(completion_length, val, dtype=np.int32),
        completion_mask=np.ones(completion_length, dtype=np.float32),
        advantages=np.full(completion_length, 1.0, dtype=np.float32),
        metadata=metadata,
    )

  def test_feed_cross_group_packing_and_auto_flush(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=16,
        pad_id=0,
        group_size=1,
        mini_batch_size=3,  # total_step_rollouts = 3
    )
    # 3 groups with 4 tokens each (total 12 tokens < 16)
    # Group 1: 4 tokens -> buffers
    res1 = assembler.feed([self._make_streaming_payload(prompt_length=2, completion_length=2, val=1)])
    self.assertEmpty(res1)

    # Group 2: 4 tokens -> buffers (8 tokens total)
    res2 = assembler.feed([self._make_streaming_payload(prompt_length=2, completion_length=2, val=2)])
    self.assertEmpty(res2)

    # Group 3: 4 tokens -> hits total_step_rollouts = 3, auto-flushes!
    res3 = assembler.feed([self._make_streaming_payload(prompt_length=2, completion_length=2, val=3)])
    self.assertLen(res3, 1)
    batch = res3[0]
    self.assertTrue(batch.is_final_batch)
    self.assertEqual(batch.payload.completion_ids.shape, (1, 16))
    # Verify 3 segments are packed in the same buffer
    seg_ids = batch.payload.segment_ids[0]
    np.testing.assert_array_equal(seg_ids[0:4], 1)
    np.testing.assert_array_equal(seg_ids[4:8], 2)
    np.testing.assert_array_equal(seg_ids[8:12], 3)
    np.testing.assert_array_equal(seg_ids[12:16], 0)  # trailing pad

  def test_feed_early_emission_when_bin_is_full(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=16,
        pad_id=0,
        group_size=1,
        mini_batch_size=3,  # total_step_rollouts = 3
    )
    # Item 1: 8 tokens -> buffers (8 < 16)
    res1 = assembler.feed([self._make_streaming_payload(prompt_length=4, completion_length=4, val=1)])
    self.assertEmpty(res1)

    # Item 2: 8 tokens -> 8 + 8 = 16 tokens >= 16 (chunk capacity)!
    # Emits early before step boundary
    res2 = assembler.feed([self._make_streaming_payload(prompt_length=4, completion_length=4, val=2)])
    self.assertLen(res2, 1)
    self.assertFalse(res2[0].is_final_batch)
    self.assertEqual(res2[0].payload.completion_ids.shape, (1, 16))

    # Item 3: 4 tokens -> hits step boundary (rollouts = 3), auto-flushes open bin
    res3 = assembler.feed([self._make_streaming_payload(prompt_length=2, completion_length=2, val=3)])
    self.assertLen(res3, 1)
    self.assertTrue(res3[0].is_final_batch)
    self.assertEqual(res3[0].payload.completion_ids.shape, (1, 16))

  def test_feed_overflow_opens_new_bin(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=2,
        max_packed_len=16,
        pad_id=0,
        group_size=1,
        mini_batch_size=2,
    )
    # Item 1 has 10 tokens
    res1 = assembler.feed([self._make_streaming_payload(prompt_length=4, completion_length=6, val=1)])
    self.assertEmpty(res1)

    # Item 2 has 8 tokens (10 + 8 = 18 > 16, cannot fit!)
    # Should place Item 1 in bin 1 and Item 2 in bin 2.
    # Reaching total_step_rollouts = 2 auto-flushes bin 2!
    # With batch_size=2, the 2 bins form 1 microbatch of shape [2, 16]!
    res2 = assembler.feed([self._make_streaming_payload(prompt_length=4, completion_length=4, val=2)])
    self.assertLen(res2, 1)
    self.assertTrue(res2[0].is_final_batch)
    self.assertEqual(res2[0].payload.completion_ids.shape, (2, 16))
    np.testing.assert_array_equal(res2[0].payload.segment_ids[0, :10], 1)
    np.testing.assert_array_equal(res2[0].payload.segment_ids[0, 10:], 0)
    np.testing.assert_array_equal(res2[0].payload.segment_ids[1, :8], 1)
    np.testing.assert_array_equal(res2[0].payload.segment_ids[1, 8:], 0)

  def test_manual_flush_for_early_eof(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=16,
        pad_id=0,
        group_size=1,
        mini_batch_size=4,
    )
    assembler.feed([self._make_streaming_payload(prompt_length=2, completion_length=2, val=1)])
    flushed = assembler.flush()
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].payload.completion_ids.shape, (1, 16))
    self.assertEmpty(assembler.flush())

  def test_reset_clears_state(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=16,
        pad_id=0,
        group_size=1,
        mini_batch_size=4,
    )
    assembler.feed([self._make_streaming_payload(prompt_length=2, completion_length=2, val=1)])
    assembler.reset()
    self.assertEmpty(assembler.flush())

  def test_sequence_packed_batch_assembler_tracks_trajectory_ids(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=16,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,  # total_step_rollouts = 4
    )
    # Group 1: 2 items of 4 tokens each (8 tokens total) -> buffers
    res1 = assembler.feed([
        self._make_streaming_payload(prompt_length=2, completion_length=2, prompt_id="p0", group_index=0),
        self._make_streaming_payload(prompt_length=2, completion_length=2, prompt_id="p0", group_index=1),
    ])
    self.assertEmpty(res1)

    # Group 2: 2 items of 4 tokens each (8 tokens total). Reaches step boundary.
    res2 = assembler.feed([
        self._make_streaming_payload(prompt_length=2, completion_length=2, prompt_id="p1", group_index=0),
        self._make_streaming_payload(
            prompt_length=2, completion_length=2, prompt_id="p1", group_index=1),
    ])
    self.assertLen(res2, 1)
    self.assertTrue(res2[0].is_final_batch)
    self.assertEqual(
        res2[0].trajectory_ids,
        ("traj_p0_g0", "traj_p0_g1", "traj_p1_g0", "traj_p1_g1"),
    )
    self.assertEqual(
        res2[0].payload.metadata["trajectory_ids"],
        ("traj_p0_g0", "traj_p0_g1", "traj_p1_g0", "traj_p1_g1"),
    )

  def test_feed_emits_packed_sequence_across_input_batch_boundaries_with_segment_verification(
      self,
  ):
    """Verifies that an emitted packed sequence combines items across separate feed() calls."""
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=16,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,
    )

    # Feed 1 (Input batch 1): 2 items with tokens [10, 11] and [12, 13] (total 4 tokens)
    item1 = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros(0, dtype=np.int32),
        prompt_mask=np.zeros(0, dtype=np.float32),
        completion_ids=np.array([10, 11], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.array([0.5, 0.5], dtype=np.float32),
    )
    item2 = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros(0, dtype=np.int32),
        prompt_mask=np.zeros(0, dtype=np.float32),
        completion_ids=np.array([12, 13], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.array([0.5, 0.5], dtype=np.float32),
    )
    res1 = assembler.feed([item1, item2])
    self.assertEmpty(res1)  # 4 tokens < 16, stays buffered

    # Feed 2 (Input batch 2): 2 items with tokens [20, 21, 22] and [23, 24, 25] (total 6 tokens)
    item3 = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros(0, dtype=np.int32),
        prompt_mask=np.zeros(0, dtype=np.float32),
        completion_ids=np.array([20, 21, 22], dtype=np.int32),
        completion_mask=np.ones(3, dtype=np.float32),
        advantages=np.array([1.5, 1.5, 1.5], dtype=np.float32),
    )
    item4 = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros(0, dtype=np.int32),
        prompt_mask=np.zeros(0, dtype=np.float32),
        completion_ids=np.array([23, 24, 25], dtype=np.int32),
        completion_mask=np.ones(3, dtype=np.float32),
        advantages=np.array([1.5, 1.5, 1.5], dtype=np.float32),
    )
    # Total rollouts = 2 + 2 = 4 == total_step_rollouts; reaches step boundary!
    res2 = assembler.feed([item3, item4])
    self.assertLen(res2, 1)
    batch = res2[0]
    self.assertTrue(batch.is_final_batch)

    payload = batch.payload
    self.assertEqual(payload.completion_ids.shape, (1, 16))

    # Under First-Fit-Decreasing (FFD) packing, items are sorted by length descending:
    # item3 (len 3), item4 (len 3), item1 (len 2), item2 (len 2).
    expected_tokens = np.array(
        [20, 21, 22, 23, 24, 25, 10, 11, 12, 13, 0, 0, 0, 0, 0, 0],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(payload.completion_ids[0], expected_tokens)

    # Verify segment boundaries across the 4 items
    expected_segments = np.array(
        [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0], dtype=np.int32
    )
    np.testing.assert_array_equal(payload.segment_ids[0], expected_segments)

    # Verify segment positions reset for each item
    expected_positions = np.array(
        [0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32
    )
    np.testing.assert_array_equal(
        payload.segment_positions[0], expected_positions
    )

  def test_feed_final_batch_broken_down_into_multiple_packed_sequences(
      self,
  ):
    """Verifies final batch breaking into multiple sequences when tokens exceed max_packed_len."""
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=16,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,  # total_step_rollouts = 4
    )

    # Group 1 (2 items, 6 tokens each = 12 tokens): buffered (12 < 16)
    group1 = [
        self._make_streaming_payload(prompt_length=3, completion_length=3, val=1),
        self._make_streaming_payload(prompt_length=3, completion_length=3, val=1),
    ]
    res1 = assembler.feed(group1)
    self.assertEmpty(res1)

    # Group 2 (final batch, 2 items, 6 tokens each):
    # Total tokens = 12 + 12 = 24 tokens.
    # Chunk 1 packs 2 items (12 tokens <= 16).
    # Chunk 2 packs remaining 2 items (12 tokens <= 16).
    # Reaching step rollouts = 4 -> drains entire buffer!
    group2 = [
        self._make_streaming_payload(prompt_length=3, completion_length=3, val=2),
        self._make_streaming_payload(prompt_length=3, completion_length=3, val=2),
    ]
    res2 = assembler.feed(group2)
    self.assertLen(res2, 2)
    self.assertFalse(res2[0].is_final_batch)
    self.assertTrue(res2[1].is_final_batch)
    self.assertEqual(res2[0].payload.completion_ids.shape, (1, 16))
    self.assertEqual(res2[1].payload.completion_ids.shape, (1, 16))

  def test_feed_packs_multiple_items_into_bin(self):
    """Verifies that multiple unbatched items pack into the first row of batch_size=2."""
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=2,
        max_packed_len=16,
        pad_id=0,
        group_size=2,
        mini_batch_size=1,
    )
    item1 = self._make_streaming_payload(prompt_length=3, completion_length=3, val=1)
    item2 = self._make_streaming_payload(prompt_length=3, completion_length=3, val=2)
    res = assembler.feed([item1, item2])
    self.assertLen(res, 1)
    self.assertTrue(res[0].is_final_batch)
    self.assertEqual(res[0].payload.completion_ids.shape, (2, 16))
    np.testing.assert_array_equal(res[0].payload.segment_ids[0, :6], 1)
    np.testing.assert_array_equal(res[0].payload.segment_ids[0, 6:12], 2)
    np.testing.assert_array_equal(res[0].payload.segment_ids[0, 12:], 0)
    np.testing.assert_array_equal(res[0].payload.segment_ids[1], 0)

  def test_feed_rejects_2d_batched_payload(self):
    """Verifies that 2D payloads with shape (1, N) are rejected by sequence packing."""
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=2,
        max_packed_len=16,
        pad_id=0,
        group_size=2,
        mini_batch_size=1,
    )
    item = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((1, 0), dtype=np.int32),
        prompt_mask=np.zeros((1, 0), dtype=np.float32),
        completion_ids=np.ones((1, 6), dtype=np.int32),
        completion_mask=np.ones((1, 6), dtype=np.float32),
        advantages=np.ones((1, 6), dtype=np.float32),
    )
    with self.assertRaisesRegex(ValueError, "UNBATCHED payloads"):
      assembler.feed([item])

  def test_feed_and_flush_with_batch_size_greater_than_1(self):
    """Verifies streaming feed and flush with batch_size > 1."""
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=2,
        max_packed_len=16,
        pad_id=0,
        group_size=1,
        mini_batch_size=4,  # total_step_rollouts = 4
    )
    # chunk_capacity = batch_size * max_packed_len = 2 * 16 = 32 tokens.
    # Item 1: 16 tokens -> buffers (16 < 32)
    res1 = assembler.feed([self._make_streaming_payload(prompt_length=8, completion_length=8, val=1)])
    self.assertEmpty(res1)

    # Item 2: 16 tokens -> 16 + 16 = 32 tokens >= 32 (chunk capacity).
    # Emits early microbatch of shape [2, 16], is_final_batch=False
    res2 = assembler.feed([self._make_streaming_payload(prompt_length=8, completion_length=8, val=2)])
    self.assertLen(res2, 1)
    self.assertFalse(res2[0].is_final_batch)
    self.assertEqual(res2[0].payload.completion_ids.shape, (2, 16))

    # Item 3: 16 tokens -> buffers (16 < 32)
    res3 = assembler.feed([self._make_streaming_payload(prompt_length=8, completion_length=8, val=3)])
    self.assertEmpty(res3)

    # Item 4: 16 tokens -> Step done (rollouts = 4)!
    # Emits final microbatch of shape [2, 16], is_final_batch=True
    res4 = assembler.feed([self._make_streaming_payload(prompt_length=8, completion_length=8, val=4)])
    self.assertLen(res4, 1)
    self.assertTrue(res4[0].is_final_batch)
    self.assertEqual(res4[0].payload.completion_ids.shape, (2, 16))

    # Early EOF flush test: 1 item fed into a fresh step, flush pads to [2, 16]
    assembler.feed([
        self._make_streaming_payload(
            prompt_length=3, completion_length=3, val=5
        )
    ])
    flushed = assembler.flush()
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].payload.completion_ids.shape, (2, 16))
    np.testing.assert_array_equal(flushed[0].payload.segment_ids[0, :6], 1)
    np.testing.assert_array_equal(flushed[0].payload.segment_ids[0, 6:], 0)
    np.testing.assert_array_equal(flushed[0].payload.segment_ids[1], 0)


def _make_payload(
    prompt_len: int,
    completion_len: int,
    *,
    advantage=1.0,
    ref_logps=None,
    old_logps=None,
    returns=None,
    old_values=None,
    sampler_is_weights=None,
    prompt_mask=None,
    completion_mask=None,
):
  """Builds an unbatched payload shaped like `AlgorithmAdapter` output."""
  prompt = np.arange(1, prompt_len + 1, dtype=np.int32)
  completion = np.arange(101, 101 + completion_len, dtype=np.int32)
  total_seq_len = prompt_len + completion_len
  comp_mask = (
      completion_mask
      if completion_mask is not None
      else np.ones(completion_len, dtype=np.float32)
  )
  prompt_valid_mask = (
      prompt_mask
      if prompt_mask is not None
      else np.ones(prompt_len, dtype=np.float32)
  )
  seq_returns = (
      np.full(total_seq_len, float(returns), dtype=np.float32)
      if returns is not None and np.ndim(returns) == 0
      else (
          np.asarray(returns, dtype=np.float32) if returns is not None else None
      )
  )
  seq_old_values = (
      np.full(total_seq_len, float(old_values), dtype=np.float32)
      if old_values is not None and np.ndim(old_values) == 0
      else (
          np.asarray(old_values, dtype=np.float32)
          if old_values is not None
          else None
      )
  )
  seq_sampler_is = (
      np.full(total_seq_len, float(sampler_is_weights), dtype=np.float32)
      if sampler_is_weights is not None and np.ndim(sampler_is_weights) == 0
      else (
          np.asarray(sampler_is_weights, dtype=np.float32)
          if sampler_is_weights is not None
          else None
      )
  )
  return datatypes.RLTrainerPayload(
      advantages=advantage,
      prompt_ids=prompt,
      prompt_mask=prompt_valid_mask,
      completion_ids=completion,
      completion_mask=comp_mask,
      ref_per_token_logps=ref_logps,
      old_per_token_logps=old_logps,
      returns=seq_returns,
      old_values=seq_old_values,
      sampler_is_weights=seq_sampler_is,
  )


class PaddedBatchAssemblerTest(absltest.TestCase):

  def _assembler(self, **kwargs):
    defaults = dict(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=5,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,
    )
    defaults.update(kwargs)
    return batch_assembly.PaddedBatchAssembler(**defaults)

  def test_rejects_non_positive_dimensions(self):
    for bad in (
        dict(batch_size=0),
        dict(max_prompt_length=0),
        dict(max_response_length=-1),
        dict(group_size=0),
        dict(mini_batch_size=0),
    ):
      with self.assertRaises(ValueError):
        self._assembler(**bad)

  def test_requires_group_size_and_mini_batch_size(self):
    with self.assertRaises(TypeError):
      batch_assembly.PaddedBatchAssembler(  # pyrefly: ignore[missing-parameter]
          batch_size=2,
          max_prompt_length=4,
          max_response_length=5,
          pad_id=0,
      )
    with self.assertRaises(TypeError):
      batch_assembly.PaddedBatchAssembler(  # pyrefly: ignore[missing-parameter]
          batch_size=2,
          max_prompt_length=4,
          max_response_length=5,
          pad_id=0,
          group_size=1,
      )



  def test_max_seq_len_is_sum_of_prompt_and_response_lengths(self):
    assembler = self._assembler(max_prompt_length=128, max_response_length=256)
    self.assertEqual(assembler.max_seq_len, 384)

  def test_empty_input_returns_empty_list(self):
    self.assertEmpty(self._assembler().pack([]))

  def test_row_layout_is_left_padded_prompt_and_right_padded_completion(self):
    payload = self._assembler().pack([_make_payload(2, 3)])[0]

    self.assertEqual(payload.prompt_ids.shape, (2, 4))
    self.assertEqual(payload.prompt_mask.shape, (2, 4))
    self.assertEqual(payload.completion_ids.shape, (2, 5))
    self.assertEqual(payload.completion_mask.shape, (2, 5))
    self.assertEqual(payload.advantages.shape, (2, 5))

    np.testing.assert_array_equal(payload.prompt_ids[0], [0, 0, 1, 2])
    np.testing.assert_array_equal(payload.prompt_mask[0], [0, 0, 1, 1])
    np.testing.assert_array_equal(
        payload.completion_ids[0], [101, 102, 103, 0, 0]
    )
    np.testing.assert_array_equal(payload.completion_mask[0], [1, 1, 1, 0, 0])
    np.testing.assert_allclose(payload.advantages[0], [1.0, 1.0, 1.0, 0.0, 0.0])

  def test_completion_mask_excludes_tool_observation_tokens(self):
    # Middle completion token is a tool observation: attended, but not trained.
    item = _make_payload(
        2, 3, completion_mask=np.array([1, 0, 1], dtype=np.float32)
    )
    payload = self._assembler().pack([item])[0]

    np.testing.assert_array_equal(payload.completion_mask[0], [1, 0, 1, 0, 0])

  def test_completion_aligned_logps_do_not_crash_on_length_mismatch(self):
    # Regression: ref logps are [C] while token_ids are [P + C]; a single
    # shared pad length used to produce ragged rows and fail np.stack.
    items = [
        _make_payload(2, 3, ref_logps=np.full(3, -0.1, dtype=np.float32)),
        _make_payload(4, 2, ref_logps=np.full(2, -0.2, dtype=np.float32)),
    ]
    payload = self._assembler().pack(items)[0]

    self.assertEqual(payload.ref_per_token_logps.shape, (2, 5))
    np.testing.assert_allclose(
        payload.ref_per_token_logps[0], [-0.1, -0.1, -0.1, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        payload.ref_per_token_logps[1], [-0.2, -0.2, 0.0, 0.0, 0.0]
    )

  def test_partially_present_optional_fields_stay_row_aligned(self):
    # Regression: appending only for items that carried the field shifted the
    # surviving rows onto the wrong sequences.
    items = [
        _make_payload(2, 3),
        _make_payload(2, 3, old_logps=np.full(3, -0.7, dtype=np.float32)),
    ]
    with self.assertLogs(level="WARNING") as logs:
      payload = self._assembler().pack(items)[0]

    self.assertIsNone(payload.old_per_token_logps)
    self.assertIn("Partially present optional fields", logs.output[0])

  def test_optional_fields_absent_everywhere_stay_none(self):
    payload = self._assembler().pack([_make_payload(2, 3)])[0]

    self.assertIsNone(payload.ref_per_token_logps)
    self.assertIsNone(payload.old_per_token_logps)
    self.assertIsNone(payload.returns)

  def test_returns_field_is_propagated(self):
    payload = self._assembler().pack([_make_payload(2, 3, returns=4.0)])[0]

    self.assertEqual(payload.returns.shape, (2, 5))
    np.testing.assert_allclose(payload.returns[0], [4, 4, 4, 0, 0])

  def test_scalar_advantage_broadcasts_over_completion(self):
    payload = self._assembler().pack([_make_payload(2, 3, advantage=2.5)])[0]

    self.assertEqual(payload.advantages.shape, (2, 5))
    np.testing.assert_allclose(payload.advantages[0], [2.5, 2.5, 2.5, 0, 0])

  def test_sequence_aligned_advantage_is_sliced_to_completion(self):
    item = _make_payload(2, 3).replace(
        advantages=np.array([0, 0, 2, 2, 2], dtype=np.float32)
    )
    payload = self._assembler().pack([item])[0]

    np.testing.assert_allclose(payload.advantages[0], [2, 2, 2, 0, 0])

  def test_truncates_overlong_prompt_from_the_left(self):
    with self.assertLogs(level="WARNING") as logs:
      payload = self._assembler().pack([_make_payload(6, 8)])[0]

    self.assertIn(
        "PaddedBatchAssembler truncated 1 prompt(s) to 4 tokens and 1"
        " completion(s) to 5 tokens",
        logs.output[0],
    )
    # Keeps the most recent prompt tokens.
    np.testing.assert_array_equal(payload.prompt_ids[0], [3, 4, 5, 6])
    # Keeps the earliest completion tokens.
    np.testing.assert_array_equal(
        payload.completion_ids[0], [101, 102, 103, 104, 105]
    )
    np.testing.assert_array_equal(payload.completion_mask[0], np.ones(5))

  def test_logs_warning_with_truncated_counts(self):
    items = [
        _make_payload(prompt_len=6, completion_len=7),
        _make_payload(prompt_len=5, completion_len=6),
        _make_payload(prompt_len=5, completion_len=6),
        _make_payload(prompt_len=4, completion_len=5),
        _make_payload(prompt_len=3, completion_len=4),
    ]
    with self.assertLogs(level="WARNING") as logs:
      self._assembler(batch_size=5).pack(items)

    self.assertIn(
        "PaddedBatchAssembler truncated 3 prompt(s) to 4 tokens and 3"
        " completion(s) to 5 tokens",
        logs.output[0],
    )

  def test_logs_warning_when_only_prompts_truncated(self):
    items = [
        _make_payload(prompt_len=6, completion_len=5),
        _make_payload(prompt_len=5, completion_len=4),
        _make_payload(prompt_len=5, completion_len=4),
        _make_payload(prompt_len=4, completion_len=3),
        _make_payload(prompt_len=3, completion_len=2),
    ]
    with self.assertLogs(level="WARNING") as logs:
      self._assembler(batch_size=5).pack(items)

    self.assertIn(
        "PaddedBatchAssembler truncated 3 prompt(s) to 4 tokens and 0"
        " completion(s) to 5 tokens",
        logs.output[0],
    )

  def test_logs_warning_when_only_completions_truncated(self):
    items = [
        _make_payload(prompt_len=4, completion_len=7),
        _make_payload(prompt_len=3, completion_len=6),
        _make_payload(prompt_len=3, completion_len=6),
        _make_payload(prompt_len=2, completion_len=5),
        _make_payload(prompt_len=1, completion_len=4),
    ]
    with self.assertLogs(level="WARNING") as logs:
      self._assembler(batch_size=5).pack(items)

    self.assertIn(
        "PaddedBatchAssembler truncated 0 prompt(s) to 4 tokens and 3"
        " completion(s) to 5 tokens",
        logs.output[0],
    )

  def test_no_warning_when_no_truncation(self):
    items = [
        _make_payload(prompt_len=4, completion_len=5),
        _make_payload(prompt_len=3, completion_len=4),
    ]
    with self.assertNoLogs(level="WARNING"):
      self._assembler().pack(items)

  def test_trailing_rows_are_masked_out(self):
    payload = self._assembler(batch_size=3).pack([_make_payload(2, 3)])[0]

    for row in (1, 2):
      np.testing.assert_array_equal(payload.advantages[row], np.zeros(5))

  def test_chunks_into_multiple_microbatches(self):
    payloads = self._assembler(batch_size=2).pack(
        [_make_payload(2, 3) for _ in range(5)]
    )

    self.assertLen(payloads, 3)
    for p in payloads:
      self.assertEqual(p.prompt_ids.shape, (2, 4))
      self.assertEqual(p.completion_ids.shape, (2, 5))

  def test_sequence_aligned_fields_are_sliced_to_completion(self):
    item = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([3], dtype=np.int32),
        completion_mask=np.ones(1, dtype=np.float32),
        advantages=np.full(3, 2.0, dtype=np.float32),
    )
    payload = self._assembler().pack([item])[0]

    self.assertEqual(payload.prompt_ids.shape, (2, 4))
    self.assertEqual(payload.prompt_mask.shape, (2, 4))
    self.assertEqual(payload.completion_ids.shape, (2, 5))
    self.assertEqual(payload.completion_mask.shape, (2, 5))
    self.assertEqual(payload.advantages.shape, (2, 5))

    np.testing.assert_array_equal(payload.prompt_ids[0], [0, 0, 1, 2])
    np.testing.assert_array_equal(payload.prompt_mask[0], [0, 0, 1, 1])
    np.testing.assert_array_equal(payload.completion_ids[0], [3, 0, 0, 0, 0])
    np.testing.assert_array_equal(payload.completion_mask[0], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(payload.advantages[0], [2, 0, 0, 0, 0])

  def test_completion_mask_with_only_action_tokens_masked(
      self,
  ):
    item = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([101, 102, 103], dtype=np.int32),
        completion_mask=np.array([1, 0, 1], dtype=np.float32),
        advantages=np.full(3, 1.5, dtype=np.float32),
    )
    payload = self._assembler().pack([item])[0]

    np.testing.assert_array_equal(payload.completion_mask[0], [1, 0, 1, 0, 0])

  def test_valid_prompt_mask_is_left_padded(self):
    item = _make_payload(
        prompt_len=3,
        completion_len=2,
        prompt_mask=np.array([1, 0, 1], dtype=np.float32),
    )
    payload = self._assembler(max_prompt_length=5).pack([item])[0]

    np.testing.assert_array_equal(payload.prompt_ids[0], [0, 0, 1, 2, 3])
    np.testing.assert_array_equal(payload.prompt_mask[0], [0, 0, 1, 0, 1])

  def test_prompt_mask_with_mismatched_length_falls_back_to_default_mask(self):
    item = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.array([1, 1, 1], dtype=np.float32),
        completion_ids=np.array([101, 102], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=1.0,
    )
    payload = self._assembler().pack([item])[0]

    np.testing.assert_array_equal(payload.prompt_mask[0], [0, 0, 1, 1])

  def test_all_optional_fields_are_propagated(self):
    item = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([101, 102, 103], dtype=np.int32),
        completion_mask=np.ones(3, dtype=np.float32),
        advantages=np.full(3, 1.5, dtype=np.float32),
        ref_per_token_logps=np.full(3, -0.1, dtype=np.float32),
        old_per_token_logps=np.full(3, -0.2, dtype=np.float32),
        returns=np.full(3, 2.0, dtype=np.float32),
        old_values=np.full(3, 0.5, dtype=np.float32),
        sampler_is_weights=np.full(3, 1.0, dtype=np.float32),
    )
    payload = self._assembler().pack([item])[0]

    self.assertEqual(payload.old_values.shape, (2, 5))
    self.assertEqual(payload.sampler_is_weights.shape, (2, 5))
    np.testing.assert_allclose(payload.old_values[0], [0.5, 0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(
        payload.sampler_is_weights[0], [1.0, 1.0, 1.0, 0.0, 0.0]
    )

  def test_underlength_completion_aligned_field_is_padded(self):
    item = _make_payload(
        prompt_len=2,
        completion_len=4,
        advantage=np.array([1.5, 2.5], dtype=np.float32),
        ref_logps=np.array([-0.5, -0.2], dtype=np.float32),
    )
    payload = self._assembler().pack([item])[0]

    self.assertEqual(payload.advantages.shape, (2, 5))
    self.assertEqual(payload.ref_per_token_logps.shape, (2, 5))
    np.testing.assert_allclose(payload.advantages[0], [1.5, 2.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        payload.ref_per_token_logps[0], [-0.5, -0.2, 0.0, 0.0, 0.0]
    )

  def test_none_advantages_defaults_to_zeros(self):
    item = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([101, 102, 103], dtype=np.int32),
        completion_mask=np.ones(3, dtype=np.float32),
        advantages=None,
    )
    payload = self._assembler().pack([item])[0]

    self.assertEqual(payload.advantages.shape, (2, 5))
    np.testing.assert_allclose(payload.advantages[0], [0.0, 0.0, 0.0, 0.0, 0.0])


class SequencePackedConversionTest(absltest.TestCase):
  """Test for the payload <-> PackItem adapter boundary."""

  def _make_assembler(self, max_packed_len=16, **kwargs):
    defaults = dict(
        batch_size=1,
        group_size=2,
        mini_batch_size=2,
        max_packed_len=max_packed_len,
    )
    defaults.update(kwargs)
    return batch_assembly.SequencePackedBatchAssembler(**defaults)

  def _drain(self, assembler, items):
    """Fully packs `items` via the streaming contract (feed then flush)."""
    return [
        batch.payload
        for batch in (assembler.feed(items) + assembler.flush())
    ]

  def test_basic_packing(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.arange(4, dtype=np.int32),
        prompt_mask=np.ones(4, dtype=np.float32),
        completion_ids=np.arange(4, 10, dtype=np.int32),
        completion_mask=np.ones(6, dtype=np.float32),
        advantages=np.float32(1.0),
    )
    [out] = self._drain(self._make_assembler(max_packed_len=12), [payload])
    np.testing.assert_array_equal(
        out.completion_mask[0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    )

  def test_batched_payload_is_rejected(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((2, 2), dtype=np.int32),
        prompt_mask=np.ones((2, 2), dtype=np.float32),
        completion_ids=np.zeros((2, 3), dtype=np.int32),
        completion_mask=np.ones((2, 3), dtype=np.float32),
        advantages=np.array([1.0, 2.0], dtype=np.float32),
    )
    with self.assertRaisesRegex(ValueError, "UNBATCHED payloads"):
      self._drain(self._make_assembler(max_packed_len=16), [payload])

  def test_mixed_rank_payload_raises(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((1, 2), dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.zeros(3, dtype=np.int32),
        completion_mask=np.ones(3, dtype=np.float32),
        advantages=np.float32(1.0),
    )
    with self.assertRaisesRegex(ValueError, "has rank 2"):
      self._drain(self._make_assembler(max_packed_len=16), [payload])

  def test_batch_size_packs_content_into_both_rows(self):
    payloads = [
        _make_payload(2, 2, advantage=1.0),
        _make_payload(2, 2, advantage=2.0),
    ]
    [out] = self._drain(
        self._make_assembler(max_packed_len=4, batch_size=2), payloads
    )
    self.assertEqual(out.prompt_ids.shape, (2, 0))
    self.assertEqual(out.completion_ids.shape, (2, 4))
    np.testing.assert_array_equal(
        np.sort(np.unique(out.advantages)), [0.0, 1.0, 2.0]
    )

  def test_metadata_with_static_segment_bound(self):
    [out] = self._drain(
        self._make_assembler(
            max_packed_len=16, max_segments_per_packed_row=4
        ),
        [_make_payload(2, 2)],
    )
    self.assertEqual(out.num_segments, 5)

  def test_partial_optional_field_raises(self):
    payloads = [
        _make_payload(1, 2, returns=np.ones(2, dtype=np.float32)),
        _make_payload(1, 2),
    ]
    with self.assertRaisesRegex(ValueError, "Some but not all"):
      self._drain(self._make_assembler(max_packed_len=16), payloads)

  def test_oversized_sequence_raises(self):
    with self.assertRaisesRegex(ValueError, "exceeding budget"):
      self._drain(
          self._make_assembler(max_packed_len=8), [_make_payload(10, 10)]
      )

  def test_constructor_invalid_arguments_raise(self):
    with self.assertRaisesRegex(ValueError, "max_packed_len must be positive"):
      self._make_assembler(max_packed_len=0)
    with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
      self._make_assembler(max_packed_len=16, batch_size=0)
    with self.assertRaisesRegex(ValueError, "group_size must be positive"):
      self._make_assembler(max_packed_len=16, group_size=0)
    with self.assertRaisesRegex(ValueError, "mini_batch_size must be positive"):
      self._make_assembler(max_packed_len=16, mini_batch_size=0)
    with self.assertRaisesRegex(
        ValueError, "max_segments_per_packed_row must be positive"
    ):
      self._make_assembler(
          max_packed_len=16, max_segments_per_packed_row=0
      )

  def test_to_pack_item_missing_completion_ids_raises(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=None,
        completion_mask=None,
        advantages=1.0,
    )
    with self.assertRaisesRegex(
        ValueError, "RLTrainerPayload.completion_ids is required"
    ):
      batch_assembly.to_pack_item(payload)

  def test_to_pack_item_whole_sequence_advantages_sliced(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([10, 20, 30], dtype=np.int32),
        completion_mask=np.ones(3, dtype=np.float32),
        advantages=np.array([99.0, 99.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )
    pack_item = batch_assembly.to_pack_item(payload)
    np.testing.assert_allclose(pack_item.advantages, [1.0, 2.0, 3.0])

  def test_to_pack_item_completion_length_advantages(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([10, 20, 30], dtype=np.int32),
        completion_mask=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        advantages=np.array([0.5, 1.5, 2.5], dtype=np.float32),
    )
    pack_item = batch_assembly.to_pack_item(payload)
    np.testing.assert_array_equal(pack_item.prompt_ids, [1, 2])
    np.testing.assert_array_equal(pack_item.completion_ids, [10, 20, 30])
    np.testing.assert_allclose(pack_item.completion_mask, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(pack_item.advantages, [0.5, 1.5, 2.5])
    self.assertEqual(pack_item.advantages.dtype, np.float32)

  def test_to_pack_item_scalar_advantages_and_default_mask(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=None,
        completion_ids=np.array([10, 20, 30], dtype=np.int32),
        completion_mask=None,
        advantages=2.0,
    )
    pack_item = batch_assembly.to_pack_item(payload)
    np.testing.assert_allclose(pack_item.completion_mask, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(pack_item.advantages, [2.0, 2.0, 2.0])

  def test_to_pack_item_none_advantages_defaults_to_zeros(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=None,
        completion_ids=np.array([10, 20, 30], dtype=np.int32),
        completion_mask=None,
        advantages=None,
    )
    pack_item = batch_assembly.to_pack_item(payload)
    np.testing.assert_allclose(pack_item.completion_mask, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(pack_item.advantages, [0.0, 0.0, 0.0])
    self.assertEqual(pack_item.advantages.dtype, np.float32)

  def test_to_pack_item_whole_sequence_completion_mask_sliced(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([10, 20, 30], dtype=np.int32),
        completion_mask=np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        advantages=1.0,
    )
    pack_item = batch_assembly.to_pack_item(payload)
    np.testing.assert_allclose(pack_item.completion_mask, [1.0, 0.0, 1.0])

  def test_to_pack_item_none_prompt_ids_defaults_to_empty(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=None,
        prompt_mask=None,
        completion_ids=np.array([10, 20, 30], dtype=np.int32),
        completion_mask=None,
        advantages=1.0,
    )
    pack_item = batch_assembly.to_pack_item(payload)
    self.assertEqual(pack_item.prompt_ids.shape, (0,))
    self.assertEqual(pack_item.prompt_ids.dtype, np.int32)
    np.testing.assert_array_equal(pack_item.completion_ids, [10, 20, 30])

  def test_to_pack_item_with_per_token_fields(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([10, 20, 30], dtype=np.int32),
        completion_mask=np.ones(3, dtype=np.float32),
        advantages=1.0,
        old_per_token_logps=np.array([-0.1, -0.2, -0.3], dtype=np.float32),
        ref_per_token_logps=np.array(
            [0.0, 0.0, -0.4, -0.5, -0.6], dtype=np.float32
        ),  # whole sequence: length 5 sliced to 3
    )
    pack_item = batch_assembly.to_pack_item(payload)
    self.assertEqual(
        pack_item.per_token["old_per_token_logps"].dtype, np.float32
    )
    self.assertEqual(
        pack_item.per_token["ref_per_token_logps"].dtype, np.float32
    )
    self.assertEqual(pack_item.advantages.dtype, np.float32)
    self.assertEqual(pack_item.completion_mask.dtype, np.float32)
    np.testing.assert_allclose(
        pack_item.per_token["old_per_token_logps"], [-0.1, -0.2, -0.3]
    )
    np.testing.assert_allclose(
        pack_item.per_token["ref_per_token_logps"], [-0.4, -0.5, -0.6]
    )

  def test_to_pack_item_unexpected_advantages_shape_raises(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([10, 20, 30], dtype=np.int32),
        completion_mask=np.ones(3, dtype=np.float32),
        advantages=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    with self.assertRaisesRegex(ValueError, "unexpected size"):
      batch_assembly.to_pack_item(payload)

  def test_sequence_packing_with_all_per_token_fields(self):
    payload1 = _make_payload(
        prompt_len=2,
        completion_len=2,
        advantage=1.0,
        ref_logps=np.array([-0.1, -0.2], dtype=np.float32),
        old_logps=np.array([-0.3, -0.4], dtype=np.float32),
        returns=np.array([2.0, 2.5], dtype=np.float32),
        old_values=np.array([1.5, 1.6], dtype=np.float32),
        sampler_is_weights=np.array([1.0, 1.0], dtype=np.float32),
    )
    payload2 = _make_payload(
        prompt_len=1,
        completion_len=1,
        advantage=-0.5,
        ref_logps=np.array([-0.5], dtype=np.float32),
        old_logps=np.array([-0.6], dtype=np.float32),
        returns=np.array([0.5], dtype=np.float32),
        old_values=np.array([0.2], dtype=np.float32),
        sampler_is_weights=np.array([0.9], dtype=np.float32),
    )
    assembler = self._make_assembler(max_packed_len=8)
    [out] = self._drain(assembler, [payload1, payload2])

    self.assertEqual(out.completion_ids.shape, (1, 8))
    self.assertEqual(out.ref_per_token_logps.shape, (1, 8))
    self.assertEqual(out.old_per_token_logps.shape, (1, 8))
    self.assertEqual(out.returns.shape, (1, 8))
    self.assertEqual(out.old_values.shape, (1, 8))
    self.assertEqual(out.sampler_is_weights.shape, (1, 8))
    self.assertEqual(out.completion_ids.dtype, np.int32)

    # Segment 1: prompt [0:2], completion [2:4]
    # Segment 2: prompt [4:5], completion [5:6]
    # Padded: [6:8]
    np.testing.assert_allclose(
        out.returns[0], [0.0, 0.0, 2.0, 2.5, 0.0, 0.5, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        out.old_values[0], [0.0, 0.0, 1.5, 1.6, 0.0, 0.2, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        out.ref_per_token_logps[0], [0.0, 0.0, -0.1, -0.2, 0.0, -0.5, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        out.old_per_token_logps[0], [0.0, 0.0, -0.3, -0.4, 0.0, -0.6, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        out.sampler_is_weights[0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.9, 0.0, 0.0]
    )

  def test_sequence_packing_multiple_chunks(self):
    payloads = [_make_payload(2, 2), _make_payload(2, 2)]
    assembler = self._make_assembler(max_packed_len=6, batch_size=1)
    chunks = self._drain(assembler, payloads)
    self.assertLen(chunks, 2)
    for c in chunks:
      self.assertEqual(c.completion_ids.shape, (1, 6))
      self.assertEqual(c.num_segments, 7)


_ROUTING_LAYERS = 2
_ROUTING_TOP_K = 2
_UNSET = datatypes.UNSET_ROUTED_EXPERT


def _routing(length, fill):
  """`[length, num_layers, top_k]` routing where every slot holds `fill`."""
  shape = (length, _ROUTING_LAYERS, _ROUTING_TOP_K)
  return np.full(shape, fill, dtype=np.int32)


class RoutedExpertsAlignmentTest(absltest.TestCase):
  """Replayed routing must be padded the same way the token ids are."""

  def test_prompt_is_right_aligned_and_completion_left_aligned(self):
    """Routing must follow the token padding, not sit at the row start.

    Prompts are left-padded and completions right-padded; if the routing does
    not follow, every replayed expert lands on the wrong token.
    """
    prompt_len, completion_len = 2, 3
    max_prompt, max_response = 5, 6
    routed = np.concatenate(
        [_routing(prompt_len, 7), _routing(completion_len, 9)], axis=0
    )

    out = batch_assembly._routed_experts_aligned(  # pylint: disable=protected-access
        routed, prompt_len, completion_len, max_prompt, max_response
    )

    self.assertEqual(
        out.shape, (max_prompt + max_response, _ROUTING_LAYERS, _ROUTING_TOP_K)
    )
    # Prompt window: leading pad unset, prompt routing flush against the end.
    np.testing.assert_array_equal(out[: max_prompt - prompt_len], _UNSET)
    np.testing.assert_array_equal(out[max_prompt - prompt_len : max_prompt], 7)
    # Response window: completion routing first, trailing pad unset.
    np.testing.assert_array_equal(
        out[max_prompt : max_prompt + completion_len], 9
    )
    np.testing.assert_array_equal(out[max_prompt + completion_len :], _UNSET)

  def test_overlong_prompt_keeps_the_tail(self):
    """`_left_pad` keeps the last tokens, so routing must keep its last rows."""
    # One prompt row per position, so a dropped row is visible by value.
    prompt = np.broadcast_to(
        np.arange(4, dtype=np.int32).reshape(4, 1, 1),
        (4, _ROUTING_LAYERS, _ROUTING_TOP_K),
    )
    routed = np.concatenate([prompt, _routing(1, 9)], axis=0)
    out = batch_assembly._routed_experts_aligned(routed, 4, 1, 2, 3)  # pylint: disable=protected-access
    # Prompt rows 0 and 1 are dropped; 2 and 3 survive, in order.
    np.testing.assert_array_equal(out[0], 2)
    np.testing.assert_array_equal(out[1], 3)


class PaddedBatchAssemblerRoutingTest(absltest.TestCase):
  """The assembler must emit `[B, P + C, num_layers, top_k]`, or nothing."""

  MAX_PROMPT = 4
  MAX_RESPONSE = 4

  def _assembler(self, batch_size=2, group_size=1, mini_batch_size=1):
    return batch_assembly.PaddedBatchAssembler(
        batch_size=batch_size,
        max_prompt_length=self.MAX_PROMPT,
        max_response_length=self.MAX_RESPONSE,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,
    )

  def _payload(self, fill, with_routing=True):
    prompt = np.array([1, 2], dtype=np.int32)
    completion = np.array([3, 4, 5], dtype=np.int32)
    routed = None
    if with_routing:
      routed = np.concatenate(
          [_routing(len(prompt), fill), _routing(len(completion), fill)],
          axis=0,
      )
    return datatypes.RLTrainerPayload(
        advantages=np.zeros(len(completion), dtype=np.float32),
        prompt_ids=prompt,
        prompt_mask=np.ones(len(prompt), dtype=np.float32),
        completion_ids=completion,
        completion_mask=np.ones(len(completion), dtype=np.float32),
        routed_experts=routed,
    )

  def test_batches_routing_across_rows(self):
    packed = self._assembler().pack([self._payload(1), self._payload(2)])
    self.assertLen(packed, 1)
    routed = packed[0].routed_experts
    self.assertIsNotNone(routed, "assembler dropped the replayed routing")
    self.assertEqual(
        routed.shape,
        (
            2,
            self.MAX_PROMPT + self.MAX_RESPONSE,
            _ROUTING_LAYERS,
            _ROUTING_TOP_K,
        ),
    )
    # Row order must be preserved, or rows train on each other's routing.
    self.assertEqual(int(routed[0, self.MAX_PROMPT, 0, 0]), 1)
    self.assertEqual(int(routed[1, self.MAX_PROMPT, 0, 0]), 2)

  def test_partial_capture_disables_replay_for_the_batch(self):
    """A half-replayed batch would silently mix replayed and fresh routing."""
    packed = self._assembler().pack(
        [self._payload(1), self._payload(2, with_routing=False)]
    )
    self.assertIsNone(packed[0].routed_experts)

  def test_short_batch_pads_rows_as_unset(self):
    """Filler rows must not replay a real expert id."""
    packed = self._assembler(batch_size=2).pack([self._payload(1)])
    routed = packed[0].routed_experts
    self.assertEqual(routed.shape[0], 2)
    np.testing.assert_array_equal(routed[1], _UNSET)


  def _make_streaming_payload(
      self,
      val: int = 1,
      prompt_id: str = "",
      group_index: int = 0,
  ) -> datatypes.RLTrainerPayload:
    metadata = {}
    if prompt_id:
      metadata = {"traj_id": datatypes.format_traj_id(prompt_id, group_index)}
    return datatypes.RLTrainerPayload(
        prompt_ids=np.array([val, val], dtype=np.int32),
        completion_ids=np.array([val + 1, val + 2], dtype=np.int32),
        prompt_mask=np.array([1.0, 1.0], dtype=np.float32),
        completion_mask=np.array([1.0, 1.0], dtype=np.float32),
        advantages=np.array([0.5, 0.5], dtype=np.float32),
        metadata=metadata,
    )

  def test_feed_buffering_and_steady_state(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=4,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,  # total_step_rollouts = 4
    )
    # Feed 2 items (half step): should buffer and return empty list
    items_group1 = [
        self._make_streaming_payload(1),
        self._make_streaming_payload(2),
    ]
    res1 = assembler.feed(items_group1)
    self.assertEmpty(res1)

    # Feed 2 items (second half): reaches total_step_rollouts = 4 and batch_size = 4
    items_group2 = [
        self._make_streaming_payload(3),
        self._make_streaming_payload(4),
    ]
    res2 = assembler.feed(items_group2)
    self.assertLen(res2, 1)
    batch = res2[0]
    self.assertTrue(batch.is_final_batch)
    self.assertEqual(batch.payload.prompt_ids.shape, (4, 2))
    self.assertEqual(batch.payload.completion_ids.shape, (4, 2))

  def test_feed_multiple_microbatches_in_step(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,  # total_step_rollouts = 4, emits 2 microbatches
    )
    # Feed 2 items: reaches batch_size=2, but total_step_rollouts is 4, so is_final_batch=False
    res1 = assembler.feed(
        [self._make_streaming_payload(1), self._make_streaming_payload(2)]
    )
    self.assertLen(res1, 1)
    self.assertFalse(res1[0].is_final_batch)

    # Feed 2 items: reaches total_step_rollouts=4, so is_final_batch=True
    res2 = assembler.feed(
        [self._make_streaming_payload(3), self._make_streaming_payload(4)]
    )
    self.assertLen(res2, 1)
    self.assertTrue(res2[0].is_final_batch)

  def test_feed_auto_flush_with_remainder(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=4,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        group_size=3,
        mini_batch_size=1,  # total_step_rollouts = 3 (less than batch_size=4!)
    )
    # Feed 3 items: hits total_step_rollouts=3, auto-flushes with 1 padded row
    res = assembler.feed([
        self._make_streaming_payload(1),
        self._make_streaming_payload(2),
        self._make_streaming_payload(3),
    ])
    self.assertLen(res, 1)
    batch = res[0]
    self.assertTrue(batch.is_final_batch)
    self.assertEqual(batch.payload.prompt_ids.shape, (4, 2))
    # 4th row should be zero-padded
    np.testing.assert_array_equal(batch.payload.prompt_mask[3], np.zeros(2))
    np.testing.assert_array_equal(batch.payload.completion_mask[3], np.zeros(2))

  def test_manual_flush_for_early_eof(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=4,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,  # total_step_rollouts = 4
    )
    # Feed only 2 items mid-step
    res1 = assembler.feed(
        [self._make_streaming_payload(1), self._make_streaming_payload(2)]
    )
    self.assertEmpty(res1)

    # Dataset runs out: manual flush
    flushed = assembler.flush()
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].payload.prompt_ids.shape, (4, 2))
    # Subsequent flush on empty buffer returns empty
    self.assertEmpty(assembler.flush())

  def test_reset_clears_state(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=4,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,
    )
    assembler.feed(
        [self._make_streaming_payload(1), self._make_streaming_payload(2)]
    )
    assembler.reset()
    self.assertEmpty(assembler.flush())

  def test_padded_batch_assembler_tracks_trajectory_ids(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=3,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        group_size=2,
        mini_batch_size=2,  # total_step_rollouts = 4
    )
    items = [
        self._make_streaming_payload(1, prompt_id="p0", group_index=0),
        self._make_streaming_payload(1, prompt_id="p0", group_index=1),
        self._make_streaming_payload(1, prompt_id="p1", group_index=0),
        self._make_streaming_payload(1, prompt_id="p1", group_index=1),
    ]
    # Feed 4 items: first batch gets 3 items, second auto-flushed gets 1 item + 2 padding
    res = assembler.feed(items)
    self.assertLen(res, 2)
    self.assertEqual(
        res[0].trajectory_ids,
        ("traj_p0_g0", "traj_p0_g1", "traj_p1_g0"),
    )
    # 2nd batch has 1 real trajectory and 2 padded rows
    self.assertEqual(res[1].trajectory_ids, ("traj_p1_g1",))
    self.assertTrue(res[1].is_final_batch)


if __name__ == "__main__":
  absltest.main()
