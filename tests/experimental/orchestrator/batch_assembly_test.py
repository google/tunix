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
import tracemalloc
from unittest import mock

from absl.testing import absltest
import jax
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import lineage
from tunix.experimental.orchestrator import batch_assembly
from tunix.rl import packing


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
        num_generations=2,
        mini_batch_size=2,
        max_packed_len=max_packed_len,
    )
    defaults.update(kwargs)
    return batch_assembly.SequencePackedBatchAssembler(**defaults)

  def _drain(self, assembler, items):
    """Fully packs `items` via the streaming contract (feed then flush)."""
    return [
        batch.payload for batch in [*assembler.feed(items), *assembler.flush()]
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
        num_generations=1,
        mini_batch_size=3,
        max_packed_len=10,
    )
    p1 = dataclasses.replace(_make_payload(2, 2), metadata={"traj_id": "t1"})
    p2 = dataclasses.replace(_make_payload(2, 3), metadata={"traj_id": "t2"})
    p3 = dataclasses.replace(_make_payload(1, 2), metadata={"traj_id": "t3"})

    # Feed 1 (4 tokens < 10): buffers without emitting.
    self.assertEmpty(list(assembler.feed([p1])))

    # Feed 2 (4 + 5 = 9 tokens < 10): still held back, cannot fill a chunk yet.
    self.assertEmpty(list(assembler.feed([p2])))

    # Feed 3 reaches rollouts_per_optimizer_update=3 -> drains the whole buffer (12
    # tokens) into a full chunk plus a final remainder chunk.
    batches3 = list(assembler.feed([p3]))
    self.assertLen(batches3, 2)
    self.assertFalse(batches3[0].is_final_batch)
    self.assertEqual(set(batches3[0].trajectory_ids), {"t1", "t2"})
    self.assertEqual(
        set(batches3[0].payload.metadata["trajectory_ids"]), {"t1", "t2"}
    )
    self.assertTrue(batches3[1].is_final_batch)
    self.assertEqual(batches3[1].trajectory_ids, ("t3",))

  def test_feed_yields_microbatches_lazily_one_at_a_time(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        num_generations=3,
        mini_batch_size=1,
        max_packed_len=6,
    )
    items = [
        dataclasses.replace(_make_payload(2, 2), metadata={"traj_id": f"t{i}"})
        for i in range(3)
    ]
    emit_calls = 0
    orig_emit = assembler._emit_one_chunk

    def counting_emit(*args, **kwargs):
      nonlocal emit_calls
      emit_calls += 1
      return orig_emit(*args, **kwargs)

    assembler._emit_one_chunk = counting_emit

    batch_iter = assembler.feed(items)
    # Calling feed() validates/ingests items into _buffer without packing any
    # chunk yet.
    self.assertEqual(emit_calls, 0)
    self.assertLen(assembler._buffer, 3)

    mb0 = next(batch_iter)
    self.assertEqual(emit_calls, 1)
    self.assertFalse(mb0.is_final_batch)
    self.assertLen(assembler._buffer, 2)

    mb1 = next(batch_iter)
    self.assertEqual(emit_calls, 2)
    self.assertFalse(mb1.is_final_batch)
    self.assertLen(assembler._buffer, 1)

    mb2 = next(batch_iter)
    self.assertEqual(emit_calls, 3)
    self.assertTrue(mb2.is_final_batch)
    self.assertEmpty(assembler._buffer)
    self.assertIsNone(next(batch_iter, None))

  def test_feed_merges_lineage_contexts(self):
    ctx1 = lineage.LineageContext(
        tracking_id="traj_p1_g0", parent_tracking_ids=["p1"]
    )
    ctx1.add_event("engine.dispatch", "rollout", {"group_index": 0})
    ctx1.add_event("worker.rollout", "generate", {"worker_id": "w0"})

    ctx2 = lineage.LineageContext(
        tracking_id="traj_p1_g1", parent_tracking_ids=["p1"]
    )
    ctx2.add_event("engine.dispatch", "rollout", {"group_index": 1})
    ctx2.add_event("worker.rollout", "generate", {"worker_id": "w1"})

    payload1 = _make_payload(2, 2, metadata={"lineage": ctx1})
    payload2 = _make_payload(2, 2, metadata={"lineage": ctx2})

    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1, num_generations=2, mini_batch_size=1, max_packed_len=16
    )
    batches = list(assembler.feed([payload1, payload2]))

    self.assertLen(batches, 1)
    batch_payload = batches[0].payload
    self.assertIn("lineage", batch_payload.metadata)
    batch_ctx = batch_payload.metadata["lineage"]
    self.assertEqual(batch_ctx.tracking_id, "batch_0")
    self.assertEqual(
        sorted(batch_ctx.parent_tracking_ids), ["traj_p1_g0", "traj_p1_g1"]
    )
    self.assertLen(batch_ctx.events, 1)
    merge_event = batch_ctx.events[0]
    self.assertEqual(merge_event.component, "orchestrator.assembler")
    self.assertEqual(merge_event.operation, "pack")
    self.assertEqual(
        merge_event.attributes.get("packing_type"), "sequence_packed"
    )
    self.assertEqual(merge_event.attributes.get("num_items"), 2)

  def test_feed_multi_bin_creates_distinct_lineage_contexts(self):
    ctx1 = lineage.LineageContext(
        tracking_id="traj_p1_g0", parent_tracking_ids=["p1"]
    )
    ctx2 = lineage.LineageContext(
        tracking_id="traj_p2_g0", parent_tracking_ids=["p2"]
    )

    p1 = _make_payload(5, 5, metadata={"lineage": ctx1})
    p2 = _make_payload(4, 4, metadata={"lineage": ctx2})

    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1, num_generations=2, mini_batch_size=1, max_packed_len=12
    )
    batches = list(assembler.feed([p1, p2]))

    self.assertLen(batches, 2)
    self.assertEqual(
        batches[0].payload.metadata["lineage"].tracking_id, "batch_0"
    )
    self.assertEqual(
        batches[0].payload.metadata["lineage"].parent_tracking_ids,
        ["traj_p1_g0"],
    )
    self.assertEqual(
        batches[1].payload.metadata["lineage"].tracking_id, "batch_1"
    )
    self.assertEqual(
        batches[1].payload.metadata["lineage"].parent_tracking_ids,
        ["traj_p2_g0"],
    )

  def test_feed_without_lineage_returns_clean_payload(self):
    p = _make_payload(1, 1)
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1, num_generations=1, mini_batch_size=1, max_packed_len=8
    )
    batches = list(assembler.feed([p]))
    self.assertLen(batches, 1)
    self.assertNotIn("lineage", batches[0].payload.metadata)

  def test_feed_sequential_increments_monotonic_batch_ids(self):
    ctx = lineage.LineageContext(
        tracking_id="traj_t0_g0", parent_tracking_ids=["t0"]
    )
    p = _make_payload(1, 1, metadata={"lineage": ctx})
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1, num_generations=1, mini_batch_size=1, max_packed_len=8
    )
    out1 = list(assembler.feed([p]))
    out2 = list(assembler.feed([p]))
    self.assertEqual(out1[0].payload.metadata["lineage"].tracking_id, "batch_0")
    self.assertEqual(out2[0].payload.metadata["lineage"].tracking_id, "batch_1")

  def test_custom_start_batch_index_on_init_and_reset(self):
    ctx = lineage.LineageContext(
        tracking_id="traj_t0_g0", parent_tracking_ids=["t0"]
    )
    p = _make_payload(1, 1, metadata={"lineage": ctx})
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        num_generations=1,
        mini_batch_size=1,
        max_packed_len=8,
        start_batch_index=42,
    )
    out = list(assembler.feed([p]))
    self.assertEqual(out[0].payload.metadata["lineage"].tracking_id, "batch_42")
    assembler.reset(start_batch_index=100)
    out2 = list(assembler.feed([p]))
    self.assertEqual(
        out2[0].payload.metadata["lineage"].tracking_id, "batch_100"
    )
    assembler.reset()
    out3 = list(assembler.feed([p]))
    self.assertEqual(
        out3[0].payload.metadata["lineage"].tracking_id, "batch_101"
    )

  def test_streaming_feed_emits_full_chunk_mid_step(self):
    # batch_size=1, max_packed_len=6 -> a full chunk holds 6 tokens.
    # mini_batch_size is large, so the optimizer-update boundary is not reached.
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        num_generations=1,
        mini_batch_size=10,
        max_packed_len=6,
    )
    p1 = dataclasses.replace(_make_payload(2, 2), metadata={"traj_id": "t1"})
    p2 = dataclasses.replace(_make_payload(2, 2), metadata={"traj_id": "t2"})

    # 4 tokens < 6: buffered.
    self.assertEmpty(list(assembler.feed([p1])))

    # 4 + 4 = 8 >= 6: emits one full chunk mid-step (not final), holding the
    # remainder back for the next feed.
    mid = list(assembler.feed([p2]))
    self.assertLen(mid, 1)
    self.assertFalse(mid[0].is_final_batch)
    self.assertEqual(mid[0].trajectory_ids, ("t1",))

    # The held-back remainder is drained (and marked final) on flush.
    flushed = list(assembler.flush())
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].trajectory_ids, ("t2",))

  def test_streaming_flush_and_reset(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        num_generations=2,
        mini_batch_size=2,
        max_packed_len=16,
    )
    p1 = dataclasses.replace(_make_payload(2, 2), metadata={"traj_id": "t1"})
    batches = list(assembler.feed([p1]))
    self.assertEmpty(batches)

    # Explicit flush should emit buffered items and mark final
    flushed = list(assembler.flush())
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].trajectory_ids, ("t1",))

    # Reset clears state
    assembler.reset()
    self.assertEmpty(list(assembler.flush()))

  def test_to_rl_trainer_payload_reuses_contiguous_pack_chunk_buffers(self):
    item1 = batch_assembly.to_pack_item(_make_payload(2, 2, advantage=1.0))
    item2 = batch_assembly.to_pack_item(_make_payload(2, 2, advantage=2.0))
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
      tracemalloc.start()
    try:
      base_bytes, _ = tracemalloc.get_traced_memory()
      tracemalloc.reset_peak()
      chunk = packing.pack_chunk(
          [[item1], [item2]], budget=4096, pad_id=0, carried=()
      )
      after_pack_bytes, _ = tracemalloc.get_traced_memory()
      tracemalloc.reset_peak()
      payload = batch_assembly.to_rl_trainer_payload(chunk, max_segments=2)
      retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
      if not was_tracing:
        tracemalloc.stop()

    # 6 pre-allocated [2, 4096] 4-byte arrays = 196,608 bytes; while both
    # `chunk` and `payload` are alive, `to_rl_trainer_payload` passes `chunk`'s
    # 2D arrays directly by reference instead of allocating 163,840 extra bytes.
    # Under `Py_LIMITED_API` (`STABLE_ABI`), `_packing_ext` uses `std::malloc`
    # which is not tracked by `tracemalloc`, whereas non-limited-API builds use
    # `PyMem_RawMalloc`.
    pack_allocated = after_pack_bytes - base_bytes
    to_payload_peak = peak_bytes - after_pack_bytes
    if pack_allocated > 65_536:
      self.assertGreater(pack_allocated, 190_000)
      self.assertLess(
          peak_bytes - base_bytes, (retained_bytes - base_bytes) * 1.05
      )
    self.assertLess(to_payload_peak, 4_096)

    chunk = packing.pack_chunk(
        [[item1], [item2]], budget=6, pad_id=0, carried=()
    )
    payload = batch_assembly.to_rl_trainer_payload(chunk, max_segments=2)
    self.assertIs(payload.completion_ids, chunk.ids)
    self.assertIs(payload.completion_mask, chunk.completion_mask)
    self.assertIs(payload.advantages, chunk.advantages)
    self.assertIs(payload.segment_ids, chunk.segment_ids)
    self.assertIs(payload.segment_positions, chunk.segment_positions)

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
        num_generations=1,
        mini_batch_size=3,  # rollouts_per_optimizer_update = 3
    )
    # 3 groups with 4 tokens each (total 12 tokens < 16)
    # Group 1: 4 tokens -> buffers
    res1 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=2, completion_length=2, val=1
            )
        ])
    )
    self.assertEmpty(res1)

    # Group 2: 4 tokens -> buffers (8 tokens total)
    res2 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=2, completion_length=2, val=2
            )
        ])
    )
    self.assertEmpty(res2)

    # Group 3: 4 tokens -> hits rollouts_per_optimizer_update = 3, auto-flushes!
    res3 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=2, completion_length=2, val=3
            )
        ])
    )
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
        num_generations=1,
        mini_batch_size=3,  # rollouts_per_optimizer_update = 3
    )
    # Item 1: 8 tokens -> buffers (8 < 16)
    res1 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=4, completion_length=4, val=1
            )
        ])
    )
    self.assertEmpty(res1)

    # Item 2: 8 tokens -> 8 + 8 = 16 tokens >= 16 (chunk capacity)!
    # Emits early before the optimizer-update boundary.
    res2 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=4, completion_length=4, val=2
            )
        ])
    )
    self.assertLen(res2, 1)
    self.assertFalse(res2[0].is_final_batch)
    self.assertEqual(res2[0].payload.completion_ids.shape, (1, 16))

    # Item 3 hits the optimizer-update boundary and auto-flushes the open bin.
    res3 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=2, completion_length=2, val=3
            )
        ])
    )
    self.assertLen(res3, 1)
    self.assertTrue(res3[0].is_final_batch)
    self.assertEqual(res3[0].payload.completion_ids.shape, (1, 16))

  def test_feed_overflow_opens_new_bin(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=2,
        max_packed_len=16,
        pad_id=0,
        num_generations=1,
        mini_batch_size=2,
    )
    # Item 1 has 10 tokens
    res1 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=4, completion_length=6, val=1
            )
        ])
    )
    self.assertEmpty(res1)

    # Item 2 has 8 tokens (10 + 8 = 18 > 16, cannot fit!)
    # Should place Item 1 in bin 1 and Item 2 in bin 2.
    # Reaching rollouts_per_optimizer_update = 2 auto-flushes bin 2!
    # With batch_size=2, the 2 bins form 1 microbatch of shape [2, 16]!
    res2 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=4, completion_length=4, val=2
            )
        ])
    )
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
        num_generations=1,
        mini_batch_size=4,
    )
    list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=2, completion_length=2, val=1
            )
        ])
    )
    flushed = list(assembler.flush())
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].payload.completion_ids.shape, (1, 16))
    self.assertEmpty(list(assembler.flush()))

  def test_reset_clears_state(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=16,
        pad_id=0,
        num_generations=1,
        mini_batch_size=4,
    )
    list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=2, completion_length=2, val=1
            )
        ])
    )
    assembler.reset()
    self.assertEmpty(list(assembler.flush()))

  def test_sequence_packed_batch_assembler_tracks_trajectory_ids(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=16,
        pad_id=0,
        num_generations=2,
        mini_batch_size=2,  # rollouts_per_optimizer_update = 4
    )
    # Group 1: 2 items of 4 tokens each (8 tokens total) -> buffers
    res1 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=2,
                completion_length=2,
                prompt_id="p0",
                group_index=0,
            ),
            self._make_streaming_payload(
                prompt_length=2,
                completion_length=2,
                prompt_id="p0",
                group_index=1,
            ),
        ])
    )
    self.assertEmpty(res1)

    # Group 2: 2 items of 4 tokens each (8 tokens total). Reaches the optimizer-update boundary.
    res2 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=2,
                completion_length=2,
                prompt_id="p1",
                group_index=0,
            ),
            self._make_streaming_payload(
                prompt_length=2,
                completion_length=2,
                prompt_id="p1",
                group_index=1,
            ),
        ])
    )
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
        num_generations=2,
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
    res1 = list(assembler.feed([item1, item2]))
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
    # Total rollouts = 2 + 2 = 4 == rollouts_per_optimizer_update.
    res2 = list(assembler.feed([item3, item4]))
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
        num_generations=2,
        mini_batch_size=2,  # rollouts_per_optimizer_update = 4
    )

    # Group 1 (2 items, 6 tokens each = 12 tokens): buffered (12 < 16)
    group1 = [
        self._make_streaming_payload(
            prompt_length=3, completion_length=3, val=1
        ),
        self._make_streaming_payload(
            prompt_length=3, completion_length=3, val=1
        ),
    ]
    res1 = list(assembler.feed(group1))
    self.assertEmpty(res1)

    # Group 2 (final batch, 2 items, 6 tokens each):
    # Total tokens = 12 + 12 = 24 tokens.
    # Chunk 1 packs 2 items (12 tokens <= 16).
    # Chunk 2 packs remaining 2 items (12 tokens <= 16).
    # Reaching the optimizer-update boundary drains the entire buffer.
    group2 = [
        self._make_streaming_payload(
            prompt_length=3, completion_length=3, val=2
        ),
        self._make_streaming_payload(
            prompt_length=3, completion_length=3, val=2
        ),
    ]
    res2 = list(assembler.feed(group2))
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
        num_generations=2,
        mini_batch_size=1,
    )
    item1 = self._make_streaming_payload(
        prompt_length=3, completion_length=3, val=1
    )
    item2 = self._make_streaming_payload(
        prompt_length=3, completion_length=3, val=2
    )
    res = list(assembler.feed([item1, item2]))
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
        num_generations=2,
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
        num_generations=1,
        mini_batch_size=4,  # rollouts_per_optimizer_update = 4
    )
    # chunk_capacity = batch_size * max_packed_len = 2 * 16 = 32 tokens.
    # Item 1: 16 tokens -> buffers (16 < 32)
    res1 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=8, completion_length=8, val=1
            )
        ])
    )
    self.assertEmpty(res1)

    # Item 2: 16 tokens -> 16 + 16 = 32 tokens >= 32 (chunk capacity).
    # Emits early microbatch of shape [2, 16], is_final_batch=False
    res2 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=8, completion_length=8, val=2
            )
        ])
    )
    self.assertLen(res2, 1)
    self.assertFalse(res2[0].is_final_batch)
    self.assertEqual(res2[0].payload.completion_ids.shape, (2, 16))

    # Item 3: 16 tokens -> buffers (16 < 32)
    res3 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=8, completion_length=8, val=3
            )
        ])
    )
    self.assertEmpty(res3)

    # Item 4: 16 tokens -> Step done (rollouts = 4)!
    # Emits final microbatch of shape [2, 16], is_final_batch=True
    res4 = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=8, completion_length=8, val=4
            )
        ])
    )
    self.assertLen(res4, 1)
    self.assertTrue(res4[0].is_final_batch)
    self.assertEqual(res4[0].payload.completion_ids.shape, (2, 16))

    # Early EOF flush test: 1 item fed into a fresh step, flush pads to [2, 16]
    list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=3, completion_length=3, val=5
            )
        ])
    )
    flushed = list(assembler.flush())
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].payload.completion_ids.shape, (2, 16))
    np.testing.assert_array_equal(flushed[0].payload.segment_ids[0, :6], 1)
    np.testing.assert_array_equal(flushed[0].payload.segment_ids[0, 6:], 0)
    np.testing.assert_array_equal(flushed[0].payload.segment_ids[1], 0)

  def test_padding_stats_reflect_bin_occupancy(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=2,
        max_packed_len=16,
        pad_id=0,
        num_generations=2,
        mini_batch_size=1,
    )
    res = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=3, completion_length=3, val=1
            ),
            self._make_streaming_payload(
                prompt_length=2, completion_length=2, val=2
            ),
        ])
    )

    stats = res[0].padding_stats
    # First-fit packs both items into row 0; row 1 is a filler row.
    np.testing.assert_array_equal(stats.row_valid_tokens, [10, 0])
    np.testing.assert_array_equal(stats.row_num_sequences, [2, 0])
    self.assertEqual(stats.row_capacity, 16)
    self.assertAlmostEqual(stats.padding_ratio, 1.0 - 10 / 32)
    self.assertAlmostEqual(stats.row_imbalance, 2.0)

  def test_padding_stats_with_one_segment_per_row(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=2,
        max_packed_len=16,
        pad_id=0,
        num_generations=2,
        mini_batch_size=1,
        max_segments_per_packed_row=1,
    )
    res = list(
        assembler.feed([
            self._make_streaming_payload(
                prompt_length=3, completion_length=3, val=1
            ),
            self._make_streaming_payload(
                prompt_length=2, completion_length=2, val=2
            ),
        ])
    )

    stats = res[0].padding_stats
    np.testing.assert_array_equal(stats.row_valid_tokens, [6, 4])
    np.testing.assert_array_equal(stats.row_num_sequences, [1, 1])
    self.assertAlmostEqual(stats.row_imbalance, 1.2)


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
    metadata=None,
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
  payload_metadata = dict(metadata) if metadata is not None else {}
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
      metadata=payload_metadata,
  )


def _stats(row_valid_tokens, row_num_sequences, row_capacity):
  return batch_assembly.PaddingStats(
      row_valid_tokens=np.asarray(row_valid_tokens, dtype=np.int64),
      row_num_sequences=np.asarray(row_num_sequences, dtype=np.int64),
      row_capacity=row_capacity,
  )


class PaddingStatsTest(absltest.TestCase):

  def test_properties(self):
    stats = _stats([8, 2, 0, 6], [3, 1, 0, 2], 10)

    self.assertEqual(stats.num_rows, 4)
    self.assertEqual(stats.valid_tokens, 16)
    self.assertEqual(stats.capacity_tokens, 40)
    self.assertAlmostEqual(stats.padding_ratio, 0.6)
    np.testing.assert_allclose(stats.row_fill, [0.8, 0.2, 0.0, 0.6])
    self.assertAlmostEqual(stats.row_imbalance, 2.0)

  def test_rejects_invalid_inputs(self):
    for valid, nseq, cap in (
        ([[1, 2]], [[1, 1]], 4),  # Not 1D.
        ([1, 2], [1], 4),  # Shape mismatch.
        ([], [], 4),  # No rows.
        ([1], [1], 0),  # Non-positive capacity.
        ([5], [1], 4),  # Exceeds capacity.
        ([-1, 2], [1, 1], 4),  # Negative tokens.
        ([1], [-1], 4),  # Negative sequences.
        ([0, 0], [0, 0], 4),  # No valid tokens.
    ):
      with self.subTest(valid=valid, nseq=nseq, cap=cap):
        with self.assertRaises(ValueError):
          _stats(valid, nseq, cap)

  def test_summarize_rejects_empty(self):
    with self.assertRaises(ValueError):
      batch_assembly.summarize_padding_stats([])

  def test_summarize(self):
    summary = batch_assembly.summarize_padding_stats([
        _stats([10, 10], [1, 2], 10),  # Fully packed, balanced.
        _stats([6, 0], [3, 0], 10),  # Half padded, one filler row.
    ])

    self.assertAlmostEqual(summary["ratio"], 1.0 - 26 / 40)
    self.assertAlmostEqual(summary["microbatch_ratio_mean"], 0.35)
    self.assertAlmostEqual(summary["microbatch_ratio_max"], 0.7)
    self.assertAlmostEqual(summary["microbatch_ratio_min"], 0.0)
    self.assertAlmostEqual(summary["microbatch_imbalance"], 20 / 13)
    self.assertAlmostEqual(summary["row_imbalance_mean"], 1.5)
    self.assertAlmostEqual(summary["row_imbalance_max"], 2.0)
    self.assertAlmostEqual(summary["row_fill_min"], 0.0)
    self.assertAlmostEqual(summary["row_fill_max"], 1.0)
    self.assertEqual(summary["empty_rows"], 1.0)
    self.assertAlmostEqual(summary["sequences_per_row_mean"], 1.5)
    self.assertEqual(summary["sequences_per_row_max"], 3.0)
    self.assertEqual(summary["valid_tokens"], 26.0)
    self.assertEqual(summary["capacity_tokens"], 40.0)


class PaddedBatchAssemblerTest(absltest.TestCase):

  def _assembler(self, **kwargs):
    defaults = dict(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=5,
        pad_id=0,
        num_generations=2,
        mini_batch_size=2,
    )
    defaults.update(kwargs)
    return batch_assembly.PaddedBatchAssembler(**defaults)

  def test_rejects_non_positive_dimensions(self):
    for bad in (
        dict(batch_size=0),
        dict(max_prompt_length=0),
        dict(max_response_length=-1),
        dict(num_generations=0),
        dict(mini_batch_size=0),
    ):
      with self.assertRaises(ValueError):
        self._assembler(**bad)

  def test_requires_num_generations_and_mini_batch_size(self):
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
          num_generations=1,
      )

  def test_max_seq_len_is_sum_of_prompt_and_response_lengths(self):
    assembler = self._assembler(max_prompt_length=128, max_response_length=256)
    self.assertEqual(assembler.max_seq_len, 384)

  def test_padding_stats_count_truncated_tokens_per_row(self):
    # P=4, C=5: the second item is truncated to 4 + 5 = 9 valid tokens.
    res = list(
        self._assembler(num_generations=1, mini_batch_size=2).feed(
            [_make_payload(2, 3), _make_payload(6, 7)]
        )
    )

    self.assertLen(res, 1)
    stats = res[0].padding_stats
    np.testing.assert_array_equal(stats.row_valid_tokens, [5, 9])
    np.testing.assert_array_equal(stats.row_num_sequences, [1, 1])
    self.assertEqual(stats.row_capacity, 9)
    self.assertAlmostEqual(stats.padding_ratio, 1.0 - 14 / 18)
    self.assertAlmostEqual(stats.row_imbalance, 9 / 7)

  def test_padding_stats_mark_filler_rows_in_remainder(self):
    res = list(
        self._assembler(num_generations=1, mini_batch_size=1).feed(
            [_make_payload(2, 3)]
        )
    )

    self.assertLen(res, 1)
    self.assertTrue(res[0].is_final_batch)
    stats = res[0].padding_stats
    np.testing.assert_array_equal(stats.row_valid_tokens, [5, 0])
    np.testing.assert_array_equal(stats.row_num_sequences, [1, 0])

  def test_flush_reports_padding_stats(self):
    assembler = self._assembler()
    self.assertEmpty(list(assembler.feed([_make_payload(1, 1)])))

    flushed = list(assembler.flush())

    np.testing.assert_array_equal(
        flushed[0].padding_stats.row_valid_tokens, [2, 0]
    )

  def test_final_marking_preserves_padding_stats(self):
    # A full chunk lands on the update boundary, so `feed` re-marks the last
    # batch final; its padding stats must survive.
    res = list(
        self._assembler(num_generations=2, mini_batch_size=1).feed(
            [_make_payload(1, 1), _make_payload(2, 2)]
        )
    )

    self.assertLen(res, 1)
    self.assertTrue(res[0].is_final_batch)
    np.testing.assert_array_equal(res[0].padding_stats.row_valid_tokens, [2, 4])

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

  def test_pack_merges_lineage_contexts(self):
    ctx1 = lineage.LineageContext(
        tracking_id="traj_p1_g0", parent_tracking_ids=["p1"]
    )
    ctx1.add_event("worker.rollout", "generate", {"worker_id": "w0"})
    ctx2 = lineage.LineageContext(
        tracking_id="traj_p1_g1", parent_tracking_ids=["p1"]
    )
    ctx2.add_event("worker.rollout", "generate", {"worker_id": "w1"})

    item1 = _make_payload(2, 2, metadata={"lineage": ctx1})
    item2 = _make_payload(2, 2, metadata={"lineage": ctx2})

    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=4,
        pad_id=0,
        num_generations=1,
        mini_batch_size=1,
    )
    payloads = assembler.pack([item1, item2])

    self.assertLen(payloads, 1)
    batch_payload = payloads[0]
    self.assertIn("lineage", batch_payload.metadata)
    batch_ctx = batch_payload.metadata["lineage"]
    self.assertEqual(batch_ctx.tracking_id, "batch_0")
    self.assertEqual(
        sorted(batch_ctx.parent_tracking_ids), ["traj_p1_g0", "traj_p1_g1"]
    )
    self.assertLen(batch_ctx.events, 1)
    merge_event = batch_ctx.events[0]
    self.assertEqual(merge_event.component, "orchestrator.assembler")
    self.assertEqual(merge_event.operation, "pack")
    self.assertEqual(merge_event.attributes.get("packing_type"), "padded")
    self.assertEqual(merge_event.attributes.get("num_items"), 2)

  def test_pack_sequential_increments_monotonic_batch_ids(self):
    ctx = lineage.LineageContext(
        tracking_id="traj_t0_g0", parent_tracking_ids=["t0"]
    )
    item = _make_payload(2, 2, metadata={"lineage": ctx})
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=4,
        pad_id=0,
        num_generations=1,
        mini_batch_size=1,
    )
    out1 = assembler.pack([item])
    out2 = assembler.pack([item])
    self.assertEqual(out1[0].metadata["lineage"].tracking_id, "batch_0")
    self.assertEqual(out2[0].metadata["lineage"].tracking_id, "batch_1")

  def test_custom_start_batch_index_on_init_and_reset(self):
    ctx = lineage.LineageContext(
        tracking_id="traj_t0_g0", parent_tracking_ids=["t0"]
    )
    item = _make_payload(2, 2, metadata={"lineage": ctx})
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=1,
        max_prompt_length=4,
        max_response_length=4,
        pad_id=0,
        num_generations=1,
        mini_batch_size=1,
        start_batch_index=10,
    )
    out = list(assembler.feed([item]))
    self.assertEqual(out[0].payload.metadata["lineage"].tracking_id, "batch_10")
    assembler.reset(start_batch_index=20)
    out2 = list(assembler.feed([item]))
    self.assertEqual(
        out2[0].payload.metadata["lineage"].tracking_id, "batch_20"
    )
    assembler.reset()
    out3 = list(assembler.feed([item]))
    self.assertEqual(
        out3[0].payload.metadata["lineage"].tracking_id, "batch_21"
    )


class SequencePackedConversionTest(absltest.TestCase):
  """Test for the payload <-> PackItem adapter boundary."""

  def _make_assembler(self, max_packed_len=16, **kwargs):
    defaults = dict(
        batch_size=1,
        num_generations=2,
        mini_batch_size=2,
        max_packed_len=max_packed_len,
    )
    defaults.update(kwargs)
    return batch_assembly.SequencePackedBatchAssembler(**defaults)

  def _drain(self, assembler, items):
    """Fully packs `items` via the streaming contract (feed then flush)."""
    return [
        batch.payload for batch in [*assembler.feed(items), *assembler.flush()]
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
    with self.assertRaisesRegex(ValueError, "num_generations must be positive"):
      self._make_assembler(max_packed_len=16, num_generations=0)
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

    out = np.full(
        (max_prompt + max_response, _ROUTING_LAYERS, _ROUTING_TOP_K),
        _UNSET,
        dtype=np.int16,
    )
    batch_assembly._routed_experts_aligned(  # pylint: disable=protected-access
        routed, prompt_len, completion_len, max_prompt, max_response, out=out
    )

    self.assertEqual(out.dtype, np.int16)
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
    out = np.full((5, _ROUTING_LAYERS, _ROUTING_TOP_K), _UNSET, dtype=np.int16)
    batch_assembly._routed_experts_aligned(  # pylint: disable=protected-access
        routed, 4, 1, 2, 3, out=out
    )
    # Prompt rows 0 and 1 are dropped; 2 and 3 survive, in order.
    np.testing.assert_array_equal(out[0], 2)
    np.testing.assert_array_equal(out[1], 3)

  def test_prefix_capture_ending_in_completion_leaves_tail_unset(self):
    """Prefix routing of length `p + c - 1` leaves the final completion token unset."""
    prompt_len, completion_len = 3, 4
    max_prompt, max_response = 5, 6
    # Capture covers 3 prompt tokens + 3 of the 4 completion tokens.
    routed = np.concatenate(
        [_routing(prompt_len, 7), _routing(completion_len - 1, 9)], axis=0
    )
    out = np.full(
        (max_prompt + max_response, _ROUTING_LAYERS, _ROUTING_TOP_K),
        _UNSET,
        dtype=np.int16,
    )
    batch_assembly._routed_experts_aligned(  # pylint: disable=protected-access
        routed, prompt_len, completion_len, max_prompt, max_response, out=out
    )
    np.testing.assert_array_equal(out[: max_prompt - prompt_len], _UNSET)
    np.testing.assert_array_equal(out[max_prompt - prompt_len : max_prompt], 7)
    np.testing.assert_array_equal(
        out[max_prompt : max_prompt + completion_len - 1], 9
    )
    np.testing.assert_array_equal(
        out[max_prompt + completion_len - 1 :], _UNSET
    )

  def test_prefix_capture_single_token_completion_leaves_response_unset(self):
    """When `completion_len == 1`, a `p + c - 1 == p` capture aligns prompt and leaves response unset."""
    prompt_len, completion_len = 3, 1
    max_prompt, max_response = 5, 4
    routed = _routing(prompt_len, 7)
    out = np.full(
        (max_prompt + max_response, _ROUTING_LAYERS, _ROUTING_TOP_K),
        _UNSET,
        dtype=np.int16,
    )
    batch_assembly._routed_experts_aligned(  # pylint: disable=protected-access
        routed, prompt_len, completion_len, max_prompt, max_response, out=out
    )
    prompt_start = max_prompt - prompt_len
    np.testing.assert_array_equal(out[:prompt_start], _UNSET)
    np.testing.assert_array_equal(out[prompt_start:max_prompt], 7)
    np.testing.assert_array_equal(out[max_prompt:], _UNSET)

  def test_rejects_capture_shorter_than_sequence_minus_one(self):
    """Completion-only or truncated captures (`< p + c - 1`) are rejected."""
    prompt_len, completion_len = 3, 4
    max_prompt, max_response = 5, 6
    out = np.full(
        (max_prompt + max_response, _ROUTING_LAYERS, _ROUTING_TOP_K),
        _UNSET,
        dtype=np.int16,
    )
    with self.assertRaisesRegex(
        ValueError, "routed_experts length must be >= 6"
    ):
      batch_assembly._routed_experts_aligned(  # pylint: disable=protected-access
          _routing(completion_len, 9),
          prompt_len,
          completion_len,
          max_prompt,
          max_response,
          out=out,
      )


class PaddedBatchAssemblerRoutingTest(absltest.TestCase):
  """The assembler must emit `[B, P + C, num_layers, top_k]`, or nothing."""

  MAX_PROMPT = 4
  MAX_RESPONSE = 4

  def _assembler(self, batch_size=2, num_generations=1, mini_batch_size=1):
    return batch_assembly.PaddedBatchAssembler(
        batch_size=batch_size,
        max_prompt_length=self.MAX_PROMPT,
        max_response_length=self.MAX_RESPONSE,
        pad_id=0,
        num_generations=2,
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
    self.assertEqual(routed.dtype, np.int16)
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

  def test_pack_does_not_double_allocate_routed_experts_buffer(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=4,
        max_prompt_length=512,
        max_response_length=512,
        pad_id=0,
        num_generations=2,
        mini_batch_size=2,
    )
    items = [
        dataclasses.replace(
            self._payload(i + 1, with_routing=False),
            routed_experts=np.full((5, 16, 4), i + 1, dtype=np.int16),
        )
        for i in range(4)
    ]
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
      tracemalloc.start()
    try:
      base_bytes, _ = tracemalloc.get_traced_memory()
      tracemalloc.reset_peak()
      [packed] = assembler.pack(items)
      retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
      if not was_tracing:
        tracemalloc.stop()

    self.assertIsNotNone(packed.routed_experts)
    self.assertEqual(packed.routed_experts.shape, (4, 1024, 16, 4))
    net_retained = retained_bytes - base_bytes
    net_peak = peak_bytes - base_bytes
    # Pre-allocating the batch buffer (~565 KB for [4, 1024, 16, 4] int16 +
    # token/mask arrays) avoids the ~2x np.stack peak spike (~1.09 MB).
    # Under `Py_LIMITED_API` (`STABLE_ABI`), `_packing_ext` uses `std::malloc`
    # which is not tracked by `tracemalloc`; in that case `net_peak` in the
    # Python heap remains well below a single routed_experts buffer.
    if net_retained > 65_536:
      self.assertGreater(net_retained, 500_000)
      self.assertLess(net_peak, net_retained * 1.10)
    else:
      self.assertLess(net_peak, 65_536)

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
        num_generations=2,
        mini_batch_size=2,  # rollouts_per_optimizer_update = 4
    )
    # Feed half an optimizer update: should buffer and return an empty iterator.
    items_group1 = [
        self._make_streaming_payload(1),
        self._make_streaming_payload(2),
    ]
    res1 = list(assembler.feed(items_group1))
    self.assertEmpty(res1)

    # Feed 2 items (second half): reaches rollouts_per_optimizer_update = 4.
    items_group2 = [
        self._make_streaming_payload(3),
        self._make_streaming_payload(4),
    ]
    res2 = list(assembler.feed(items_group2))
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
        num_generations=2,
        mini_batch_size=2,  # rollouts_per_optimizer_update = 4
    )
    # Feed 2 items: reaches batch_size=2, but not the optimizer-update boundary.
    res1 = list(
        assembler.feed(
            [self._make_streaming_payload(1), self._make_streaming_payload(2)]
        )
    )
    self.assertLen(res1, 1)
    self.assertFalse(res1[0].is_final_batch)

    # Feed 2 items: reaches rollouts_per_optimizer_update=4, so is_final_batch=True
    res2 = list(
        assembler.feed(
            [self._make_streaming_payload(3), self._make_streaming_payload(4)]
        )
    )
    self.assertLen(res2, 1)
    self.assertTrue(res2[0].is_final_batch)

  def test_feed_yields_microbatches_lazily_one_at_a_time(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        num_generations=2,
        mini_batch_size=2,  # rollouts_per_optimizer_update = 4
    )
    items = [self._make_streaming_payload(i) for i in range(1, 5)]
    with mock.patch.object(
        assembler, "_pack_chunk", wraps=assembler._pack_chunk
    ) as spy_pack:
      it = assembler.feed(items)
      self.assertEqual(spy_pack.call_count, 0)

      mb0 = next(it)
      self.assertEqual(spy_pack.call_count, 1)
      self.assertFalse(mb0.is_final_batch)

      mb1 = next(it)
      self.assertEqual(spy_pack.call_count, 2)
      self.assertTrue(mb1.is_final_batch)

      with self.assertRaises(StopIteration):
        next(it)

  def test_feed_auto_flush_with_remainder(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=4,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        num_generations=3,
        mini_batch_size=1,  # rollouts_per_optimizer_update = 3
    )
    # Feed 3 items: hits rollouts_per_optimizer_update=3 and auto-flushes.
    res = list(
        assembler.feed([
            self._make_streaming_payload(1),
            self._make_streaming_payload(2),
            self._make_streaming_payload(3),
        ])
    )
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
        num_generations=2,
        mini_batch_size=2,  # rollouts_per_optimizer_update = 4
    )
    # Feed only 2 items mid-step
    res1 = list(
        assembler.feed(
            [self._make_streaming_payload(1), self._make_streaming_payload(2)]
        )
    )
    self.assertEmpty(res1)

    # Dataset runs out: manual flush
    flushed = list(assembler.flush())
    self.assertLen(flushed, 1)
    self.assertTrue(flushed[0].is_final_batch)
    self.assertEqual(flushed[0].payload.prompt_ids.shape, (4, 2))
    # Subsequent flush on empty buffer returns empty
    self.assertEmpty(list(assembler.flush()))

  def test_reset_clears_state(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=4,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        num_generations=2,
        mini_batch_size=2,
    )
    list(
        assembler.feed(
            [self._make_streaming_payload(1), self._make_streaming_payload(2)]
        )
    )
    assembler.reset()
    self.assertEmpty(list(assembler.flush()))

  def test_padded_batch_assembler_tracks_trajectory_ids(self):
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=3,
        max_prompt_length=2,
        max_response_length=2,
        pad_id=0,
        num_generations=2,
        mini_batch_size=2,  # rollouts_per_optimizer_update = 4
    )
    items = [
        self._make_streaming_payload(1, prompt_id="p0", group_index=0),
        self._make_streaming_payload(1, prompt_id="p0", group_index=1),
        self._make_streaming_payload(1, prompt_id="p1", group_index=0),
        self._make_streaming_payload(1, prompt_id="p1", group_index=1),
    ]
    # Feed 4 items: first batch gets 3 items, second auto-flushed gets 1 item + 2 padding
    res = list(assembler.feed(items))
    self.assertLen(res, 2)
    self.assertEqual(
        res[0].trajectory_ids,
        ("traj_p0_g0", "traj_p0_g1", "traj_p1_g0"),
    )
    # 2nd batch has 1 real trajectory and 2 padded rows
    self.assertEqual(res[1].trajectory_ids, ("traj_p1_g1",))
    self.assertTrue(res[1].is_final_batch)


class CreateBatchAssemblerTest(absltest.TestCase):

  def test_create_sequence_packed_assembler_with_mesh_dims(self):
    assembler = batch_assembly.create_batch_assembler(
        num_generations=2,
        mini_batch_size=2,
        train_micro_batch_size=1,
        batch_config=batch_assembly.BatchConfig(
            pad_id=42,
            max_prompt_length=256,
            max_response_length=256,
            max_seq_token_per_tpu=1024,
            max_segments_per_packed_row=8,
            trainer_fsdp=2,
            trainer_dp=4,
        ),
    )
    self.assertIsInstance(
        assembler, batch_assembly.SequencePackedBatchAssembler
    )
    self.assertEqual(assembler.batch_size, 8)
    self.assertEqual(assembler.max_packed_len, 1024)
    self.assertEqual(assembler.pad_id, 42)
    self.assertEqual(assembler.max_segments_per_packed_row, 8)
    self.assertEqual(assembler.num_generations, 2)
    self.assertEqual(assembler.mini_batch_size, 2)

  def test_create_sequence_packed_assembler_defaults_pack_size(self):
    assembler = batch_assembly.create_batch_assembler(
        num_generations=4,
        mini_batch_size=2,
        train_micro_batch_size=3,
        batch_config=batch_assembly.BatchConfig(
            max_seq_token_per_tpu=512,
        ),
    )
    self.assertIsInstance(
        assembler, batch_assembly.SequencePackedBatchAssembler
    )
    self.assertEqual(assembler.batch_size, 3)
    self.assertEqual(assembler.max_packed_len, 512)

  def test_create_sequence_packed_assembler_validates_budget(self):
    with self.assertRaises(ValueError):
      batch_assembly.create_batch_assembler(
          num_generations=2,
          mini_batch_size=2,
          train_micro_batch_size=1,
          batch_config=batch_assembly.BatchConfig(
              max_prompt_length=512,
              max_response_length=512,
              max_seq_token_per_tpu=256,  # 256 < 512 + 512
          ),
      )

  def test_create_padded_batch_assembler(self):
    assembler = batch_assembly.create_batch_assembler(
        num_generations=2,
        mini_batch_size=2,
        train_micro_batch_size=4,
        batch_config=batch_assembly.BatchConfig(
            max_prompt_length=128,
            max_response_length=256,
            pad_id=7,
        ),
    )
    self.assertIsInstance(assembler, batch_assembly.PaddedBatchAssembler)
    self.assertEqual(assembler.batch_size, 4)
    self.assertEqual(assembler.max_prompt_length, 128)
    self.assertEqual(assembler.max_response_length, 256)
    self.assertEqual(assembler.pad_id, 7)

  def test_create_padded_batch_assembler_with_config_response_length(self):
    assembler = batch_assembly.create_batch_assembler(
        num_generations=2,
        mini_batch_size=2,
        train_micro_batch_size=4,
        batch_config=batch_assembly.BatchConfig(
            max_prompt_length=128,
            max_response_length=512,
            pad_id=7,
        ),
    )
    self.assertIsInstance(assembler, batch_assembly.PaddedBatchAssembler)
    self.assertEqual(assembler.max_prompt_length, 128)
    self.assertEqual(assembler.max_response_length, 512)

  def test_create_sequence_packed_assembler_validates_budget_from_config(self):
    with self.assertRaises(ValueError):
      batch_assembly.create_batch_assembler(
          num_generations=2,
          mini_batch_size=2,
          train_micro_batch_size=1,
          batch_config=batch_assembly.BatchConfig(
              max_prompt_length=512,
              max_response_length=512,
              max_seq_token_per_tpu=256,  # 256 < 512 + 512
          ),
      )

  def test_create_default_batch_assembler(self):
    assembler = batch_assembly.create_batch_assembler(
        num_generations=2,
        mini_batch_size=1,
        train_micro_batch_size=2,
        batch_config=batch_assembly.BatchConfig(),
    )
    self.assertIsInstance(
        assembler, batch_assembly.SequencePackedBatchAssembler
    )
    self.assertEqual(assembler.batch_size, 2)
    self.assertEqual(assembler.max_packed_len, 8192)
    self.assertEqual(assembler.segment_alignment_boundary, 1)

  def test_create_sequence_packed_assembler_forwards_segment_alignment(self):
    for max_seq_token_per_tpu in (1024, None):
      assembler = batch_assembly.create_batch_assembler(
          num_generations=2,
          mini_batch_size=1,
          train_micro_batch_size=2,
          batch_config=batch_assembly.BatchConfig(
              max_seq_token_per_tpu=max_seq_token_per_tpu,
              segment_alignment_boundary=64,
          ),
      )
      self.assertIsInstance(
          assembler, batch_assembly.SequencePackedBatchAssembler
      )
      self.assertEqual(assembler.segment_alignment_boundary, 64)


class SequencePackedRoutingTest(absltest.TestCase):
  """Packed payloads must replay rollout routing on the packed token layout."""

  def _assembler(self, **kwargs):
    defaults = dict(
        batch_size=1,
        num_generations=2,
        mini_batch_size=1,
        max_packed_len=8,
        pad_id=0,
    )
    defaults.update(kwargs)
    return batch_assembly.SequencePackedBatchAssembler(**defaults)

  def _payload(self, prompt_len, completion_len, *, routed_fill=None):
    payload = _make_payload(prompt_len, completion_len)
    if routed_fill is None:
      return payload
    return dataclasses.replace(
        payload,
        routed_experts=_routing(prompt_len + completion_len, routed_fill),
    )

  def test_to_pack_item_carries_sequence_aligned_routing(self):
    item = batch_assembly.to_pack_item(self._payload(1, 2, routed_fill=3))
    self.assertEqual(item.routed_experts.dtype, np.int16)
    self.assertEqual(
        item.routed_experts.shape, (3, _ROUTING_LAYERS, _ROUTING_TOP_K)
    )

  def test_to_pack_item_rejects_completion_only_routing(self):
    payload = dataclasses.replace(
        _make_payload(2, 2), routed_experts=_routing(2, 3)
    )
    with self.assertRaisesRegex(ValueError, "routed_experts"):
      batch_assembly.to_pack_item(payload)

  def test_packed_payload_carries_routing_on_token_positions(self):
    payloads = [
        self._payload(1, 2, routed_fill=3),  # 3 tokens -> [0:3]
        self._payload(1, 2, routed_fill=5),  # 3 tokens -> [3:6]
    ]
    [batch] = self._assembler().feed(payloads)
    routed = batch.payload.routed_experts
    self.assertIsNotNone(routed, "packed assembler dropped the routing")
    self.assertEqual(routed.dtype, np.int16)
    self.assertEqual(routed.shape, (1, 8, _ROUTING_LAYERS, _ROUTING_TOP_K))
    np.testing.assert_array_equal(
        batch.payload.segment_ids[0], [1, 1, 1, 2, 2, 2, 0, 0]
    )
    np.testing.assert_array_equal(routed[0, 0:3], 3)
    np.testing.assert_array_equal(routed[0, 3:6], 5)
    np.testing.assert_array_equal(routed[0, 6:], _UNSET)

  def test_to_rl_trainer_payload_reuses_chunk_routing_buffer(self):
    items = [
        batch_assembly.to_pack_item(self._payload(1, 2, routed_fill=3)),
        batch_assembly.to_pack_item(self._payload(1, 2, routed_fill=5)),
    ]
    chunk = packing.pack_chunk(
        [[items[0]], [items[1]]], budget=4, pad_id=0, carried=()
    )
    payload = batch_assembly.to_rl_trainer_payload(chunk, max_segments=2)
    self.assertIs(payload.routed_experts, chunk.routed_experts)

  def test_no_routing_leaves_payload_without_routed_experts(self):
    [batch] = self._assembler().feed(
        [self._payload(1, 2), self._payload(1, 2)]
    )
    self.assertIsNone(batch.payload.routed_experts)

  def test_routing_less_trajectory_is_unset_and_warned(self):
    payloads = [
        self._payload(1, 2, routed_fill=3),
        self._payload(1, 2),
    ]
    with self.assertLogs(level="WARNING") as logs:
      [batch] = self._assembler().feed(payloads)
    self.assertIn("1 of 2 packed trajectories carry no", logs.output[0])
    routed = batch.payload.routed_experts
    np.testing.assert_array_equal(routed[0, 0:3], 3)
    np.testing.assert_array_equal(routed[0, 3:], _UNSET)

  def test_dummy_rows_carry_unset_routing(self):
    [batch] = self._assembler(batch_size=2).feed(
        [self._payload(1, 2, routed_fill=3), self._payload(1, 2, routed_fill=4)]
    )
    routed = batch.payload.routed_experts
    self.assertEqual(routed.shape[0], 2)
    np.testing.assert_array_equal(routed[1], _UNSET)

  def test_segment_alignment_pads_gaps_and_routing(self):
    payloads = [
        self._payload(10, 20, routed_fill=3),  # 30 tokens -> [0:30]
        self._payload(5, 15, routed_fill=5),  # 20 tokens -> [64:84]
    ]
    [batch] = self._assembler(
        max_packed_len=128, pad_id=42, segment_alignment_boundary=64
    ).feed(payloads)
    payload = batch.payload
    seg_ids = payload.segment_ids[0]
    np.testing.assert_array_equal(seg_ids[:30], 1)
    np.testing.assert_array_equal(seg_ids[30:64], 0)
    np.testing.assert_array_equal(seg_ids[64:84], 2)
    np.testing.assert_array_equal(seg_ids[84:], 0)
    np.testing.assert_array_equal(payload.completion_mask[0, 30:64], 0)
    np.testing.assert_array_equal(payload.completion_ids[0, 30:64], 42)
    np.testing.assert_array_equal(payload.routed_experts[0, :30], 3)
    np.testing.assert_array_equal(payload.routed_experts[0, 30:64], _UNSET)
    np.testing.assert_array_equal(payload.routed_experts[0, 64:84], 5)

  def test_invalid_segment_alignment_boundary_raises(self):
    with self.assertRaisesRegex(
        ValueError, "segment_alignment_boundary must be positive"
    ):
      self._assembler(segment_alignment_boundary=0)

  def test_router_replay_coverage_logs_for_top_k_1_2_and_4(self):
    for top_k in (1, 2, 4):
      routed = np.zeros((1, 4, 2, top_k), dtype=np.int16)
      routed[0, :3] = np.arange(top_k, dtype=np.int16)
      routed[0, 3:] = _UNSET
      # Token 1 has duplicate expert ID when top_k >= 2.
      if top_k >= 2:
        routed[0, 1, 0, 1] = routed[0, 1, 0, 0]
      seg_ids = np.array([[1, 1, 1, 0]], dtype=np.int32)
      payload = datatypes.RLTrainerPayload(
          prompt_ids=np.zeros((1, 0), dtype=np.int32),
          prompt_mask=np.zeros((1, 0), dtype=np.float32),
          completion_ids=np.ones((1, 4), dtype=np.int32),
          completion_mask=np.ones((1, 4), dtype=np.float32),
          advantages=np.zeros((1, 4), dtype=np.float32),
          segment_ids=seg_ids,
          routed_experts=routed,
      )
      with self.assertLogs(level="INFO") as logs:
        batch_assembly._log_router_replay_coverage(
            payload, batch_id="test_b0", num_segments=1
        )
      expected = (
          "3/3 real tokens forced" if top_k == 1 else "2/3 real tokens forced"
      )
      self.assertTrue(
          any(expected in msg for msg in logs.output),
          f"Expected {expected!r} in {logs.output}",
      )

  def _assert_batches_identical(
      self,
      cpp_batches: list[batch_assembly.AssembledBatch],
      py_batches: list[batch_assembly.AssembledBatch],
  ) -> None:
    self.assertEqual(len(cpp_batches), len(py_batches))
    for cb, pb in zip(cpp_batches, py_batches):
      self.assertEqual(cb.is_final_batch, pb.is_final_batch)
      self.assertEqual(cb.trajectory_ids, pb.trajectory_ids)
      self.assertEqual(
          cb.padding_stats.row_capacity, pb.padding_stats.row_capacity
      )
      np.testing.assert_array_equal(
          cb.padding_stats.row_valid_tokens, pb.padding_stats.row_valid_tokens
      )
      np.testing.assert_array_equal(
          cb.padding_stats.row_num_sequences, pb.padding_stats.row_num_sequences
      )
      cp, pp = cb.payload, pb.payload
      for name in (
          "prompt_ids",
          "prompt_mask",
          "completion_ids",
          "completion_mask",
          "advantages",
          "segment_ids",
          "segment_positions",
          "ref_per_token_logps",
          "old_per_token_logps",
          "returns",
          "old_values",
          "sampler_is_weights",
          "routed_experts",
      ):
        ca, pa = getattr(cp, name), getattr(pp, name)
        if pa is None:
          self.assertIsNone(ca, f"Field {name} expected None")
        else:
          self.assertIsNotNone(ca, f"Field {name} unexpectedly None")
          self.assertEqual(ca.dtype, pa.dtype, f"Dtype mismatch on {name}")
          np.testing.assert_array_equal(ca, pa, err_msg=f"Mismatch on {name}")

  def test_cpp_and_python_parity_for_to_pack_item_and_assemblers(self):
    self.assertIsNotNone(packing._packing_ext)

    rng = np.random.default_rng(77)
    payloads = []
    for i in range(18):
      p_len = int(rng.integers(3, 28))
      c_len = int(rng.integers(4, 36))
      full_len = p_len + c_len
      routed_len = full_len - (i % 2)
      routed_dtype = np.int32 if i % 3 == 0 else np.int16
      payloads.append(
          datatypes.RLTrainerPayload(
              prompt_ids=rng.integers(1, 500, size=p_len, dtype=np.int32),
              prompt_mask=(
                  np.ones(p_len, dtype=np.float32) if i % 2 == 0 else None
              ),
              completion_ids=rng.integers(1, 500, size=c_len, dtype=np.int32),
              completion_mask=(
                  rng.integers(0, 2, size=full_len).astype(np.float32)
                  if i % 3 == 0
                  else np.ones(c_len, dtype=np.float32)
              ),
              advantages=(
                  np.array([float(i) * 0.25], dtype=np.float32)
                  if i % 4 == 0
                  else rng.standard_normal(full_len).astype(np.float32)
              ),
              old_per_token_logps=rng.standard_normal(c_len).astype(np.float32),
              returns=rng.standard_normal(full_len).astype(np.float32),
              routed_experts=rng.integers(
                  0,
                  16,
                  size=(routed_len, _ROUTING_LAYERS, _ROUTING_TOP_K),
                  dtype=routed_dtype,
              ),
              metadata={"traj_id": f"traj_{i}"},
          )
      )

    orig_ext = packing._packing_ext
    # 1. Verify `to_pack_item` bit-for-bit parity.
    cpp_items = [batch_assembly.to_pack_item(p) for p in payloads]
    try:
      packing._packing_ext = None
      py_items = [batch_assembly.to_pack_item(p) for p in payloads]
    finally:
      packing._packing_ext = orig_ext
    for ci, pi in zip(cpp_items, py_items):
      np.testing.assert_array_equal(ci.prompt_ids, pi.prompt_ids)
      np.testing.assert_array_equal(ci.completion_ids, pi.completion_ids)
      np.testing.assert_array_equal(ci.completion_mask, pi.completion_mask)
      np.testing.assert_array_equal(ci.advantages, pi.advantages)
      self.assertEqual(set(ci.per_token.keys()), set(pi.per_token.keys()))
      for k in pi.per_token:
        np.testing.assert_array_equal(ci.per_token[k], pi.per_token[k])
      np.testing.assert_array_equal(ci.routed_experts, pi.routed_experts)

    # 2. Verify `SequencePackedBatchAssembler` bit-for-bit parity.
    for boundary in (1, 64):
      def _run_seq_packed(bd: int) -> list[batch_assembly.AssembledBatch]:
        asm = batch_assembly.SequencePackedBatchAssembler(
            batch_size=2,
            num_generations=4,
            mini_batch_size=2,
            max_packed_len=128,
            pad_id=9,
            segment_alignment_boundary=bd,
        )
        out = []
        out.extend(asm.feed(payloads[:5]))
        out.extend(asm.feed(payloads[5:13]))
        out.extend(asm.feed(payloads[13:]))
        out.extend(asm.flush())
        return out

      cpp_seq = _run_seq_packed(boundary)
      try:
        packing._packing_ext = None
        py_seq = _run_seq_packed(boundary)
      finally:
        packing._packing_ext = orig_ext
      self._assert_batches_identical(cpp_seq, py_seq)

    # 3. Verify `PaddedBatchAssembler` bit-for-bit parity.
    def _run_padded() -> list[batch_assembly.AssembledBatch]:
      asm = batch_assembly.PaddedBatchAssembler(
          batch_size=4,
          max_prompt_length=16,
          max_response_length=20,
          pad_id=7,
          num_generations=4,
          mini_batch_size=2,
      )
      out = []
      out.extend(asm.feed(payloads[:6]))
      out.extend(asm.feed(payloads[6:14]))
      out.extend(asm.feed(payloads[14:]))
      out.extend(asm.flush())
      return out

    cpp_padded = _run_padded()
    try:
      packing._packing_ext = None
      py_padded = _run_padded()
    finally:
      packing._packing_ext = orig_ext
    self._assert_batches_identical(cpp_padded, py_padded)


if __name__ == "__main__":
  absltest.main()
