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


from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import batch_assembly
from tunix.rl import common as rl_common


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


class WithRefPerTokenLogpsTest(absltest.TestCase):

  def _make_train_example(self, b=2, p=3, c=4):
    return rl_common.TrainExample(
        prompt_ids=np.ones((b, p), dtype=np.int32),
        prompt_mask=np.ones((b, p), dtype=np.float32),
        completion_ids=np.ones((b, c), dtype=np.int32),
        completion_mask=np.ones((b, c), dtype=np.float32),
        advantages=np.ones((b, c), dtype=np.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
    )

  def test_success_with_ndarray(self):
    batch = self._make_train_example(b=2, p=3, c=4)
    ref_logps = np.full((2, 4), -0.5, dtype=np.float32)
    updated = batch_assembly.with_ref_per_token_logps(batch, ref_logps)

    self.assertIsNotNone(updated.ref_per_token_logps)
    self.assertEqual(updated.ref_per_token_logps.shape, (2, 4))
    np.testing.assert_allclose(updated.ref_per_token_logps, ref_logps)
    np.testing.assert_array_equal(updated.prompt_ids, batch.prompt_ids)
    np.testing.assert_array_equal(updated.completion_ids, batch.completion_ids)

  def test_success_with_logprobs_response(self):
    batch = self._make_train_example(b=2, p=3, c=4)
    resp = datatypes.LogprobsResponse(
        per_token_logps=np.full((2, 4), -0.8, dtype=np.float32)
    )
    updated = batch_assembly.with_ref_per_token_logps(batch, resp)

    self.assertIsNotNone(updated.ref_per_token_logps)
    self.assertEqual(updated.ref_per_token_logps.shape, (2, 4))
    np.testing.assert_allclose(
        updated.ref_per_token_logps, resp.per_token_logps
    )

  def test_error_in_logprobs_response_raises_runtime_error(self):
    batch = self._make_train_example(b=2, p=3, c=4)
    resp = datatypes.LogprobsResponse(
        per_token_logps=None,
        error=datatypes.ErrorInfo(
            error_type="InferenceError", message="inference worker failed"
        ),
    )
    with self.assertRaisesRegex(RuntimeError, "inference worker failed"):
      batch_assembly.with_ref_per_token_logps(batch, resp)

  def test_rejects_non_train_example(self):
    payload = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2], dtype=np.int32),
        token_mask=np.array([1, 1], dtype=np.float32),
        loss_mask=np.array([0, 1], dtype=np.float32),
        advantages=np.array([1.0, 1.0], dtype=np.float32),
    )
    with self.assertRaisesRegex(TypeError, "expects a padded TrainExample"):
      batch_assembly.with_ref_per_token_logps(payload, np.zeros((2, 2)))

  def test_mismatched_shape_raises_value_error(self):
    batch = self._make_train_example(b=2, p=3, c=4)
    bad_shape_logps = np.zeros((2, 3), dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        "Reference logps shape must match padded completion_ids shape",
    ):
      batch_assembly.with_ref_per_token_logps(batch, bad_shape_logps)


class SequencePackedBatchAssemblerTest(absltest.TestCase):

  def test_empty_input_returns_empty_list(self):
    assembler = batch_assembly.SequencePackedBatchAssembler(max_packed_len=16)
    self.assertEmpty(assembler.pack([]))

  def test_sequence_packed_assembler_with_trainer_payload(self):
    payload1 = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        token_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        advantages=np.full(4, 1.5, dtype=np.float32),
    )
    payload2 = datatypes.RLTrainerPayload(
        token_ids=np.array([5, 6, 7, 8], dtype=np.int32),
        token_mask=np.array([0, 0, 0, 1], dtype=np.float32),
        loss_mask=np.array([0, 0, 0, 1], dtype=np.float32),
        action_mask=np.array([0, 0, 0, 1], dtype=np.float32),
        advantages=np.full(4, -0.5, dtype=np.float32),
    )

    assembler = batch_assembly.SequencePackedBatchAssembler(max_packed_len=16)
    payloads = assembler.pack([payload1, payload2])

    self.assertLen(payloads, 1)
    payload = payloads[0]
    self.assertEqual(payload.token_ids.shape, (1, 16))
    self.assertEqual(payload.loss_mask.shape, (1, 16))
    self.assertEqual(payload.segment_ids.shape, (1, 16))
    self.assertEqual(payload.segment_positions.shape, (1, 16))
    self.assertEqual(payload.advantages.shape, (1, 16))

    # Check segment boundaries
    seg_ids = payload.segment_ids[0]
    self.assertTrue(np.all(seg_ids[:4] == 1))
    self.assertTrue(np.all(seg_ids[4:8] == 2))
    self.assertTrue(np.all(seg_ids[8:] == 0))

    # Check segment positions
    seg_pos = payload.segment_positions[0]
    np.testing.assert_array_equal(seg_pos[:4], [0, 1, 2, 3])
    np.testing.assert_array_equal(seg_pos[4:8], [0, 1, 2, 3])

  def test_sequence_packed_assembler_all_optional_fields(self):
    payload1 = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        token_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        advantages=np.full(4, 1.5, dtype=np.float32),
        old_per_token_logps=np.full(4, -1.0, dtype=np.float32),
        ref_per_token_logps=np.full(4, -1.2, dtype=np.float32),
    )
    payload2 = datatypes.RLTrainerPayload(
        token_ids=np.array([5, 6, 7], dtype=np.int32),
        token_mask=np.array([0, 1, 1], dtype=np.float32),
        loss_mask=np.array([0, 1, 1], dtype=np.float32),
        action_mask=np.array([0, 1, 1], dtype=np.float32),
        advantages=np.full(3, 2.0, dtype=np.float32),
        old_per_token_logps=np.full(3, -0.5, dtype=np.float32),
        ref_per_token_logps=np.full(3, -0.7, dtype=np.float32),
    )

    assembler = batch_assembly.SequencePackedBatchAssembler(max_packed_len=12)
    payloads = assembler.pack([payload1, payload2])

    self.assertLen(payloads, 1)
    payload = payloads[0]
    self.assertEqual(payload.token_ids.shape, (1, 12))
    self.assertEqual(payload.loss_mask.shape, (1, 12))
    self.assertEqual(payload.action_mask.shape, (1, 12))
    self.assertEqual(payload.advantages.shape, (1, 12))
    self.assertEqual(payload.old_per_token_logps.shape, (1, 12))
    self.assertEqual(payload.ref_per_token_logps.shape, (1, 12))

    np.testing.assert_allclose(
        payload.old_per_token_logps[0],
        [-1.0, -1.0, -1.0, -1.0, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        payload.ref_per_token_logps[0],
        [-1.2, -1.2, -1.2, -1.2, -0.7, -0.7, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

  def test_sequence_packed_assembler_multiple_bins(self):
    payload1 = datatypes.RLTrainerPayload(
        token_ids=np.arange(10, dtype=np.int32),
        token_mask=np.ones(10, dtype=np.float32),
        loss_mask=np.ones(10, dtype=np.float32),
        advantages=np.ones(10, dtype=np.float32),
    )
    payload2 = datatypes.RLTrainerPayload(
        token_ids=np.arange(8, dtype=np.int32),
        token_mask=np.ones(8, dtype=np.float32),
        loss_mask=np.ones(8, dtype=np.float32),
        advantages=np.ones(8, dtype=np.float32),
    )

    assembler = batch_assembly.SequencePackedBatchAssembler(max_packed_len=12)
    payloads = assembler.pack([payload1, payload2])

    self.assertLen(payloads, 2)
    self.assertEqual(payloads[0].token_ids.shape, (1, 12))
    self.assertEqual(payloads[1].token_ids.shape, (1, 12))


class GRPOTrainExampleAssemblerTest(absltest.TestCase):

  def test_rejects_non_positive_batch_size(self):
    with self.assertRaisesRegex(ValueError, "batch size must be positive"):
      batch_assembly.GRPOTrainExampleAssembler(
          batch_size=0,
          max_prompt_length=4,
          max_response_length=5,
          pad_id=0,
      )

  def test_empty_input_returns_empty_list(self):
    assembler = batch_assembly.GRPOTrainExampleAssembler(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=5,
        pad_id=0,
    )
    self.assertEmpty(assembler.pack([]))

  def test_grpo_train_example_assembler_basic(self):
    payload = datatypes.RLTrainerPayload(
        token_ids=np.array([10, 11, 20, 21, 22], dtype=np.int32),
        token_mask=np.ones(5, dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1, 0], dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1, 0], dtype=np.float32),
        advantages=np.array([0, 0, 2, 2, 2], dtype=np.float32),
        prompt_ids=np.array([10, 11], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([20, 21, 22], dtype=np.int32),
        completion_mask=np.array([1, 1, 0], dtype=np.float32),
    )

    assembler = batch_assembly.GRPOTrainExampleAssembler(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=5,
        pad_id=0,
    )
    train_example = assembler.pack([payload])[0]

    self.assertEqual(train_example.prompt_ids.shape, (2, 4))
    self.assertEqual(train_example.completion_ids.shape, (2, 5))
    np.testing.assert_array_equal(
        train_example.prompt_ids[0], np.array([0, 0, 10, 11])
    )
    np.testing.assert_array_equal(
        train_example.completion_ids[0], np.array([20, 21, 22, 0, 0])
    )
    np.testing.assert_array_equal(
        train_example.completion_mask[0], np.array([1, 1, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        train_example.advantages[0], np.array([2, 2, 2, 0, 0])
    )

  def test_grpo_assembler_optional_fields_propagation(self):
    payload = datatypes.RLTrainerPayload(
        token_ids=np.array([10, 11, 20, 21, 22], dtype=np.int32),
        token_mask=np.ones(5, dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1, 1], dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1, 1], dtype=np.float32),
        advantages=np.array([2, 2, 2], dtype=np.float32),
        prompt_ids=np.array([10, 11], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([20, 21, 22], dtype=np.int32),
        completion_mask=np.ones(3, dtype=np.float32),
        ref_per_token_logps=np.array([-0.3, -0.4, -0.5], dtype=np.float32),
        old_per_token_logps=np.array([-0.1, -0.2, -0.3], dtype=np.float32),
    )

    assembler = batch_assembly.GRPOTrainExampleAssembler(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=5,
        pad_id=0,
    )
    train_example = assembler.pack([payload])[0]

    self.assertIsNotNone(train_example.ref_per_token_logps)
    self.assertIsNotNone(train_example.old_per_token_logps)

    self.assertEqual(train_example.ref_per_token_logps.shape, (2, 5))
    self.assertEqual(train_example.old_per_token_logps.shape, (2, 5))

    np.testing.assert_allclose(
        train_example.ref_per_token_logps[0], [-0.3, -0.4, -0.5, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        train_example.old_per_token_logps[0], [-0.1, -0.2, -0.3, 0.0, 0.0]
    )

  def test_grpo_assembler_chunks_multiple_microbatches(self):
    payload = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 3], dtype=np.int32),
        token_mask=np.ones(3, dtype=np.float32),
        loss_mask=np.array([0, 1, 1], dtype=np.float32),
        advantages=np.array([1.0, 1.0], dtype=np.float32),
        prompt_ids=np.array([1], dtype=np.int32),
        completion_ids=np.array([2, 3], dtype=np.int32),
    )

    assembler = batch_assembly.GRPOTrainExampleAssembler(
        batch_size=2,
        max_prompt_length=3,
        max_response_length=4,
        pad_id=0,
    )
    train_examples = assembler.pack([payload, payload, payload])

    self.assertLen(train_examples, 2)
    self.assertEqual(train_examples[0].prompt_ids.shape, (2, 3))
    self.assertEqual(train_examples[1].prompt_ids.shape, (2, 3))


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
    action_mask=None,
):
  """Builds an unbatched payload shaped like `AlgorithmAdapter` output."""
  prompt = np.arange(1, prompt_len + 1, dtype=np.int32)
  completion = np.arange(101, 101 + completion_len, dtype=np.int32)
  token_ids = np.concatenate([prompt, completion])
  token_mask = np.ones_like(token_ids, dtype=np.float32)
  completion_action_mask = (
      action_mask
      if action_mask is not None
      else np.ones(completion_len, dtype=np.float32)
  )
  seq_loss_mask = np.concatenate(
      [np.zeros(prompt_len, dtype=np.float32), completion_action_mask]
  )
  seq_adv = (
      np.full(len(token_ids), float(advantage), dtype=np.float32)
      if np.ndim(advantage) == 0
      else np.asarray(advantage, dtype=np.float32)
  )
  seq_returns = (
      np.full(len(token_ids), float(returns), dtype=np.float32)
      if returns is not None and np.ndim(returns) == 0
      else (
          np.asarray(returns, dtype=np.float32)
          if returns is not None
          else None
      )
  )
  seq_old_values = (
      np.full(len(token_ids), float(old_values), dtype=np.float32)
      if old_values is not None and np.ndim(old_values) == 0
      else (
          np.asarray(old_values, dtype=np.float32)
          if old_values is not None
          else None
      )
  )
  seq_sampler_is = (
      np.full(len(token_ids), float(sampler_is_weights), dtype=np.float32)
      if sampler_is_weights is not None and np.ndim(sampler_is_weights) == 0
      else (
          np.asarray(sampler_is_weights, dtype=np.float32)
          if sampler_is_weights is not None
          else None
      )
  )
  return datatypes.RLTrainerPayload(
      token_ids=token_ids,
      token_mask=token_mask,
      loss_mask=seq_loss_mask,
      action_mask=seq_loss_mask,
      advantages=seq_adv,
      prompt_ids=prompt,
      prompt_mask=np.ones(prompt_len, dtype=np.float32),
      completion_ids=completion,
      completion_mask=completion_action_mask,
      ref_per_token_logps=ref_logps,
      old_per_token_logps=old_logps,
      returns=seq_returns,
      old_values=seq_old_values,
      sampler_is_weights=seq_sampler_is,
  )


class PaddedBatchAssemblerTest(absltest.TestCase):

  def _assembler(self, **kwargs):
    defaults = dict(batch_size=2, max_seq_len=9, pad_id=0)
    defaults.update(kwargs)
    return batch_assembly.PaddedBatchAssembler(**defaults)

  def test_rejects_non_positive_dimensions(self):
    for bad in (
        dict(batch_size=0),
        dict(max_seq_len=0),
        dict(max_seq_len=-1),
    ):
      with self.assertRaises(ValueError):
        batch_assembly.PaddedBatchAssembler(**bad)

  def test_max_seq_len_attribute(self):
    assembler = batch_assembly.PaddedBatchAssembler(max_seq_len=384)
    self.assertEqual(assembler.max_seq_len, 384)

  def test_empty_input_returns_empty_list(self):
    self.assertEmpty(self._assembler().pack([]))

  def test_row_layout_is_right_padded_into_max_seq_len(self):
    payload = self._assembler().pack([_make_payload(2, 3)])[0]

    self.assertEqual(payload.token_ids.shape, (2, 9))
    self.assertEqual(payload.token_mask.shape, (2, 9))
    self.assertEqual(payload.loss_mask.shape, (2, 9))
    self.assertEqual(payload.action_mask.shape, (2, 9))
    self.assertEqual(payload.advantages.shape, (2, 9))

    np.testing.assert_array_equal(
        payload.token_ids[0], [1, 2, 101, 102, 103, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        payload.token_mask[0], [1, 1, 1, 1, 1, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        payload.loss_mask[0], [0, 0, 1, 1, 1, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        payload.action_mask[0], [0, 0, 1, 1, 1, 0, 0, 0, 0]
    )
    np.testing.assert_allclose(
        payload.advantages[0], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    )

  def test_action_mask_excludes_tool_observation_tokens(self):
    # Middle completion token is a tool observation: attended, but not trained.
    item = _make_payload(
        2, 3, action_mask=np.array([1, 0, 1], dtype=np.float32)
    )
    payload = self._assembler().pack([item])[0]

    np.testing.assert_array_equal(
        payload.loss_mask[0], [0, 0, 1, 0, 1, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        payload.action_mask[0], [0, 0, 1, 0, 1, 0, 0, 0, 0]
    )

  def test_ignores_prompt_and_completion_fields(self):
    item = _make_payload(2, 3)
    payload = self._assembler().pack([item])[0]

    self.assertIsNone(payload.prompt_ids)
    self.assertIsNone(payload.prompt_mask)
    self.assertIsNone(payload.completion_ids)
    self.assertIsNone(payload.completion_mask)

  def test_partially_present_optional_fields_stay_row_aligned(self):
    items = [
        _make_payload(2, 3),
        _make_payload(2, 3, old_logps=np.full(5, -0.7, dtype=np.float32)),
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

    self.assertEqual(payload.returns.shape, (2, 9))
    np.testing.assert_allclose(payload.returns[0], [4, 4, 4, 4, 4, 0, 0, 0, 0])

  def test_scalar_advantage_broadcasts_over_valid_tokens(self):
    payload = self._assembler().pack([_make_payload(2, 3, advantage=2.5)])[0]

    self.assertEqual(payload.advantages.shape, (2, 9))
    np.testing.assert_allclose(
        payload.advantages[0], [2.5, 2.5, 2.5, 2.5, 2.5, 0, 0, 0, 0]
    )

  def test_truncates_overlong_sequences(self):
    payload = self._assembler(batch_size=2, max_seq_len=6).pack(
        [_make_payload(4, 4)]
    )[0]

    self.assertEqual(payload.token_ids.shape, (2, 6))
    self.assertEqual(payload.loss_mask.shape, (2, 6))
    np.testing.assert_array_equal(payload.token_ids[0], [1, 2, 3, 4, 101, 102])
    np.testing.assert_array_equal(payload.token_mask[0], [1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(payload.loss_mask[0], [0, 0, 0, 0, 1, 1])

  def test_trailing_rows_are_masked_out(self):
    payload = self._assembler(batch_size=3, max_seq_len=9).pack(
        [_make_payload(2, 3)]
    )[0]

    self.assertEqual(payload.loss_mask.shape, (3, 9))
    for row in (1, 2):
      np.testing.assert_array_equal(payload.token_ids[row], np.zeros(9))
      np.testing.assert_array_equal(payload.token_mask[row], np.zeros(9))
      np.testing.assert_array_equal(payload.loss_mask[row], np.zeros(9))
      np.testing.assert_array_equal(payload.action_mask[row], np.zeros(9))
      np.testing.assert_array_equal(payload.advantages[row], np.zeros(9))

  def test_chunks_into_multiple_microbatches(self):
    payloads = self._assembler(batch_size=2, max_seq_len=9).pack(
        [_make_payload(2, 3) for _ in range(5)]
    )

    self.assertLen(payloads, 3)
    for p in payloads:
      self.assertEqual(p.token_ids.shape, (2, 9))
      self.assertEqual(p.token_mask.shape, (2, 9))

  def test_all_optional_fields_are_propagated(self):
    item = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 101, 102, 103], dtype=np.int32),
        token_mask=np.ones(5, dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1, 1], dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1, 1], dtype=np.float32),
        advantages=np.full(5, 1.5, dtype=np.float32),
        ref_per_token_logps=np.full(5, -0.1, dtype=np.float32),
        old_per_token_logps=np.full(5, -0.2, dtype=np.float32),
        returns=np.full(5, 2.0, dtype=np.float32),
        old_values=np.full(5, 0.5, dtype=np.float32),
        sampler_is_weights=np.full(5, 1.0, dtype=np.float32),
    )
    payload = self._assembler(batch_size=2, max_seq_len=9).pack([item])[0]

    self.assertEqual(payload.old_values.shape, (2, 9))
    self.assertEqual(payload.sampler_is_weights.shape, (2, 9))
    np.testing.assert_allclose(
        payload.old_values[0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        payload.sampler_is_weights[0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )

  def test_underlength_field_is_padded(self):
    item = _make_payload(
        prompt_len=2,
        completion_len=4,
        advantage=np.array([1.5, 2.5], dtype=np.float32),
        ref_logps=np.array([-0.5, -0.2], dtype=np.float32),
    )
    payload = self._assembler(batch_size=2, max_seq_len=9).pack([item])[0]

    self.assertEqual(payload.advantages.shape, (2, 9))
    self.assertEqual(payload.ref_per_token_logps.shape, (2, 9))
    np.testing.assert_allclose(
        payload.advantages[0], [1.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        payload.ref_per_token_logps[0],
        [-0.5, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

  def test_none_advantages_defaults_to_zeros(self):
    item = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 101, 102, 103], dtype=np.int32),
        token_mask=np.ones(5, dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1, 1], dtype=np.float32),
        advantages=None,
    )
    payload = self._assembler(batch_size=2, max_seq_len=9).pack([item])[0]

    self.assertEqual(payload.advantages.shape, (2, 9))
    np.testing.assert_allclose(payload.advantages[0], np.zeros(9))


if __name__ == "__main__":
  absltest.main()