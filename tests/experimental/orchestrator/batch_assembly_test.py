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

"""Unit tests for Universal BatchAssembler (SequencePacked & Padded)."""


from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import batch_assembly


class BatchAssemblyTest(absltest.TestCase):

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

  def test_grpo_train_example_assembler(self):
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


def _make_payload(
    prompt_len: int,
    completion_len: int,
    *,
    advantage=1.0,
    ref_logps=None,
    old_logps=None,
    returns=None,
    action_mask=None,
):
  """Builds an unbatched payload shaped like `AlgorithmAdapter` output."""
  prompt = np.arange(1, prompt_len + 1, dtype=np.int32)
  completion = np.arange(101, 101 + completion_len, dtype=np.int32)
  tokens = np.concatenate([prompt, completion])
  if action_mask is None:
    action_mask = np.ones(completion_len, dtype=np.float32)
  seq_loss_mask = np.concatenate(
      [np.zeros(prompt_len, dtype=np.float32), action_mask]
  )
  return datatypes.RLTrainerPayload(
      token_ids=tokens,
      token_mask=np.ones_like(tokens, dtype=np.float32),
      loss_mask=seq_loss_mask,
      action_mask=seq_loss_mask,
      advantages=advantage,
      prompt_ids=prompt,
      prompt_mask=np.ones(prompt_len, dtype=np.float32),
      completion_ids=completion,
      completion_mask=action_mask,
      ref_per_token_logps=ref_logps,
      old_per_token_logps=old_logps,
      returns=returns,
  )


class PaddedBatchAssemblerTest(absltest.TestCase):

  def _assembler(self, **kwargs):
    defaults = dict(
        batch_size=2, max_prompt_length=4, max_response_length=5, pad_id=0
    )
    defaults.update(kwargs)
    return batch_assembly.PaddedBatchAssembler(**defaults)

  def test_rejects_non_positive_dimensions(self):
    for bad in (
        dict(batch_size=0),
        dict(max_prompt_length=0),
        dict(max_response_length=-1),
    ):
      with self.assertRaises(ValueError):
        self._assembler(**bad)

  def test_empty_input_returns_empty_list(self):
    self.assertEmpty(self._assembler().pack([]))

  def test_row_layout_is_left_padded_prompt_and_right_padded_completion(self):
    payload = self._assembler().pack([_make_payload(2, 3)])[0]

    self.assertEqual(payload.token_ids.shape, (2, 9))
    self.assertEqual(payload.prompt_ids.shape, (2, 4))
    self.assertEqual(payload.completion_ids.shape, (2, 5))
    np.testing.assert_array_equal(payload.prompt_ids[0], [0, 0, 1, 2])
    np.testing.assert_array_equal(
        payload.completion_ids[0], [101, 102, 103, 0, 0]
    )
    np.testing.assert_array_equal(
        payload.token_ids[0], [0, 0, 1, 2, 101, 102, 103, 0, 0]
    )

  def test_token_mask_marks_real_tokens_and_loss_mask_skips_prompt(self):
    payload = self._assembler().pack([_make_payload(2, 3)])[0]

    # token_mask: real tokens (prompt included), not the loss mask.
    np.testing.assert_array_equal(
        payload.token_mask[0], [0, 0, 1, 1, 1, 1, 1, 0, 0]
    )
    # loss_mask: zero over the prompt, action mask over the completion.
    np.testing.assert_array_equal(
        payload.loss_mask[0], [0, 0, 0, 0, 1, 1, 1, 0, 0]
    )

  def test_action_mask_excludes_tool_observation_tokens(self):
    # Middle completion token is a tool observation: attended, but not trained.
    item = _make_payload(
        2, 3, action_mask=np.array([1, 0, 1], dtype=np.float32)
    )
    payload = self._assembler().pack([item])[0]

    np.testing.assert_array_equal(payload.completion_mask[0], [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(
        payload.loss_mask[0], [0, 0, 0, 0, 1, 0, 1, 0, 0]
    )
    # The observation token is still a real token for the attention mask.
    np.testing.assert_array_equal(
        payload.token_mask[0], [0, 0, 1, 1, 1, 1, 1, 0, 0]
    )

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
    payload = self._assembler().pack(items)[0]

    self.assertEqual(payload.old_per_token_logps.shape, (2, 5))
    np.testing.assert_allclose(payload.old_per_token_logps[0], np.zeros(5))
    np.testing.assert_allclose(
        payload.old_per_token_logps[1], [-0.7, -0.7, -0.7, 0.0, 0.0]
    )

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
    item = _make_payload(2, 3)
    item.advantages = np.array([0, 0, 2, 2, 2], dtype=np.float32)
    payload = self._assembler().pack([item])[0]

    np.testing.assert_allclose(payload.advantages[0], [2, 2, 2, 0, 0])

  def test_truncates_overlong_prompt_from_the_left(self):
    payload = self._assembler().pack([_make_payload(6, 8)])[0]

    self.assertEqual(payload.token_ids.shape, (2, 9))
    # Keeps the most recent prompt tokens.
    np.testing.assert_array_equal(payload.prompt_ids[0], [3, 4, 5, 6])
    # Keeps the earliest completion tokens.
    np.testing.assert_array_equal(
        payload.completion_ids[0], [101, 102, 103, 104, 105]
    )
    np.testing.assert_array_equal(payload.completion_mask[0], np.ones(5))

  def test_trailing_rows_are_masked_out(self):
    payload = self._assembler(batch_size=3).pack([_make_payload(2, 3)])[0]

    self.assertEqual(payload.token_ids.shape, (3, 9))
    self.assertEqual(payload.metadata["num_real_rows"], 1)
    for row in (1, 2):
      np.testing.assert_array_equal(payload.token_mask[row], np.zeros(9))
      np.testing.assert_array_equal(payload.loss_mask[row], np.zeros(9))
      np.testing.assert_array_equal(payload.advantages[row], np.zeros(5))

  def test_chunks_into_multiple_microbatches(self):
    payloads = self._assembler(batch_size=2).pack(
        [_make_payload(2, 3) for _ in range(5)]
    )

    self.assertLen(payloads, 3)
    self.assertEqual([p.metadata["num_real_rows"] for p in payloads], [2, 2, 1])
    for p in payloads:
      self.assertEqual(p.token_ids.shape, (2, 9))

  def test_infers_prompt_boundary_from_loss_mask_when_split_fields_absent(self):
    item = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 3], dtype=np.int32),
        token_mask=np.ones(3, dtype=np.float32),
        loss_mask=np.array([0, 0, 1], dtype=np.float32),
        action_mask=np.array([0, 0, 1], dtype=np.float32),
        advantages=np.full(3, 2.0, dtype=np.float32),
    )
    payload = self._assembler().pack([item])[0]

    np.testing.assert_array_equal(payload.prompt_ids[0], [0, 0, 1, 2])
    np.testing.assert_array_equal(payload.completion_ids[0], [3, 0, 0, 0, 0])
    np.testing.assert_allclose(payload.advantages[0], [2, 0, 0, 0, 0])


if __name__ == "__main__":
  absltest.main()
