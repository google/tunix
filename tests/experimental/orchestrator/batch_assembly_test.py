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