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
    payload1 = _make_payload(2, 2, advantage=1.5)
    payload2 = _make_payload(3, 1, advantage=-0.5)

    assembler = batch_assembly.SequencePackedBatchAssembler(max_packed_len=16)
    payloads = assembler.pack([payload1, payload2])

    self.assertLen(payloads, 1)
    payload = payloads[0]
    self.assertEqual(payload.token_ids.shape, (1, 16))
    self.assertEqual(payload.token_mask.shape, (1, 16))
    self.assertEqual(payload.loss_mask.shape, (1, 16))
    self.assertEqual(payload.segment_ids.shape, (1, 16))
    self.assertEqual(payload.segment_positions.shape, (1, 16))
    self.assertEqual(payload.advantages.shape, (1, 16))

    # Both sequences are 4 tokens long, so the decreasing sort is a no-op and
    # the stable order is preserved: payload1 is segment 1, payload2 segment 2.
    seg_ids = payload.segment_ids[0]
    self.assertTrue(np.all(seg_ids[:4] == 1))
    self.assertTrue(np.all(seg_ids[4:8] == 2))
    self.assertTrue(np.all(seg_ids[8:] == 0))
    # Positions restart at 0 within each segment.
    np.testing.assert_array_equal(
        payload.segment_positions[0][:8], [0, 1, 2, 3, 0, 1, 2, 3]
    )
    # token_mask marks real tokens, NOT segment ids, and is 0 in the slack.
    np.testing.assert_array_equal(
        payload.token_mask[0], [1] * 8 + [0] * 8
    )
    # payload1 (prompt 2, completion 2) then payload2 (prompt 3, completion 1):
    # loss applies to completion positions only, per segment.
    np.testing.assert_array_equal(
        payload.loss_mask[0], [0, 0, 1, 1, 0, 0, 0, 1] + [0] * 8
    )
    # Advantages sit on completion positions only, and each segment keeps its
    # own value rather than bleeding across the packing boundary.
    np.testing.assert_allclose(
        payload.advantages[0], [0, 0, 1.5, 1.5, 0, 0, 0, -0.5] + [0] * 8
    )

  def test_sequence_packed_assembler_rejects_partial_optional_fields(self):
    items = [
        _make_payload(2, 2),
        _make_payload(2, 2, ref_logps=np.full(2, -0.3, dtype=np.float32)),
    ]
    assembler = batch_assembly.SequencePackedBatchAssembler(max_packed_len=16)

    with self.assertRaisesRegex(ValueError, "ref_per_token_logps"):
      assembler.pack(items)

  def test_grpo_train_example_assembler(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([10, 11], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([20, 21, 22], dtype=np.int32),
        completion_mask=np.array([1, 1, 0], dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1, 0], dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1, 0], dtype=np.float32),
        advantages=np.array([0, 0, 2, 2, 2], dtype=np.float32),
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
  """Builds an unbatched payload shaped like `AlgorithmAdapter` output.

  Unbatched payloads carry no concatenated `token_ids`: prompt/completion is
  the canonical RL representation and the token stream is built by assembly.
  """
  prompt = np.arange(1, prompt_len + 1, dtype=np.int32)
  completion = np.arange(101, 101 + completion_len, dtype=np.int32)
  if action_mask is None:
    action_mask = np.ones(completion_len, dtype=np.float32)
  seq_loss_mask = np.concatenate(
      [np.zeros(prompt_len, dtype=np.float32), action_mask]
  )
  return datatypes.RLTrainerPayload(
      prompt_ids=prompt,
      prompt_mask=np.ones(prompt_len, dtype=np.float32),
      completion_ids=completion,
      completion_mask=action_mask,
      loss_mask=seq_loss_mask,
      action_mask=seq_loss_mask,
      advantages=advantage,
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

  def test_partially_present_optional_field_is_rejected(self):
    # Zero is not a neutral log-probability (exp(0) == 1), and dropping the
    # field would deactivate the KL term, so mixed presence must fail loudly.
    items = [
        _make_payload(2, 3),
        _make_payload(2, 3, old_logps=np.full(3, -0.7, dtype=np.float32)),
    ]

    with self.assertRaisesRegex(ValueError, "old_per_token_logps"):
      self._assembler().pack(items)

  def test_uniformly_present_optional_field_is_emitted(self):
    items = [
        _make_payload(2, 3, old_logps=np.full(3, -0.7, dtype=np.float32)),
        _make_payload(2, 3, old_logps=np.full(3, -0.4, dtype=np.float32)),
    ]
    payload = self._assembler().pack(items)[0]

    self.assertEqual(payload.old_per_token_logps.shape, (2, 5))
    np.testing.assert_allclose(
        payload.old_per_token_logps[0], [-0.7, -0.7, -0.7, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        payload.old_per_token_logps[1], [-0.4, -0.4, -0.4, 0.0, 0.0]
    )

  def test_optional_fields_absent_everywhere_stay_none(self):
    payload = self._assembler().pack([_make_payload(2, 3)])[0]

    self.assertIsNone(payload.ref_per_token_logps)
    self.assertIsNone(payload.old_per_token_logps)
    self.assertIsNone(payload.returns)
    self.assertIsNone(payload.old_values)
    self.assertIsNone(payload.sampler_is_weights)

  def test_unused_optional_fields_allocate_nothing(self):
    # A GRPO-shaped batch must not ship PPO-only buffers to the accelerator.
    items = [
        _make_payload(2, 3, ref_logps=np.full(3, -0.1, dtype=np.float32))
        for _ in range(2)
    ]
    payload = self._assembler().pack(items)[0]

    per_token_bytes = sum(
        getattr(payload, name).nbytes
        for name in (
            "ref_per_token_logps",
            "old_per_token_logps",
            "returns",
            "old_values",
            "sampler_is_weights",
        )
        if getattr(payload, name) is not None
    )
    # Only ref_per_token_logps: [2, 5] float32.
    self.assertEqual(per_token_bytes, 2 * 5 * 4)

  def test_segment_tensors_are_views_into_the_row_buffers(self):
    payload = self._assembler().pack([_make_payload(2, 3)])[0]

    self.assertIs(payload.prompt_ids.base, payload.token_ids)
    self.assertIs(payload.completion_ids.base, payload.token_ids)
    self.assertIs(payload.prompt_mask.base, payload.token_mask)
    self.assertIs(payload.completion_mask.base, payload.loss_mask)
    # Views must still agree with the buffers they alias.
    np.testing.assert_array_equal(
        payload.token_ids[:, :4], payload.prompt_ids
    )
    np.testing.assert_array_equal(
        payload.token_ids[:, 4:], payload.completion_ids
    )

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

  def test_prompt_completion_boundary_is_explicit_not_inferred(self):
    # A completion whose first token is a tool observation (action_mask[0] == 0)
    # used to be misfiled into the prompt when the boundary was inferred from
    # where loss_mask becomes non-zero. It is now read off the required fields.
    item = _make_payload(
        2, 3, action_mask=np.array([0, 1, 1], dtype=np.float32)
    )
    payload = self._assembler().pack([item])[0]

    np.testing.assert_array_equal(payload.prompt_ids[0], [0, 0, 1, 2])
    np.testing.assert_array_equal(
        payload.completion_ids[0], [101, 102, 103, 0, 0]
    )
    np.testing.assert_array_equal(payload.completion_mask[0], [0, 1, 1, 0, 0])


if __name__ == "__main__":
  absltest.main()
