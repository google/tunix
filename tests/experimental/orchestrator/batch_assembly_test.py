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
        prompt_ids=np.array([1, 2], dtype=np.int32),
        completion_ids=np.array([3, 4], dtype=np.int32),
    )
    payload2 = datatypes.RLTrainerPayload(
        token_ids=np.array([5, 6, 7, 8], dtype=np.int32),
        token_mask=np.array([0, 0, 0, 1], dtype=np.float32),
        loss_mask=np.array([0, 0, 0, 1], dtype=np.float32),
        action_mask=np.array([0, 0, 0, 1], dtype=np.float32),
        advantages=np.full(4, -0.5, dtype=np.float32),
        prompt_ids=np.array([5, 6], dtype=np.int32),
        completion_ids=np.array([7, 8], dtype=np.int32),
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

  def test_padded_batch_assembler(self):
    payload1 = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 3], dtype=np.int32),
        token_mask=np.array([0, 0, 1], dtype=np.float32),
        loss_mask=np.array([0, 0, 1], dtype=np.float32),
        action_mask=np.array([0, 0, 1], dtype=np.float32),
        advantages=np.full(3, 2.0, dtype=np.float32),
    )
    payload2 = datatypes.RLTrainerPayload(
        token_ids=np.array([4, 5, 6], dtype=np.int32),
        token_mask=np.array([0, 1, 1], dtype=np.float32),
        loss_mask=np.array([0, 1, 1], dtype=np.float32),
        action_mask=np.array([0, 1, 1], dtype=np.float32),
        advantages=np.full(3, 1.0, dtype=np.float32),
    )

    assembler = batch_assembly.PaddedBatchAssembler(batch_size=2, max_seq_len=8)
    payloads = assembler.pack([payload1, payload2])

    self.assertLen(payloads, 1)
    payload = payloads[0]
    self.assertEqual(payload.token_ids.shape, (2, 8))
    self.assertEqual(payload.loss_mask.shape, (2, 8))
    self.assertEqual(payload.action_mask.shape, (2, 8))
    self.assertEqual(payload.advantages.shape, (2, 8))

  def test_grpo_train_example_assembler_keeps_prompt_completion_fields(self):
    payload = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 3, 4, 5], dtype=np.int32),
        token_mask=np.array([1, 1, 1, 1, 1], dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1, 1], dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1, 1], dtype=np.float32),
        advantages=np.array([0, 0, 2, 2, 2], dtype=np.float32),
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.array([1, 1], dtype=np.float32),
        completion_ids=np.array([3, 4, 5], dtype=np.int32),
        completion_mask=np.array([1, 1, 1], dtype=np.float32),
    )

    assembler = batch_assembly.GRPOTrainExampleAssembler(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=5,
        pad_id=0,
    )

    batches = assembler.pack([payload])

    self.assertLen(batches, 1)
    batch = batches[0]
    np.testing.assert_array_equal(batch.prompt_ids[0], np.array([0, 0, 1, 2]))
    np.testing.assert_array_equal(batch.prompt_mask[0], np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(
        batch.completion_ids[0], np.array([3, 4, 5, 0, 0])
    )
    np.testing.assert_array_equal(
        batch.completion_mask[0], np.array([1, 1, 1, 0, 0])
    )
    np.testing.assert_array_equal(
        batch.advantages[0], np.array([2, 2, 2, 0, 0])
    )
    self.assertEqual(batch.prompt_ids.shape, (2, 4))
    self.assertEqual(batch.completion_ids.shape, (2, 5))


if __name__ == "__main__":
  absltest.main()
