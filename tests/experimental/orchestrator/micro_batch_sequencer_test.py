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

"""Tests for the micro-batch sequencer."""

import numpy as np

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import micro_batch_sequencer as seq


def _mb(loss_mask) -> datatypes.TrainerPayload:
  return datatypes.TrainerPayload(loss_mask=np.asarray(loss_mask, dtype=np.int32))


class MicroBatchSequencerTest(absltest.TestCase):

  def test_token_mean_denominator_counts_masked_tokens(self):
    payload = _mb([[1, 1, 1, 0], [1, 0, 0, 0]])
    self.assertEqual(seq.micro_batch_denominator(payload, "token-mean"), 4.0)

  def test_sequence_mean_denominator_counts_contributing_rows(self):
    payload = _mb([[1, 1, 0], [0, 0, 0], [1, 0, 0]])  # rows 0 and 2 contribute
    self.assertEqual(
        seq.micro_batch_denominator(payload, "sequence-mean-token-mean"), 2.0
    )

  def test_plan_indices_accum_id_and_token_mean_scales(self):
    micro_batches = [_mb([[1, 1, 1, 0]]), _mb([[1, 1, 0, 0]])]  # denoms 3, 2
    steps = seq.plan_micro_steps(micro_batches, accum_id="s0")

    self.assertEqual([s.micro_index for s in steps], [0, 1])
    self.assertEqual([s.accum_id for s in steps], ["s0", "s0"])
    self.assertAlmostEqual(steps[0].loss_scale, 0.6)
    self.assertAlmostEqual(steps[1].loss_scale, 0.4)
    self.assertAlmostEqual(sum(s.loss_scale for s in steps), 1.0)

  def test_equal_micro_batches_reproduce_mean(self):
    micro_batches = [_mb([[1, 1]]), _mb([[1, 1]]), _mb([[1, 1]])]
    steps = seq.plan_micro_steps(micro_batches, accum_id="s0")
    for step in steps:
      self.assertAlmostEqual(step.loss_scale, 1 / 3)

  def test_empty_raises(self):
    with self.assertRaises(ValueError):
      seq.plan_micro_steps([], accum_id="s0")


if __name__ == "__main__":
  absltest.main()
