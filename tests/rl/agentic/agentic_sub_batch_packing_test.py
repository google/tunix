# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequence-packing wiring for sub-batch checkpointing.

Learner-side packed-mode machinery: tag-based identity through the FFD packer
(identity rides each item -- the packer sorts and carries items over, so
consumption order cannot recover it) and per-segment masking of
already-trained segments, which replaces the epoch-uniformity constraint
standard mode relies on. The packer hooks themselves are covered in
rl_utils_test; the loop integration by the preemption drill.
"""

import collections

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.rl.agentic import agentic_rl_learner

from tunix.rl.agentic import agentic_sub_batch_resume_test


_make_learner = agentic_sub_batch_resume_test._make_learner


def _packed_chunk(seg_rows, *, is_update_step=False):
  """A [n_rows, width] packed TrainExample from explicit segment_ids rows.

  completion_mask is 1 wherever a segment exists and advantages equal the
  segment id per token, so masking effects are directly readable.
  """
  seg = np.asarray(seg_rows, dtype=np.int32)
  mask = (seg > 0).astype(np.float32)
  chunk = agentic_rl_learner.TrainExample(
      prompt_ids=jnp.zeros((seg.shape[0], 0), dtype=jnp.int32),
      prompt_mask=jnp.zeros((seg.shape[0], 0), dtype=jnp.float32),
      completion_ids=jnp.asarray(seg),
      completion_mask=jnp.asarray(mask),
      advantages=jnp.asarray(seg.astype(np.float32)),
      ref_per_token_logps=None,
      old_per_token_logps=None,
      segment_ids=jnp.asarray(seg),
      segment_positions=jnp.zeros_like(jnp.asarray(seg)),
  )
  return chunk.replace(is_update_step=jnp.array([is_update_step]))


class LearnerTagPlumbingTest(absltest.TestCase):

  def test_claim_item_tag_pops_fifo_and_underflow_raises(self):
    learner = _make_learner(mgr=None, k=1)
    learner._sb_item_fifo = collections.deque([(7, 0), (7, 1)])
    self.assertEqual(learner._sb_claim_item_tag(), (7, 0))
    self.assertEqual(learner._sb_claim_item_tag(), (7, 1))
    with self.assertRaisesRegex(RuntimeError, "FIFO underflow"):
      learner._sb_claim_item_tag()

  def test_packed_stream_stages_identities_per_batch(self):
    # process_in_consumer=False: one identity list per queue item, so a
    # micro-batch of n items dequeues n lists, extended in row order.
    learner = _make_learner(mgr=None, k=1)
    learner._sb_item_fifo = collections.deque()
    staged = [[(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)]]
    learner._sb_dequeue_identities = lambda n: [
        staged.pop(0) for _ in range(n)
    ]
    out = list(learner._sb_packed_stream(iter([["mb1a", "mb1b"], ["mb2"]])))
    self.assertEqual(out, [["mb1a", "mb1b"], ["mb2"]])
    self.assertEqual(
        list(learner._sb_item_fifo),
        [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
    )
    self.assertEqual(staged, [])

  def test_packed_stream_consumer_path_dequeues_one_per_batch(self):
    # process_in_consumer=True: the converted micro-batch is one queue item
    # whose single identity list covers every row.
    learner = _make_learner(mgr=None, k=1, process_in_consumer=True)
    learner._sb_item_fifo = collections.deque()
    staged = [[(1, 0), (1, 1), (2, 0), (2, 1)]]
    learner._sb_dequeue_identities = lambda n: [
        staged.pop(0) for _ in range(n)
    ]
    out = list(learner._sb_packed_stream(iter([["converted"]])))
    self.assertEqual(out, [["converted"]])
    self.assertEqual(
        list(learner._sb_item_fifo), [(1, 0), (1, 1), (2, 0), (2, 1)]
    )
    self.assertEqual(staged, [])


class SegmentMaskingTest(absltest.TestCase):
  """Per-segment masking: what makes an epoch-mixed pack harmless."""

  def _learner_with_counts(self, counts):
    learner = _make_learner(mgr=None, k=1)
    learner._sb_counts = dict(counts)
    return learner

  def test_trained_segments_masked_untrained_kept(self):
    # Row 0: segments 1,2 -> A (trained), B (untrained). Row 1: C (trained).
    chunk = _packed_chunk([[1, 1, 2, 2, 0, 0], [1, 1, 1, 0, 0, 0]])
    bins = [[("A", 0), ("B", 0)], [("C", 0)]]
    learner = self._learner_with_counts({("A", 0): 1, ("C", 0): 1})
    masked, trained_ids, has_untrained = learner._sb_mask_trained_segments(
        chunk, bins, epoch=1
    )
    self.assertTrue(has_untrained)
    self.assertEqual(trained_ids, [("B", 0)])
    mask = np.asarray(masked.completion_mask)
    adv = np.asarray(masked.advantages)
    seg = np.asarray(chunk.segment_ids)
    # A's span (row 0, seg 1) and C's span (row 1, seg 1) zeroed.
    self.assertTrue((mask[0][seg[0] == 1] == 0).all())
    self.assertTrue((adv[0][seg[0] == 1] == 0).all())
    self.assertTrue((mask[1][seg[1] == 1] == 0).all())
    # B's span (row 0, seg 2) untouched.
    self.assertTrue((mask[0][seg[0] == 2] == 1).all())
    self.assertTrue((adv[0][seg[0] == 2] == 2.0).all())
    # Everything else rides through unchanged.
    np.testing.assert_array_equal(np.asarray(masked.segment_ids), seg)
    np.testing.assert_array_equal(
        np.asarray(masked.completion_ids), np.asarray(chunk.completion_ids)
    )

  def test_masking_preserves_update_flag(self):
    # _sb_chunk_will_apply reads the flag off the (masked) chunk it trains.
    chunk = _packed_chunk([[1, 1, 2, 2]], is_update_step=True)
    learner = self._learner_with_counts({("A", 0): 1})
    masked, _, _ = learner._sb_mask_trained_segments(
        chunk, [[("A", 0), ("B", 0)]], epoch=1
    )
    self.assertTrue(learner._sb_chunk_will_apply(masked))

  def test_nothing_trained_returns_chunk_unchanged(self):
    chunk = _packed_chunk([[1, 1, 2, 2]])
    bins = [[("A", 0), ("B", 0)]]
    learner = self._learner_with_counts({})
    masked, trained_ids, has_untrained = learner._sb_mask_trained_segments(
        chunk, bins, epoch=1
    )
    self.assertIs(masked, chunk)  # fast path: no rebuild
    self.assertTrue(has_untrained)
    self.assertCountEqual(trained_ids, [("A", 0), ("B", 0)])

  def test_everything_trained_reports_no_untrained(self):
    chunk = _packed_chunk([[1, 1, 2, 2]])
    bins = [[("A", 0), ("B", 0)]]
    learner = self._learner_with_counts({("A", 0): 1, ("B", 0): 1})
    masked, trained_ids, has_untrained = learner._sb_mask_trained_segments(
        chunk, bins, epoch=1
    )
    self.assertFalse(has_untrained)
    self.assertEqual(trained_ids, [])
    self.assertTrue((np.asarray(masked.completion_mask) == 0).all())

  def test_epoch_argument_gates_replay_masking(self):
    # A segment at count 1 is masked for epoch 1 but NOT for epoch 2: replay
    # sweeps re-train each segment exactly once per epoch.
    chunk = _packed_chunk([[1, 1]])
    bins = [[("A", 0)]]
    learner = self._learner_with_counts({("A", 0): 1})
    _, _, has_untrained_e1 = learner._sb_mask_trained_segments(
        chunk, bins, epoch=1
    )
    self.assertFalse(has_untrained_e1)
    masked_e2, trained_e2, has_untrained_e2 = (
        learner._sb_mask_trained_segments(chunk, bins, epoch=2)
    )
    self.assertTrue(has_untrained_e2)
    self.assertEqual(trained_e2, [("A", 0)])
    self.assertIs(masked_e2, chunk)

  def test_dummy_rows_have_empty_bins(self):
    # pack_size padding produces all-zero dummy rows; their bin is empty and
    # masking passes them through untouched.
    chunk = _packed_chunk([[1, 1, 0, 0], [0, 0, 0, 0]])
    bins = [[("A", 0)], []]
    learner = self._learner_with_counts({("A", 0): 1})
    masked, trained_ids, has_untrained = learner._sb_mask_trained_segments(
        chunk, bins, epoch=1
    )
    self.assertFalse(has_untrained)
    self.assertEqual(trained_ids, [])
    self.assertTrue((np.asarray(masked.completion_mask)[1] == 0).all())

  def test_pack_has_untrained_is_the_skip_predicate(self):
    # The pre-logps skip decision must agree with what masking would train.
    learner = self._learner_with_counts({("A", 0): 2, ("B", 0): 1})
    bins = [[("A", 0)], [("B", 0)]]
    self.assertFalse(learner._sb_pack_has_untrained(bins, 1))
    self.assertTrue(learner._sb_pack_has_untrained(bins, 2))
    self.assertTrue(learner._sb_pack_has_untrained(bins, 3))
    self.assertFalse(learner._sb_pack_has_untrained([[], []], 1))  # dummies
    for epoch in (1, 2, 3):
      _, _, has_untrained = learner._sb_mask_trained_segments(
          _packed_chunk([[1, 1], [1, 1]]), bins, epoch=epoch
      )
      self.assertEqual(
          has_untrained, learner._sb_pack_has_untrained(bins, epoch)
      )


if __name__ == "__main__":
  absltest.main()
