#!/usr/bin/env python3
"""Microbatch-level compact-filter provenance under CANON_P32_LENGTH_SORT.

`ObservedTrainExample.all_compact_filtered` is a `pytree_node=False` scalar, so
the per-microbatch `jax.tree.map` row gather copies the *batch* answer verbatim.
Length sorting reorders the rows of an update by sequence length, and a
compact-filtered row (`MAX_CONTEXT_LIMIT_REACHED`) sits at the context ceiling
by construction, so once a step filters more such rows than a microbatch holds,
one microbatch ends up wholly filtered while the batch does not.  Observed for
real on `canon-p58-128s-zsoptt-full-timbd11` step 12: 14 of 128 rows were
compact-filtered (11 MAX_CONTEXT_LIMIT_REACHED + 3 MODEL_TIMEOUT), every row of
one microbatch carried an empty action mask, the TPU correctly returned
grad_norm=0 for it, and `alignment.check_batch` still raised
`AlignmentGateError: ['N_action=0']` on a row set that legitimately carried no
learning signal.

These tests pin the microbatch-level re-derivation and, just as importantly,
the three places where it must stay inert or fail closed.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import types
from unittest import mock
import unittest

import jax
import numpy as np

from tunix.rl import alignment
from tunix.rl.agentic import agentic_rl_learner


def _signed_env(arm: str) -> dict[str, str]:
  """The signed P58 training signature that admits a no-signal transaction."""
  return {
      "CANON_P34_DEEPSWE": "1",
      "CANON_P34_RUN_STAGE": "full",
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_P58_EXPECTED_UPDATES": "1000",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1" if arm == "native" else "0",
      "CANON_ALIGNMENT_GATE": "1",
      "CANON_ALIGNMENT_GATE_ONLY": "0",
      "CANON_ALIGNMENT_UPDATE_CANARY": "0",
      "CANON_ALIGNMENT_TRAIN": "1",
      "CANON_ALIGNMENT_AUDIT_EVERY": "10",
      "CANON_P32_LENGTH_SORT": "1",
      alignment.PRE_GATE_ENV: "1",
  }


ROWS = 128
MICRO = 8
SEQ = 4
# bd11 step 12: 11 MAX_CONTEXT_LIMIT_REACHED + 3 MODEL_TIMEOUT out of 128 rows.
FILTERED_ROWS = 14
# Step 192 = update 12 * 16 microbatches + microbatch 0.
STEP = 12 * 16


def _batch_sidecar(*, filtered, all_compact_filtered: bool):
  """A 128-row host sidecar whose filtered rows carry an empty action mask."""
  logps = np.zeros((ROWS, SEQ), dtype=np.float32)
  action_mask = np.ones((ROWS, SEQ), dtype=np.bool_)
  action_mask[np.asarray(filtered, dtype=np.int64)] = False
  return alignment.ObservedTrainExample(
      train_example=types.SimpleNamespace(),
      s_decode=logps,
      s_prefill=logps.copy(),
      t_old=logps.copy(),
      action_mask=action_mask,
      completion_valid_mask=action_mask.copy(),
      prompt_mask=np.zeros((ROWS, SEQ), dtype=np.bool_),
      tokens=np.zeros((ROWS, SEQ), dtype=np.int32),
      policy_version=np.zeros((ROWS,), dtype=np.int32),
      sampling_values=np.tile(
          np.asarray([1.0, 0.0, 1.0], dtype=np.float32), (ROWS, 1)
      ),
      all_compact_filtered=all_compact_filtered,
      # Step 12 is not a multiple of CANON_ALIGNMENT_AUDIT_EVERY=10, so the
      # bit-for-bit boundaries are not evaluated on this row set.
      audited=False,
  )


def _slice_rows(sidecar, rows):
  """Reproduces the learner's per-microbatch host row gather exactly."""
  row_index = np.asarray(rows, dtype=np.int32)
  return jax.tree.map(
      lambda value: (
          value[row_index]
          if hasattr(value, "shape") and value.shape and value.shape[0] == ROWS
          else value
      ),
      sidecar,
  )


class P58MicrobatchCompactFilterTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Length sort is what lets the filtered rows of an update collect into one
    # microbatch; the count only has to exceed the microbatch size.
    self.filtered = tuple(range(FILTERED_ROWS))
    self.sidecar = _batch_sidecar(
        filtered=self.filtered, all_compact_filtered=False
    )
    self.host_action_tokens = int(np.asarray(self.sidecar.action_mask).sum())
    self.all_filtered_rows = self.filtered[:MICRO]
    self.mixed_rows = tuple(range(MICRO, 2 * MICRO))

  def test_premise_batch_has_signal_while_the_microbatch_has_none(self):
    self.assertGreater(FILTERED_ROWS, MICRO)
    self.assertGreater(self.host_action_tokens, 0)
    pair = _slice_rows(self.sidecar, self.all_filtered_rows)
    self.assertEqual(int(np.asarray(pair.action_mask).sum()), 0)
    # The inherited batch answer is what made check_batch call this a red.
    self.assertFalse(pair.all_compact_filtered)

  def test_wholly_filtered_microbatch_is_marked_no_signal(self):
    pair = _slice_rows(self.sidecar, self.all_filtered_rows)
    for arm in ("zero", "native"):
      with self.subTest(arm=arm):
        rewritten, flipped = (
            agentic_rl_learner._p58_microbatch_all_compact_filtered(
                pair,
                host_action_tokens=self.host_action_tokens,
                values=_signed_env(arm),
            )
        )
        self.assertTrue(flipped)
        self.assertTrue(rewritten.all_compact_filtered)
        self.assertEqual(int(np.asarray(rewritten.action_mask).sum()), 0)
    # The batch keeps its own answer, so the optimizer still commits the step.
    self.assertFalse(self.sidecar.all_compact_filtered)

  def test_microbatch_with_action_tokens_is_untouched(self):
    pair = _slice_rows(self.sidecar, self.mixed_rows)
    self.assertGreater(int(np.asarray(pair.action_mask).sum()), 0)
    rewritten, flipped = (
        agentic_rl_learner._p58_microbatch_all_compact_filtered(
            pair,
            host_action_tokens=self.host_action_tokens,
            values=_signed_env("zero"),
        )
    )
    self.assertFalse(flipped)
    self.assertFalse(rewritten.all_compact_filtered)

  def test_empty_batch_keeps_the_batch_level_transaction(self):
    """A genuinely signal-free batch must not be rewritten microbatch-wise."""
    sidecar = _batch_sidecar(
        filtered=tuple(range(ROWS)), all_compact_filtered=False
    )
    pair = _slice_rows(sidecar, tuple(range(MICRO)))
    rewritten, flipped = (
        agentic_rl_learner._p58_microbatch_all_compact_filtered(
            pair, host_action_tokens=0, values=_signed_env("zero")
        )
    )
    self.assertFalse(flipped)
    self.assertFalse(rewritten.all_compact_filtered)

  def test_unsigned_lane_is_inert(self):
    pair = _slice_rows(self.sidecar, self.all_filtered_rows)
    for key in (
        "CANON_P34_DEEPSWE",
        "CANON_P58_DEEPSWE_TIM",
        "CANON_P58_TIM_ADMITTED",
        "CANON_ALIGNMENT_TRAIN",
        "CANON_P58_TIM_ARM",
    ):
      values = _signed_env("zero")
      values.pop(key)
      with self.subTest(missing=key):
        _, flipped = agentic_rl_learner._p58_microbatch_all_compact_filtered(
            pair, host_action_tokens=self.host_action_tokens, values=values
        )
        self.assertFalse(flipped)

  def test_already_marked_sidecar_is_not_rewritten_twice(self):
    sidecar = _batch_sidecar(
        filtered=tuple(range(ROWS)), all_compact_filtered=True
    )
    pair = _slice_rows(sidecar, tuple(range(MICRO)))
    rewritten, flipped = (
        agentic_rl_learner._p58_microbatch_all_compact_filtered(
            pair, host_action_tokens=0, values=_signed_env("zero")
        )
    )
    self.assertFalse(flipped)
    self.assertTrue(rewritten.all_compact_filtered)

  def test_marked_microbatch_passes_check_batch_with_zero_gradient(self):
    pair = _slice_rows(self.sidecar, self.all_filtered_rows)
    with tempfile.TemporaryDirectory() as root:
      values = {
          **_signed_env("native"),
          alignment.REPORT_ENV: str(Path(root) / "post.jsonl"),
      }
      with mock.patch.dict(os.environ, values, clear=True):
        rewritten, flipped = (
            agentic_rl_learner._p58_microbatch_all_compact_filtered(
                pair,
                host_action_tokens=self.host_action_tokens,
                values=os.environ,
            )
        )
        record = alignment.check_batch(
            rewritten,
            t_current=np.zeros((MICRO, SEQ), dtype=np.float32),
            gradient_norm=np.asarray(0.0, dtype=np.float32),
            optimizer_skipped=np.asarray(0, dtype=np.int32),
            step=STEP,
            fail_closed=True,
        )
    self.assertTrue(flipped)
    self.assertEqual(record["verdict"], "PASS")
    self.assertEqual(record["N_action"], 0)
    self.assertTrue(record["no_signal_admitted"])
    self.assertEqual(record["blocking_reds"], [])
    # CANON_ALIGNMENT_AUDIT_EVERY=10 skips the bit-for-bit boundaries here.
    self.assertFalse(record["audited"])
    self.assertEqual(record["boundaries"], {})

  def test_marked_microbatch_still_fails_closed_on_nonzero_gradient(self):
    """The zero-gradient contract is the reason this rewrite is safe."""
    pair = _slice_rows(self.sidecar, self.all_filtered_rows)
    with tempfile.TemporaryDirectory() as root:
      values = {
          **_signed_env("native"),
          alignment.REPORT_ENV: str(Path(root) / "post.jsonl"),
      }
      with mock.patch.dict(os.environ, values, clear=True):
        rewritten, _ = (
            agentic_rl_learner._p58_microbatch_all_compact_filtered(
                pair,
                host_action_tokens=self.host_action_tokens,
                values=os.environ,
            )
        )
        with self.assertRaisesRegex(
            alignment.AlignmentGateError, "compact_filtered_gradient_nonzero"
        ):
          alignment.check_batch(
              rewritten,
              t_current=np.zeros((MICRO, SEQ), dtype=np.float32),
              gradient_norm=np.asarray(0.25, dtype=np.float32),
              optimizer_skipped=np.asarray(0, dtype=np.int32),
              step=STEP,
              fail_closed=True,
          )

  def test_unrewritten_microbatch_reproduces_the_bd11_step12_failure(self):
    """Without the rewrite the identical slice is a blocking N_action=0 red."""
    pair = _slice_rows(self.sidecar, self.all_filtered_rows)
    with tempfile.TemporaryDirectory() as root:
      values = {
          **_signed_env("native"),
          alignment.REPORT_ENV: str(Path(root) / "post.jsonl"),
      }
      with mock.patch.dict(os.environ, values, clear=True):
        with self.assertRaisesRegex(
            alignment.AlignmentGateError, "N_action=0"
        ):
          alignment.check_batch(
              pair,
              t_current=np.zeros((MICRO, SEQ), dtype=np.float32),
              gradient_norm=np.asarray(0.0, dtype=np.float32),
              optimizer_skipped=np.asarray(0, dtype=np.int32),
              step=STEP,
              fail_closed=True,
          )


if __name__ == "__main__":
  unittest.main()
