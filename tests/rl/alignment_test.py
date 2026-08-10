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

"""Tests for the fail-closed zero-TIM alignment sidecar."""

import contextlib
import io
import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import alignment
from tunix.rl import canonical_forward


@flax.struct.dataclass(frozen=True)
class _Example:
  completion_ids: object
  completion_mask: object
  advantages: object
  is_update_step: object


def _real_rescore():
  pass


_real_rescore.is_real_rescore = True


class _CanonicalAdapter:
  implementation_id = "test.tpu_inference.qwen3"
  is_engine_module = True
  supports_value_and_grad = True

  def compute_per_token_logps(self, **kwargs):
    raise AssertionError(f"unexpected compute in host-only test: {kwargs}")


class AlignmentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = mock.patch.dict(
        os.environ,
        {
            canonical_forward.ENV: "1",
            alignment.ALIGN_ENV: "1",
            alignment.GATE_ONLY_ENV: "1",
        },
        clear=False,
    )
    self.env.start()
    canonical_forward.register(_CanonicalAdapter())

  def tearDown(self):
    canonical_forward._clear_for_test()  # pylint: disable=protected-access
    self.env.stop()
    super().tearDown()

  def _wrapped(self, rows=2):
    ids = np.arange(rows * 3, dtype=np.int32).reshape(rows, 3)
    mask = np.ones_like(ids, dtype=np.bool_)
    values = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) / 8
    example = _Example(
        completion_ids=jnp.asarray(ids),
        completion_mask=jnp.asarray(mask),
        advantages=jnp.ones((rows,), dtype=jnp.float32),
        is_update_step=None,
    )
    return alignment.wrap_train_example(
        example,
        s_decode=values,
        s_prefill=values.copy(),
        t_old=values.copy(),
        action_mask=mask,
        tokens=ids,
        policy_version=np.zeros((rows,), dtype=np.int32),
        temperature=0.7,
        top_k=0,
        top_p=1.0,
        s_prefill_source=_real_rescore,
    )

  def test_exact_gate_passes_and_writes_report(self):
    wrapped = self._wrapped()
    with tempfile.TemporaryDirectory() as tmpdir:
      report = os.path.join(tmpdir, "report.jsonl")
      with mock.patch.dict(
          os.environ,
          {
              alignment.ALIGN_ENV: "1",
              alignment.GATE_ONLY_ENV: "1",
              alignment.REPORT_ENV: report,
          },
          clear=False,
      ):
        result = alignment.check_batch(
            wrapped,
            t_current=wrapped.t_old.copy(),
            gradient_norm=np.asarray(2.0, np.float32),
            optimizer_skipped=np.asarray(1, np.int32),
            step=0,
        )
      self.assertEqual(result["verdict"], "PASS")
      self.assertTrue(result["exact"]["wr_all_exactly_1"])
      with open(report, encoding="utf-8") as report_file:
        self.assertEqual(json.loads(report_file.readline())["verdict"], "PASS")

  def test_pre_backward_gate_passes_and_writes_two_boundaries(self):
    wrapped = self._wrapped()
    with tempfile.TemporaryDirectory() as tmpdir:
      report = os.path.join(tmpdir, "pre.jsonl")
      with mock.patch.dict(
          os.environ,
          {
              alignment.PRE_GATE_ENV: "1",
              alignment.PRE_REPORT_ENV: report,
          },
          clear=False,
      ):
        result = alignment.check_pre_backward(wrapped, step=3)
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["step"], 3)
      self.assertEqual(
          set(result["boundaries"]),
          {"S_decode_vs_S_prefill", "S_prefill_vs_T_old"},
      )
      for boundary in result["boundaries"].values():
        self.assertTrue(boundary["valid"])
        self.assertEqual(boundary["differing_bytes"], 0)
        self.assertEqual(boundary["differing_elements"], 0)
        self.assertEqual(boundary["total_elements"], 6)
        self.assertEqual(boundary["total_bytes"], 24)
        self.assertEqual(boundary["byte_fraction"], 0.0)
        self.assertEqual(boundary["element_fraction"], 0.0)
      self.assertEqual(
          result["masked_hashes"]["S_decode"],
          result["masked_hashes"]["S_prefill"],
      )
      with open(report, encoding="utf-8") as report_file:
        self.assertEqual(json.loads(report_file.readline()), result)

  def test_pre_backward_gate_prints_complete_json_and_evidence_sha(self):
    wrapped = self._wrapped()
    stdout = io.StringIO()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.PRE_GATE_ENV: "1",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ), contextlib.redirect_stdout(stdout):
      result = alignment.check_pre_backward(wrapped, step=7)
    lines = stdout.getvalue().splitlines()
    json_line = next(
        line for line in lines if line.startswith("[CANON_ALIGN_PRE_JSON] ")
    )
    printed = json.loads(json_line.split(" ", 1)[1])
    self.assertEqual(printed, result)
    evidence_line = next(
        line
        for line in lines
        if line.startswith("[CANON_ALIGN_PRE_EVIDENCE] ")
    )
    self.assertIn("sha256=", evidence_line)

  def test_pre_backward_gate_localizes_first_red_boundary(self):
    wrapped = self._wrapped()
    drift = wrapped.s_prefill.copy()
    drift[0, 0] = np.nextafter(drift[0, 0], np.float32(np.inf))
    wrapped = wrapped.replace(s_prefill=drift)
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.PRE_GATE_ENV: "1",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ):
      with self.assertRaisesRegex(
          alignment.AlignmentGateError,
          "S_decode_vs_S_prefill.*S_prefill_vs_T_old",
      ):
        alignment.check_pre_backward(wrapped, step=0)

  def test_pre_backward_gate_rejects_missing_admission(self):
    with mock.patch.dict(
        os.environ,
        {alignment.PRE_GATE_ENV: "0"},
        clear=False,
    ):
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, alignment.PRE_GATE_ENV
      ):
        alignment.check_pre_backward(self._wrapped(), step=0)

  def test_one_ulp_drift_fails_closed(self):
    wrapped = self._wrapped()
    drift = wrapped.t_old.copy()
    drift[0, 0] = np.nextafter(drift[0, 0], np.float32(np.inf))
    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch.dict(
          os.environ,
          {
              alignment.ALIGN_ENV: "1",
              alignment.GATE_ONLY_ENV: "1",
              alignment.REPORT_ENV: os.path.join(tmpdir, "report.jsonl"),
          },
          clear=False,
      ):
        with self.assertRaisesRegex(
            alignment.AlignmentGateError, "T_old_vs_T_current"
        ):
          alignment.check_batch(
              wrapped,
              t_current=drift,
              gradient_norm=np.asarray(2.0, np.float32),
              optimizer_skipped=np.asarray(1, np.int32),
              step=0,
          )

  def test_rejects_cached_alias_and_missing_skip_attestation(self):
    wrapped = self._wrapped()
    ids = np.asarray(wrapped.tokens)
    with self.assertRaisesRegex(alignment.AlignmentGateError, "does not declare"):
      alignment.wrap_train_example(
          wrapped.train_example,
          s_decode=wrapped.s_decode,
          s_prefill=wrapped.s_prefill,
          t_old=wrapped.t_old,
          action_mask=wrapped.action_mask,
          tokens=ids,
          policy_version=wrapped.policy_version,
          temperature=0.7,
          top_k=0,
          top_p=1.0,
          s_prefill_source=lambda: None,
      )
    with mock.patch.dict(
        os.environ,
        {alignment.GATE_ONLY_ENV: "1"},
        clear=False,
    ):
      with self.assertRaisesRegex(alignment.AlignmentGateError, "optimizer_skipped"):
        alignment.check_batch(
            wrapped,
            t_current=wrapped.t_old,
            gradient_norm=np.asarray(2.0, np.float32),
            optimizer_skipped=np.asarray(0, np.int32),
            step=0,
        )

  def test_update_canary_requires_optimizer_execution(self):
    wrapped = self._wrapped()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "1",
            alignment.REPORT_ENV: os.path.join(tmpdir, "report.jsonl"),
        },
        clear=False,
    ):
      result = alignment.check_batch(
          wrapped,
          t_current=wrapped.t_old,
          gradient_norm=np.asarray(2.0, np.float32),
          optimizer_skipped=np.asarray(0, np.int32),
          step=0,
      )
      self.assertEqual(result["execution_mode"], "update-canary")
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, "optimizer attestation mismatch"
      ):
        alignment.check_batch(
            wrapped,
            t_current=wrapped.t_old,
            gradient_norm=np.asarray(2.0, np.float32),
            optimizer_skipped=np.asarray(1, np.int32),
            step=0,
        )

  def test_train_mode_allows_real_zero_signal_but_not_numerical_drift(self):
    wrapped = self._wrapped()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "1",
            alignment.REPORT_ENV: os.path.join(tmpdir, "report.jsonl"),
        },
        clear=False,
    ):
      result = alignment.check_batch(
          wrapped,
          t_current=wrapped.t_old,
          gradient_norm=np.asarray(0.0, np.float32),
          optimizer_skipped=np.asarray(0, np.int32),
          step=0,
      )
      self.assertEqual(result["execution_mode"], "train")
      self.assertEqual(result["verdict"], "PASS")
      self.assertFalse(result["gradient"]["nonzero"])

      drift = wrapped.t_old.copy()
      drift[0, 0] = np.nextafter(drift[0, 0], np.float32(np.inf))
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, "T_old_vs_T_current"
      ):
        alignment.check_batch(
            wrapped,
            t_current=drift,
            gradient_norm=np.asarray(0.0, np.float32),
            optimizer_skipped=np.asarray(0, np.int32),
            step=1,
        )

  def test_execution_mode_rejects_multiple_modes(self):
    with mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "1",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "1",
        },
        clear=False,
    ):
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, "exactly one alignment execution mode"
      ):
        alignment.execution_mode()

  def test_merge_and_slice_keep_sampling_metadata_batch_aligned(self):
    a = self._wrapped(rows=1)
    b = self._wrapped(rows=2)
    merged = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), a, b)
    self.assertEqual(merged.sampling_values.shape, (3, 3))
    sliced = jax.tree.map(
        lambda x: x[1:] if hasattr(x, "shape") and x.shape else x, merged
    )
    self.assertEqual(sliced.sampling_values.shape, (2, 3))
    core, sidecar = alignment.unwrap_train_example(sliced)
    self.assertIs(sidecar, sliced)
    self.assertEqual(core.completion_ids.shape, (2, 3))

  def test_signed_zero_reports_bitwise_mismatch_without_index_error(self):
    result = alignment._masked_bitwise_difference(  # pylint: disable=protected-access
        np.asarray([[0.0]], np.float32),
        np.asarray([[-0.0]], np.float32),
        np.asarray([[True]]),
    )
    self.assertGreater(result["differing_bytes"], 0)
    self.assertEqual(result["differing_elements"], 1)
    self.assertEqual(result["total_elements"], 1)
    self.assertEqual(result["first_mismatch"]["masked_index"], 0)
    self.assertEqual(result["first_mismatch"]["coordinate"], [0, 0])
    self.assertEqual(result["first_mismatch"]["a_bits"], "0x00000000")
    self.assertEqual(result["first_mismatch"]["b_bits"], "0x80000000")
    self.assertEqual(result["first_mismatch"]["xor_bits"], "0x80000000")
    self.assertEqual(result["first_mismatch"]["ulp_distance"], 1)

  def test_one_ulp_is_one_differing_element(self):
    left = np.asarray([[1.0, 2.0]], np.float32)
    right = left.copy()
    right[0, 1] = np.nextafter(right[0, 1], np.float32(np.inf))
    result = alignment._masked_bitwise_difference(  # pylint: disable=protected-access
        left, right, np.asarray([[True, True]])
    )
    self.assertEqual(result["differing_elements"], 1)
    self.assertEqual(result["total_elements"], 2)
    self.assertGreater(result["differing_bytes"], 0)
    mismatch = result["first_mismatch"]
    self.assertEqual(mismatch["coordinate"], [0, 1])
    self.assertEqual(mismatch["sequence_row"], 0)
    self.assertEqual(mismatch["completion_position"], 1)
    self.assertEqual(mismatch["ulp_distance"], 1)

  def test_pre_backward_mismatch_includes_token_and_sparse_max_abs(self):
    wrapped = self._wrapped()
    drift = wrapped.s_prefill.copy()
    drift[1, 2] = drift[1, 2] + np.float32(0.1039)
    wrapped = wrapped.replace(s_prefill=drift)
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.PRE_GATE_ENV: "1",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ):
      result = alignment.check_pre_backward(
          wrapped, step=0, fail_closed=False
      )
    boundary = result["boundaries"]["S_decode_vs_S_prefill"]
    self.assertEqual(boundary["differing_elements"], 1)
    self.assertAlmostEqual(boundary["max_abs"], 0.1039, places=6)
    mismatch = boundary["first_mismatch"]
    self.assertEqual(mismatch["coordinate"], [1, 2])
    self.assertEqual(mismatch["token_id"], int(wrapped.tokens[1, 2]))
    self.assertGreater(mismatch["abs_delta"], 0.1)

  def test_mismatch_details_are_bounded_and_marked_truncated(self):
    left = np.zeros((1, 1025), dtype=np.float32)
    right = np.ones_like(left)
    result = alignment._masked_bitwise_difference(  # pylint: disable=protected-access
        left, right, np.ones_like(left, dtype=np.bool_)
    )
    self.assertEqual(result["differing_elements"], 1025)
    self.assertEqual(result["reported_mismatches"], 1024)
    self.assertLen(result["mismatches"], 1024)
    self.assertTrue(result["mismatches_truncated"])

  def test_pre_backward_invalid_shape_is_a_hard_failure(self):
    wrapped = self._wrapped().replace(
        s_prefill=np.zeros((1, 1), dtype=np.float32)
    )
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.PRE_GATE_ENV: "1",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ):
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, "S_decode_vs_S_prefill"
      ):
        alignment.check_pre_backward(wrapped, step=0)

  def test_pre_backward_nonfinite_mismatch_preserves_strict_json_evidence(self):
    wrapped = self._wrapped()
    drift = wrapped.s_prefill.copy()
    drift[0, 0] = np.nan
    wrapped = wrapped.replace(s_prefill=drift)
    stdout = io.StringIO()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.PRE_GATE_ENV: "1",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ), contextlib.redirect_stdout(stdout):
      result = alignment.check_pre_backward(
          wrapped, step=0, fail_closed=False
      )
    boundary = result["boundaries"]["S_decode_vs_S_prefill"]
    self.assertEqual(boundary["first_mismatch"]["b"], "nan")
    self.assertEqual(boundary["max_abs"], "nan")
    json_line = next(
        line
        for line in stdout.getvalue().splitlines()
        if line.startswith("[CANON_ALIGN_PRE_JSON] ")
    )
    self.assertEqual(json.loads(json_line.split(" ", 1)[1]), result)

  def test_full_hash_can_differ_while_masked_boundary_is_exact(self):
    wrapped = self._wrapped(rows=1)
    mask = wrapped.action_mask.copy()
    mask[0, 0] = False
    drift = wrapped.s_prefill.copy()
    drift[0, 0] = np.nextafter(drift[0, 0], np.float32(np.inf))
    wrapped = wrapped.replace(action_mask=mask, s_prefill=drift)
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.PRE_GATE_ENV: "1",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ):
      result = alignment.check_pre_backward(wrapped, step=0)
    self.assertEqual(result["verdict"], "PASS")
    self.assertNotEqual(
        result["hashes"]["S_decode"], result["hashes"]["S_prefill"]
    )
    self.assertEqual(
        result["masked_hashes"]["S_decode"],
        result["masked_hashes"]["S_prefill"],
    )


if __name__ == "__main__":
  absltest.main()
