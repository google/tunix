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
  prompt_ids: object | None = None


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
        prompt_ids=jnp.arange(rows * 2, dtype=jnp.int32).reshape(rows, 2),
    )
    return alignment.wrap_train_example(
        example,
        s_decode=values,
        s_prefill=values.copy(),
        t_old=values.copy(),
        action_mask=mask,
        completion_valid_mask=mask,
        prompt_mask=np.ones((rows, 2), dtype=np.bool_),
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

  def test_p38_precheck_only_stops_after_exact_record(self):
    record = {"verdict": "PASS", "step": 4, "N_action": 7}
    stdout = io.StringIO()
    with mock.patch.dict(
        os.environ,
        {alignment.PRECHECK_ONLY_ENV: "1"},
        clear=False,
    ), contextlib.redirect_stdout(stdout), self.assertRaisesRegex(
        alignment.PreAlignmentProbeComplete, "before backward"
    ):
      alignment.stop_after_exact_precheck(record)
    self.assertIn(
        "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD step=4 N_action=7",
        stdout.getvalue(),
    )

  def test_p38_diagnostic_precheck_stops_after_finite_ab_red(self):
    record = {
        "verdict": "FAIL",
        "step": 0,
        "N_action": 17,
        "boundaries": {
            "S_decode_vs_S_prefill": {
                "valid": True,
                "finite": True,
                "differing_bytes": 3,
            },
            "S_prefill_vs_T_old": {
                "valid": True,
                "finite": True,
                "differing_bytes": 0,
            },
        },
    }
    stdout = io.StringIO()
    with mock.patch.dict(
        os.environ,
        {alignment.PRECHECK_ONLY_ENV: "1"},
        clear=False,
    ), contextlib.redirect_stdout(stdout), self.assertRaisesRegex(
        alignment.PreAlignmentProbeComplete, "before backward"
    ):
      alignment.stop_after_diagnostic_precheck(record)
    self.assertIn("verdict=FAIL a_b_differing_bytes=3", stdout.getvalue())

  def test_p38_diagnostic_precheck_rejects_bc_red(self):
    record = {
        "verdict": "FAIL",
        "step": 0,
        "N_action": 17,
        "boundaries": {
            "S_decode_vs_S_prefill": {
                "valid": True,
                "finite": True,
                "differing_bytes": 3,
            },
            "S_prefill_vs_T_old": {
                "valid": True,
                "finite": True,
                "differing_bytes": 1,
            },
        },
    }
    with mock.patch.dict(
        os.environ,
        {alignment.PRECHECK_ONLY_ENV: "1"},
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "exact B/C"
    ):
      alignment.stop_after_diagnostic_precheck(record)

  def test_p38_precheck_only_is_default_off_and_rejects_bad_values(self):
    with mock.patch.dict(
        os.environ,
        {alignment.PRECHECK_ONLY_ENV: "0"},
        clear=False,
    ):
      alignment.stop_after_exact_precheck({"verdict": "PASS"})
    with mock.patch.dict(
        os.environ,
        {alignment.PRECHECK_ONLY_ENV: "yes"},
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, alignment.PRECHECK_ONLY_ENV
    ):
      alignment.stop_after_exact_precheck({"verdict": "PASS"})

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
          completion_valid_mask=wrapped.completion_valid_mask,
          prompt_mask=wrapped.prompt_mask,
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

  def test_gsm8k_full_reports_bounded_ab_drift_and_keeps_hard_boundaries(self):
    wrapped = self._wrapped(rows=100)
    decode = wrapped.s_decode.copy()
    decode[0, 1] = np.nextafter(decode[0, 1], np.float32(np.inf))
    wrapped = wrapped.replace(s_decode=decode)
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "1",
            alignment.PRE_GATE_ENV: "1",
            alignment.GSM8K_AB_REPORT_ONLY_ENV: "1",
            "CANON_P32_WORKLOAD": "gsm8k",
            "CANON_P33_RUN_STAGE": "full",
            "CANON_P33_NO_COMMIT": "0",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
            alignment.REPORT_ENV: os.path.join(tmpdir, "post.jsonl"),
        },
        clear=False,
    ):
      pre = alignment.check_pre_backward(wrapped, step=0)
      post = alignment.check_batch(
          wrapped,
          t_current=wrapped.t_old,
          gradient_norm=np.asarray(1.0, np.float32),
          optimizer_skipped=np.asarray(0, np.int32),
          step=0,
      )
    self.assertEqual(pre["verdict"], "PASS_WITH_REPORTED_DRIFT")
    self.assertEqual(pre["blocking_reds"], [])
    self.assertEqual(pre["reported_reds"], ["S_decode_vs_S_prefill"])
    self.assertEqual(post["verdict"], "PASS_WITH_REPORTED_DRIFT")
    self.assertEqual(post["blocking_reds"], [])
    self.assertIn("S_decode_vs_S_prefill", post["reported_reds"])
    self.assertTrue(post["exact"]["r_all_exactly_1"])
    self.assertFalse(post["exact"]["w_all_exactly_1"])
    self.assertFalse(post["exact"]["wr_all_exactly_1"])
    self.assertEqual(post["clip_hits"], 0)
    self.assertEqual(post["tis_hits"], 0)

  def test_gsm8k_ab_policy_rejects_wrong_scope_and_out_of_budget_drift(self):
    wrapped = self._wrapped()
    with mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "1",
            alignment.PRE_GATE_ENV: "1",
            alignment.GSM8K_AB_REPORT_ONLY_ENV: "1",
            "CANON_P32_WORKLOAD": "frozenlake",
            "CANON_P33_RUN_STAGE": "full",
            "CANON_P33_NO_COMMIT": "0",
        },
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "only for committed GSM8K"
    ):
      alignment.check_pre_backward(wrapped, step=0)

    decode = wrapped.s_decode.copy()
    decode[0, 1] += np.float32(1.0e-2)
    wrapped = wrapped.replace(s_decode=decode)
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "1",
            alignment.PRE_GATE_ENV: "1",
            alignment.GSM8K_AB_REPORT_ONLY_ENV: "1",
            "CANON_P32_WORKLOAD": "gsm8k",
            "CANON_P33_RUN_STAGE": "full",
            "CANON_P33_NO_COMMIT": "0",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "S_decode_vs_S_prefill"
    ):
      alignment.check_pre_backward(wrapped, step=0)

  def test_gsm8k_ab_policy_rejects_nonfinite_values(self):
    wrapped = self._wrapped()
    decode = wrapped.s_decode.copy()
    decode[0, 1] = np.nan
    wrapped = wrapped.replace(s_decode=decode)
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "1",
            alignment.PRE_GATE_ENV: "1",
            alignment.GSM8K_AB_REPORT_ONLY_ENV: "1",
            "CANON_P32_WORKLOAD": "gsm8k",
            "CANON_P33_RUN_STAGE": "full",
            "CANON_P33_NO_COMMIT": "0",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "S_decode_vs_S_prefill"
    ):
      alignment.check_pre_backward(wrapped, step=0)

  def test_gsm8k_full_warning_policy_never_blocks_finite_alignment_drift(self):
    wrapped = self._wrapped()
    wrapped = wrapped.replace(
        s_decode=wrapped.s_decode - np.float32(3.0),
        s_prefill=wrapped.s_prefill + np.float32(0.5),
        t_old=wrapped.t_old + np.float32(0.25),
    )
    t_current = wrapped.t_old - np.float32(0.75)
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "1",
            alignment.PRE_GATE_ENV: "1",
            alignment.GSM8K_AB_REPORT_ONLY_ENV: "0",
            alignment.GSM8K_ALIGNMENT_WARN_ONLY_ENV: "1",
            "CANON_P32_WORKLOAD": "gsm8k",
            "CANON_P33_RUN_STAGE": "full",
            "CANON_P33_NO_COMMIT": "0",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
            alignment.REPORT_ENV: os.path.join(tmpdir, "post.jsonl"),
        },
        clear=False,
    ):
      pre = alignment.check_pre_backward(wrapped, step=0)
      post = alignment.check_batch(
          wrapped,
          t_current=t_current,
          gradient_norm=np.asarray(1.0, np.float32),
          optimizer_skipped=np.asarray(0, np.int32),
          step=0,
      )
    self.assertEqual(pre["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
    self.assertEmpty(pre["blocking_reds"])
    self.assertSameElements(
        pre["warning_reds"],
        ("S_decode_vs_S_prefill", "S_prefill_vs_T_old"),
    )
    self.assertEqual(post["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
    self.assertEmpty(post["blocking_reds"])
    self.assertIn("T_old_vs_T_current", post["warning_reds"])
    self.assertIn("w_all_exactly_1", post["warning_reds"])
    self.assertIn("r_all_exactly_1", post["warning_reds"])
    self.assertIn("wr_all_exactly_1", post["warning_reds"])
    self.assertGreater(post["clip_hits"], 0)
    self.assertGreater(post["tis_hits"], 0)
    self.assertTrue(post["ratio_finite"])
    self.assertEqual(post["admission_policy"]["claim_level"], "convergence-only")

  def test_gsm8k_warning_policy_keeps_scope_and_nonfinite_fail_closed(self):
    wrapped = self._wrapped()
    common = {
        alignment.GATE_ONLY_ENV: "0",
        alignment.UPDATE_CANARY_ENV: "0",
        alignment.TRAIN_ENV: "1",
        alignment.PRE_GATE_ENV: "1",
        alignment.GSM8K_AB_REPORT_ONLY_ENV: "0",
        alignment.GSM8K_ALIGNMENT_WARN_ONLY_ENV: "1",
        "CANON_P33_RUN_STAGE": "full",
        "CANON_P33_NO_COMMIT": "0",
    }
    with mock.patch.dict(
        os.environ,
        {**common, "CANON_P32_WORKLOAD": "frozenlake"},
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "only for committed GSM8K"
    ):
      alignment.check_pre_backward(wrapped, step=0)

    nonfinite = wrapped.replace(s_decode=np.full_like(wrapped.s_decode, np.nan))
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            **common,
            "CANON_P32_WORKLOAD": "gsm8k",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "S_decode_vs_S_prefill"
    ):
      alignment.check_pre_backward(nonfinite, step=0)

  def test_frozenlake_full_warning_policy_is_scoped_and_finite_only(self):
    wrapped = self._wrapped().replace(
        s_decode=self._wrapped().s_decode - np.float32(0.3)
    )
    common = {
        alignment.GATE_ONLY_ENV: "0",
        alignment.UPDATE_CANARY_ENV: "0",
        alignment.TRAIN_ENV: "1",
        alignment.PRE_GATE_ENV: "1",
        alignment.GSM8K_AB_REPORT_ONLY_ENV: "0",
        alignment.GSM8K_ALIGNMENT_WARN_ONLY_ENV: "0",
        alignment.FROZENLAKE_ALIGNMENT_WARN_ONLY_ENV: "1",
        "CANON_P33_RUN_STAGE": "full",
        "CANON_P33_NO_COMMIT": "0",
    }
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            **common,
            "CANON_P32_WORKLOAD": "frozenlake",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ):
      record = alignment.check_pre_backward(wrapped, step=0)
    self.assertEqual(record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
    self.assertEmpty(record["blocking_reds"])
    self.assertEqual(
        record["admission_policy"]["id"],
        "frozenlake-full-alignment-warning-v1",
    )

    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            **common,
            "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ):
      p45_record = alignment.check_pre_backward(wrapped, step=0)
    self.assertEqual(
        p45_record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS"
    )
    self.assertEqual(
        p45_record["admission_policy"]["workload"], "frozenlake"
    )

    with mock.patch.dict(
        os.environ,
        {
            **common,
            "CANON_P32_WORKLOAD": "frozenlake",
            "CANON_P33_RUN_STAGE": "backward-no-commit",
            "CANON_P33_NO_COMMIT": "1",
        },
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "committed FrozenLake full training"
    ):
      alignment.gsm8k_ab_report_policy()

    with mock.patch.dict(
        os.environ,
        {
            **common,
            alignment.GSM8K_ALIGNMENT_WARN_ONLY_ENV: "1",
            "CANON_P32_WORKLOAD": "gsm8k",
        },
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "mutually exclusive"
    ):
      alignment.check_pre_backward(wrapped, step=0)

  def test_deepswe_full_warning_policy_continues_finite_bc_only(self):
    wrapped = self._wrapped().replace(
        s_decode=self._wrapped().s_decode - np.float32(0.5),
        t_old=self._wrapped().t_old + np.float32(0.25),
    )
    common = {
        alignment.GATE_ONLY_ENV: "0",
        alignment.UPDATE_CANARY_ENV: "0",
        alignment.TRAIN_ENV: "1",
        alignment.PRE_GATE_ENV: "1",
        alignment.GSM8K_AB_REPORT_ONLY_ENV: "0",
        alignment.GSM8K_ALIGNMENT_WARN_ONLY_ENV: "0",
        alignment.FROZENLAKE_ALIGNMENT_WARN_ONLY_ENV: "0",
        alignment.DEEPSWE_ALIGNMENT_WARN_ONLY_ENV: "1",
        "CANON_P34_DEEPSWE": "1",
        "CANON_P34_RUN_STAGE": "full",
        "CANON_P34_NO_COMMIT": "0",
        "CANON_P39_64CHIP_PILOT": "0",
        "CANON_P43_DEEPSWE_DEBUG": "0",
        "CANON_P44_DEEPSWE_PARITY": "0",
    }
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            **common,
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
            alignment.REPORT_ENV: os.path.join(tmpdir, "post.jsonl"),
        },
        clear=False,
    ):
      pre = alignment.check_pre_backward(wrapped, step=0)
      post = alignment.check_batch(
          wrapped,
          t_current=wrapped.t_old - np.float32(0.125),
          gradient_norm=np.asarray(0.0, np.float32),
          optimizer_skipped=np.asarray(0, np.int32),
          step=0,
      )
    self.assertEqual(pre["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
    self.assertEmpty(pre["blocking_reds"])
    self.assertIn("S_decode_vs_S_prefill", pre["warning_reds"])
    self.assertIn("S_prefill_vs_T_old", pre["warning_reds"])
    self.assertEqual(post["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
    self.assertEmpty(post["blocking_reds"])
    self.assertFalse(post["gradient"]["nonzero"])
    self.assertEqual(
        pre["admission_policy"]["claim_level"], "convergence-only"
    )

    nonfinite = wrapped.replace(t_old=np.full_like(wrapped.t_old, np.nan))
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            **common,
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
        },
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "S_prefill_vs_T_old"
    ):
      alignment.check_pre_backward(nonfinite, step=0)

    with mock.patch.dict(
        os.environ,
        {**common, "CANON_P34_RUN_STAGE": "one-update"},
        clear=False,
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "P34 full training"
    ):
      alignment.gsm8k_ab_report_policy()

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

  def test_pre_backward_mismatch_includes_multiturn_chunk_context(self):
    ids = np.arange(6, dtype=np.int32).reshape(1, 6)
    action_mask = np.asarray(
        [[True, True, False, False, True, True]], dtype=np.bool_
    )
    completion_valid_mask = np.ones_like(action_mask)
    prompt_mask = np.ones((1, 255), dtype=np.bool_)
    values = np.arange(6, dtype=np.float32).reshape(1, 6) / 8
    example = _Example(
        completion_ids=jnp.asarray(ids),
        completion_mask=jnp.asarray(action_mask),
        advantages=jnp.ones((1,), dtype=jnp.float32),
        is_update_step=None,
        prompt_ids=jnp.arange(255, dtype=jnp.int32).reshape(1, 255),
    )
    wrapped = alignment.wrap_train_example(
        example,
        s_decode=values,
        s_prefill=values.copy(),
        t_old=values.copy(),
        action_mask=action_mask,
        completion_valid_mask=completion_valid_mask,
        prompt_mask=prompt_mask,
        tokens=ids,
        policy_version=np.zeros((1,), dtype=np.int32),
        temperature=0.7,
        top_k=0,
        top_p=1.0,
        s_prefill_source=_real_rescore,
    )
    drift = wrapped.s_prefill.copy()
    drift[0, 4] = drift[0, 4] + np.float32(0.1039)
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

    mismatch = result["boundaries"]["S_decode_vs_S_prefill"][
        "first_mismatch"
    ]
    self.assertEqual(mismatch["coordinate"], [0, 4])
    self.assertEqual(mismatch["prompt_length"], 255)
    self.assertEqual(mismatch["completion_valid_length"], 6)
    self.assertEqual(mismatch["logical_kv_prefix_length"], 259)
    self.assertEqual(mismatch["completion_chunk_index"], 0)
    self.assertEqual(mismatch["sequence_chunk_index"], 1)
    self.assertEqual(mismatch["offset_in_sequence_chunk"], 3)
    self.assertEqual(mismatch["distance_to_next_sequence_chunk"], 253)
    self.assertEqual(mismatch["turn_index"], 1)
    self.assertTrue(mismatch["action_run_start"])
    self.assertFalse(mismatch["action_run_end"])
    self.assertEqual(mismatch["offset_in_action_run"], 0)
    self.assertTrue(mismatch["previous_token_is_environment"])

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

  def test_p38_mismatch_capsule_persists_bounded_replay_inputs(self):
    wrapped = self._wrapped(rows=3)
    drift = wrapped.s_prefill.copy()
    drift[2, 1] = drift[2, 1] + np.float32(0.125)
    drift[1, 0] = drift[1, 0] + np.float32(0.25)
    wrapped = wrapped.replace(s_prefill=drift)
    stdout = io.StringIO()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.PRE_GATE_ENV: "1",
            alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
            alignment.P38_MISMATCH_CAPSULE_ENV: os.path.join(
                tmpdir, "capsule.npz"
            ),
            alignment.P38_MISMATCH_CAPSULE_MAX_ROWS_ENV: "1",
            "CANON_NUM_GENERATIONS": "8",
        },
        clear=False,
    ), contextlib.redirect_stdout(stdout):
      result = alignment.check_pre_backward(
          wrapped, step=4, fail_closed=False
      )

      capsule_path = os.path.join(tmpdir, "capsule.npz")
      with np.load(capsule_path, allow_pickle=False) as capsule:
        self.assertEqual(capsule["selected_rows"].tolist(), [1])
        self.assertEqual(capsule["prompt_ids"].shape, (1, 2))
        self.assertEqual(capsule["completion_ids"].shape, (1, 3))
        self.assertEqual(capsule["s_decode"].shape, (1, 3))
        metadata = json.loads(capsule["metadata_json"].tobytes())
      self.assertEqual(metadata["schema"], "p38-frozenlake-mismatch-capsule-v1")
      self.assertEqual(metadata["selected_rows"], [1])
      self.assertEqual(metadata["num_generations"], 8)
      self.assertEqual(
          metadata["row_identity"],
          [{
              "source_row": 1,
              "batch_group_index": 0,
              "generation_index": 1,
          }],
      )
      self.assertEqual(result["mismatch_capsule"]["selected_rows"], [1])
      self.assertIn("[CANON_P38_CAPSULE]", stdout.getvalue())

  def test_p38_mismatch_capsule_rejects_collision(self):
    wrapped = self._wrapped(rows=1)
    drift = wrapped.s_prefill.copy()
    drift[0, 0] = drift[0, 0] + np.float32(0.125)
    wrapped = wrapped.replace(s_prefill=drift)
    with tempfile.TemporaryDirectory() as tmpdir:
      capsule_path = os.path.join(tmpdir, "capsule.npz")
      with open(capsule_path, "wb") as capsule_file:
        capsule_file.write(b"occupied")
      with mock.patch.dict(
          os.environ,
          {
              alignment.PRE_GATE_ENV: "1",
              alignment.PRE_REPORT_ENV: os.path.join(tmpdir, "pre.jsonl"),
              alignment.P38_MISMATCH_CAPSULE_ENV: capsule_path,
              "CANON_NUM_GENERATIONS": "8",
          },
          clear=False,
      ), self.assertRaisesRegex(
          alignment.AlignmentGateError, "already exists"
      ):
        alignment.check_pre_backward(wrapped, step=0, fail_closed=False)


if __name__ == "__main__":
  absltest.main()
