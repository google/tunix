"""Runtime routing and postflight gates for P57 stock-fast runs."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "canon-zero-tim/cluster/steps/p57_runtime_contract.sh"
ENTRYPOINT = ROOT / "canon-zero-tim/cluster/entrypoint.sh"
RUNNER = ROOT / "canon-zero-tim/cluster/steps/90_run.sh"
INSTALL_STOCK = ROOT / "canon-zero-tim/cluster/steps/37_install_stock_runtime.sh"
VERIFY_STOCK = ROOT / "canon-zero-tim/cluster/steps/38_verify_stock_engine.sh"
PROFILE = "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"


def _bash(body: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
  return subprocess.run(
      [
          "bash",
          "-c",
          f"set -euo pipefail; source {shlex.quote(str(LIB))}; {body}",
      ],
      cwd=ROOT,
      env={**os.environ, **(env or {})},
      text=True,
      capture_output=True,
      check=False,
  )


class P57RuntimeContractTest(unittest.TestCase):

  def test_only_exact_stock_calibration_selects_stock_engine(self):
    good = {
        "CANON_PROFILE_FILE": PROFILE,
        "CANON_P57_RUN_KIND": "calibration",
        "CANON_P57_INFERENCE_REGIME": "stock-fast",
    }
    result = _bash(
        "p57_is_stock_fast_calibration; p57_is_nontraining_runtime", good
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    for changed in (
        {"CANON_PROFILE_FILE": "cluster/profiles/qwen3-8b.env"},
        {"CANON_P57_RUN_KIND": "train"},
        {"CANON_P57_INFERENCE_REGIME": "canonical"},
    ):
      with self.subTest(changed=changed):
        result = _bash(
            "if p57_is_stock_fast_calibration; then exit 9; fi",
            {**good, **changed},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

  def test_exact_stock_training_selects_stock_engine_but_is_training(self):
    good = {
        "CANON_PROFILE_FILE": PROFILE,
        "CANON_P57_RUN_KIND": "train",
        "CANON_P57_TIM_ARM": "mismatch",
        "CANON_P57_INFERENCE_REGIME": "stock-fast",
    }
    result = _bash(
        "p57_is_stock_fast_training; p57_is_stock_fast_runtime; "
        "if p57_is_nontraining_runtime; then exit 9; fi",
        good,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    is_result = _bash(
        "p57_is_stock_fast_training; p57_is_stock_fast_runtime; "
        "if p57_is_nontraining_runtime; then exit 9; fi",
        {**good, "CANON_P57_TIM_ARM": "is"},
    )
    self.assertEqual(is_result.returncode, 0, is_result.stderr)
    for changed in (
        {"CANON_P57_TIM_ARM": "zero"},
        {"CANON_P57_RUN_KIND": "eval"},
        {"CANON_P57_INFERENCE_REGIME": ""},
    ):
      with self.subTest(changed=changed):
        result = _bash(
            "if p57_is_stock_fast_training; then exit 9; fi",
            {**good, **changed},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

  def test_exact_stock_eval_is_stock_runtime_and_nontraining(self):
    result = _bash(
        "p57_is_stock_fast_evaluation; p57_is_stock_fast_runtime; "
        "p57_is_nontraining_runtime",
        {
            "CANON_PROFILE_FILE": PROFILE,
            "CANON_P57_RUN_KIND": "eval",
            "CANON_P57_TIM_ARM": "mismatch",
            "CANON_P57_INFERENCE_REGIME": "stock-fast",
        },
    )
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_only_exact_p57_eval_bypasses_training_admission(self):
    result = _bash(
        "p57_is_nontraining_runtime; ! p57_is_stock_fast_calibration",
        {
            "CANON_PROFILE_FILE": PROFILE,
            "CANON_P57_RUN_KIND": "eval",
        },
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    result = _bash(
        "if p57_is_nontraining_runtime; then exit 9; fi",
        {
            "CANON_PROFILE_FILE": "cluster/profiles/qwen3-8b.env",
            "CANON_P57_RUN_KIND": "eval",
        },
    )
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_stock_postflight_requires_every_canonical_marker_absent(self):
    result = _bash("p57_validate_stock_fast_runtime_markers 0 0 0 0 0 0")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("RUNTIME_PATH_PASS canonical_markers=0", result.stdout)
    for index in range(6):
      counts = ["0"] * 6
      counts[index] = "1"
      with self.subTest(index=index):
        result = _bash(
            "p57_validate_stock_fast_runtime_markers " + " ".join(counts)
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical runtime marker leaked", result.stderr)

  def test_stock_postflight_rejects_malformed_counts(self):
    result = _bash("p57_validate_stock_fast_runtime_markers 0 0 x 0 0 0")
    self.assertEqual(result.returncode, 2)
    self.assertIn("invalid logprob_m marker count", result.stderr)

  def test_entrypoint_stock_branch_never_installs_canonical_overlay(self):
    text = ENTRYPOINT.read_text(encoding="utf-8")
    start = text.index("elif p57_is_stock_fast_runtime; then")
    end = text.index("\nelse\n", start)
    branch = text[start:end]
    self.assertIn("step 35_install_r2egym.sh", branch)
    self.assertIn("step 37_install_stock_runtime.sh", branch)
    self.assertIn("step 38_verify_stock_engine.sh", branch)
    self.assertIn("canonical_overlay=skipped", branch)
    self.assertNotIn("30_install_canon", branch)
    self.assertNotIn("40_overlay_engine", branch)
    self.assertNotIn("50_verify_overlay", branch)

  def test_stock_install_and_verify_admit_the_complete_runtime_tuple(self):
    for path in (INSTALL_STOCK, VERIFY_STOCK):
      with self.subTest(path=path.name):
        text = path.read_text(encoding="utf-8")
        self.assertIn("if ! p57_is_stock_fast_runtime; then", text)
        self.assertNotIn("if ! p57_is_stock_fast_calibration; then", text)

  def test_runner_keeps_distinct_stock_and_canonical_postflights(self):
    text = RUNNER.read_text(encoding="utf-8")
    self.assertIn("p57_validate_stock_fast_runtime_markers", text)
    self.assertIn("elif p57_is_stock_fast_runtime; then", text)
    self.assertIn('n_p57_stock_sync" -ne 1', text)
    self.assertIn('n_p57_stock_segment_complete" -ne 1', text)
    self.assertIn('n_p57_tim_purity_none" -ne 1', text)
    self.assertIn('n_p57_tim_purity_is" -ne 1', text)
    self.assertIn("P57 no-IS purity marker contract failed", text)
    self.assertIn("P57 IS purity marker contract failed", text)
    self.assertIn('elif [ "$n_ar" -eq 0 ] || [ "$n_emb" -eq 0 ]', text)


if __name__ == "__main__":
  unittest.main()
