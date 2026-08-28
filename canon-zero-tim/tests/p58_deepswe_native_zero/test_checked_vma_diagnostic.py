#!/usr/bin/env python3
"""Contracts for the exact-geometry P58 checked-VMA-off classifier."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts"
    / "classify_p58_checked_vma_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("p58_vma_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLASSIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLASSIFIER)


def _fixture(
    root: Path, *, selector: str = "off", a_b_bytes: int = 2,
    b_c_bytes: int = 0,
) -> dict:
  debug = root / "debug"
  debug.mkdir()
  manifest = {
      "schema": "canon.p58.deepswe.run-manifest.v1",
      "trajectory_schema": "canon.p58.deepswe.trajectory.v1",
      "metrics_schema": "canon.p58.deepswe.batch-metrics.v1",
      "source_commit": "1" * 40,
      "model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "contract_name": "p58-qwen4b-tim-128",
      "tim_arm": "zero",
      "checked_vma_diagnostic": selector,
      "stage": "full",
      "slice_topology": "4x4x8",
      "role_topology": {"dp": 8, "tp": 8, "devices": 64},
      "global_prompts": 8,
      "generations": 16,
      "global_trajectories": 128,
      "max_response_length": 16384,
      "max_turns": 50,
      "whitelist_sha256": (
          "ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973"
      ),
  }
  (debug / "run_manifest.json").write_text(json.dumps(manifest))
  rows = []
  for index in range(128):
    rows.append({
        "schema": "canon.p58.deepswe.trajectory.v1",
        "status": "SUCCEEDED",
        "compact_filtered": False,
        "trajectory": {
            "prompt_length": 5 if index == 0 else 6,
            "conversation_tokens": [12 + index],
            "conversation_masks": [1],
            "old_logprobs": [-0.25 - index],
        },
    })
  trajectory_path = debug / "batch-000000.trajectories.jsonl.gz"
  with gzip.open(trajectory_path, "wt", encoding="utf-8") as stream:
    for row in rows:
      stream.write(json.dumps(row) + "\n")
  (debug / "batch_metrics.jsonl").write_text(json.dumps({
      "trajectory_path": str(trajectory_path),
      "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
      "trajectories": 128,
  }) + "\n")

  mismatches = []
  if a_b_bytes:
    mismatches.append({
        "coordinate": [0, 0],
        "completion_position": 0,
        "completion_valid_length": 1,
        "prompt_length": 5,
        "token_id": 12,
        "a": -0.25,
        "b": -0.20,
    })
  prealignment = root / "pre_alignment.jsonl"
  prealignment.write_text(json.dumps({
      "N_action": 128,
      "boundaries": {
          "S_decode_vs_S_prefill": {
              "valid": True,
              "finite": True,
              "differing_elements": 1 if a_b_bytes else 0,
              "differing_bytes": a_b_bytes,
              "total_elements": 128,
              "max_abs": 0.05 if a_b_bytes else 0.0,
              "mismatches": mismatches,
              "first_mismatch": mismatches[0] if mismatches else None,
          },
          "S_prefill_vs_T_old": {
              "valid": True,
              "finite": True,
              "differing_elements": 1 if b_c_bytes else 0,
              "differing_bytes": b_c_bytes,
              "total_elements": 128,
              "max_abs": 0.01 if b_c_bytes else 0.0,
              "mismatches": [],
              "first_mismatch": None,
          },
      },
  }) + "\n")
  run_log = root / "run.log"
  run_log.write_text(
      CLASSIFIER._profile_marker(selector) + "\n"
      "[CANON_P38] PRECHECK_ROUND_COMPLETE round=0 step=0 N_action=128 "
      "verdict=PASS backward=0 optimizer_commits=0\n"
      "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD rounds=1 step=0 "
      "N_action=128 verdict=PASS a_b_differing_bytes=2\n"
      + CLASSIFIER._CONTROLLED_EXIT + "\n"
  )
  return {
      "run_log": run_log,
      "pre_alignment": prealignment,
      "debug_dir": debug,
      "update_report": root / "updates.jsonl",
      "selector": selector,
  }


class P58CheckedVmaDiagnosticTest(unittest.TestCase):

  def test_finite_red_is_a_valid_zero_commit_discriminator(self):
    with tempfile.TemporaryDirectory() as directory:
      result = CLASSIFIER.classify(**_fixture(Path(directory)))
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["outcome"], "A_B_RED_WITH_CHECKED_VMA_OFF")
      self.assertEqual(result["B_C_differing_bytes"], 0)

  def test_exact_a_b_is_a_valid_zero_commit_discriminator(self):
    with tempfile.TemporaryDirectory() as directory:
      result = CLASSIFIER.classify(
          **_fixture(Path(directory), a_b_bytes=0)
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["outcome"], "A_B_EXACT_WITH_CHECKED_VMA_OFF")

  def test_on_arm_is_a_valid_zero_commit_control(self):
    with tempfile.TemporaryDirectory() as directory:
      result = CLASSIFIER.classify(
          **_fixture(Path(directory), selector="on")
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["selector"], "on")
      self.assertEqual(result["outcome"], "A_B_RED_WITH_CHECKED_VMA_ON")

  def test_b_c_drift_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      result = CLASSIFIER.classify(
          **_fixture(Path(directory), b_c_bytes=2)
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("B-C_bytes=2", result["reasons"])

  def test_optimizer_activity_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = _fixture(Path(directory))
      paths["update_report"].write_text('{"commits":1}\n')
      result = CLASSIFIER.classify(**paths)
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("update_report_nonempty", result["reasons"])

  def test_any_fixed_head_vjp_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = _fixture(Path(directory))
      with paths["run_log"].open("a", encoding="utf-8") as stream:
        stream.write(
            "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 semantic_M=4096\n"
        )
      result = CLASSIFIER.classify(**paths)
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn(
          "forbidden_runtime=['fixed_lm_head_vjp']", result["reasons"]
      )

  def test_postflight_skips_full_training_receipt_gate_before_classifier(self):
    postflight = (
        ROOT / "canon-zero-tim/cluster/steps/90_run.sh"
    ).read_text()
    self.assertIn('CANON_P58_CHECKED_VMA_DIAGNOSTIC:-}" ] &&', postflight)
    self.assertIn('CANON_P58_SEAM_LOCALIZATION:-}" ]; then', postflight)
    self.assertIn("classify_p58_checked_vma_diagnostic.py", postflight)
    self.assertIn("n_p58_continue_decode_observer_bypass", postflight)
    self.assertIn("P58 observer round budget receipt is absent", postflight)
    self.assertIn(
        "P58 continue-decode observer bypass was not observed", postflight
    )
    self.assertIn(
        "foreign P58 continue-decode observer bypass marker", postflight
    )


if __name__ == "__main__":
  unittest.main()
