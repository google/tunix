#!/usr/bin/env python3
"""Regression tests for the bounded P58 decode/prefill evidence join."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts"
    / "classify_decode_prefill_probe.py"
)
SOURCE_SHA = "1" * 40


def _load():
  spec = importlib.util.spec_from_file_location("decode_prefill_probe", SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PROBE = _load()


def _write_fixture(
    root: Path,
    *,
    differing: int = 1,
    b_c_differing: int = 0,
    mismatch_token: int = 12,
    n_action: int = 2,
    zero_admission: bool = False,
    seam_diagnostic: str = "",
    alignment_only: bool = False,
    short_backward: bool = False,
) -> None:
  manifest = {
      "schema": "canon.local.deepswe.run-manifest.v1",
      "source_commit": SOURCE_SHA,
      "expected_hostname": "v5p-host",
      "model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "contract_name": "local-qwen4b-dp1-tp4-seam-probe",
      "role_topology": {"dp": 1, "tp": 4, "devices": 4},
      "onehost_seam_probe": True,
      "onehost_xprof_arm": "zero-hp",
      "stage": "backward-no-commit",
      "whitelist_sha256": "2" * 64,
  }
  if zero_admission:
    manifest.update({
        "contract_name": "local-qwen4b-dp1-tp4-zero-admission",
        "q4_tp4_zero_admission": True,
        "q4_tp4_seam_diagnostic": seam_diagnostic,
        "q4_tp4_continue_kv_diagnostic": alignment_only,
        "alignment_precheck_only": alignment_only,
        "alignment_controlled_exit": alignment_only,
        "continue_decode_steps": "" if seam_diagnostic else "8",
        "sampling_contract": {
            "source": "explicit-cli",
            "temperature": 0.7,
            "top_k": 0,
            "top_p": 1.0,
        },
    })
  if short_backward:
    manifest.update({
        "q4_tp4_short_backward": True,
        "compilation_cache_dir": (
            "/mnt/disks/tunix-data/jax-compilation-cache/"
            "p58-q4-tp4-short-backward"
        ),
        "max_prompt_length": 1792,
        "max_response_length": 2880,
        "max_turns": 16,
        "task_image": (
            "namanjain12/pillow_final:"
            "52079cb2975fda98476c7a7f172e5519e67ba612"
        ),
        "whitelist_sha256": (
            "7294da90559ebace771b7bd3fd8be01de"
            "87e0ae9bcb7ae1e317dbe5a6ed0db9f"
        ),
    })
  (root / "run_manifest.json").write_text(json.dumps(manifest))
  rows = [
      {
          "schema": "canon.local.deepswe.trajectory.v1",
          "status": "SUCCEEDED",
          "compact_filtered": False,
          "trajectory": {
              "prompt_length": 5,
              "conversation_tokens": [11, 12, 13],
              "conversation_masks": [0, 1, 1],
              "old_logprobs": [0.0, -0.25, -0.75],
          },
      },
      {
          "schema": "canon.local.deepswe.trajectory.v1",
          "status": "MODEL_TIMEOUT",
          "compact_filtered": True,
          "trajectory": {
              "prompt_length": None,
              "conversation_tokens": [],
              "conversation_masks": [],
              "old_logprobs": [],
          },
      },
  ]
  with gzip.open(root / "batch-000000.trajectories.jsonl.gz", "wt") as out:
    for row in rows:
      out.write(json.dumps(row) + "\n")
  mismatches = []
  if differing:
    mismatches.append({
        "coordinate": [7, 1],
        "completion_position": 1,
        "completion_valid_length": 3,
        "prompt_length": 5,
        "token_id": mismatch_token,
        "a": -0.25,
        "b": -0.20,
        "action_run_start": True,
        "previous_token_is_environment": True,
    })
  boundary = {
      "valid": True,
      "finite": True,
      "differing_elements": differing,
      "differing_bytes": differing * 2,
      "total_elements": 2,
      "element_fraction": differing / 2,
      "byte_fraction": differing / 4,
      "max_abs": 0.05 if differing else 0.0,
      "first_mismatch": mismatches[0] if mismatches else None,
      "mismatches": mismatches,
      "mismatches_truncated": False,
  }
  b_c_boundary = {
      "valid": True,
      "finite": True,
      "differing_elements": b_c_differing,
      "differing_bytes": b_c_differing * 2,
      "total_elements": 2,
      "element_fraction": b_c_differing / 2,
      "byte_fraction": b_c_differing / 4,
      "max_abs": 0.05 if b_c_differing else 0.0,
      "first_mismatch": None,
      "mismatches": [],
      "mismatches_truncated": False,
  }
  prealignment = {
      "N_action": n_action,
      "boundaries": {
          "S_decode_vs_S_prefill": boundary,
          "S_prefill_vs_T_old": b_c_boundary,
      },
  }
  (root / "pre_alignment.jsonl").write_text(json.dumps(prealignment) + "\n")
  (root / "batch_metrics.jsonl").write_text(
      json.dumps({"schema": "canon.local.deepswe.batch-metrics.v1"}) + "\n"
  )
  if zero_admission:
    (root / "probe_process_status.json").write_text(json.dumps({
        "profile": "seam",
        "training_process_status": 42 if alignment_only else 0,
    }))
    if alignment_only:
      (root / "raw.log").write_text(
          "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD "
          "rounds=1 step=0 N_action=2 verdict=PASS "
          "a_b_differing_bytes=0\n"
          "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 "
          "optimizer_commits=0\n"
      )
      return
    backward = {
        "verdict": "PASS",
        "commits": 0,
        "gradient_finite": True,
        "gradient_nonzero": True,
        "gradient_repeat_exact": True,
        "repeat_count": 2,
        "xprof_arm": "zero-hp",
        "model_changed_paths": [],
        "optimizer_changed_paths": [],
        "accumulator_changed_paths": [],
        "reference_changed_paths": [],
        "train_steps_before": 0,
        "train_steps_after": 0,
        "work_hashes": {"actor_update_calls": 2},
    }
    (root / "backward_no_commit.json").write_text(json.dumps(backward))
    (root / "alignment.jsonl").write_text(json.dumps({
        "N_action": n_action,
        "boundaries": {
            "S_decode_vs_S_prefill": boundary,
            "S_prefill_vs_T_old": b_c_boundary,
        },
    }) + "\n")


class DecodePrefillProbeTest(unittest.TestCase):

  def test_red_joins_durable_row_and_skips_compact_timeout(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root)
      report = PROBE.classify(
          root, source_sha=SOURCE_SHA, expected_hostname="v5p-host"
      )
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(report["outcome"], "FINITE_RED_REPRODUCED")
      self.assertEqual(report["N_action"], 2)
      self.assertEqual(report["compact_filtered_rows"], 1)
      boundary = report["S_decode_vs_S_prefill"]
      self.assertEqual(boundary["joined_artifact_rows"], {"0": 1})
      self.assertAlmostEqual(
          boundary["shift_discriminator"][1]["median_abs"], 0.05
      )

  def test_exact_action_boundary_is_a_bounded_pass(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, differing=0)
      report = PROBE.classify(root)
      self.assertEqual(report["outcome"], "EXACT_ON_THIS_CARRIER")
      self.assertEqual(report["verdict"], "PASS")

  def test_zero_admission_requires_both_boundaries_and_backward(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, differing=0, zero_admission=True)
      report = PROBE.classify(root)
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(report["outcome"], "ZERO_TIM_BACKWARD_NO_COMMIT_PASS")
      self.assertTrue(report["zero_admission"])
      self.assertEqual(report["zero_red_boundaries"], [])
      self.assertEqual(report["backward_no_commit"]["report"]["commits"], 0)

  def test_short_backward_requires_exact_clean_task_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(
          root, differing=0, zero_admission=True, short_backward=True
      )
      report = PROBE.classify(root)
      self.assertEqual(report["outcome"], "ZERO_TIM_BACKWARD_NO_COMMIT_PASS")

      manifest_path = root / "run_manifest.json"
      for key, value, message in (
          ("max_response_length", 3072, "max_response_length"),
          ("task_image", "wrong", "clean task image"),
          ("whitelist_sha256", "3" * 64, "clean whitelist"),
      ):
        manifest = json.loads(manifest_path.read_text())
        original = manifest[key]
        manifest[key] = value
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, message):
          PROBE.classify(root)
        manifest[key] = original
        manifest_path.write_text(json.dumps(manifest))

  def test_zero_alignment_only_requires_exact_controlled_stop(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(
          root, differing=0, zero_admission=True, alignment_only=True
      )
      report = PROBE.classify(root)
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(report["outcome"], "ZERO_TIM_ALIGNMENT_ONLY_PASS")
      self.assertIsNone(report["backward_no_commit"])
      self.assertIn("does not certify backward", report["claim"])

      (root / "raw.log").write_text("missing controlled marker\n")
      with self.assertRaisesRegex(ValueError, "lacks marker"):
        PROBE.classify(root)

  def test_zero_admission_red_is_never_a_pass(self):
    for differing, b_c_differing, name in (
        (1, 0, "S_decode_vs_S_prefill"),
        (0, 1, "S_prefill_vs_T_old"),
    ):
      with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_fixture(
            root,
            differing=differing,
            b_c_differing=b_c_differing,
            zero_admission=True,
        )
        report = PROBE.classify(root)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["outcome"], "ZERO_TIM_ALIGNMENT_RED")
        self.assertIn(name, report["zero_red_boundaries"])

  def test_standard_decode_control_has_diagnostic_outcomes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(
          root,
          differing=0,
          zero_admission=True,
          seam_diagnostic="standard-decode",
      )
      report = PROBE.classify(root)
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(
          report["outcome"], "ZERO_TIM_STANDARD_DECODE_CONTROL_PASS"
      )
      self.assertIn("causal control only", report["claim"])
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(
          root,
          differing=1,
          zero_admission=True,
          seam_diagnostic="standard-decode",
      )
      report = PROBE.classify(root)
      self.assertEqual(report["verdict"], "FAIL")
      self.assertEqual(
          report["outcome"], "ZERO_TIM_STANDARD_DECODE_ALIGNMENT_RED"
      )

  def test_mismatch_token_must_join_exact_durable_value(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, mismatch_token=99)
      with self.assertRaisesRegex(ValueError, "does not join exactly one"):
        PROBE.classify(root)

  def test_action_count_must_match_admitted_rows(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, n_action=3)
      with self.assertRaisesRegex(ValueError, "action count differs"):
        PROBE.classify(root)

  def test_manifest_requires_whitelist_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root)
      manifest_path = root / "run_manifest.json"
      manifest = json.loads(manifest_path.read_text())
      manifest["whitelist_sha256"] = ""
      manifest_path.write_text(json.dumps(manifest))
      with self.assertRaisesRegex(ValueError, "whitelist identity"):
        PROBE.classify(root)

  def test_cli_packages_every_root_evidence_file_with_checksums(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root)
      (root / "raw.log").write_text("strict gate stopped\n")
      subprocess.run(
          [
              sys.executable,
              str(SCRIPT),
              "--artifact-dir",
              str(root),
              "--source-sha",
              SOURCE_SHA,
              "--expected-hostname",
              "v5p-host",
              "--package",
          ],
          check=True,
          capture_output=True,
          text=True,
      )
      checksums = (root / "SHA256SUMS").read_text().splitlines()
      names = set()
      for line in checksums:
        digest, name = line.split("  ", 1)
        names.add(name)
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        self.assertEqual(observed, digest)
      self.assertIn("raw.log", names)
      self.assertIn("RETURN_TO_AGENT.md", names)
      self.assertIn("decode_prefill_probe.classification.json", names)


if __name__ == "__main__":
  unittest.main()
