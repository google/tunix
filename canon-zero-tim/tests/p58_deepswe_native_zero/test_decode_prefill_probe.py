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
    mismatch_token: int = 12,
    n_action: int = 2,
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
  prealignment = {
      "N_action": n_action,
      "boundaries": {
          "S_decode_vs_S_prefill": {
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
      },
  }
  (root / "pre_alignment.jsonl").write_text(json.dumps(prealignment) + "\n")
  (root / "batch_metrics.jsonl").write_text(
      json.dumps({"schema": "canon.local.deepswe.batch-metrics.v1"}) + "\n"
  )


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
