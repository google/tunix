#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("classify_m15_e0v_onehost_pair.py")
SPEC = importlib.util.spec_from_file_location("classify_m15_e0v_onehost_pair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_arm(root: Path, arm: str) -> None:
  directory = root / arm
  directory.mkdir()
  lines = [
      "[M15.E0V.ONEHOST] exact TITO enabled mode=exact "
      f"arm={arm} topology=DP1xTP4 rounds=3"
  ]
  for index in range(1, 4):
    lines.extend((
        "[CANON_M15_TOKEN_CONTINUITY] mode=exact turn=1 "
        "verdict=TOKEN_STREAM_EQUAL actual_tokens=10 expected_tokens=10 "
        f"actual_sha256={'a' * 64} expected_sha256={'a' * 64} "
        "first_mismatch=-1 actual_token=NA expected_token=NA",
        "[CANON_P38] PRECHECK_ROUND_COMPLETE "
        f"round={index}/3 step=0 N_action=12 verdict=PASS "
        "a_b_differing_bytes=0 backward=0 optimizer_commits=0",
    ))
  (directory / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
  (directory / "source.diff").write_text("bounded source diff\n", encoding="utf-8")
  (directory / "diagnostic_round").write_text("2\n", encoding="ascii")
  records = []
  for index in range(3):
    records.append(json.dumps({"diagnostic_round": index}))
  (directory / "pre_alignment.jsonl").write_text(
      "\n".join(records) + "\n", encoding="utf-8"
  )
  raw_sha = _sha256(directory / "raw.log")
  report_sha = _sha256(directory / "pre_alignment.jsonl")
  alignment = {
      "schema": "m15-e0v-tito-onehost-arm-classification-v1",
      "status": "CONTROL_GREEN" if arm == "off" else "TREATMENT_EXACT",
      "records": 3,
      "diagnostic_rounds": [0, 1, 2],
      "a_b_differing_bytes": [0, 0, 0],
      "b_c_differing_bytes": [0, 0, 0],
      "max_prefix_cache_hit_rate_percent": None if arm == "off" else 80.0,
      "b_full_reset_receipt_counts": [1, 1, 1],
      "raw_sha256": raw_sha,
      "pre_alignment_sha256": report_sha,
  }
  (directory / "alignment.classification.json").write_text(
      json.dumps(alignment) + "\n", encoding="utf-8"
  )
  tito = {
      "status": "PASS",
      "scope": "onehost",
      "arm": arm,
      "topology": "DP1xTP4",
      "diagnostic_rounds": 3,
      "round_receipt_counts": [1, 1, 1],
      "total_exact_equal_receipts": 3,
      "different_or_malformed_receipts": 0,
      "run_log_sha256": raw_sha,
  }
  (directory / "tito.classification.json").write_text(
      json.dumps(tito) + "\n", encoding="utf-8"
  )
  contract = {
      "schema": "m15-e0v-tito-onehost-arm-v1",
      "arm": arm,
      "apc": 1 if arm == "on" else 0,
      "topology": "DP1xTP4",
      "rounds": 3,
      "docker_exit": 42,
      "elapsed_seconds": 10,
      "backward": 0,
      "optimizer_commits": 0,
      "m15_token_continuity": "exact",
      "source_commit": "1" * 40,
      "source_diff_sha256": "2" * 64,
      "image_id": "sha256:" + "3" * 64,
      "vllm_rollout_sha256": "4" * 64,
  }
  (directory / "RUN_CONTRACT.json").write_text(
      json.dumps(contract) + "\n", encoding="utf-8"
  )
  names = sorted(classifier._ARM_FILES)
  (directory / "SHA256SUMS").write_text(
      "".join(f"{_sha256(directory / name)}  {name}\n" for name in names),
      encoding="ascii",
  )


class OnehostPairClassifierTest(unittest.TestCase):

  def _root(self, directory: str) -> Path:
    root = Path(directory)
    _write_arm(root, "off")
    _write_arm(root, "on")
    return root

  def test_exact_matched_pair_passes(self):
    with tempfile.TemporaryDirectory() as directory:
      report = classifier.classify(self._root(directory))
      self.assertEqual(report["status"], "ONEHOST_PAIR_EXACT")
      self.assertTrue(report["tito_exact_both_arms"])
      self.assertFalse(report["target_executed"])

  def test_treatment_a_b_red_is_preserved(self):
    with tempfile.TemporaryDirectory() as directory:
      root = self._root(directory)
      path = root / "on" / "alignment.classification.json"
      report = json.loads(path.read_text())
      report["a_b_differing_bytes"] = [0, 1, 0]
      manifest = root / "on" / "SHA256SUMS"
      report["status"] = "TREATMENT_RED"
      path.write_text(json.dumps(report) + "\n", encoding="utf-8")
      lines = manifest.read_text().splitlines()
      manifest.write_text(
          "\n".join(
              f"{_sha256(path)}  alignment.classification.json"
              if line.endswith("  alignment.classification.json") else line
              for line in lines
          ) + "\n",
          encoding="ascii",
      )
      classified = classifier.classify(root)
      self.assertEqual(classified["status"], "ONEHOST_RED_REPRODUCED")
      self.assertFalse(classified["treatment_a_b_zero"])

  def test_manifest_hash_drift_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = self._root(directory)
      with (root / "off" / "raw.log").open("a", encoding="utf-8") as stream:
        stream.write("drift\n")
      with self.assertRaisesRegex(classifier.OnehostPairError, "hash drifted"):
        classifier.classify(root)

  def test_missing_b_reset_round_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = self._root(directory)
      path = root / "on" / "alignment.classification.json"
      report = json.loads(path.read_text())
      report["b_full_reset_receipt_counts"] = [1, 0, 1]
      path.write_text(json.dumps(report) + "\n", encoding="utf-8")
      manifest = root / "on" / "SHA256SUMS"
      lines = manifest.read_text().splitlines()
      manifest.write_text(
          "\n".join(
              f"{_sha256(path)}  alignment.classification.json"
              if line.endswith("  alignment.classification.json") else line
              for line in lines
          ) + "\n",
          encoding="ascii",
      )
      with self.assertRaisesRegex(classifier.OnehostPairError, "B full-reset"):
        classifier.classify(root)

  def test_image_mismatch_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = self._root(directory)
      path = root / "on" / "RUN_CONTRACT.json"
      report = json.loads(path.read_text())
      report["image_id"] = "sha256:" + "4" * 64
      path.write_text(json.dumps(report) + "\n", encoding="utf-8")
      manifest = root / "on" / "SHA256SUMS"
      lines = manifest.read_text().splitlines()
      manifest.write_text(
          "\n".join(
              f"{_sha256(path)}  RUN_CONTRACT.json"
              if line.endswith("  RUN_CONTRACT.json") else line
              for line in lines
          ) + "\n",
          encoding="ascii",
      )
      with self.assertRaisesRegex(classifier.OnehostPairError, "image_id"):
        classifier.classify(root)


if __name__ == "__main__":
  unittest.main()
