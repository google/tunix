#!/usr/bin/env python3
"""Host positives and negatives for the three-round small GCS return."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_m15_multiround_gcs_return import MultiRoundAuditError, audit


SOURCE = "a" * 40


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class MultiRoundReturnTest(unittest.TestCase):

  def setUp(self) -> None:
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)
    self.off = self.root / "off"
    self.on = self.root / "on"
    for arm_root in (self.off, self.on):
      (arm_root / "root").mkdir(parents=True)
      for round_index in range(3):
        (arm_root / f"round-{round_index:06d}").mkdir()

  def tearDown(self) -> None:
    self.holder.cleanup()

  def _round(self, arm: str, round_index: int, *, red: bool = False) -> None:
    root = (self.off if arm == "off" else self.on) / f"round-{round_index:06d}"
    classification_name = (
        "M15_LAYER_FIRST_RED_LOCALIZED"
        if red else (
            "M15_OBSERVER_CONTROL_EXACT"
            if arm == "off" else "M15_OBSERVER_TREATMENT_EXACT"
        )
    )
    receipt = {
        "schema": "m15-wide-sealed-input-v1",
        "status": "PASS",
        "diagnostic_round": round_index,
        "expected_source_commit": SOURCE,
        "runtime_source_commit": SOURCE,
        "record_pairs": 2,
        "replay_records": 2,
        "shards": [{"sequence": round_index}],
    }
    classification = {
        "schema": "m15-apc-wide-seam-classification-v1",
        "status": "PASS",
        "arm": arm,
        "diagnostic_round": round_index,
        "classification": classification_name,
        "alignment": {
            "a_b_differing_bytes": 7 if red else 0,
            "b_c_differing_bytes": 0,
        },
    }
    (root / "ROUND_INPUT_RECEIPT.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    (root / "p38_seam.classification.json").write_text(
        json.dumps(classification), encoding="utf-8"
    )
    bundle_sha = hashlib.sha256(f"bundle-{arm}-{round_index}".encode()).hexdigest()
    manifest = root / "WIDE_SHA256SUMS"
    manifest.write_text(
        f"{_sha(root / 'ROUND_INPUT_RECEIPT.json')}  ROUND_INPUT_RECEIPT.json\n"
        f"{_sha(root / 'p38_seam.classification.json')}  p38_seam.classification.json\n"
        f"{bundle_sha}  m15_wide_seam_bundle.tar\n",
        encoding="ascii",
    )
    completion = {
        "schema": "m15-wide-round-completion-v1",
        "status": "classified-and-uploaded",
        "diagnostic_round": round_index,
        "expected_source_commit": SOURCE,
        "runtime_source_commit": SOURCE,
        "classification": classification_name,
        "manifest_sha256": _sha(manifest),
        "record_pairs": 2,
        "shards": receipt["shards"],
    }
    (root / "WIDE_ROUND_COMPLETE.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    (root / "remote-inventory.txt").write_text(
        "ROUND_INPUT_RECEIPT.json present\n"
        "p38_seam.classification.json present\n"
        "WIDE_SHA256SUMS present\n"
        "WIDE_ROUND_COMPLETE.json present\n"
        "m15_wide_seam_bundle.tar present\n",
        encoding="utf-8",
    )

  def _markers(self, arm_root: Path, *, terminal: bool) -> None:
    names = ["PREFLIGHT.json"]
    if terminal:
      names.extend(("COLLECTED.json", "COMPLETE.json"))
    for name in names:
      (arm_root / "root" / name).write_text(json.dumps({
          "source_commit": SOURCE,
          "status": "PASS",
      }), encoding="utf-8")

  def _all_rounds(self) -> None:
    for round_index in range(3):
      self._round("off", round_index)
      self._round("on", round_index, red=round_index == 1)

  def test_complete_pair_returns_six_hash_bound_classifiers(self) -> None:
    self._all_rounds()
    self._markers(self.off, terminal=True)
    self._markers(self.on, terminal=True)
    output = self.root / "return"
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=output,
    )
    self.assertEqual(result["status"], "COMPLETE")
    self.assertEqual(len(list(output.glob("*.classification.json"))), 6)

  def test_all_rounds_survive_missing_root_terminal_markers(self) -> None:
    self._all_rounds()
    self._markers(self.off, terminal=False)
    self._markers(self.on, terminal=False)
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=self.root / "return",
    )
    self.assertEqual(result["status"], "ROUNDS_RECOVERED_ROOT_INCOMPLETE")
    self.assertEqual(result["arms"]["off"]["sealed_rounds"], 3)

  def test_one_sealed_round_is_reported_as_partial_not_discarded(self) -> None:
    self._round("off", 0)
    self._round("on", 0, red=True)
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=self.root / "return",
    )
    self.assertEqual(result["status"], "PARTIAL_ROUNDS_RECOVERED")

  def test_tampered_classifier_is_rejected(self) -> None:
    self._all_rounds()
    path = self.off / "round-000001/p38_seam.classification.json"
    path.write_text("{}", encoding="utf-8")
    with self.assertRaisesRegex(MultiRoundAuditError, "failed SHA"):
      audit(
          source_commit=SOURCE,
          rounds=3,
          off_root=self.off,
          on_root=self.on,
          output=self.root / "return",
      )


if __name__ == "__main__":
  unittest.main()
