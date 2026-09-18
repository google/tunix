#!/usr/bin/env python3
"""Host positives and negatives for the Attempt-13 return analyzer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_m15_attempt13_return import (
    Attempt13ReturnError,
    SOURCE_COMMIT,
    analyze,
)


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class Attempt13ReturnAnalysisTest(unittest.TestCase):

  def setUp(self) -> None:
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)

  def tearDown(self) -> None:
    self.holder.cleanup()

  def _classifier(
      self,
      arm: str,
      round_index: int,
      *,
      checkpoint: str = "rpa_output",
      treatment_exact: bool = False,
  ) -> dict:
    exact = arm == "off" or treatment_exact
    value = {
        "schema": "m15-apc-wide-seam-classification-v1",
        "status": "PASS",
        "arm": arm,
        "diagnostic_round": round_index,
        "classification": (
            "M15_OBSERVER_CONTROL_EXACT" if arm == "off"
            else (
                "M15_OBSERVER_TREATMENT_EXACT" if treatment_exact
                else "M15_INTERNAL_FIRST_RED_LOCALIZED"
            )
        ),
        "alignment": {
            "a_b_differing_bytes": 0 if exact else 9,
            "b_c_differing_bytes": 0,
        },
    }
    if arm == "on" and not treatment_exact:
      value.update({
          "observer_mode": "full",
          "expected_layer": 0,
          "mixed_first_difference_signatures": False,
          "anchors": [{"diagnostic_round": round_index}],
          "first_difference_signatures": [
              {"layer": 0, "checkpoint": checkpoint}
          ],
          "replay_ledger_receipts": [{"diagnostic_round": round_index}],
          "last_exact_boundary": {"layer": 0, "checkpoint": "k_post_rope"},
          "first_red_boundary": {"layer": 0, "checkpoint": checkpoint},
          "source_interval": {"last_exact": {}, "first_red": {}},
      })
    return value

  def _return(
      self,
      *,
      unstable: bool = False,
      omit_official: bool = False,
      treatment_exact: bool = False,
  ) -> None:
    arms = {}
    for arm in ("off", "on"):
      rounds = []
      for round_index in range(3):
        value = self._classifier(
            arm,
            round_index,
            checkpoint=("o_proj" if unstable and round_index == 2 else "rpa_output"),
            treatment_exact=treatment_exact,
        )
        if omit_official and arm == "on" and round_index == 1:
          value.pop("anchors")
        path = self.root / f"{arm}.round-{round_index:06d}.classification.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        rounds.append({"diagnostic_round": round_index, "status": "SEALED"})
      arms[arm] = {"sealed_rounds": 3, "rounds": rounds, "root_markers": {}}
    summary = {
        "schema": "m15-apc-multiround-small-return-v1",
        "status": "COMPLETE",
        "source_commit": SOURCE_COMMIT,
        "expected_rounds_per_arm": 3,
        "arms": arms,
    }
    (self.root / "MULTIROUND_SUMMARY.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (self.root / "PACKAGING.txt").write_text("test\n", encoding="utf-8")
    names = sorted(path.name for path in self.root.iterdir())
    (self.root / "SHA256SUMS").write_text(
        "".join(f"{_sha(self.root / name)}  {name}\n" for name in names),
        encoding="ascii",
    )

  def test_three_stable_red_rounds_are_analysis_ready(self) -> None:
    self._return()
    result = analyze(self.root)
    self.assertEqual(
        result["decision"], "THREE_ROUND_ATTENTION_INTERVAL_REPEAT_READY"
    )
    self.assertFalse(result["numerical_repair_authorized"])

  def test_mixed_first_red_signatures_are_reported(self) -> None:
    self._return(unstable=True)
    result = analyze(self.root)
    self.assertEqual(result["decision"], "THREE_ROUND_SIGNATURE_UNSTABLE")

  def test_three_exact_treatment_rounds_do_not_claim_attention_interval(self) -> None:
    self._return(treatment_exact=True)
    result = analyze(self.root)
    self.assertEqual(result["decision"], "THREE_ROUND_TREATMENT_EXACT")

  def test_minimized_classifier_without_official_fields_is_rejected(self) -> None:
    self._return(omit_official=True)
    with self.assertRaisesRegex(
        Attempt13ReturnError, "official full-observer provenance"
    ):
      analyze(self.root)

  def test_manifest_tamper_is_rejected(self) -> None:
    self._return()
    path = self.root / "on.round-000001.classification.json"
    path.write_text("{}", encoding="utf-8")
    with self.assertRaisesRegex(Attempt13ReturnError, "SHA failed"):
      analyze(self.root)


if __name__ == "__main__":
  unittest.main()
