#!/usr/bin/env python3
"""Host tests for the Attempt-17 offline candidate review."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from review_m15_attempt17_d36_candidate import (
    M15Attempt17ReviewError,
    review,
)
from test_classify_m15_apc_wide_seam import Fixture, PACKAGER


SOURCE = "1" * 40


class M15Attempt17OfflineReviewTest(unittest.TestCase):

  def setUp(self):
    self.holder = tempfile.TemporaryDirectory()
    self.addCleanup(self.holder.cleanup)
    self.root = Path(self.holder.name)

  def _bundle(self, *, add_binding: bool) -> tuple[Path, Path, Fixture]:
    fixture = Fixture(mode="full")
    self.addCleanup(fixture.close)
    fixture.duplicate_seam(
        arm="A",
        request_id="request-a-exact",
        call_index=20,
        numeric_source_arm="B",
    )
    original = fixture.classify()
    self.assertEqual(original["gate"], "FIRST_RED_CANDIDATE_SET")
    classification = self.root / "classification.json"
    classification.write_text(
        json.dumps(original, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    if add_binding:
      fixture.append_future_request_identity(
          request_id="request-a", tokens=fixture.tokens[:4], call_index=30
      )
      fixture.append_future_request_identity(
          request_id="request-a-exact", tokens=[10, 11, 12, 99], call_index=31
      )
    bundle = self.root / ("bound.tar" if add_binding else "unresolved.tar")
    PACKAGER.package(
        directory=fixture.capture,
        classification_path=classification,
        alignment_report=fixture.report,
        capsules=[fixture.capsule],
        replay_ledger=fixture.ledger,
        output=bundle,
    )
    return bundle, classification, fixture

  def test_future_prefix_binding_localizes_immutable_candidate_bundle(self):
    bundle, classification, _ = self._bundle(add_binding=True)
    output = self.root / "return"
    core_summary = self.root / "MULTIROUND_SUMMARY.json"
    core_summary.write_text(json.dumps({
        "schema": "m15-apc-multiround-small-return-v1",
        "source_commit": SOURCE,
        "status": "PARTIAL_ROUNDS_RECOVERED",
        "arms": {"on": {"rounds": [
            {
                "status": "SEALED",
                "classification": "M15_INTERNAL_FIRST_RED_CANDIDATE_SET",
            },
            {"status": "UNSEALED"},
            {"status": "ABSENT"},
        ]}},
    }, sort_keys=True) + "\n", encoding="utf-8")
    result = review(
        bundle=bundle,
        expected_classification=classification,
        source_commit=SOURCE,
        analysis_commit=SOURCE,
        output=output,
        scratch_parent=self.root,
        core_summary=core_summary,
    )
    self.assertEqual(result["status"], "FIRST_RED_LOCALIZED")
    self.assertEqual(result["reclassification_gate"], "FIRST_RED_LOCALIZED")
    self.assertEqual(result["source_request_binding_statuses"], [
        "UNIQUE_FUTURE_PREFIX_BINDING"
    ])
    self.assertTrue((output / "SHA256SUMS").is_file())
    self.assertTrue((output / "REMOTE_MULTIROUND_SUMMARY.json").is_file())

  def test_missing_future_evidence_preserves_candidate_set(self):
    bundle, classification, _ = self._bundle(add_binding=False)
    result = review(
        bundle=bundle,
        expected_classification=classification,
        source_commit=SOURCE,
        analysis_commit=SOURCE,
        output=self.root / "return",
        scratch_parent=self.root,
    )
    self.assertEqual(result["status"], "FIRST_RED_CANDIDATE_SET_PRESERVED")
    self.assertEqual(result["source_request_binding_statuses"], ["UNRESOLVED"])

  def test_committed_classification_mismatch_fails_closed(self):
    bundle, classification, _ = self._bundle(add_binding=True)
    classification.write_text("{}\n", encoding="utf-8")
    with self.assertRaisesRegex(
        M15Attempt17ReviewError, "differs from the committed receipt"
    ):
      review(
          bundle=bundle,
          expected_classification=classification,
          source_commit=SOURCE,
          analysis_commit=SOURCE,
          output=self.root / "return",
          scratch_parent=self.root,
      )


if __name__ == "__main__":
  unittest.main()
