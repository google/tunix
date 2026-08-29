#!/usr/bin/env python3
"""Host tests for the Attempt-17 offline candidate review."""

from __future__ import annotations

import hashlib
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
SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent


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
    self.assertEqual(result["decision_scope"], "COMPLETION_POSITION_ZERO")
    self.assertFalse(result["numerical_repair_authorized"])
    self.assertTrue(result["pinned_exact_image_required"])
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

  def test_d3e_wrapper_is_read_only_and_delegates_verified_recovery(self):
    wrapper = (
        SCRIPT_DIR / "run_m15_attempt17_d3e_canonical_action.sh"
    ).read_text(encoding="utf-8")
    self.assertIn("run_m15_attempt17_d36_offline_binding.sh", wrapper)
    self.assertIn("M15_D3E_CANONICAL_ACTION_REVIEW_PASS", wrapper)
    self.assertIn("gcs_write=0 kubernetes=0 tpu=0", wrapper)
    self.assertNotIn("kubectl", wrapper)
    self.assertNotIn("gcs_write=1", wrapper)

  def test_committed_d3d_return_requires_canonical_action_scope(self):
    evidence = (
        TASK_DIR / "evidence" /
        "v1_apc_m15_attempt17_d36_offline_binding_20260829"
    )
    for line in (evidence / "SHA256SUMS").read_text(
        encoding="ascii"
    ).splitlines():
      digest, separator, name = line.partition("  ")
      self.assertEqual(separator, "  ")
      self.assertEqual(
          hashlib.sha256((evidence / name).read_bytes()).hexdigest(), digest
      )
    classification = json.loads(
        (evidence / "D36_RECLASSIFICATION.json").read_text(encoding="utf-8")
    )
    self.assertEqual(classification["gate"], "FIRST_RED_CANDIDATE_SET")
    self.assertEqual(classification["alignment"]["a_b_differing_bytes"], 207)
    self.assertEqual(classification["alignment"]["b_c_differing_bytes"], 0)
    self.assertEqual(classification["coverage"]["total_red_points"], 95)
    self.assertEqual(
        classification["coverage"]["first_action_joinable_red_points"], 1
    )
    self.assertEqual(classification["coverage"]["candidate_anchors"], 7)
    self.assertEqual(classification["coverage"]["unobserved_red_points"], 88)
    self.assertEqual(
        classification["first_difference_signatures"],
        [
            {"layer": 0, "checkpoint": "rpa_output"},
            {"layer": None, "checkpoint": "final_norm"},
        ],
    )
    self.assertEqual(len(classification["anchors"]), 1)
    anchor = classification["anchors"][0]
    self.assertEqual(anchor["completion_position"], 0)
    self.assertEqual(anchor["first_difference"]["checkpoint"], "rpa_output")
    self.assertEqual(
        anchor["source_request_binding"]["status"],
        "UNIQUE_FUTURE_PREFIX_BINDING",
    )


if __name__ == "__main__":
  unittest.main()
