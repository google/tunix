#!/usr/bin/env python3
"""Host contracts for the Attempt-13 registered-root inventory audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from audit_m15_attempt13_d32_inventory import (
    ARM_CONTRACTS,
    SOURCE_COMMIT,
    audit,
)


TASK_DIR = Path(__file__).resolve().parents[1]
RECEIPT = TASK_DIR / (
    "evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json"
)


class FakeStorageClient:

  def __init__(self) -> None:
    self.fail_arm: str | None = None
    self.live_arm: str | None = None
    self.missing_member_arm: str | None = None
    self.observed_totals = {"off": 2445, "on": 2188}
    self.list_calls: list[str] = []

  @staticmethod
  def _arm(root: str) -> str:
    return "off" if "-off-" in root else "on"

  @staticmethod
  def _query(exit_code: int, rows: list[str]) -> dict[str, object]:
    stdout = ("\n".join(rows) + ("\n" if rows else "")).encode()
    return {
        "tool": "fake",
        "exit_code": exit_code,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_bytes": 0,
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }

  def list_recursive(self, root: str):
    arm = self._arm(root)
    self.list_calls.append(arm)
    if self.fail_arm == arm:
      return self._query(1, []), []
    rows = [root + "/PREFLIGHT.json"]
    for sequence in range(int(ARM_CONTRACTS[arm]["shards"])):
      members = ["SHARD_ARCHIVE.tar", "SHA256SUMS", "SHARD_COMPLETE.json"]
      if self.missing_member_arm == arm and sequence == 0:
        members.remove("SHARD_ARCHIVE.tar")
      rows.extend(
          root + f"/wide/shards/{sequence:06d}/{name}" for name in members
      )
    if self.live_arm == arm:
      rows.extend((
          root + "/live/000123/LIVE_ARCHIVE.tar",
          root + "/live/000123/SHA256SUMS",
          root + "/live/000123/LIVE.json",
      ))
    return self._query(0, rows), rows

  def copy(self, uri: str, destination: Path):
    arm = self._arm(uri)
    match = re.search(r"/wide/shards/([0-9]{6})/SHARD_COMPLETE\.json$", uri)
    assert match is not None
    sequence = int(match.group(1))
    count = int(ARM_CONTRACTS[arm]["shards"])
    total = int(self.observed_totals[arm])
    record_pairs = total // count + (1 if sequence < total % count else 0)
    completion = {
        "schema": "m15-wide-observer-shard-completion-v1",
        "status": "sealed-uploaded-verified",
        "sequence": sequence,
        "diagnostic_round": 0,
        "record_pairs": record_pairs,
        "payload_bytes": 4096 + sequence,
        "manifest_sha256": "a" * 64,
        "archive_sha256": "b" * 64,
        "expected_source_commit": SOURCE_COMMIT,
        "runtime_source_commit": SOURCE_COMMIT,
    }
    destination.write_text(json.dumps(completion), encoding="utf-8")
    return self._query(0, [])


class Attempt13InventoryAuditTest(unittest.TestCase):

  def setUp(self) -> None:
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)
    self.client = FakeStorageClient()

  def tearDown(self) -> None:
    self.holder.cleanup()

  def _audit(self):
    return audit(
        receipt_path=RECEIPT,
        output=self.root / "return",
        scratch=self.root / "scratch",
        client=self.client,
    )

  def test_successful_recursive_inventory_proves_both_live_roots_absent(self) -> None:
    result = self._audit()
    self.assertEqual(result["decision"], "D32_LIVE_ABSENT_WITH_COUNT_DRIFT")
    self.assertEqual(result["count_contract_status"], "DRIFT")
    self.assertTrue(result["d33_preparation_eligible"])
    self.assertFalse(result["d33_launch_authorized"])
    self.assertEqual(self.client.list_calls, ["off", "on"])
    self.assertEqual(result["arms"]["off"]["shard_record_pairs"], 2445)
    self.assertEqual(result["arms"]["off"]["receipt_seam_records"], 2474)
    self.assertEqual(result["arms"]["off"]["record_count_delta"], -29)
    self.assertEqual(result["arms"]["on"]["shard_record_pairs"], 2188)
    self.assertEqual(result["arms"]["on"]["receipt_seam_records"], 2087)
    self.assertEqual(result["arms"]["on"]["record_count_delta"], 101)
    output = self.root / "return"
    self.assertEqual(len(list(output.iterdir())), 7)
    self.assertNotIn("gs://", "".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
    ))
    for row in (output / "SHA256SUMS").read_text().splitlines():
      digest, name = row.split("  ", 1)
      self.assertEqual(
          hashlib.sha256((output / name).read_bytes()).hexdigest(), digest
      )

  def test_successful_inventory_with_live_objects_does_not_claim_absence(self) -> None:
    self.client.live_arm = "on"
    result = self._audit()
    self.assertEqual(
        result["decision"], "D32_LIVE_PRESENT_WITH_COUNT_DRIFT"
    )
    self.assertFalse(result["arms"]["on"]["live_absence_proven"])

  def test_query_failure_is_red_and_other_arm_is_still_audited(self) -> None:
    self.client.fail_arm = "off"
    result = self._audit()
    self.assertEqual(result["decision"], "D32_INVENTORY_AUDIT_RED")
    self.assertEqual(result["arms"]["off"]["failure"],
                     "RECURSIVE_LIST_QUERY_FAILED")
    self.assertEqual(self.client.list_calls, ["off", "on"])
    self.assertFalse(result["arms"]["off"]["live_absence_proven"])

  def test_missing_flat_shard_object_is_red(self) -> None:
    self.client.missing_member_arm = "on"
    result = self._audit()
    self.assertEqual(result["status"], "RED")
    self.assertIn("triples", result["arms"]["on"]["failure"])

  def test_matching_metrics_are_reported_without_moving_the_gate(self) -> None:
    self.client.observed_totals = {"off": 2474, "on": 2087}
    result = self._audit()
    self.assertEqual(result["decision"], "D32_LIVE_ABSENT_COUNTS_MATCH")
    self.assertEqual(result["count_contract_status"], "MATCH")
    self.assertEqual(result["arms"]["off"]["record_count_delta"], 0)
    self.assertEqual(result["arms"]["on"]["record_count_delta"], 0)


if __name__ == "__main__":
  unittest.main()
