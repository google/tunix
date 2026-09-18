#!/usr/bin/env python3
"""Unit tests for Attempt 14 (d33) inventory audit tool."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from audit_m15_attempt14_d33_inventory import (
    EXPECTED_CAMPAIGN,
    EXPECTED_SOURCE,
    Attempt14InventoryError,
    audit,
)


class FakeStorageClient:

  def __init__(self) -> None:
    self.fail_mode: dict[str, str] = {}
    self.files_by_arm: dict[str, list[str]] = {
        "off": ["PREFLIGHT.json", "wide/rounds/000000/ROUND_INPUT_RECEIPT.json", "run.log"],
        "on": ["PREFLIGHT.json", "wide/rounds/000000/ROUND_INPUT_RECEIPT.json", "run.log"],
    }

  @staticmethod
  def _arm(uri: str) -> str:
    return "off" if "-off-" in uri else "on"

  def list_recursive(self, root: str):
    arm = self._arm(root)
    mode = self.fail_mode.get(arm, "PASS")
    if mode == "QUERY_FAILED":
      return {
          "tool": "fake",
          "exit_code": 1,
          "outcome": "QUERY_FAILED",
          "stdout_bytes": 0,
          "stdout_sha256": "0" * 64,
          "stderr_bytes": 10,
          "stderr_sha256": "1" * 64,
          "sanitized_stderr": "AccessDenied: Permission missing",
      }, []
    elif mode == "NOT_FOUND":
      return {
          "tool": "fake",
          "exit_code": 1,
          "outcome": "NOT_FOUND",
          "stdout_bytes": 0,
          "stdout_sha256": "0" * 64,
          "stderr_bytes": 15,
          "stderr_sha256": "2" * 64,
          "sanitized_stderr": "NotFoundException: 404 No objects found",
      }, []
    rows = [f"{root.rstrip('/')}/{f}" for f in self.files_by_arm.get(arm, [])]
    return {
        "tool": "fake",
        "exit_code": 0,
        "outcome": "PASS",
        "stdout_bytes": sum(len(r) for r in rows),
        "stdout_sha256": "3" * 64,
        "stderr_bytes": 0,
        "stderr_sha256": "0" * 64,
        "sanitized_stderr": "",
    }, rows

  def stat_object(self, uri: str):
    arm = self._arm(uri)
    return {
        "tool": "fake",
        "exit_code": 0,
        "outcome": "PASS",
        "size_bytes": 12345,
        "stdout_bytes": 20,
        "stdout_sha256": "a" * 64,
        "stderr_bytes": 0,
        "stderr_sha256": "0" * 64,
        "sanitized_stderr": "",
    }


class FakeKubernetesClient:

  def __init__(self, fail: bool = False) -> None:
    self.fail = fail

  def get_jobset(self, jobset_name: str):
    if self.fail:
      return {
          "status": "NOT_FOUND",
          "exit_code": 1,
          "terminal_state": None,
          "sanitized_stderr": 'Error from server (NotFound): jobsets.jobset.x-k8s.io "..." not found',
      }
    return {
        "status": "PRESENT",
        "exit_code": 0,
        "terminal_state": "Completed",
        "sanitized_stderr": "",
    }


class TestAuditAttempt14Inventory(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir = tempfile.TemporaryDirectory()
    self.root = Path(self.temp_dir.name)
    self.receipt_path = self.root / "RECOVERY_INPUT_RECEIPT.json"
    receipt_data = {
        "schema": "m15-apc-attempt14-recovery-input-v1",
        "source_commit": EXPECTED_SOURCE,
        "campaign_root": EXPECTED_CAMPAIGN,
        "jobsets": {
            "off": f"canon-v1-apc-m15-off-d33-{EXPECTED_SOURCE[:8]}",
            "on": f"canon-v1-apc-m15-on-d33-{EXPECTED_SOURCE[:8]}",
        },
        "status": "LOCATOR_ONLY",
        "submitted_manifest_sha256": "a" * 64,
        "submitted_receipt_sha256": "b" * 64,
    }
    self.receipt_path.write_text(json.dumps(receipt_data), encoding="utf-8")

  def tearDown(self) -> None:
    self.temp_dir.cleanup()

  def test_success(self) -> None:
    out_dir = self.root / "inventory_out"
    summary = audit(
        self.receipt_path,
        out_dir,
        storage_client=FakeStorageClient(),
        k8s_client=FakeKubernetesClient(),
    )
    self.assertEqual(summary["source_commit"], EXPECTED_SOURCE)
    self.assertTrue((out_dir / "SHA256SUMS").is_file())
    self.assertTrue((out_dir / "off.inventory.json").is_file())
    self.assertTrue((out_dir / "on.inventory.json").is_file())
    self.assertTrue((out_dir / "JOBSET_RECEIPTS.json").is_file())

  def test_query_failed_handled(self) -> None:
    storage = FakeStorageClient()
    storage.fail_mode["off"] = "QUERY_FAILED"
    out_dir = self.root / "inventory_fail"
    summary = audit(
        self.receipt_path,
        out_dir,
        storage_client=storage,
        k8s_client=FakeKubernetesClient(),
    )
    self.assertEqual(summary["arms"]["off"]["gcs_query"]["outcome"], "QUERY_FAILED")

  def test_invalid_source_rejected(self) -> None:
    bad_receipt = self.root / "bad_receipt.json"
    bad_receipt.write_text(json.dumps({
        "schema": "m15-apc-attempt14-recovery-input-v1",
        "source_commit": "1234567890123456789012345678901234567890",
        "campaign_root": EXPECTED_CAMPAIGN,
        "jobsets": {"off": "a", "on": "b"},
    }), encoding="utf-8")
    with self.assertRaises(Attempt14InventoryError):
      audit(bad_receipt, self.root / "should_not_exist", storage_client=FakeStorageClient(), k8s_client=FakeKubernetesClient())


if __name__ == "__main__":
  unittest.main()
