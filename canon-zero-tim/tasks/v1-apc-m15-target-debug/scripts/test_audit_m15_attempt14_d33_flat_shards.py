#!/usr/bin/env python3
"""Comprehensive unit tests for Attempt 14 (d33) flat-shard content audit tool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_m15_attempt14_d33_flat_shards import (
    EXPECTED_CAMPAIGN,
    EXPECTED_SHARD_COUNTS,
    EXPECTED_SOURCE,
    Attempt14FlatShardAuditError,
    audit_flat_shards,
)


def _sha_text(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeStorageClient:

  def __init__(self) -> None:
    self.fail_mode: dict[str, str] = {}
    self.objects_by_arm: dict[str, list[str]] = {}
    self.file_contents: dict[str, str] = {}

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
    rows = [f"{root.rstrip('/')}/{f}" for f in self.objects_by_arm.get(arm, [])]
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

  def copy_file(self, uri: str, destination: Path):
    arm = self._arm(uri)
    mode = self.fail_mode.get(arm, "PASS")
    if mode == "COPY_FAILED":
      return {
          "tool": "fake",
          "exit_code": 1,
          "outcome": "COPY_FAILED",
          "stderr_bytes": 10,
          "sanitized_stderr": "Download error",
      }
    content = self.file_contents.get(uri)
    if content is None:
      return {
          "tool": "fake",
          "exit_code": 1,
          "outcome": "NOT_FOUND",
          "stderr_bytes": 10,
          "sanitized_stderr": "Object not found in fake storage",
      }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return {
        "tool": "fake",
        "exit_code": 0,
        "outcome": "PASS",
        "stdout_bytes": len(content),
        "stderr_bytes": 0,
        "sanitized_stderr": "",
    }


class TestAuditM15Attempt14D33FlatShards(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir = tempfile.TemporaryDirectory()
    self.root = Path(self.temp_dir.name)
    self.recovery_receipt = self.root / "RECOVERY_INPUT_RECEIPT.json"
    self.output_dir = self.root / "output_evidence"

    self.receipt_data = {
        "campaign_root": EXPECTED_CAMPAIGN,
        "claim_ceiling": "Locator only",
        "jobsets": {
            "off": "canon-v1-apc-m15-off-d33-003276a3",
            "on": "canon-v1-apc-m15-on-d33-003276a3",
        },
        "schema": "m15-apc-attempt14-recovery-input-v1",
        "source_commit": EXPECTED_SOURCE,
        "status": "LOCATOR_ONLY",
        "submitted_manifest_sha256": "0" * 64,
        "submitted_receipt_sha256": "1" * 64,
    }
    self.recovery_receipt.write_text(json.dumps(self.receipt_data), encoding="utf-8")
    self.storage = FakeStorageClient()
    self._populate_valid_fake_shards()

  def tearDown(self) -> None:
    self.temp_dir.cleanup()

  def _populate_valid_fake_shards(self, rounds_fn=lambda seq, arm: 0) -> None:
    self.storage.objects_by_arm = {}
    self.storage.file_contents = {}

    for arm in ("off", "on"):
      jobset = self.receipt_data["jobsets"][arm]
      gcs_root = f"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{jobset}/attempt-0"
      count = EXPECTED_SHARD_COUNTS[arm]
      obj_list = ["PREFLIGHT.json"]

      for seq in range(count):
        seq_str = f"{seq:06d}"
        obj_list.extend([
            f"wide/shards/{seq_str}/SHA256SUMS",
            f"wide/shards/{seq_str}/SHARD_ARCHIVE.tar",
            f"wide/shards/{seq_str}/SHARD_COMPLETE.json",
        ])

        sums_text = (
            f"05c7b3096b8e656fbecf880fce029e7ce0d1503811b32913a67cc53bfc32b15a  SHARD_INVENTORY.json\n"
            f"cdbadb55bfeeca29b54c41f7c2d44ac087ed210d8bef434129d53c880375d29e  p38_seam_{seq_str}.json\n"
            f"064d74addcec01857598d7a82792c015628f1b1522a8a5eea9eca566fd639cdb  p38_seam_{seq_str}.npz\n"
        )
        manifest_sha = _sha_text(sums_text)

        comp_data = {
            "archive_sha256": "f" * 64,
            "claim_ceiling": "INCONCLUSIVE_PARTIAL_LIVE_EVIDENCE_UNTIL_WIDE_ROUND_COMPLETE",
            "diagnostic_round": rounds_fn(seq, arm),
            "expected_source_commit": EXPECTED_SOURCE,
            "manifest_sha256": manifest_sha,
            "payload_bytes": 1000 + seq,
            "record_pairs": 5 + (seq % 10),
            "runtime_source_commit": EXPECTED_SOURCE,
            "schema": "m15-wide-observer-shard-completion-v1",
            "sequence": seq,
            "status": "sealed-uploaded-verified",
        }

        self.storage.file_contents[f"{gcs_root}/wide/shards/{seq_str}/SHA256SUMS"] = sums_text
        self.storage.file_contents[f"{gcs_root}/wide/shards/{seq_str}/SHARD_COMPLETE.json"] = json.dumps(comp_data)

      self.storage.objects_by_arm[arm] = obj_list

  def test_audit_round0_only_pass(self) -> None:
    summary = audit_flat_shards(
        recovery_receipt_path=self.recovery_receipt,
        output_dir=self.output_dir,
        storage_client=self.storage,
    )
    self.assertEqual(summary["decision"], "D33_FLAT_SHARDS_ROUND0_ONLY")
    self.assertEqual(summary["all_rounds_observed"], [0])
    self.assertEqual(summary["arms"]["off"]["total_shards"], 88)
    self.assertEqual(summary["arms"]["on"]["total_shards"], 74)
    self.assertTrue((self.output_dir / "SHA256SUMS").is_file())
    self.assertTrue((self.output_dir / "FLAT_SHARD_AUDIT_SUMMARY.json").is_file())

  def test_audit_three_rounds_verified_pass(self) -> None:
    def three_rounds(seq: int, arm: str) -> int:
      if seq < 20:
        return 0
      elif seq < 40:
        return 1
      else:
        return 2

    self._populate_valid_fake_shards(rounds_fn=three_rounds)
    summary = audit_flat_shards(
        recovery_receipt_path=self.recovery_receipt,
        output_dir=self.output_dir,
        storage_client=self.storage,
    )
    self.assertEqual(summary["decision"], "D33_FLAT_SHARDS_THREE_ROUNDS_VERIFIED")
    self.assertEqual(summary["all_rounds_observed"], [0, 1, 2])

  def test_reject_query_failed(self) -> None:
    self.storage.fail_mode["off"] = "QUERY_FAILED"
    with self.assertRaises(Attempt14FlatShardAuditError) as ctx:
      audit_flat_shards(
          recovery_receipt_path=self.recovery_receipt,
          output_dir=self.output_dir,
          storage_client=self.storage,
      )
    self.assertIn("GCS recursive list failed", str(ctx.exception))

  def test_reject_missing_triple_member(self) -> None:
    # Remove tar member from shard 000005
    self.storage.objects_by_arm["off"] = [
        o for o in self.storage.objects_by_arm["off"]
        if o != "wide/shards/000005/SHARD_ARCHIVE.tar"
    ]
    with self.assertRaises(Attempt14FlatShardAuditError) as ctx:
      audit_flat_shards(
          recovery_receipt_path=self.recovery_receipt,
          output_dir=self.output_dir,
          storage_client=self.storage,
      )
    self.assertIn("incomplete triple members", str(ctx.exception))

  def test_reject_non_contiguous_sequence(self) -> None:
    # Remove entire shard 000010
    self.storage.objects_by_arm["off"] = [
        o for o in self.storage.objects_by_arm["off"]
        if not o.startswith("wide/shards/000010/")
    ]
    with self.assertRaises(Attempt14FlatShardAuditError) as ctx:
      audit_flat_shards(
          recovery_receipt_path=self.recovery_receipt,
          output_dir=self.output_dir,
          storage_client=self.storage,
      )
    self.assertIn("sequences not contiguous", str(ctx.exception))

  def test_reject_duplicate_sequence_in_listing(self) -> None:
    self.storage.objects_by_arm["off"].append("wide/shards/000000/SHARD_COMPLETE.json")
    with self.assertRaises(Attempt14FlatShardAuditError) as ctx:
      audit_flat_shards(
          recovery_receipt_path=self.recovery_receipt,
          output_dir=self.output_dir,
          storage_client=self.storage,
      )
    self.assertIn("duplicate objects in listing", str(ctx.exception))

  def test_reject_source_mismatch(self) -> None:
    jobset = self.receipt_data["jobsets"]["off"]
    gcs_root = f"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{jobset}/attempt-0"
    comp_uri = f"{gcs_root}/wide/shards/000002/SHARD_COMPLETE.json"
    comp_data = json.loads(self.storage.file_contents[comp_uri])
    comp_data["runtime_source_commit"] = "bad" * 13 + "0"
    self.storage.file_contents[comp_uri] = json.dumps(comp_data)

    with self.assertRaises(Attempt14FlatShardAuditError) as ctx:
      audit_flat_shards(
          recovery_receipt_path=self.recovery_receipt,
          output_dir=self.output_dir,
          storage_client=self.storage,
      )
    self.assertIn("source commit mismatch", str(ctx.exception))

  def test_reject_manifest_mismatch(self) -> None:
    jobset = self.receipt_data["jobsets"]["off"]
    gcs_root = f"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{jobset}/attempt-0"
    comp_uri = f"{gcs_root}/wide/shards/000003/SHARD_COMPLETE.json"
    comp_data = json.loads(self.storage.file_contents[comp_uri])
    comp_data["manifest_sha256"] = "e" * 64
    self.storage.file_contents[comp_uri] = json.dumps(comp_data)

    with self.assertRaises(Attempt14FlatShardAuditError) as ctx:
      audit_flat_shards(
          recovery_receipt_path=self.recovery_receipt,
          output_dir=self.output_dir,
          storage_client=self.storage,
      )
    self.assertIn("manifest SHA mismatch", str(ctx.exception))

  def test_reject_archive_digest_mismatch(self) -> None:
    jobset = self.receipt_data["jobsets"]["on"]
    gcs_root = f"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{jobset}/attempt-0"
    comp_uri = f"{gcs_root}/wide/shards/000001/SHARD_COMPLETE.json"
    comp_data = json.loads(self.storage.file_contents[comp_uri])
    comp_data["archive_sha256"] = "not-a-valid-sha"
    self.storage.file_contents[comp_uri] = json.dumps(comp_data)

    with self.assertRaises(Attempt14FlatShardAuditError) as ctx:
      audit_flat_shards(
          recovery_receipt_path=self.recovery_receipt,
          output_dir=self.output_dir,
          storage_client=self.storage,
      )
    self.assertIn("malformed archive_sha256", str(ctx.exception))

  def test_reject_invalid_round(self) -> None:
    jobset = self.receipt_data["jobsets"]["off"]
    gcs_root = f"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{jobset}/attempt-0"
    comp_uri = f"{gcs_root}/wide/shards/000004/SHARD_COMPLETE.json"
    comp_data = json.loads(self.storage.file_contents[comp_uri])
    comp_data["diagnostic_round"] = "not_a_round"
    self.storage.file_contents[comp_uri] = json.dumps(comp_data)

    with self.assertRaises(Attempt14FlatShardAuditError) as ctx:
      audit_flat_shards(
          recovery_receipt_path=self.recovery_receipt,
          output_dir=self.output_dir,
          storage_client=self.storage,
      )
    self.assertIn("non-integer diagnostic_round", str(ctx.exception))

  def test_reject_one_bit_mutation_in_sums(self) -> None:
    jobset = self.receipt_data["jobsets"]["off"]
    gcs_root = f"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{jobset}/attempt-0"
    sums_uri = f"{gcs_root}/wide/shards/000005/SHA256SUMS"
    self.storage.file_contents[sums_uri] += "# mutated\n"

    with self.assertRaises(Attempt14FlatShardAuditError) as ctx:
      audit_flat_shards(
          recovery_receipt_path=self.recovery_receipt,
          output_dir=self.output_dir,
          storage_client=self.storage,
      )
    self.assertIn("manifest SHA mismatch", str(ctx.exception))


if __name__ == "__main__":
  unittest.main()
