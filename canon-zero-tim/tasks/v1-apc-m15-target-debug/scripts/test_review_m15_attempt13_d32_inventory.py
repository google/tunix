#!/usr/bin/env python3
"""Contracts for the offline Attempt-13 inventory review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from review_m15_attempt13_d32_inventory import (
    Attempt13ReviewError,
    _write_return,
    review,
)


TASK_DIR = Path(__file__).resolve().parents[1]
INVENTORY = TASK_DIR / (
    "evidence/v1_apc_m15_attempt13_d32_inventory_20260828"
)
RECEIPT = TASK_DIR / (
    "evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json"
)


class Attempt13InventoryReviewTest(unittest.TestCase):

  def test_checked_in_return_preserves_count_drift(self) -> None:
    result = review(inventory_root=INVENTORY, receipt_path=RECEIPT)
    self.assertEqual(result["decision"], "D32_LIVE_ABSENT_WITH_COUNT_DRIFT")
    self.assertEqual(result["count_contract_status"], "DRIFT")
    self.assertEqual(result["arms"]["off"]["record_count_delta"], -29)
    self.assertEqual(result["arms"]["on"]["record_count_delta"], 101)
    self.assertTrue(result["d33_preparation_eligible"])
    self.assertFalse(result["d33_launch_authorized"])
    self.assertFalse(result["numerical_repair_authorized"])

  def test_review_return_is_self_hashed(self) -> None:
    result = review(inventory_root=INVENTORY, receipt_path=RECEIPT)
    with tempfile.TemporaryDirectory() as holder:
      output = Path(holder) / "return"
      _write_return(output, result)
      self.assertEqual(len(list(output.iterdir())), 3)
      for row in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = row.split("  ", 1)
        self.assertEqual(
            hashlib.sha256((output / name).read_bytes()).hexdigest(), digest
        )

  def test_tampered_inventory_member_is_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as holder:
      copied = Path(holder) / "inventory"
      shutil.copytree(INVENTORY, copied)
      with (copied / "off.objects.txt").open("a", encoding="utf-8") as stream:
        stream.write("live/forged/LIVE.json\n")
      with self.assertRaisesRegex(Attempt13ReviewError, "failed SHA"):
        review(inventory_root=copied, receipt_path=RECEIPT)

  def test_rehashed_live_object_cannot_pass_the_no_live_gate(self) -> None:
    with tempfile.TemporaryDirectory() as holder:
      copied = Path(holder) / "inventory"
      shutil.copytree(INVENTORY, copied)
      objects = (copied / "off.objects.txt").read_text().splitlines()
      objects.append("live/000001/LIVE.json")
      (copied / "off.objects.txt").write_text(
          "\n".join(sorted(objects)) + "\n", encoding="utf-8"
      )
      source = json.loads((copied / "D32_INVENTORY.json").read_text())
      source["arms"]["off"]["live_objects"] = 1
      source["arms"]["off"]["live_absence_proven"] = False
      (copied / "D32_INVENTORY.json").write_text(
          json.dumps(source, sort_keys=True, indent=2) + "\n", encoding="utf-8"
      )
      names = []
      for row in (copied / "SHA256SUMS").read_text().splitlines():
        _, name = row.split("  ", 1)
        names.append(name)
      (copied / "SHA256SUMS").write_text(
          "".join(
              f"{hashlib.sha256((copied / name).read_bytes()).hexdigest()}  {name}\n"
              for name in names
          ),
          encoding="ascii",
      )
      with self.assertRaisesRegex(Attempt13ReviewError, "exactly PREFLIGHT"):
        review(inventory_root=copied, receipt_path=RECEIPT)


if __name__ == "__main__":
  unittest.main()
