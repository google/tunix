#!/usr/bin/env python3
"""Tests for the immutable d33 recovery locator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from prepare_m15_attempt14_d33_recovery_contract import (
    RecoveryContractError,
    build,
)


class D33RecoveryContractTest(unittest.TestCase):

  def setUp(self) -> None:
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)
    self.evidence = Path(__file__).parents[1] / (
        "evidence/v1_apc_m15_attempt14_paired_d33_20260828"
    )

  def tearDown(self) -> None:
    self.holder.cleanup()

  def test_committed_receipt_builds_exact_locator(self) -> None:
    output = self.root / "contract"
    result = build(self.evidence, output)
    self.assertEqual(result["status"], "LOCATOR_ONLY")
    self.assertEqual(set(result["jobsets"]), {"off", "on"})
    for arm in ("off", "on"):
      path = output / f"jobset-v1-apc-m15-{arm}-full.yaml"
      document = yaml.safe_load(path.read_text(encoding="utf-8"))
      self.assertEqual(document["metadata"]["name"], result["jobsets"][arm])
    self.assertNotIn(
        "gs://",
        (output / "RECOVERY_INPUT_RECEIPT.json").read_text(encoding="utf-8"),
    )

  def test_tampered_manifest_is_rejected(self) -> None:
    copied = self.root / "evidence"
    copied.mkdir()
    for path in self.evidence.iterdir():
      if path.is_file():
        (copied / path.name).write_bytes(path.read_bytes())
    (copied / "receipt.json").write_text("{}\n", encoding="utf-8")
    with self.assertRaisesRegex(RecoveryContractError, "submitted SHA failed"):
      build(copied, self.root / "contract")

  def test_wrong_jobset_is_rejected_even_with_rehashed_receipt(self) -> None:
    copied = self.root / "evidence"
    copied.mkdir()
    values = {}
    for path in self.evidence.iterdir():
      if path.is_file() and path.name != "SHA256SUMS":
        target = copied / path.name
        target.write_bytes(path.read_bytes())
        values[path.name] = target
    receipt = json.loads(values["receipt.json"].read_text(encoding="utf-8"))
    receipt["control_arm_off"]["jobset_name"] = "wrong"
    values["receipt.json"].write_text(json.dumps(receipt), encoding="utf-8")
    import hashlib
    (copied / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(values[name].read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(values)
    ), encoding="ascii")
    with self.assertRaisesRegex(RecoveryContractError, "JobSet identity drifted"):
      build(copied, self.root / "contract")


if __name__ == "__main__":
  unittest.main()
