#!/usr/bin/env python3
"""End-to-end tests for the P38s22 independent-round salvage audit."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "tasks/p38-pathways-decode-prefill-carrier/scripts"
WRAPPER = SCRIPT_DIR / "run_p38s22_round_salvage.sh"
FAKE_GCLOUD = ROOT / "tests/p38_serving/fake_gcloud.sh"


def _load_module(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


OFFSITE_TEST = _load_module(
    "p38s22_offsite_fixture",
    ROOT / "tests/p38_serving/test_p38s22_offsite_audit.py",
)


class P38s22RoundSalvageTest(unittest.TestCase):

  def _fixture(
      self,
      temp: Path,
      *,
      incident_scope_drift: bool = False,
  ) -> tuple[Path, Path]:
    owner = OFFSITE_TEST.P38s22OffsiteAuditTest(methodName="runTest")
    old_contract, remote = owner._fixture(
        temp, incident_scope_drift=incident_scope_drift)
    contract = json.loads(old_contract.read_text(encoding="utf-8"))
    contract["schema"] = "p38s22-round-salvage-contract-v1"
    contract["root_postflight_required_for_round_verdict"] = False
    for expected in contract["expected_rounds"]:
      round_index = expected["diagnostic_round"]
      marker = json.loads((
          remote / f"rounds/{round_index:06d}/ROUND_COMPLETE.json"
      ).read_text(encoding="utf-8"))
      expected.update({
          "archive_sha256": marker["archive_sha256"],
          "b_c_max_abs": 0.0,
          "logical_file_count": marker["logical_file_count"],
          "manifest_sha256": marker["manifest_sha256"],
      })
    salvage_contract = temp / "salvage-contract.json"
    salvage_contract.write_text(
        json.dumps(contract, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return salvage_contract, remote

  def _run(self, temp: Path, contract: Path) -> subprocess.CompletedProcess[str]:
    fake_bin = temp / "bin"
    fake_bin.mkdir()
    shutil.copyfile(FAKE_GCLOUD, fake_bin / "gcloud")
    (fake_bin / "gcloud").chmod(0o755)
    output = temp / "return"
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "FAKE_GCS_ROOT": str(temp / "gcs"),
        "CANON_P38S22_SALVAGE_ALLOW_DIRTY_FOR_TEST": "1",
    })
    return subprocess.run(
        ["bash", str(WRAPPER), str(contract), str(temp), str(output)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

  @staticmethod
  def _verify_seal(output: Path) -> None:
    subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
        cwd=output,
        check=True,
    )

  def test_three_rounds_pass_without_root_postflight(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, remote = self._fixture(temp)
      for name in ("COLLECTED.json", "COMPLETE.json", "SHA256SUMS"):
        (remote / name).unlink()
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      output = temp / "return"
      audit = json.loads((output / "AUDIT.json").read_text())
      self.assertEqual(audit["status"], "PASS")
      self.assertEqual(
          audit["verdict"],
          "ROUND_SEALED_GENERIC_LM_HEAD_ALGORITHM_PRESET_REJECTED",
      )
      self.assertEqual(audit["totals"]["n_action"], 12)
      self.assertEqual(audit["totals"]["a_b_differing_elements"], 3)
      self.assertEqual(audit["totals"]["b_c_differing_elements"], 0)
      self.assertFalse(audit["root_postflight"]["receipts_present"])
      self.assertFalse(audit["root_postflight"]["admitted"])
      acquisition = (output / "ACQUISITION.jsonl").read_text()
      self.assertNotIn("gs://", acquisition)
      self._verify_seal(output)

  def test_present_root_receipts_do_not_expand_claim(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, _ = self._fixture(temp)
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      audit = json.loads((temp / "return/AUDIT.json").read_text())
      self.assertTrue(audit["root_postflight"]["receipts_present"])
      self.assertFalse(audit["root_postflight"]["admitted"])
      self._verify_seal(temp / "return")

  def test_archive_bit_flip_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, remote = self._fixture(temp)
      archive = remote / "rounds/000001/ROUND_ARCHIVE.tar"
      payload = bytearray(archive.read_bytes())
      payload[len(payload) // 2] ^= 1
      archive.write_bytes(payload)
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
      audit = json.loads((temp / "return/AUDIT.json").read_text())
      self.assertEqual(audit["status"], "INCONCLUSIVE")
      self.assertIn("archive SHA differs", audit["failure"])
      self._verify_seal(temp / "return")

  def test_missing_archive_has_sealed_acquisition_failure(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, remote = self._fixture(temp)
      (remote / "rounds/000002/ROUND_ARCHIVE.tar").unlink()
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
      output = temp / "return"
      audit = json.loads((output / "AUDIT.json").read_text())
      self.assertIn("required source object is unavailable", audit["failure"])
      ledger = [
          json.loads(line)
          for line in (output / "ACQUISITION.jsonl").read_text().splitlines()
      ]
      failed = [
          item for item in ledger
          if item["label"] == "rounds/000002/ROUND_ARCHIVE.tar"
      ]
      self.assertEqual(failed[0]["status"], "missing_or_unreadable")
      self._verify_seal(output)

  def test_cross_round_incident_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, _ = self._fixture(temp, incident_scope_drift=True)
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
      audit = json.loads((temp / "return/AUDIT.json").read_text())
      self.assertIn("incident-ledger scope/schema drifted", audit["failure"])
      self._verify_seal(temp / "return")


if __name__ == "__main__":
  unittest.main()
