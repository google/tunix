#!/usr/bin/env python3
"""End-to-end tests for the one-command P38s22 offsite audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "tasks/p38-pathways-decode-prefill-carrier/scripts"
WRAPPER = SCRIPT_DIR / "run_p38s22_offsite_audit.sh"
FAKE_GCLOUD = ROOT / "tests/p38_serving/fake_gcloud.sh"


def _load_module(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


archive_module = _load_module(
    "p38s22_archive_fixture", SCRIPT_DIR / "p38_evidence_archive.py")


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _byte_diff(left: np.ndarray, right: np.ndarray) -> int:
  return int(np.count_nonzero(
      np.ascontiguousarray(left).view(np.uint8)
      != np.ascontiguousarray(right).view(np.uint8)))


class P38s22OffsiteAuditTest(unittest.TestCase):

  def _fixture(
      self,
      temp: Path,
      *,
      corrupt_receipt: bool = False,
      raw_npz_archive: bool = False,
      staged_pre_alignment_drift: bool = False,
      incident_scope_drift: bool = False,
      orphan_observer_npz: bool = False,
  ) -> tuple[Path, Path]:
    remote = temp / "gcs/test-p38/p38s22/attempt-0"
    remote.mkdir(parents=True)
    source_commit = "e" * 40
    jobset = "canon-p38-test"
    rounds = []
    pre_records = []
    root_payloads: dict[str, bytes] = {}
    run_lines = ["[PATHTRACE] CANON_MM_ALGO on preset=BF16_BF16_F32 (test)"]

    for round_index in range(3):
      a = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
      b = a.copy()
      b[0, round_index] = np.nextafter(
          b[0, round_index], np.float32(np.inf), dtype=np.float32)
      c = b.copy()
      mask = np.ones_like(a, dtype=np.bool_)
      capsule = temp / f"capsule-{round_index}.npz"
      np.savez(
          capsule,
          action_mask=mask,
          s_decode=a,
          s_prefill=b,
          t_old=c,
      )
      capsule_sha = _sha(capsule)
      ab_elements = int(np.count_nonzero(a != b))
      ab_bytes = _byte_diff(a, b)
      max_abs = float(np.max(np.abs(a - b)))
      rounds.append({
          "diagnostic_round": round_index,
          "n_action": 4,
          "a_b_differing_elements": ab_elements,
          "a_b_differing_bytes": ab_bytes,
          "a_b_max_abs": max_abs,
          "b_c_differing_elements": 0,
          "b_c_differing_bytes": 0,
          "capsule_sha256": capsule_sha,
      })
      pre_records.append({
          "diagnostic_round": round_index,
          "N_action": 4,
          "boundaries": {
              "S_decode_vs_S_prefill": {
                  "differing_elements": ab_elements,
                  "differing_bytes": ab_bytes,
                  "max_abs": max_abs,
              },
              "S_prefill_vs_T_old": {
                  "differing_elements": 0,
                  "differing_bytes": 0,
                  "max_abs": 0.0,
              },
          },
      })
      run_lines.append(
          f"[CANON_P38] PRECHECK_ROUND_COMPLETE round={round_index + 1}/3 "
          f"step={round_index} N_action=4 verdict=FAIL "
          f"a_b_differing_bytes={ab_bytes} backward=0 optimizer_commits=0")

      stage = temp / f"stage-{round_index}"
      stage.mkdir()
      shutil.copyfile(capsule, stage / "mismatch-capsule.npz")
      (stage / "run.log").write_text(
          "\n".join(run_lines) + "\n", encoding="utf-8")
      staged_pre_alignment = dict(pre_records[-1])
      if staged_pre_alignment_drift and round_index == 1:
        staged_pre_alignment["staged_only_drift"] = True
      (stage / "pre-alignment.jsonl").write_text(
          json.dumps(staged_pre_alignment) + "\n", encoding="utf-8")
      (stage / "request-journal.jsonl").write_text(json.dumps({
          "schema": "p38-request-journal-v1",
          "request_id": f"request-{round_index}",
      }) + "\n", encoding="utf-8")
      (stage / "incident-ledger.jsonl").write_text(json.dumps({
          "diagnostic_round": (
              round_index + 1
              if incident_scope_drift and round_index == 1 else round_index
          ),
          "schema": "p38-incident-ledger-v1",
      }) + "\n", encoding="utf-8")
      kv_npz = stage / f"p38_kv_observer_{round_index:06d}.npz"
      kv_npz.write_bytes(f"kv-{round_index}\n".encode())
      (stage / f"p38_kv_observer_{round_index:06d}.json").write_text(
          json.dumps({
              "diagnostic_round": round_index,
              "npz_sha256": _sha(kv_npz),
              "schema": "p38-live-kv-prefix-table-v1",
          }) + "\n",
          encoding="utf-8",
      )
      if orphan_observer_npz and round_index == 1:
        (stage / "p38_kv_observer_orphan.npz").write_bytes(b"orphan\n")
      (stage / "ROUND_INVENTORY.json").write_text(json.dumps({
          "schema": "canon-p38-round-stage-v1",
          "diagnostic_round": round_index,
          "incident_records": 1,
          "journal_scope": "cumulative-unscoped",
          "journal_records": 1,
          "kv_records": 1,
          "pre_alignment_records": 1,
          "seam_records": 0,
          "tail_records": 0,
          "terminal_records": 0,
      }) + "\n", encoding="utf-8")
      names = sorted(path.name for path in stage.iterdir())
      (stage / "SHA256SUMS").write_text("".join(
          f"{_sha(stage / name)}  {name}\n" for name in names
      ), encoding="utf-8")
      round_remote = remote / f"rounds/{round_index:06d}"
      round_remote.mkdir(parents=True)
      archive = round_remote / "ROUND_ARCHIVE.tar"
      count, archive_sha = archive_module.create_archive(
          stage, stage / "SHA256SUMS", archive)
      shutil.copyfile(stage / "SHA256SUMS", round_remote / "SHA256SUMS")
      if raw_npz_archive:
        archive.write_bytes(capsule.read_bytes())
        archive_sha = capsule_sha
      marker_archive_sha = capsule_sha if corrupt_receipt else archive_sha
      (round_remote / "ROUND_COMPLETE.json").write_text(json.dumps({
          "archive_name": "ROUND_ARCHIVE.tar",
          "archive_sha256": marker_archive_sha,
          "attempt": "0",
          "diagnostic_round": round_index,
          "logical_file_count": count,
          "manifest_sha256": _sha(stage / "SHA256SUMS"),
          "schema": "canon-p38-round-completion-v1",
          "source_commit": source_commit,
          "status": "sealed-and-verified",
          "transport": "single-deterministic-tar-v1",
      }, sort_keys=True) + "\n", encoding="utf-8")
      round_name = f"mismatch-capsule.round-{round_index:06d}.npz"
      root_payloads[round_name] = capsule.read_bytes()
      if round_index == 2:
        root_payloads["mismatch-capsule.npz"] = capsule.read_bytes()
      else:
        run_lines.append(
            "[CANON_P38] DIAGNOSTIC_ROUND_SKIPPED_UPDATE "
            f"completed={round_index + 1}/3 backward=0 optimizer_commits=0 "
            "weights=frozen next_round=queued"
        )

    run_lines.append(
        "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0")
    root_payloads["run.log"] = ("\n".join(run_lines) + "\n").encode()
    root_payloads["pre-alignment.jsonl"] = (
        "".join(json.dumps(value) + "\n" for value in pre_records).encode())
    root_payloads["serving-classification.json"] = b"{}\n"
    root_payloads["serving-capture.tar"] = b"capture\n"
    for name, payload in root_payloads.items():
      (remote / name).write_bytes(payload)
    root_manifest = remote / "SHA256SUMS"
    root_manifest.write_text("".join(
        f"{_sha(remote / name)}  {name}\n" for name in root_payloads
    ), encoding="utf-8")
    prefix = "gs://test-p38/p38s22/attempt-0"
    (remote / "PREFLIGHT.json").write_text(json.dumps({
        "attempt": "0",
        "prefix": prefix,
        "schema": "canon-p38-gcs-preflight-v1",
        "source_commit": source_commit,
        "status": "writable",
    }) + "\n", encoding="utf-8")
    (remote / "COLLECTED.json").write_text(json.dumps({
        "attempt": "0",
        "jobset": jobset,
        "prefix": prefix,
        "schema": "canon-p38-gcs-collection-v1",
        "source_commit": source_commit,
        "status": "collected",
    }) + "\n", encoding="utf-8")
    (remote / "COMPLETE.json").write_text(json.dumps({
        "attempt": "0",
        "manifest_sha256": _sha(root_manifest),
        "prefix": prefix,
        "schema": "canon-p38-gcs-completion-v1",
        "source_commit": source_commit,
        "status": "postflight-accepted",
    }) + "\n", encoding="utf-8")
    contract = temp / "contract.json"
    contract.write_text(json.dumps({
        "schema": "p38s22-offsite-audit-contract-v1",
        "source_gcs_uri": prefix,
        "expected_source_commit": source_commit,
        "expected_jobset": jobset,
        "expected_attempt": "0",
        "expected_root_manifest_sha256": _sha(root_manifest),
        "expected_mm_algo_preset": "BF16_BF16_F32",
        "expected_rounds": rounds,
        "required_round_members": [
            "ROUND_INVENTORY.json",
            "incident-ledger.jsonl",
            "mismatch-capsule.npz",
            "pre-alignment.jsonl",
            "request-journal.jsonl",
            "run.log",
        ],
        "forbid_terminal_observer": True,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return contract, remote

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
        "CANON_P38S22_AUDIT_ALLOW_DIRTY_FOR_TEST": "1",
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

  def test_valid_remote_bundle_returns_sealed_algorithm_rejection(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, _ = self._fixture(temp)
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      output = temp / "return"
      audit = json.loads((output / "AUDIT.json").read_text())
      self.assertEqual(audit["status"], "PASS")
      self.assertEqual(
          audit["verdict"], "GENERIC_LM_HEAD_ALGORITHM_PRESET_REJECTED")
      self.assertEqual(audit["totals"]["n_action"], 12)
      self.assertEqual(audit["totals"]["a_b_differing_elements"], 3)
      self.assertEqual(audit["totals"]["b_c_differing_elements"], 0)
      self.assertFalse(audit["terminal_classification"]["admitted"])
      self.assertTrue(all(item["present"] for item in audit["returned_receipts"]))
      subprocess.run(
          ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
          cwd=output,
          check=True,
      )

  def test_capsule_sha_masquerading_as_archive_sha_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, _ = self._fixture(temp, corrupt_receipt=True)
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
      output = temp / "return"
      audit = json.loads((output / "AUDIT.json").read_text())
      self.assertEqual(audit["status"], "INCONCLUSIVE")
      self.assertIn("archive SHA receipt differs", audit["failure"])
      subprocess.run(
          ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
          cwd=output,
          check=True,
      )

  def test_missing_root_completion_marker_never_loses_failure_receipt(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, remote = self._fixture(temp)
      (remote / "COMPLETE.json").unlink()
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
      output = temp / "return"
      audit = json.loads((output / "AUDIT.json").read_text())
      self.assertEqual(audit["status"], "INCONCLUSIVE")
      self.assertIn("COMPLETE.json", audit["failure"])
      self.assertIn("GCS copy failed for COMPLETE.json", result.stderr)
      subprocess.run(
          ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
          cwd=output,
          check=True,
      )

  def test_npz_mislabeled_as_round_archive_returns_sealed_inconclusive(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, _ = self._fixture(temp, raw_npz_archive=True)
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
      output = temp / "return"
      audit = json.loads((output / "AUDIT.json").read_text())
      self.assertEqual(audit["status"], "INCONCLUSIVE")
      self.assertTrue(audit["failure"])
      subprocess.run(
          ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
          cwd=output,
          check=True,
      )

  def test_staged_pre_alignment_must_equal_root_record(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, _ = self._fixture(temp, staged_pre_alignment_drift=True)
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
      output = temp / "return"
      audit = json.loads((output / "AUDIT.json").read_text())
      self.assertIn("staged/root pre-alignment records differ", audit["failure"])
      subprocess.run(
          ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
          cwd=output,
          check=True,
      )

  def test_incident_ledger_cannot_cross_round_boundary(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, _ = self._fixture(temp, incident_scope_drift=True)
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
      output = temp / "return"
      audit = json.loads((output / "AUDIT.json").read_text())
      self.assertIn("incident-ledger scope/schema drifted", audit["failure"])
      subprocess.run(
          ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
          cwd=output,
          check=True,
      )

  def test_orphan_observer_npz_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      temp = Path(raw)
      contract, _ = self._fixture(temp, orphan_observer_npz=True)
      result = self._run(temp, contract)
      self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
      output = temp / "return"
      audit = json.loads((output / "AUDIT.json").read_text())
      self.assertIn("observer JSON/NPZ inventory differs", audit["failure"])
      subprocess.run(
          ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
          cwd=output,
          check=True,
      )


if __name__ == "__main__":
  unittest.main()
