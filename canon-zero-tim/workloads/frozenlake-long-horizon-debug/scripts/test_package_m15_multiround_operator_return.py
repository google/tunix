#!/usr/bin/env python3
"""Host positives and negatives for the complete M15 operator return."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from package_m15_multiround_operator_return import OperatorReturnError, package


SOURCE = "a" * 40


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class OperatorReturnTest(unittest.TestCase):

  def setUp(self) -> None:
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)
    self.render = self.root / "render"
    self.core = self.root / "core"
    self.jobsets = self.root / "jobsets"
    self.logs = self.root / "logs"
    for path in (self.render, self.core, self.jobsets, self.logs):
      path.mkdir()
    for arm in ("off", "on"):
      jobset = f"canon-v1-apc-m15-{arm}-test-aaaaaaaa"
      document = {
          "metadata": {"name": jobset},
          "spec": {"replicatedJobs": [{"template": {"spec": {"template": {
              "spec": {"containers": [{"env": [
                  {"name": "CANON_APC_M15_TARGET_DEBUG", "value": arm},
                  {"name": "CANON_EXPECT_COMMIT", "value": SOURCE},
                  {"name": "CANON_P38_GCS_PREFIX", "value": (
                      "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
                      f"{jobset}/attempt-0"
                  )},
                  {"name": "CANON_P38_DIAGNOSTIC_ROUNDS", "value": "3"},
                  {"name": "CANON_P38_SEAM_OBSERVER", "value": "full"},
                  {"name": "CANON_P38_SEAM_LAYER", "value": "0"},
                  {"name": "CANON_P33_RUN_STAGE", "value": "backward-no-commit"},
                  {"name": "CANON_P33_NO_COMMIT", "value": "1"},
              ]}]}
          }}}}]},
      }
      (self.render / f"jobset-v1-apc-m15-{arm}-full.yaml").write_text(
          yaml.safe_dump(document), encoding="utf-8"
      )
      (self.jobsets / f"{arm}.json").write_text(json.dumps({
          "schema": "m15-apc-jobset-status-v1",
          "arm": arm,
          "source_commit": SOURCE,
          "jobset": jobset,
          "query_status": "PASS",
          "query_exit_code": 0,
          "terminal_condition": "Failed",
          "conditions": [],
      }), encoding="utf-8")
      (self.logs / f"{arm}.json").write_text(json.dumps({
          "schema": "m15-apc-raw-log-receipt-v1",
          "arm": arm,
          "source_commit": SOURCE,
          "jobset": jobset,
          "status": "PRESENT",
          "object_identity": f"{jobset}/attempt-0/run.log",
          "sha256": "b" * 64,
          "bytes": 123,
          "payload_returned": False,
      }), encoding="utf-8")
    (self.core / "MULTIROUND_SUMMARY.json").write_text(json.dumps({
        "schema": "m15-apc-multiround-small-return-v1",
        "status": "COMPLETE",
        "source_commit": SOURCE,
    }), encoding="utf-8")
    (self.core / "PACKAGING.txt").write_text("core\n", encoding="utf-8")
    (self.core / "SHA256SUMS").write_text(
        f"{_sha(self.core / 'MULTIROUND_SUMMARY.json')}  MULTIROUND_SUMMARY.json\n"
        f"{_sha(self.core / 'PACKAGING.txt')}  PACKAGING.txt\n",
        encoding="ascii",
    )

  def tearDown(self) -> None:
    self.holder.cleanup()

  def _package(self, name: str = "return") -> dict:
    return package(
        render_dir=self.render,
        core_return=self.core,
        jobset_receipts=self.jobsets,
        raw_log_receipts=self.logs,
        output=self.root / name,
    )

  def test_complete_return_is_self_hashed(self) -> None:
    result = self._package()
    self.assertEqual(result["status"], "COMPLETE")
    output = self.root / "return"
    manifest = (output / "SHA256SUMS").read_text(encoding="ascii")
    self.assertIn("JOBSET_STATUS.json", manifest)
    self.assertIn("RAW_LOG_RECEIPTS.json", manifest)
    self.assertFalse(any(path.name == "run.log" for path in output.iterdir()))

  def test_nonterminal_jobset_is_preserved_as_incomplete(self) -> None:
    path = self.jobsets / "on.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["terminal_condition"] = None
    path.write_text(json.dumps(value), encoding="utf-8")
    result = self._package()
    self.assertEqual(result["status"], "COMPLETE_OPERATOR_RECEIPTS_INCOMPLETE")

  def test_wrong_jobset_identity_is_rejected(self) -> None:
    path = self.jobsets / "off.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["jobset"] = "wrong"
    path.write_text(json.dumps(value), encoding="utf-8")
    with self.assertRaisesRegex(OperatorReturnError, "JobSet drifted"):
      self._package()

  def test_tampered_core_is_rejected(self) -> None:
    (self.core / "PACKAGING.txt").write_text("tampered\n", encoding="utf-8")
    with self.assertRaisesRegex(OperatorReturnError, "core return SHA failed"):
      self._package()

  def test_recovery_input_is_bound_and_returned(self) -> None:
    receipt = {
        "schema": "m15-apc-attempt14-recovery-input-v1",
        "status": "LOCATOR_ONLY",
        "source_commit": SOURCE,
        "submitted_manifest_sha256": "c" * 64,
        "submitted_receipt_sha256": "d" * 64,
        "jobsets": {
            arm: f"canon-v1-apc-m15-{arm}-test-aaaaaaaa"
            for arm in ("off", "on")
        },
    }
    (self.render / "RECOVERY_INPUT_RECEIPT.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    result = self._package()
    self.assertTrue(result["recovery_input_bound"])
    self.assertTrue((self.root / "return" / "RECOVERY_INPUT_RECEIPT.json").is_file())

  def test_recovery_input_wrong_source_is_rejected(self) -> None:
    (self.render / "RECOVERY_INPUT_RECEIPT.json").write_text(json.dumps({
        "schema": "m15-apc-attempt14-recovery-input-v1",
        "status": "LOCATOR_ONLY",
        "source_commit": "f" * 40,
        "submitted_manifest_sha256": "c" * 64,
        "submitted_receipt_sha256": "d" * 64,
        "jobsets": {
            arm: f"canon-v1-apc-m15-{arm}-test-aaaaaaaa"
            for arm in ("off", "on")
        },
    }), encoding="utf-8")
    with self.assertRaisesRegex(OperatorReturnError, "recovery input source"):
      self._package()

  def _fake_remote_round(self, remote: Path, arm: str, round_index: int) -> None:
    root = remote / "wide" / "rounds" / f"{round_index:06d}"
    root.mkdir(parents=True)
    red = arm == "on" and round_index == 1
    classification_name = (
        "M15_INTERNAL_FIRST_RED_LOCALIZED"
        if red else (
            "M15_OBSERVER_CONTROL_EXACT"
            if arm == "off" else "M15_OBSERVER_TREATMENT_EXACT"
        )
    )
    receipt = {
        "schema": "m15-wide-sealed-input-v1",
        "status": "PASS",
        "diagnostic_round": round_index,
        "expected_source_commit": SOURCE,
        "runtime_source_commit": SOURCE,
        "record_pairs": 2,
        "replay_records": 2,
        "shards": [{"sequence": round_index}],
    }
    classification = {
        "schema": "m15-apc-wide-seam-classification-v1",
        "status": "PASS",
        "arm": arm,
        "diagnostic_round": round_index,
        "classification": classification_name,
        "alignment": {
            "a_b_differing_bytes": 7 if red else 0,
            "b_c_differing_bytes": 0,
        },
    }
    (root / "ROUND_INPUT_RECEIPT.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    (root / "p38_seam.classification.json").write_text(
        json.dumps(classification), encoding="utf-8"
    )
    bundle = root / "m15_wide_seam_bundle.tar"
    bundle.write_bytes(b"not-returned")
    manifest = root / "WIDE_SHA256SUMS"
    manifest.write_text(
        f"{_sha(root / 'ROUND_INPUT_RECEIPT.json')}  ROUND_INPUT_RECEIPT.json\n"
        f"{_sha(root / 'p38_seam.classification.json')}  p38_seam.classification.json\n"
        f"{_sha(bundle)}  m15_wide_seam_bundle.tar\n",
        encoding="ascii",
    )
    (root / "WIDE_ROUND_COMPLETE.json").write_text(json.dumps({
        "schema": "m15-wide-round-completion-v1",
        "status": "classified-and-uploaded",
        "diagnostic_round": round_index,
        "expected_source_commit": SOURCE,
        "runtime_source_commit": SOURCE,
        "classification": classification_name,
        "manifest_sha256": _sha(manifest),
        "record_pairs": 2,
        "shards": receipt["shards"],
    }), encoding="utf-8")

  def test_shell_wrapper_builds_complete_sanitized_return(self) -> None:
    fake_gcs = self.root / "fake-gcs"
    for arm in ("off", "on"):
      jobset = f"canon-v1-apc-m15-{arm}-test-aaaaaaaa"
      remote = fake_gcs / "canon-zero-tim" / "evidence" / "p38" / jobset / "attempt-0"
      remote.mkdir(parents=True)
      for round_index in range(3):
        self._fake_remote_round(remote, arm, round_index)
      for name in ("PREFLIGHT.json", "COLLECTED.json", "COMPLETE.json"):
        (remote / name).write_text(json.dumps({
            "source_commit": SOURCE,
            "status": "PASS",
        }), encoding="utf-8")
      (remote / "run.log").write_text("complete raw log\n", encoding="utf-8")
      (remote / "SHA256SUMS").write_text(
          f"{_sha(remote / 'run.log')}  run.log\n", encoding="ascii"
      )

    fake_bin = self.root / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text("""#!/usr/bin/env python3
import os, pathlib, shutil, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ["FAKE_GCS_ROOT"])
def local(uri):
  prefix = "gs://yuxzhang-tunix-models/"
  if not uri.startswith(prefix):
    raise SystemExit(2)
  return root / uri[len(prefix):]
if args[:2] == ["storage", "ls"]:
  raise SystemExit(0 if local(args[2]).exists() else 1)
if args[:2] == ["storage", "cp"]:
  shutil.copyfile(local(args[2]), args[3])
  raise SystemExit(0)
if args[:3] == ["storage", "objects", "describe"]:
  print(local(args[3]).stat().st_size)
  raise SystemExit(0)
raise SystemExit(2)
""", encoding="utf-8")
    gcloud.chmod(0o755)
    kubectl = fake_bin / "kubectl"
    kubectl.write_text("""#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
name = args[args.index("default") + 1]
arm = "off" if "-off-" in name else "on"
print(json.dumps({
  "metadata": {
    "name": name,
    "uid": "uid-" + arm,
    "generation": 1,
    "labels": {
      "canon.zero-tim/apc-m15-arm": arm,
      "canon.zero-tim/source": "aaaaaaaa",
    },
  },
  "status": {"conditions": [{
    "type": "Failed", "status": "True", "reason": "ControlledExit"
  }]},
}))
""", encoding="utf-8")
    kubectl.chmod(0o755)
    script = Path(__file__).with_name("run_m15_multiround_operator_return.sh")
    output = self.root / "shell-return"
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FAKE_GCS_ROOT"] = str(fake_gcs)
    result = subprocess.run(
        ["bash", str(script), str(self.render), str(output), str(self.root)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
    summary = json.loads(
        (output / "OPERATOR_RETURN_SUMMARY.json").read_text(encoding="utf-8")
    )
    self.assertEqual(summary["status"], "COMPLETE")
    for line in (output / "SHA256SUMS").read_text(encoding="ascii").splitlines():
      digest, name = line.split("  ", 1)
      self.assertEqual(_sha(output / name), digest)
    returned_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.iterdir()
        if path.suffix in (".json", ".txt")
    )
    self.assertNotIn("gs://", returned_text)
    self.assertFalse((output / "run.log").exists())


if __name__ == "__main__":
  unittest.main()
