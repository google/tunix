#!/usr/bin/env python3
"""Tests for the small, token-safe M15 wide-seam salvage return."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("audit_m15_wide_seam_gcs_salvage.py")
SPEC = importlib.util.spec_from_file_location("m15_salvage", SCRIPT)
assert SPEC and SPEC.loader
salvage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(salvage)


def _sha(payload: bytes) -> str:
  return hashlib.sha256(payload).hexdigest()


class SalvageTest(unittest.TestCase):

  def _classification(self, arm: str, classification: str, layer=None):
    return {
        "schema": "m15-apc-wide-seam-classification-v1",
        "status": "PASS",
        "arm": arm,
        "observer_mode": "layer",
        "classification": classification,
        "selected_layer": layer,
    }

  def _bundle(self, path: Path, classification: dict):
    members = {
        "classification.json": json.dumps(classification).encode(),
        "RECEIPT.json": json.dumps({
            "schema": "m15-apc-wide-seam-bundle-v1",
            "classification": classification["classification"],
            "observer_mode": "layer",
            "arm": classification["arm"],
        }).encode(),
        "selected/opaque.npz": b"token-bearing-test-payload",
    }
    members["SHA256SUMS"] = "".join(
        f"{_sha(members[name])}  {name}\n" for name in sorted(members)
    ).encode()
    with tarfile.open(path, "w") as archive:
      for name, payload in sorted(members.items()):
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

  def _root(self, root: Path, arm: str, classification: dict):
    root.mkdir()
    payload = (json.dumps(classification, sort_keys=True) + "\n").encode()
    (root / "seam-classification.json").write_bytes(payload)
    (root / "p38_seam.classification.json").write_bytes(payload)
    (root / "SHA256SUMS").write_text(
        f"{_sha(payload)}  seam-classification.json\n", encoding="ascii"
    )
    for marker in salvage._MARKERS:
      (root / marker).write_text(json.dumps({
          "schema": f"test-{marker}",
          "status": "ok",
          "source_commit": "3" * 40,
      }), encoding="utf-8")
    (root / "remote-inventory.txt").write_text("test present\n", encoding="utf-8")
    self._bundle(root / "m15_wide_seam_bundle.tar", classification)

  def _receipt(self, path: Path):
    path.write_text(json.dumps({
        "attempt": 9,
        "campaign_root": "test",
        "source_commit": "3" * 40,
    }), encoding="utf-8")

  def test_selects_layer_and_excludes_token_bundle(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      receipt = root / "receipt.json"
      self._receipt(receipt)
      off = root / "off"
      on = root / "on"
      self._root(off, "off", self._classification(
          "off", "M15_OBSERVER_CONTROL_EXACT"))
      self._root(on, "on", self._classification(
          "on", "M15_LAYER_FIRST_RED_LOCALIZED", 17))
      output = root / "return"
      result = salvage.audit(
          receipt_path=receipt, off_root=off, on_root=on, output=output
      )
      self.assertEqual(result["status"], "LAYER_SELECTED")
      self.assertIn("layer 17", result["next_action"])
      self.assertFalse((output / "m15_wide_seam_bundle.tar").exists())
      self.assertTrue((output / "off.classification.json").is_file())
      self.assertTrue((output / "on.classification.json").is_file())
      self.assertIn("token_bearing_bundle_returned=0",
                    (output / "PACKAGING.txt").read_text())

  def test_missing_classifier_returns_incomplete_package(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      receipt = root / "receipt.json"
      self._receipt(receipt)
      off = root / "off"
      on = root / "on"
      off.mkdir()
      on.mkdir()
      output = root / "return"
      result = salvage.audit(
          receipt_path=receipt, off_root=off, on_root=on, output=output
      )
      self.assertEqual(result["status"], "INCOMPLETE")
      self.assertTrue((output / "SALVAGE_SUMMARY.json").is_file())
      self.assertTrue((output / "SHA256SUMS").is_file())

  def test_rejects_conflicting_aliases(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      receipt = root / "receipt.json"
      self._receipt(receipt)
      off = root / "off"
      on = root / "on"
      off.mkdir()
      on.mkdir()
      first = self._classification("off", "M15_OBSERVER_CONTROL_EXACT")
      second = dict(first, selected_layer=4)
      (off / "seam-classification.json").write_text(json.dumps(first))
      (off / "p38_seam.classification.json").write_text(json.dumps(second))
      with self.assertRaises(salvage.SalvageError):
        salvage.audit(
            receipt_path=receipt, off_root=off, on_root=on,
            output=root / "return",
        )

  def test_returns_source_mismatch_instead_of_hiding_marker(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      receipt = root / "receipt.json"
      self._receipt(receipt)
      off = root / "off"
      on = root / "on"
      self._root(off, "off", self._classification(
          "off", "M15_OBSERVER_CONTROL_EXACT"))
      self._root(on, "on", self._classification(
          "on", "M15_LAYER_FIRST_RED_LOCALIZED", 17))
      complete = json.loads((on / "COMPLETE.json").read_text())
      complete["source_commit"] = "4" * 40
      (on / "COMPLETE.json").write_text(json.dumps(complete))
      result = salvage.audit(
          receipt_path=receipt,
          off_root=off,
          on_root=on,
          output=root / "return",
      )
      self.assertEqual(result["status"], "SOURCE_MISMATCH")
      self.assertEqual(len(result["source_commit_conflicts"]), 1)

  def test_shell_wrapper_downloads_read_only_and_returns_package(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      remote = root / "remote"
      off_uri = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
          "canon-test-off/attempt-0"
      )
      on_uri = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
          "canon-test-on/attempt-0"
      )
      off_source = root / "off-source"
      on_source = root / "on-source"
      self._root(off_source, "off", self._classification(
          "off", "M15_OBSERVER_CONTROL_EXACT"))
      self._root(on_source, "on", self._classification(
          "on", "M15_LAYER_FIRST_RED_LOCALIZED", 17))
      for uri, source in ((off_uri, off_source), (on_uri, on_source)):
        destination = remote / uri.removeprefix("gs://")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)

      receipt = root / "receipt.json"
      receipt.write_text(json.dumps({
          "attempt": 9,
          "campaign_root": "test",
          "source_commit": "3" * 40,
          "control_arm_off": {"gcs_source_uri": off_uri},
          "treatment_arm_on": {"gcs_source_uri": on_uri},
      }), encoding="utf-8")
      fake_bin = root / "bin"
      fake_bin.mkdir()
      fake_gcloud = fake_bin / "gcloud"
      fake_gcloud.write_text("""#!/usr/bin/env python3
import os
import pathlib
import shutil
import sys
args = sys.argv[1:]
if args[:2] not in ([\"storage\", \"ls\"], [\"storage\", \"cp\"]):
  raise SystemExit(2)
remote = pathlib.Path(os.environ[\"FAKE_GCS_ROOT\"])
if args[1] == \"ls\":
  path = remote / args[2].removeprefix(\"gs://\")
  raise SystemExit(0 if path.is_file() else 1)
source = args[2]
destination = pathlib.Path(args[3])
if not source.startswith(\"gs://\") or str(destination).startswith(\"gs://\"):
  raise SystemExit(3)
shutil.copyfile(remote / source.removeprefix(\"gs://\"), destination)
""", encoding="utf-8")
      fake_gcloud.chmod(0o755)
      output = root / "return"
      env = dict(os.environ)
      env["PATH"] = f"{fake_bin}:{env['PATH']}"
      env["FAKE_GCS_ROOT"] = str(remote)
      completed = subprocess.run(
          [
              "bash",
              str(Path(__file__).with_name("run_m15_wide_seam_gcs_salvage.sh")),
              str(receipt),
              str(output),
              str(root),
          ],
          check=False,
          capture_output=True,
          text=True,
          env=env,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertIn("status=LAYER_SELECTED", completed.stdout)
      self.assertFalse((output / "m15_wide_seam_bundle.tar").exists())
      self.assertTrue((output / "SHA256SUMS").is_file())


if __name__ == "__main__":
  unittest.main()
