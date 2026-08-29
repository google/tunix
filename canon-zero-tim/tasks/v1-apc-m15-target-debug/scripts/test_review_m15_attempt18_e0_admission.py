#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
MODULE = (
    ROOT / "canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/"
    "review_m15_attempt18_e0_admission.py"
)
EVIDENCE = (
    ROOT / "canon-zero-tim/tasks/v1-apc-m15-target-debug/evidence/"
    "v1_apc_m15_attempt17_d3e_canonical_action_20260829"
)
SPEC = importlib.util.spec_from_file_location("m15_e0_admission", MODULE)
assert SPEC and SPEC.loader
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)


class E0AdmissionTest(unittest.TestCase):

  _IMAGE_ID = (
      "sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a"
  )

  def _write_fake_docker(self, root: Path) -> tuple[Path, Path]:
    executable = root / "fake-docker"
    log = root / "docker.log"
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
if [ "$1" = image ] && [ "$2" = inspect ]; then
  case "${FAKE_DOCKER_INSPECT:-ok}" in
    missing) exit 1 ;;
    wrong) printf '%s\\n' 'sha256:0000000000000000000000000000000000000000000000000000000000000000' ;;
    ok) printf '%s\\n' 'sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a' ;;
    *) exit 64 ;;
  esac
  exit 0
fi
if [ "$1" = run ]; then
  exit "${FAKE_DOCKER_RUN_RC:-0}"
fi
exit 64
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, log

  def _run_classifier_helper(
      self, root: Path, mode: str, inspect: str = "ok"
  ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    helper = MODULE.with_name("run_m15_e0_kv_classifier_gate.sh")
    docker, docker_log = self._write_fake_docker(root)
    receipt = root / "KV_CLASSIFIER_RUNTIME.json"
    env = dict(os.environ)
    env.update({
        "DOCKER": str(docker),
        "FAKE_DOCKER_LOG": str(docker_log),
        "FAKE_DOCKER_INSPECT": inspect,
    })
    result = subprocess.run(
        ["bash", str(helper), str(receipt), mode],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, receipt, docker_log

  def test_prepare_wrapper_renders_without_external_mutation(self):
    wrapper = MODULE.with_name("prepare_m15_attempt18_e0_kv_pair.sh")
    text = wrapper.read_text(encoding="utf-8")
    self.assertIn("--observer kv", text)
    self.assertIn("preflight_runtime.py", text)
    self.assertIn("[M15.E0.KV] TARGET_NOT_RUN", text)
    self.assertIn("run_m15_e0_kv_classifier_gate.sh", text)
    self.assertIn("scratch_preserved=", text)
    self.assertIn("{0,14}", text)
    self.assertNotIn("tunix_base_image:latest", text)
    self.assertNotIn("kubectl apply", text)
    self.assertNotIn("gsutil ", text)
    self.assertNotIn("gcloud ", text)

  def test_classifier_runtime_host_route_writes_receipt(self):
    with tempfile.TemporaryDirectory() as directory:
      result, receipt, _ = self._run_classifier_helper(Path(directory), "host")
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      self.assertIn("KV_CLASSIFIER_RUNTIME_PASS route=host", result.stdout)
      value = json.loads(receipt.read_text(encoding="utf-8"))
      self.assertEqual(value["status"], "PASS")
      self.assertEqual(value["route"], "host")
      self.assertIsNone(value["image_id"])
      self.assertFalse(value["external_access"])

  def test_classifier_runtime_forced_docker_is_pinned_and_offline(self):
    with tempfile.TemporaryDirectory() as directory:
      result, receipt, docker_log = self._run_classifier_helper(
          Path(directory), "docker"
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      self.assertIn("KV_CLASSIFIER_RUNTIME_PASS route=docker", result.stdout)
      value = json.loads(receipt.read_text(encoding="utf-8"))
      self.assertEqual(value["image_id"], self._IMAGE_ID)
      self.assertEqual(value["pull_policy"], "never")
      self.assertEqual(value["network_mode"], "none")
      self.assertFalse(value["external_access"])
      calls = docker_log.read_text(encoding="utf-8")
      self.assertIn(f"image inspect {self._IMAGE_ID}", calls)
      self.assertIn("run --rm --pull=never --network=none", calls)
      self.assertIn(f" {self._IMAGE_ID} python3 ", calls)

  def test_classifier_runtime_missing_local_image_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      result, receipt, docker_log = self._run_classifier_helper(
          Path(directory), "docker", inspect="missing"
      )
      self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
      self.assertIn("pinned classifier image is not already local", result.stderr)
      self.assertFalse(receipt.exists())
      self.assertNotIn("run --rm", docker_log.read_text(encoding="utf-8"))

  def test_classifier_runtime_wrong_image_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      result, receipt, docker_log = self._run_classifier_helper(
          Path(directory), "docker", inspect="wrong"
      )
      self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
      self.assertIn("pinned classifier image identity mismatch", result.stderr)
      self.assertFalse(receipt.exists())
      self.assertNotIn("run --rm", docker_log.read_text(encoding="utf-8"))

  def test_prepare_rejects_long_run_id_and_preserves_scratch(self):
    wrapper = MODULE.with_name("prepare_m15_attempt18_e0_kv_pair.sh")
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "output"
      result = subprocess.run(
          ["bash", str(wrapper), "0" * 40, "abcdefghijklmnopq", str(output)],
          check=False,
          capture_output=True,
          text=True,
      )
      self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
      self.assertIn("1-16 character", result.stderr)
      match = re.search(r"scratch_preserved=(\S+)", result.stderr)
      self.assertIsNotNone(match, result.stderr)
      scratch = Path(match.group(1))
      self.assertTrue(scratch.is_dir())
      shutil.rmtree(scratch)

  def test_return_wrapper_is_read_only_and_compact(self):
    wrapper = MODULE.with_name("run_m15_attempt18_e0_kv_gcs_return.sh")
    text = wrapper.read_text(encoding="utf-8")
    self.assertIn("kv-observer-classification.json", text)
    self.assertIn("LIVE_KV_FINGERPRINT_DIFFERS", text)
    self.assertIn("LIVE_KV_FINGERPRINT_EQUAL", text)
    self.assertIn('serving.get("verdict") == "PASS"', text)
    self.assertIn('not arms["off"]["kv_all_pairs_equal"]', text)
    self.assertIn("[M15.E0.KV.RETURN] READ_ONLY", text)
    self.assertNotIn("kubectl ", text)
    self.assertNotIn("gcloud storage rsync", text)
    self.assertNotIn("gsutil -m", text)

  def test_committed_d3e_return_admits_preparation_only(self):
    report = reviewer.review(EVIDENCE)
    self.assertEqual(report["status"], "E0_PREPARATION_ADMITTED")
    self.assertEqual(report["d3e_gate"], "FIRST_RED_LOCALIZED")
    self.assertEqual(report["target_prefix"]["tokens"], 1226)
    self.assertEqual(report["target_prefix"]["aliases"], 8)
    self.assertEqual(report["target_prefix"]["logical_pages"], 77)
    self.assertFalse(report["launch_authorized"])
    self.assertFalse(report["numerical_repair_authorized"])

  def test_tampered_boundary_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for path in EVIDENCE.iterdir():
        shutil.copy2(path, root / path.name)
      classification = root / "D36_RECLASSIFICATION.json"
      value = json.loads(classification.read_text(encoding="utf-8"))
      value["first_red_boundary"]["checkpoint"] = "k_post_rope"
      classification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(reviewer.AdmissionError, "manifest member"):
        reviewer.review(root)


if __name__ == "__main__":
  unittest.main()
