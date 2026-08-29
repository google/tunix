#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
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

  def test_prepare_wrapper_renders_without_external_mutation(self):
    wrapper = MODULE.with_name("prepare_m15_attempt18_e0_kv_pair.sh")
    text = wrapper.read_text(encoding="utf-8")
    self.assertIn("--observer kv", text)
    self.assertIn("preflight_runtime.py", text)
    self.assertIn("[M15.E0.KV] TARGET_NOT_RUN", text)
    self.assertNotIn("kubectl apply", text)
    self.assertNotIn("gsutil ", text)
    self.assertNotIn("gcloud ", text)

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
