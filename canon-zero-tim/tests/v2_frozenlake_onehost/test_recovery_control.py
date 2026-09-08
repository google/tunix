"""Synthetic mutations exercise the full classifier, not mocked verdicts."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts"


def _load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


control = _load(SCRIPTS / "classify_frozenlake_recovery_control.py", "fl_recovery")
fixtures = _load(Path(__file__).with_name("test_classifier.py"), "fl_recovery_fixture")
SOURCE = "a" * 40
PRODUCER = "capsule-producer-r1"
CAPSULE = "d" * 64
BINDING = "e" * 64
SHARED_MARKER = (
    "[P59.LAYER_PROGRAM_REUSE] enabled=1 layers=36 static_keys=1 "
    "mapped_programs=1 logical_calls_per_layer=1 checked_vma=1 host_transfers=0"
)
CONTROL_MARKER = (
    "[P59.LAYER_PROGRAM_REUSE] enabled=0 layers=36 static_keys=36 "
    "mapped_programs=36 logical_calls_per_layer=1 checked_vma=1 host_transfers=0"
)
END = "[V2.FL.ONEHOST] RUN_END docker_exit=0 elapsed_seconds=948 contention=0 timeout=0"


def _edit_json(path, mutate):
  value = json.loads(path.read_text())
  mutate(value)
  path.write_text(json.dumps(value))


def _edit_raw(root, old, new):
  path = root / "raw.log"
  raw = path.read_text()
  assert old in raw, "negative control did not find its target"
  path.write_text(raw.replace(old, new))


class RecoveryControlTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name) / "input"
    self.root.mkdir()
    self.anchor = fixtures.FrozenLakeOneHostClassifierTest()._fixture(
        self.root, arm="r2", mode="certify", workload="p45"
    )
    _edit_json(self.root / "run_manifest.json", lambda manifest: manifest.update(
        source_diff_sha256=hashlib.sha256(b"").hexdigest(),
        training_capsule={"mode": "replay", "capture_run": PRODUCER,
                          "sha256": CAPSULE, "model_binding_sha256": BINDING},
    ))
    _edit_raw(self.root, SHARED_MARKER, CONTROL_MARKER)
    _edit_raw(self.root, "[V2.FL.SAMPLER] CONTRACT_PASS sampler_is=none "
              "use_rollout_logps=1 tis_weights=absent\n", "")
    _edit_raw(self.root, "[V2.FL.ANCHOR] LIVE_ACTOR_REUSE anchor_train_steps=0 "
              "actor_train_steps=0 host_transfers=0\n", "")
    with (self.root / "raw.log").open("a") as output:
      output.write(
          "[V2.CAPSULE] diagnostic_replay_ready path=/evidence/training_capsule.npz "
          f"sha256={CAPSULE} rows=16 capture_run={PRODUCER} "
          f"capture_source={'b' * 40} replay_source={SOURCE} "
          "rollout=skipped rescore_b=skipped certification=0\n"
          "[V2.CAPSULE] producer_bypass verdict=PASS environment=0 rollout=0 rescore_b=0\n"
          f"[V2.CAPSULE] model_verified mode=replay capsule_sha256={CAPSULE} "
          f"binding_sha256={BINDING} model_fingerprint_sha256={'c' * 64} sampled_model=1\n"
          f"{END}\n"
      )
    update = json.loads((self.root / "updates.json").read_text())
    self.anchor.write_text(json.dumps({
        "schema": "canon.v2-frozenlake-onehost.gradient-anchors.v2",
        "anchors": {"p45:dp2-tp2:r2": {
            "run_id": "measured-replay-r2", "capture_run_id": PRODUCER,
            "training_capsule_sha256": CAPSULE,
            "micro_gradient_norms": update["micro_gradient_norms"],
            "update_gradient_norm": update["update_gradient_norm"],
        }},
    }))

  def classify(self, source=SOURCE):
    return control.classify(self.root, docker_exit=0, anchor_registry=self.anchor,
                            expected_source_commit=source)

  def test_only_scoped_control_passes_and_standard_failure_is_unchanged(self):
    original = control.standard.classify(
        self.root, workload="p45", arm="r2", docker_exit=0,
        anchor_registry=self.anchor, require_anchor=True,
    )
    files = {path: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
    result = self.classify()
    self.assertEqual(result["verdict"], "RECOVERY_CONTROL_PASS", result["reasons"])
    self.assertEqual(result["standard_classification"], original)
    self.assertEqual(original["verdict"], "FAIL")
    self.assertEqual(original["reasons"], [control.DEFERRED_REASON])
    self.assertEqual(result["program_sharing_optimization"]["admission"], "NOT_ADMITTED")
    for path, before in files.items():
      self.assertEqual(path.read_bytes(), before)

  def test_every_reuse_field_and_missing_or_duplicate_receipt_is_rejected(self):
    raw = (self.root / "raw.log").read_text()
    variants = ["", CONTROL_MARKER + "\n" + CONTROL_MARKER, SHARED_MARKER]
    for old, new in (("enabled=0", "enabled=1"), ("layers=36", "layers=35"),
                     ("static_keys=36", "static_keys=35"),
                     ("mapped_programs=36", "mapped_programs=35"),
                     ("logical_calls_per_layer=1", "logical_calls_per_layer=2"),
                     ("checked_vma=1", "checked_vma=0"),
                     ("host_transfers=0", "host_transfers=1")):
      variants.append(CONTROL_MARKER.replace(old, new))
    for marker in variants:
      with self.subTest(marker=marker):
        (self.root / "raw.log").write_text(raw.replace(CONTROL_MARKER, marker))
        result = self.classify()
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("recovery_reuse_contract", result["reasons"])

  def test_source_is_full_pre_registered_and_clean(self):
    for source in ("", "a" * 8, "g" * 40):
      with self.subTest(source=source), self.assertRaises(ValueError):
        self.classify(source)
    self.assertIn("recovery_source_commit", self.classify("b" * 40)["reasons"])
    _edit_json(self.root / "run_manifest.json", lambda value: value.update(
        source_diff_sha256="f" * 64))
    self.assertIn("recovery_source_not_clean", self.classify()["reasons"])

  def test_other_workload_geometry_arm_or_stage_never_uses_this_contract(self):
    path = self.root / "run_manifest.json"
    original = path.read_text()
    for changed in ({"workload": "m15"}, {"arm": "r3"},
                    {"topology": {"dp": 4, "tp": 1, "devices": 4}},
                    {"stage": "full"}, {"classification_mode": "measure"},
                    {"checked_vma": False}):
      with self.subTest(changed=changed):
        value = json.loads(original)
        value.update(changed)
        path.write_text(json.dumps(value))
        result = self.classify()
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(any(reason.startswith("manifest:") for reason in result["reasons"]))

  def test_numerical_replica_commit_and_other_standard_failures_survive(self):
    path = self.root / "updates.json"
    original = path.read_text()
    for field, value in (("gradient_finite", False), ("dp_replicas_exact", False),
                         ("commits", 1), ("dp_rank_pullbacks_per_transaction", 1),
                         ("micro_gradient_norms", [123.0] * 8),
                         ("update_gradient_norm", 123.0)):
      with self.subTest(field=field):
        data = json.loads(original)
        data[field] = value
        path.write_text(json.dumps(data))
        result = self.classify()
        self.assertEqual(result["verdict"], "FAIL")
        standard_reasons = set(result["standard_classification"]["reasons"])
        self.assertTrue(standard_reasons - {control.DEFERRED_REASON})
        self.assertTrue(standard_reasons - {control.DEFERRED_REASON} <= set(result["reasons"]))

  def test_signed_zero_norm_change_is_not_bitwise_equal(self):
    _edit_json(self.root / "updates.json", lambda value: (
        value["micro_gradient_norms"].__setitem__(7, -0.0),
        value["gradient_activity"].__setitem__(7, False),
    ))
    _edit_json(self.anchor, lambda value:
               value["anchors"]["p45:dp2-tp2:r2"]["micro_gradient_norms"].__setitem__(7, 0.0))
    result = self.classify()
    self.assertTrue(result["standard_classification"]["gradient"]["anchor_exact"])
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("recovery_anchor_bits", result["reasons"])

  def test_producer_capsule_and_bypass_failures_survive(self):
    original = self.anchor.read_text()
    for field, value, reason in (
        ("capture_run_id", "foreign-producer-r1", "gradient_anchor_capture_run"),
        ("training_capsule_sha256", "f" * 64, "gradient_anchor_capsule_sha"),
    ):
      with self.subTest(field=field):
        data = json.loads(original)
        data["anchors"]["p45:dp2-tp2:r2"][field] = value
        self.anchor.write_text(json.dumps(data))
        self.assertIn(reason, self.classify()["reasons"])
    self.anchor.write_text(original)
    _edit_raw(self.root, "[V2.CAPSULE] producer_bypass verdict=PASS "
              "environment=0 rollout=0 rescore_b=0\n", "")
    self.assertIn("training_capsule_producer_bypass", self.classify()["reasons"])

  def test_missing_checked_vma_receipt_still_fails(self):
    _edit_raw(self.root, "[P66.VMA] outer_check_enabled", "[REMOVED.VMA] marker")
    result = self.classify()
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("p66_outer_check_receipts=0", result["reasons"])

  def test_missing_failed_or_duplicate_terminal_fails(self):
    raw = (self.root / "raw.log").read_text()
    for terminal in ("", END.replace("contention=0", "contention=1"),
                     END.replace("timeout=0", "timeout=1"), END + "\n" + END,
                     END.replace("docker_exit=0", "docker_exit=1") + "\n" + END):
      with self.subTest(terminal=terminal):
        (self.root / "raw.log").write_text(raw.replace(END, terminal))
        self.assertIn("recovery_clean_terminal", self.classify()["reasons"])

  def test_alignment_and_hbm_failures_are_not_performance_exemptions(self):
    path = self.root / "pre_alignment.jsonl"
    original = path.read_text()
    records = [json.loads(line) for line in original.splitlines()]
    records[0]["boundaries"]["S_prefill_vs_T_old"]["differing_bytes"] = 1
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    result = self.classify()
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("pre_alignment_not_exact", result["reasons"])
    path.write_text(original)
    _edit_json(self.root / "updates.json", lambda value:
               value["hbm_after_reverse"][0].update(peak_bytes_in_use=99))
    result = self.classify()
    self.assertEqual(result["verdict"], "FAIL")
    self.assertTrue(any(reason.startswith("hbm_headroom_ratio=") for reason in result["reasons"]))

  def test_incomplete_run_remains_inconclusive(self):
    (self.root / "runtime.json").unlink()
    result = self.classify()
    self.assertEqual(result["verdict"], "INCONCLUSIVE")
    self.assertEqual(result["standard_classification"]["verdict"], "INCONCLUSIVE")

  def test_cli_writes_new_external_output_only(self):
    output = Path(self.temporary.name) / "control.json"
    command = [sys.executable, str(SCRIPTS / "classify_frozenlake_recovery_control.py"),
               "--root", str(self.root), "--docker-exit", "0",
               "--anchor-registry", str(self.anchor), "--expected-source-commit", SOURCE,
               "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(json.loads(output.read_text())["verdict"], "RECOVERY_CONTROL_PASS")
    before = output.read_bytes()
    self.assertNotEqual(subprocess.run(command, capture_output=True, check=False).returncode, 0)
    self.assertEqual(output.read_bytes(), before)
    command[-1] = str(self.root / "new_classification.json")
    self.assertNotEqual(subprocess.run(command, capture_output=True, check=False).returncode, 0)
    self.assertFalse((self.root / "new_classification.json").exists())


if __name__ == "__main__":
  unittest.main()
