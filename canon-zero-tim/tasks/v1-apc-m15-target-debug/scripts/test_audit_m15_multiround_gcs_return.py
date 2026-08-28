#!/usr/bin/env python3
"""Host positives and negatives for the three-round small GCS return."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from audit_m15_multiround_gcs_return import MultiRoundAuditError, audit


SOURCE = "a" * 40


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class MultiRoundReturnTest(unittest.TestCase):

  def setUp(self) -> None:
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)
    self.off = self.root / "off"
    self.on = self.root / "on"
    for arm_root in (self.off, self.on):
      (arm_root / "root").mkdir(parents=True)
      for round_index in range(3):
        (arm_root / f"round-{round_index:06d}").mkdir()

  def tearDown(self) -> None:
    self.holder.cleanup()

  def _round(
      self,
      arm: str,
      round_index: int,
      *,
      red: bool = False,
      candidate: bool = False,
  ) -> None:
    root = (self.off if arm == "off" else self.on) / f"round-{round_index:06d}"
    classification_name = (
        "M15_LAYER_FIRST_RED_CANDIDATE_SET"
        if candidate else "M15_LAYER_FIRST_RED_LOCALIZED"
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
    bundle_sha = hashlib.sha256(f"bundle-{arm}-{round_index}".encode()).hexdigest()
    manifest = root / "WIDE_SHA256SUMS"
    manifest.write_text(
        f"{_sha(root / 'ROUND_INPUT_RECEIPT.json')}  ROUND_INPUT_RECEIPT.json\n"
        f"{_sha(root / 'p38_seam.classification.json')}  p38_seam.classification.json\n"
        f"{bundle_sha}  m15_wide_seam_bundle.tar\n",
        encoding="ascii",
    )
    completion = {
        "schema": "m15-wide-round-completion-v1",
        "status": "classified-and-uploaded",
        "diagnostic_round": round_index,
        "expected_source_commit": SOURCE,
        "runtime_source_commit": SOURCE,
        "classification": classification_name,
        "manifest_sha256": _sha(manifest),
        "record_pairs": 2,
        "shards": receipt["shards"],
    }
    (root / "WIDE_ROUND_COMPLETE.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    (root / "remote-inventory.txt").write_text(
        "ROUND_INPUT_RECEIPT.json present\n"
        "p38_seam.classification.json present\n"
        "WIDE_SHA256SUMS present\n"
        "WIDE_ROUND_COMPLETE.json present\n"
        "m15_wide_seam_bundle.tar present\n",
        encoding="utf-8",
    )

  def _markers(self, arm_root: Path, *, terminal: bool) -> None:
    names = ["PREFLIGHT.json"]
    if terminal:
      names.extend(("COLLECTED.json", "COMPLETE.json"))
    for name in names:
      (arm_root / "root" / name).write_text(json.dumps({
          "source_commit": SOURCE,
          "status": "PASS",
      }), encoding="utf-8")

  def _stage(
      self,
      arm: str,
      round_index: int,
      ordinal: int,
      stage: str,
      status: str,
      *,
      exit_code: int = 0,
  ) -> None:
    root = (
        (self.off if arm == "off" else self.on)
        / f"round-{round_index:06d}"
        / "stages"
    )
    root.mkdir(exist_ok=True)
    name = f"STAGE_{ordinal}_{stage}_{status}.json"
    (root / name).write_text(json.dumps({
        "diagnostic_round": round_index,
        "exit_code": exit_code,
        "runtime_source_commit": SOURCE,
        "schema": "m15-wide-round-stage-v1",
        "stage": stage,
        "status": status,
    }), encoding="utf-8")
    inventory = root.parent / "remote-inventory.txt"
    with inventory.open("a", encoding="utf-8") as stream:
      stream.write(f"stages/{name} present\n")

  def _completed_stages(self, arm: str, round_index: int) -> None:
    self._classifier_input(arm, round_index)
    for ordinal, stage in (
        (10, "assemble"),
        (15, "checkpoint-input"),
        (20, "classify"),
        (30, "package"),
        (35, "local-export"),
        (40, "manifest"),
        (50, "upload"),
        (60, "remote-verify"),
        (70, "completion"),
    ):
      self._stage(arm, round_index, ordinal, stage, "STARTED")
      self._stage(arm, round_index, ordinal, stage, "PASS")

  def _classifier_input(self, arm: str, round_index: int) -> None:
    root = (self.off if arm == "off" else self.on) / f"round-{round_index:06d}"
    receipt = json.loads((root / "ROUND_INPUT_RECEIPT.json").read_text())
    classification = json.loads(
        (root / "p38_seam.classification.json").read_text()
    )
    ab_bytes = int(classification["alignment"]["a_b_differing_bytes"])
    classifier_input = root / "classifier-input"
    classifier_input.mkdir()
    input_names = [
        "ROUND_INPUT_RECEIPT.json",
        "m15-replay-envelope.jsonl",
        "pre-alignment.jsonl",
    ]
    if ab_bytes > 0:
      input_names.append("mismatch-capsule.npz")
    input_names.sort()
    input_manifest = classifier_input / "CLASSIFIER_INPUT_SHA256SUMS"
    input_manifest.write_text(
        "".join(f"{'1' * 64}  {name}\n" for name in input_names),
        encoding="ascii",
    )
    input_receipt = {
        "schema": "m15-wide-classifier-input-v1",
        "status": "prepared-for-durable-upload",
        "arm": arm,
        "diagnostic_round": round_index,
        "a_b_differing_bytes": ab_bytes,
        "files": input_names,
        "manifest_sha256": _sha(input_manifest),
        "record_pairs": 2,
        "shards": receipt["shards"],
        "expected_source_commit": SOURCE,
        "runtime_source_commit": SOURCE,
    }
    (classifier_input / "CLASSIFIER_INPUT_RECEIPT.json").write_text(
        json.dumps(input_receipt), encoding="utf-8"
    )
    with (root / "remote-inventory.txt").open("a", encoding="utf-8") as stream:
      for name in input_names:
        stream.write(f"classifier-input/{name} present\n")
      stream.write("classifier-input/CLASSIFIER_INPUT_SHA256SUMS present\n")
      stream.write("classifier-input/CLASSIFIER_INPUT_RECEIPT.json present\n")

  def _all_rounds(self) -> None:
    for round_index in range(3):
      self._round("off", round_index)
      self._round("on", round_index, red=round_index == 1)

  def test_complete_pair_returns_six_hash_bound_classifiers(self) -> None:
    self._all_rounds()
    for arm in ("off", "on"):
      for round_index in range(3):
        self._completed_stages(arm, round_index)
    self._markers(self.off, terminal=True)
    self._markers(self.on, terminal=True)
    output = self.root / "return"
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=output,
    )
    self.assertEqual(result["status"], "COMPLETE")
    self.assertEqual(len(list(output.glob("*.classification.json"))), 6)
    self.assertEqual(len(list(output.glob("*.stage-*.json"))), 108)

  def test_candidate_set_is_preserved_without_becoming_pipeline_failure(self) -> None:
    for round_index in range(3):
      self._round("off", round_index)
      self._round(
          "on", round_index, red=round_index == 1, candidate=round_index == 1
      )
      self._completed_stages("off", round_index)
      self._completed_stages("on", round_index)
    self._markers(self.off, terminal=True)
    self._markers(self.on, terminal=True)
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=self.root / "return-candidates",
    )
    self.assertEqual(result["status"], "COMPLETE")
    self.assertEqual(
        result["arms"]["on"]["rounds"][1]["classification"],
        "M15_LAYER_FIRST_RED_CANDIDATE_SET",
    )

  def test_all_rounds_survive_missing_root_terminal_markers(self) -> None:
    self._all_rounds()
    self._markers(self.off, terminal=False)
    self._markers(self.on, terminal=False)
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=self.root / "return",
    )
    self.assertEqual(result["status"], "ROUNDS_RECOVERED_ROOT_INCOMPLETE")
    self.assertEqual(result["arms"]["off"]["sealed_rounds"], 3)

  def test_one_sealed_round_is_reported_as_partial_not_discarded(self) -> None:
    self._round("off", 0)
    self._round("on", 0, red=True)
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=self.root / "return",
    )
    self.assertEqual(result["status"], "PARTIAL_ROUNDS_RECOVERED")

  def test_tampered_classifier_is_rejected(self) -> None:
    self._all_rounds()
    path = self.off / "round-000001/p38_seam.classification.json"
    path.write_text("{}", encoding="utf-8")
    with self.assertRaisesRegex(MultiRoundAuditError, "failed SHA"):
      audit(
          source_commit=SOURCE,
          rounds=3,
          off_root=self.off,
          on_root=self.on,
          output=self.root / "return",
      )

  def test_explicit_stage_failure_is_returned_without_numerical_claim(self) -> None:
    self._stage("off", 0, 10, "assemble", "STARTED")
    self._stage("off", 0, 10, "assemble", "PASS")
    self._stage("off", 0, 15, "checkpoint-input", "STARTED")
    self._stage("off", 0, 15, "checkpoint-input", "PASS")
    self._stage("off", 0, 20, "classify", "STARTED")
    self._stage("off", 0, 20, "classify", "FAIL", exit_code=17)
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=self.root / "return",
    )
    self.assertEqual(result["status"], "ROUND_STAGE_FAILURE_IDENTIFIED")
    row = result["arms"]["off"]["rounds"][0]
    self.assertEqual(row["status"], "UNSEALED")
    self.assertEqual(row["stage_state"]["failure_stage"], "classify")
    self.assertEqual(row["stage_state"]["failure_exit_code"], 17)

  def test_started_only_stage_reports_interrupted_progress(self) -> None:
    self._stage("on", 0, 10, "assemble", "STARTED")
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=self.root / "return",
    )
    self.assertEqual(result["status"], "ROUND_STAGE_PROGRESS_ONLY")
    state = result["arms"]["on"]["rounds"][0]["stage_state"]
    self.assertEqual(state["status"], "STARTED_ONLY")
    self.assertEqual(state["active_stage"], "assemble")

  def test_partial_official_files_are_not_mistaken_for_a_sealed_round(self) -> None:
    self._stage("on", 0, 10, "assemble", "STARTED")
    partial = self.on / "round-000000/ROUND_INPUT_RECEIPT.json"
    partial.write_text("{}", encoding="utf-8")
    with (self.on / "round-000000/remote-inventory.txt").open(
        "a", encoding="utf-8"
    ) as stream:
      stream.write("ROUND_INPUT_RECEIPT.json present\n")
    result = audit(
        source_commit=SOURCE,
        rounds=3,
        off_root=self.off,
        on_root=self.on,
        output=self.root / "return",
    )
    row = result["arms"]["on"]["rounds"][0]
    self.assertEqual(row["status"], "UNSEALED")
    self.assertEqual(row["partial_round_files"], ["ROUND_INPUT_RECEIPT.json"])

  def test_stage_after_pipeline_gap_is_rejected(self) -> None:
    self._stage("off", 0, 20, "classify", "STARTED")
    with self.assertRaisesRegex(MultiRoundAuditError, "after a pipeline gap"):
      audit(
          source_commit=SOURCE,
          rounds=3,
          off_root=self.off,
          on_root=self.on,
          output=self.root / "return",
      )

  def test_boolean_failure_exit_code_is_rejected(self) -> None:
    self._stage("off", 0, 10, "assemble", "STARTED")
    self._stage("off", 0, 10, "assemble", "FAIL", exit_code=True)
    with self.assertRaisesRegex(MultiRoundAuditError, "receipt drifted"):
      audit(
          source_commit=SOURCE,
          rounds=3,
          off_root=self.off,
          on_root=self.on,
          output=self.root / "return",
      )

  def test_shell_return_downloads_stage_receipts(self) -> None:
    render = self.root / "render"
    render.mkdir()
    remote = self.root / "gcs"
    fake_bin = self.root / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "gcloud"
    shutil.copyfile(
        Path(__file__).resolve().parents[3] / "tests/p38_serving/fake_gcloud.sh",
        fake,
    )
    fake.chmod(0o755)
    for arm in ("off", "on"):
      label = f"m15-test-{arm}"
      uri = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
          f"{label}/attempt-0"
      )
      (render / f"jobset-v1-apc-m15-{arm}-full.yaml").write_text(
          "apiVersion: jobset.x-k8s.io/v1alpha2\n"
          "kind: JobSet\n"
          "metadata:\n"
          f"  name: {label}\n"
          "spec:\n"
          "  replicatedJobs:\n"
          "  - template:\n"
          "      spec:\n"
          "        template:\n"
          "          spec:\n"
          "            containers:\n"
          "            - env:\n"
          f"              - {{name: CANON_APC_M15_TARGET_DEBUG, value: '{arm}'}}\n"
          f"              - {{name: CANON_EXPECT_COMMIT, value: {SOURCE}}}\n"
          "              - {name: CANON_P38_DIAGNOSTIC_ROUNDS, value: '3'}\n"
          "              - {name: CANON_P38_SEAM_OBSERVER, value: full}\n"
          f"              - {{name: CANON_P38_GCS_PREFIX, value: {uri}}}\n",
          encoding="utf-8",
      )
    stage_root = (
        remote
        / "yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
        "m15-test-off/attempt-0/wide/rounds/000000/stages"
    )
    stage_root.mkdir(parents=True)
    for status, exit_code in (("STARTED", 0), ("FAIL", 17)):
      (stage_root / f"STAGE_10_assemble_{status}.json").write_text(
          json.dumps({
              "diagnostic_round": 0,
              "exit_code": exit_code,
              "runtime_source_commit": SOURCE,
              "schema": "m15-wide-round-stage-v1",
              "stage": "assemble",
              "status": status,
          }),
          encoding="utf-8",
      )
    output = self.root / "shell-return"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["FAKE_GCS_ROOT"] = str(remote)
    completed = subprocess.run(
        [
            "bash",
            str(Path(__file__).with_name("run_m15_multiround_gcs_return.sh")),
            str(render),
            str(output),
            str(self.root),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)
    summary = json.loads(
        (output / "MULTIROUND_SUMMARY.json").read_text(encoding="utf-8")
    )
    self.assertEqual(summary["status"], "ROUND_STAGE_FAILURE_IDENTIFIED")
    self.assertTrue(
        (output / "off.round-000000.stage-10-assemble-FAIL.json").is_file()
    )


if __name__ == "__main__":
  unittest.main()
