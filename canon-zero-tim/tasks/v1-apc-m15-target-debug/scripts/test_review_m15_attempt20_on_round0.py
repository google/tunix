#!/usr/bin/env python3

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


CANON = Path(__file__).resolve().parents[3]
REVIEW_PATH = Path(__file__).with_name(
    "review_m15_attempt20_on_round0.py"
)
REVIEW_SPEC = importlib.util.spec_from_file_location(
    "review_m15_attempt20_on_round0", REVIEW_PATH
)
assert REVIEW_SPEC is not None and REVIEW_SPEC.loader is not None
review = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(review)

ARCHIVE_PATH = (
    CANON
    / "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "p38_evidence_archive.py"
)
ARCHIVE_SPEC = importlib.util.spec_from_file_location(
    "p38_evidence_archive_test", ARCHIVE_PATH
)
assert ARCHIVE_SPEC is not None and ARCHIVE_SPEC.loader is not None
archive_tool = importlib.util.module_from_spec(ARCHIVE_SPEC)
ARCHIVE_SPEC.loader.exec_module(archive_tool)

CLASSIFIER = (
    CANON
    / "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "classify_p38_kv_observer.py"
)
WRAPPER = Path(__file__).with_name(
    "run_m15_attempt20_on_round0_offline_recovery.sh"
)
TARGET_SOURCE = "97e813de84f6c8b3e2ba911fc96ff8397b199603"
ANALYSIS_SOURCE = "18f29c56daf471cc0ac011396d7c7a09f35d695b"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class Attempt20Round0RecoveryTest(unittest.TestCase):

  def _record(
      self,
      root: Path,
      alias: int,
      arm: str,
      *,
      changed: bool,
  ) -> None:
    a_index = alias * 2
    index = a_index if arm == "A" else a_index + 1
    token_ids = np.array([1, 2, 3], dtype=np.int32)
    aggregates = np.zeros((1, 2, 4, 4), dtype=np.uint32)
    samples = np.zeros((1, 2, 4, 3, 2), dtype=np.uint16)
    if changed:
      aggregates[0, 0, 1, 0] = 1
    arrays = {
        "aggregates": aggregates,
        "samples": samples,
        "token_ids": token_ids,
        "physical_pages": np.array([7], dtype=np.int32),
        "padded_global_pages": np.array([7, 7], dtype=np.int32),
        "valid_tokens": np.array([3], dtype=np.int32),
    }
    base = root / f"p38_kv_observer_{index:04d}_{arm.lower()}"
    np.savez(str(base) + ".npz", **arrays)
    npz = Path(str(base) + ".npz")
    token_sha = hashlib.sha256(
        np.ascontiguousarray(token_ids, dtype="<i8").tobytes()
    ).hexdigest()
    request_id = f"decode-{a_index}" if arm == "A" else f"clean-{a_index}"
    record = {
        "schema": "p38-live-kv-prefix-table-v1",
        "arm": arm,
        "record_index": index,
        "request_id": request_id,
        "source_a_request_id": f"decode-{a_index}",
        "source_a_record_index": None if arm == "A" else a_index,
        "diagnostic_round": 0,
        "target_seq_len": 3,
        "token_history_sha256": token_sha,
        "block_size": 4,
        "logical_pages": 1,
        "observer_pages": 2,
        "layer_count": 1,
        "layer_indices": [0],
        "cache_shape": [8, 4, 1, 2, 4],
        "cache_dtype": "bfloat16",
        "cache_sharding": "test",
        "npz_sha256": _sha256(npz),
        "array_keys": sorted(arrays),
    }
    Path(str(base) + ".json").write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8"
    )

  def _capsule(self, root: Path) -> None:
    metadata = json.dumps({
        "schema": "p38-frozenlake-mismatch-capsule-v1",
        "diagnostic_round": 0,
    }).encode()
    np.savez(
        root / "mismatch-capsule.npz",
        metadata_json=np.frombuffer(metadata, dtype=np.uint8),
        selected_rows=np.array([217], dtype=np.int32),
        prompt_ids=np.array([[1, 2, 3]], dtype=np.int32),
        prompt_mask=np.array([[True, True, True]]),
        completion_ids=np.array([[4]], dtype=np.int32),
        completion_valid_mask=np.array([[True]]),
        action_mask=np.array([[True]]),
        s_decode=np.array([[0.1]], dtype=np.float32),
        s_prefill=np.array([[0.2]], dtype=np.float32),
    )

  def _fixture(
      self,
      root: Path,
      *,
      red: bool = True,
      fingerprint_differs: bool = True,
      ambiguous_binding: bool = False,
  ) -> tuple[Path, Path, Path]:
    stage = root / "stage"
    stage.mkdir()
    for alias in range(8):
      self._record(
          stage,
          alias,
          "A",
          changed=red and fingerprint_differs and alias == 3,
      )
      self._record(stage, alias, "B", changed=False)

    if red:
      self._capsule(stage)
    source_tokens = np.array([1, 2, 3, 4], dtype="<i8")
    conflict_tokens = np.array([1, 2, 3, 9], dtype="<i8")
    replay_rows = []
    for alias in range(8):
      a_index = alias * 2
      matches = alias == 3 or (ambiguous_binding and alias == 4)
      tokens = source_tokens if matches else conflict_tokens
      replay_rows.append({
          "schema": "m15-apc-serving-envelope-v1",
          "serving_arm": "A",
          "diagnostic_round": 0,
          "call_index": 80 + alias,
          "requests": [{
              "request_id": f"decode-{a_index}",
              "num_tokens": 4,
              "token_history_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
          }],
      })
    (stage / "m15-replay-envelope.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in replay_rows),
        encoding="utf-8",
    )
    ab_bytes = 7 if red else 0
    ab_elements = 3 if red else 0
    alignment = {
        "diagnostic_round": 0,
        "boundaries": {
            "S_decode_vs_S_prefill": {
                "differing_bytes": ab_bytes,
                "differing_elements": ab_elements,
            },
            "S_prefill_vs_T_old": {
                "differing_bytes": 0,
                "differing_elements": 0,
            },
        },
    }
    (stage / "pre-alignment.jsonl").write_text(
        json.dumps(alignment, sort_keys=True) + "\n", encoding="utf-8"
    )
    round_input = {
        "schema": "m15-e0-kv-round-input-v1",
        "status": "STAGED_FOR_CLASSIFIER_CHECKPOINT",
        "arm": "on",
        "diagnostic_round": 0,
        "a_b_differing_bytes": ab_bytes,
        "a_b_differing_elements": ab_elements,
        "b_c_differing_bytes": 0,
        "b_c_differing_elements": 0,
        "capsule_present": red,
        "expected_source_commit": TARGET_SOURCE,
        "runtime_source_commit": TARGET_SOURCE,
        "kv_records": 16,
        "kv_pairs": 8,
    }
    (stage / "ROUND_INPUT.json").write_text(
        json.dumps(round_input, sort_keys=True), encoding="utf-8"
    )
    shutil.copyfile(CLASSIFIER, stage / CLASSIFIER.name)
    (stage / "CLASSIFIER_RUNTIME.json").write_text(
        json.dumps({
            "schema": "m15-e0-kv-classifier-runtime-v2",
            "status": "source-bound",
            "path": CLASSIFIER.name,
            "sha256": _sha256(CLASSIFIER),
            "runtime_source_commit": TARGET_SOURCE,
        }, sort_keys=True),
        encoding="utf-8",
    )
    names = sorted(path.name for path in stage.iterdir() if path.is_file())
    manifest = root / "CLASSIFIER_INPUT_SHA256SUMS"
    manifest.write_text(
        "".join(f"{_sha256(stage / name)}  {name}\n" for name in names),
        encoding="ascii",
    )
    archive = root / "CLASSIFIER_INPUT_ARCHIVE.tar"
    archive_tool.create_archive(stage, manifest, archive)
    receipt = root / "CLASSIFIER_INPUT_RECEIPT.json"
    receipt.write_text(
        json.dumps({
            "schema": "m15-e0-kv-classifier-input-receipt-v1",
            "status": "uploaded-readback-verified-before-classification",
            "arm": "on",
            "diagnostic_round": 0,
            "source_commit": TARGET_SOURCE,
            "runtime_source_commit": TARGET_SOURCE,
            "kv_records": 16,
            "kv_pairs": 8,
            "a_b_differing_bytes": ab_bytes,
            "archive_sha256": _sha256(archive),
            "manifest_sha256": _sha256(manifest),
        }, sort_keys=True),
        encoding="utf-8",
    )
    return archive, manifest, receipt

  def _recover(self, root: Path, **fixture_options):
    archive, manifest, receipt = self._fixture(root, **fixture_options)
    scratch = root / "scratch"
    scratch.mkdir()
    output = root / "output"
    report = review.recover(
        archive=archive,
        manifest=manifest,
        receipt_path=receipt,
        expected_source=TARGET_SOURCE,
        analysis_source=ANALYSIS_SOURCE,
        scratch=scratch,
        output=output,
    )
    return report, output

  def test_red_round_recovers_different_live_kv_fingerprint(self):
    with tempfile.TemporaryDirectory() as tmp:
      report, output = self._recover(Path(tmp))
      self.assertEqual(report["status"], "ROUND0_LIVE_KV_FINGERPRINT_DIFFERS")
      self.assertEqual(report["a_b_differing_bytes"], 7)
      self.assertEqual(report["b_c_differing_bytes"], 0)
      self.assertEqual(report["observer_geometry"]["target_seq_len"], 3)
      self.assertEqual(report["first_difference"]["layer"], 0)
      self.assertFalse(report["three_round_verdict"])
      self.assertFalse(report["numerical_repair_authorized"])
      self.assertFalse(report["b_full_reset_runtime_receipt_available"])
      self.assertFalse(
          report["all_num_cached_tokens_zero_runtime_receipt_available"]
      )
      self.assertEqual(len((output / "SHA256SUMS").read_text().splitlines()), 4)

  def test_red_round_recovers_equal_live_kv_fingerprint(self):
    with tempfile.TemporaryDirectory() as tmp:
      report, _ = self._recover(Path(tmp), fingerprint_differs=False)
      self.assertEqual(report["status"], "ROUND0_LIVE_KV_FINGERPRINT_EQUAL")
      self.assertIsNone(report["first_difference"])
      self.assertFalse(report["terminal_pair_complete"])

  def test_exact_round_is_non_reproduction_not_target_pass(self):
    with tempfile.TemporaryDirectory() as tmp:
      report, _ = self._recover(Path(tmp), red=False)
      self.assertEqual(report["status"], "ROUND0_TARGET_NON_REPRODUCTION")
      self.assertEqual(report["a_b_differing_bytes"], 0)
      self.assertIn("NO_TARGET_PASS", report["claim_ceiling"])

  def test_receipt_source_drift_fails_closed(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      archive, manifest, receipt = self._fixture(root)
      value = json.loads(receipt.read_text(encoding="utf-8"))
      value["runtime_source_commit"] = "0" * 40
      receipt.write_text(json.dumps(value), encoding="utf-8")
      scratch = root / "scratch"
      scratch.mkdir()
      with self.assertRaisesRegex(
          review.Attempt20Round0RecoveryError, "receipt contract"
      ):
        review.recover(
            archive=archive,
            manifest=manifest,
            receipt_path=receipt,
            expected_source=TARGET_SOURCE,
            analysis_source=ANALYSIS_SOURCE,
            scratch=scratch,
            output=root / "output",
        )

  def test_ambiguous_future_binding_returns_no_classification(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      archive, manifest, receipt = self._fixture(
          root, ambiguous_binding=True
      )
      scratch = root / "scratch"
      scratch.mkdir()
      output = root / "output"
      with self.assertRaisesRegex(
          review.Attempt20Round0RecoveryError, "classifier failed"
      ):
        review.recover(
            archive=archive,
            manifest=manifest,
            receipt_path=receipt,
            expected_source=TARGET_SOURCE,
            analysis_source=ANALYSIS_SOURCE,
            scratch=scratch,
            output=output,
        )
      self.assertFalse(output.exists())

  def test_missing_original_render_refuses_without_reconstruction(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      output = root / "output"
      completed = subprocess.run(
          [
              "bash", str(WRAPPER), str(root / "missing-render"),
              str(output), str(root),
          ],
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )
      self.assertEqual(completed.returncode, 2)
      self.assertIn(
          "status=ORIGINAL_RENDER_UNAVAILABLE classification=NONE",
          completed.stdout,
      )
      self.assertFalse(output.exists())


if __name__ == "__main__":
  unittest.main()
