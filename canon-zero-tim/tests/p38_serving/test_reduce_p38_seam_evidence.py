#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REDUCER = ROOT / (
    "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "reduce_p38_seam_evidence.py"
)
AUDITOR = ROOT / (
    "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "audit_p38_seam_reduction.py"
)
SOURCE_URI = (
    "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
    "canon-p38-test/attempt-0/live/000001"
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class ReduceP38SeamEvidenceTest(unittest.TestCase):

  def _write_record(
      self,
      root: Path,
      index: int,
      arm: str,
      prefix: bytes,
      *,
      mutate: bool = False,
  ) -> None:
    layers = np.zeros((1, 2, 2, 8), dtype=np.uint32)
    if mutate:
      layers[0, 1, 1, 3] = 1
    arrays = {
        "row_indices": np.asarray([255], dtype=np.int32),
        "positions": np.asarray([2], dtype=np.int32),
        "token_ids": np.asarray([21], dtype=np.int32),
        "request_ordinals": np.asarray([0], dtype=np.int32),
        "token_prefix_sha256": np.asarray([prefix], dtype="S64"),
        "layer_fingerprints": layers,
        "final_norm_fingerprints": np.zeros((1, 8), dtype=np.uint32),
    }
    npz = root / f"p38_seam_{index:06d}.npz"
    np.savez(npz, **arrays)
    record = {
        "schema": "p38-seam-fingerprint-v1",
        "record_index": index,
        "arm": arm,
        "diagnostic_round": 0,
        "observer_mode": "layer",
        "checkpoint_names": ["layer_input", "layer_output"],
        "layer_indices": [0, 1],
        "call_index": index + 100,
        "program_path": "standard",
        "requests": [],
        "npz_sha256": _sha256(npz),
    }
    (root / f"p38_seam_{index:06d}.json").write_text(json.dumps(record))

  def _seal_source(self, root: Path) -> None:
    inputs = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.name not in ("LIVE.json", "SHA256SUMS")
    )
    (root / "SHA256SUMS").write_text("".join(
        f"{_sha256(path)}  {path.name}\n" for path in inputs))

  def _write_source(self, root: Path, *, include_b: bool = True) -> Path:
    prompt = np.asarray([[11, 12]], dtype=np.int32)
    completion = np.asarray([[21, 22, 23]], dtype=np.int32)
    tokens = np.concatenate((prompt[0], completion[0]))
    prefix = hashlib.sha256(
        np.ascontiguousarray(tokens[:3], dtype="<i8").tobytes()
    ).hexdigest().encode()
    self._write_record(root, 17, "A", prefix)
    if include_b:
      self._write_record(root, 42, "B", prefix)
    self._write_record(root, 99, "A", b"f" * 64)
    capsule = root / "p38_frozenlake_mismatch_capsule.round-000000.npz"
    np.savez(
        capsule,
        metadata_json=np.frombuffer(
            json.dumps({"diagnostic_round": 0}).encode(), dtype=np.uint8),
        selected_rows=np.asarray([255], dtype=np.int32),
        prompt_ids=prompt,
        prompt_mask=np.ones_like(prompt, dtype=np.bool_),
        completion_ids=completion,
        completion_valid_mask=np.ones_like(completion, dtype=np.bool_),
        action_mask=np.ones((1, 3), dtype=np.bool_),
        s_decode=np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32),
        s_prefill=np.asarray([[0.0, 1.5, 2.0]], dtype=np.float32),
    )
    (root / "pre-alignment.jsonl").write_text('{"step": 0}\n')
    (root / "run.log").write_text("partial run\n")
    (root / "LIVE.json").write_text(json.dumps({
        "schema": "canon-p38-gcs-live-v1",
        "sequence": 1,
        "prefix": SOURCE_URI,
    }))
    self._seal_source(root)
    return capsule

  def _write_snapshot_selection(self, root: Path) -> Path:
    path = root / "snapshot-selection.json"
    path.write_text(json.dumps({
        "schema": "p38-live-snapshot-selection-v1",
        "selection_complete": True,
        "minimum_capsule_rounds": 1,
        "selected_snapshot": "000001",
        "selected_source_gcs_uri": SOURCE_URI,
        "selected_capsule_rounds": [0],
    }))
    return path

  def _run(self, source: Path, capsule: Path, output: Path):
    snapshot_selection = self._write_snapshot_selection(output.parent)
    return subprocess.run(
        [
            sys.executable, str(REDUCER),
            "--source-dir", str(source),
            "--source-gcs-uri", SOURCE_URI,
            "--snapshot-selection", str(snapshot_selection),
            "--capsule", str(capsule),
            "--output-dir", str(output),
            "--mode", "layer",
            "--expected-rounds", "3",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

  def _audit(self, bundle: Path, output: Path):
    return subprocess.run(
        [
            sys.executable, str(AUDITOR),
            "--bundle-dir", str(bundle),
            "--output", str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

  def test_sparse_subset_is_byte_preserving_and_reclassifiable(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source"
      source.mkdir()
      capsule = self._write_source(source)
      output = root / "output"
      result = self._run(source, capsule, output)
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn("verdict=INCONCLUSIVE_PARTIAL_RUN", result.stdout)
      manifest = json.loads((output / "REDUCTION_MANIFEST.json").read_text())
      self.assertTrue(manifest["selection_complete"])
      self.assertEqual(manifest["source_seam_records"], 3)
      self.assertEqual(manifest["selected_record_indices"], [17, 42])
      self.assertEqual(manifest["candidate_record_indices"], [17, 42])
      self.assertEqual(manifest["red_points"], 1)
      self.assertEqual(manifest["matched_arm_keys"], 2)
      self.assertEqual(
          (output / "records/p38_seam_000017.npz").read_bytes(),
          (source / "p38_seam_000017.npz").read_bytes(),
      )
      report = json.loads((output / "classification.json").read_text())
      self.assertEqual(report["joined_red_points"], 1)
      self.assertEqual(
          report["classification"],
          "hidden_chain_exact_tail_localization_required",
      )
      self.assertFalse(manifest["run_contract_complete"])
      for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        self.assertEqual(_sha256(output / relative), expected)
      audit = self._audit(output, root / "bundle-audit.json")
      self.assertEqual(audit.returncode, 0, audit.stderr)
      self.assertIn(
          "scientific_verdict=INCONCLUSIVE_PARTIAL_RUN", audit.stdout)

  def test_missing_b_is_packaged_as_inconclusive(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source"
      source.mkdir()
      capsule = self._write_source(source, include_b=False)
      output = root / "output"
      result = self._run(source, capsule, output)
      self.assertEqual(result.returncode, 4, result.stderr)
      report = json.loads((output / "verdict.json").read_text())
      self.assertEqual(report["verdict"], "INCONCLUSIVE_REDUCTION_JOIN")
      manifest = json.loads((output / "REDUCTION_MANIFEST.json").read_text())
      self.assertFalse(manifest["selection_complete"])
      self.assertEqual(len(manifest["unmatched_keys"]), 1)

  def test_equivalent_duplicate_is_preserved_and_resolved_as_alias(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source"
      source.mkdir()
      capsule = self._write_source(source)
      with np.load(capsule, allow_pickle=False) as archive:
        tokens = np.concatenate((archive["prompt_ids"][0],
                                 archive["completion_ids"][0]))
      prefix = hashlib.sha256(
          np.ascontiguousarray(tokens[:3], dtype="<i8").tobytes()
      ).hexdigest().encode()
      self._write_record(source, 18, "A", prefix)
      self._seal_source(source)
      output = root / "output"
      result = self._run(source, capsule, output)
      self.assertEqual(result.returncode, 0, result.stderr)
      manifest = json.loads((output / "REDUCTION_MANIFEST.json").read_text())
      self.assertTrue(manifest["selection_complete"])
      self.assertEqual(manifest["matched_arm_keys"], 2)
      self.assertEqual(len(manifest["equivalent_alias_keys"]), 1)
      alias = manifest["equivalent_alias_keys"][0]
      self.assertEqual(alias["selected"]["record_index"], 17)
      self.assertEqual(alias["aliases"][0]["record_index"], 18)
      self.assertTrue((output / "records/p38_seam_000018.npz").is_file())
      report = json.loads((output / "classification.json").read_text())
      self.assertEqual(report["joined_red_points"], 1)

  def test_numerically_different_duplicate_is_fully_audited(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source"
      source.mkdir()
      capsule = self._write_source(source)
      with np.load(capsule, allow_pickle=False) as archive:
        tokens = np.concatenate((archive["prompt_ids"][0],
                                 archive["completion_ids"][0]))
      prefix = hashlib.sha256(
          np.ascontiguousarray(tokens[:3], dtype="<i8").tobytes()
      ).hexdigest().encode()
      self._write_record(source, 18, "A", prefix, mutate=True)
      self._seal_source(source)
      output = root / "output"
      result = self._run(source, capsule, output)
      self.assertEqual(result.returncode, 4, result.stderr)
      audit = json.loads((output / "AMBIGUITY_AUDIT.json").read_text())
      self.assertEqual(len(audit["payload_conflict_keys"]), 1)
      candidates = audit["payload_conflict_keys"][0]["candidates"]
      self.assertEqual([item["record_index"] for item in candidates], [17, 18])
      self.assertNotEqual(
          candidates[0]["numeric_payload_sha256"],
          candidates[1]["numeric_payload_sha256"],
      )
      self.assertTrue((output / "records/p38_seam_000017.npz").is_file())
      self.assertTrue((output / "records/p38_seam_000018.npz").is_file())
      self.assertFalse((output / "classification.json").exists())
      audit_result = self._audit(output, root / "bundle-audit.json")
      self.assertEqual(audit_result.returncode, 0, audit_result.stderr)
      self.assertIn(
          "scientific_verdict=INCONCLUSIVE_REDUCTION_JOIN",
          audit_result.stdout,
      )

  def test_returned_bundle_mutation_is_rejected_by_auditor(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source"
      source.mkdir()
      capsule = self._write_source(source)
      output = root / "output"
      result = self._run(source, capsule, output)
      self.assertEqual(result.returncode, 0, result.stderr)
      with (output / "records/p38_seam_000017.npz").open("ab") as stream:
        stream.write(b"tamper")
      audit = self._audit(output, root / "bundle-audit.json")
      self.assertEqual(audit.returncode, 2)
      self.assertIn("bundle SHA failed", audit.stderr)

  def test_source_sha_mutation_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source"
      source.mkdir()
      capsule = self._write_source(source)
      with capsule.open("ab") as stream:
        stream.write(b"tamper")
      result = self._run(source, capsule, root / "output")
      self.assertEqual(result.returncode, 2)
      self.assertIn("source manifest SHA failed", result.stderr)


if __name__ == "__main__":
  unittest.main()
