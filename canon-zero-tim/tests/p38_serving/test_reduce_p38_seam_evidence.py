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
        "npz_sha256": _sha256(npz),
    }
    (root / f"p38_seam_{index:06d}.json").write_text(json.dumps(record))

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
    inputs = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.name not in ("LIVE.json", "SHA256SUMS")
    )
    (root / "SHA256SUMS").write_text("".join(
        f"{_sha256(path)}  {path.name}\n" for path in inputs))
    return capsule

  def _run(self, source: Path, capsule: Path, output: Path):
    return subprocess.run(
        [
            sys.executable, str(REDUCER),
            "--source-dir", str(source),
            "--source-gcs-uri", SOURCE_URI,
            "--capsule", str(capsule),
            "--output-dir", str(output),
            "--mode", "layer",
            "--expected-rounds", "3",
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
      self.assertEqual(manifest["red_points"], 1)
      self.assertEqual(manifest["matched_arm_keys"], 2)
      self.assertEqual(
          (output / "selected/p38_seam_000017.npz").read_bytes(),
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
