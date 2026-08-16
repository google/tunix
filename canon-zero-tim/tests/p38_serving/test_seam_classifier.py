#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / (
    "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "classify_p38_seam.py"
)
SPEC = importlib.util.spec_from_file_location("p38_seam_classifier", MODULE)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classifier)


class SeamClassifierTest(unittest.TestCase):

  def _write_fixture(
      self,
      root: Path,
      mutate: bool = True,
      indices: tuple[int, int] = (0, 1),
  ) -> tuple[Path, Path]:
    prompt = np.asarray([[11, 12]], dtype=np.int32)
    completion = np.asarray([[21, 22, 23]], dtype=np.int32)
    tokens = np.concatenate((prompt[0], completion[0]))
    source_position = 2 + 1 - 1
    prefix = classifier._prefix_sha256(tokens[:source_position + 1])
    for index, arm in zip(indices, ("A", "B"), strict=True):
      layer = np.zeros((1, 2, 2, 8), dtype=np.uint32)
      if mutate and arm == "B":
        layer[0, 1, 1, 3] = 1
      arrays = {
          "row_indices": np.asarray([7], dtype=np.int32),
          "positions": np.asarray([source_position], dtype=np.int32),
          "token_ids": np.asarray([tokens[source_position]], dtype=np.int32),
          "request_ordinals": np.asarray([0], dtype=np.int32),
          "token_prefix_sha256": np.asarray([prefix], dtype="S64"),
          "layer_fingerprints": layer,
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
          "array_keys": sorted(arrays),
          "npz_sha256": classifier._sha256(npz),
      }
      (root / f"p38_seam_{index:06d}.json").write_text(json.dumps(record))
    capsule = root / "round0.npz"
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
    return root, capsule

  def _write_tail(self, root: Path, mutate_checkpoint: int | None = 1):
    seam = np.load(root / "p38_seam_000000.npz", allow_pickle=False)
    prefix = np.asarray(seam["token_prefix_sha256"])
    position = np.asarray(seam["positions"])
    source_token = np.asarray(seam["token_ids"])
    seam.close()
    for index, arm in enumerate(("A", "B")):
      values = np.zeros((1, len(classifier._TAIL_CHECKPOINTS)), np.float32)
      values[0, -1] = 1.0 if arm == "A" else 1.5
      if mutate_checkpoint is not None and arm == "B":
        values[0, mutate_checkpoint] = 0.25
      arrays = {
          "row_indices": np.asarray([7], dtype=np.int32),
          "positions": position,
          "token_ids": source_token,
          "request_ordinals": np.asarray([0], dtype=np.int32),
          "token_prefix_sha256": prefix,
          "logit_row_indices": np.asarray([9], dtype=np.int32),
          "target_ids": np.asarray([22], dtype=np.int32),
          "tail_values": values,
      }
      npz = root / f"p38_tail_{index:06d}.npz"
      np.savez(npz, **arrays)
      (root / f"p38_tail_{index:06d}.json").write_text(json.dumps({
          "schema": "p38-tail-values-v1",
          "record_index": index,
          "arm": arm,
          "diagnostic_round": 0,
          "checkpoint_names": list(classifier._TAIL_CHECKPOINTS),
          "array_keys": sorted(arrays),
          "npz_sha256": classifier._sha256(npz),
      }))

  def test_first_layer_output_difference_is_measured(self):
    with tempfile.TemporaryDirectory() as directory:
      root, capsule = self._write_fixture(Path(directory))
      report = classifier.classify(root, [capsule], "layer")
      self.assertEqual(report["joined_red_points"], 1)
      self.assertEqual(
          report["joins"][0]["first_difference"],
          {"layer": 1, "checkpoint": "layer_output",
           "differing_fingerprint_fields": [3]},
      )

  def test_exact_hidden_observer_requires_tail_localization(self):
    with tempfile.TemporaryDirectory() as directory:
      root, capsule = self._write_fixture(Path(directory), mutate=False)
      report = classifier.classify(root, [capsule], "layer")
      self.assertEqual(
          report["classification"],
          "hidden_chain_exact_tail_localization_required",
      )
      self.assertTrue(report["tail_localization_required"])
      self.assertEqual(report["joined_red_points"], 1)
      self.assertEqual(report["divergent_red_points"], 0)

  def test_exact_hidden_chain_is_localized_by_required_tail(self):
    with tempfile.TemporaryDirectory() as directory:
      root, capsule = self._write_fixture(Path(directory), mutate=False)
      self._write_tail(root, mutate_checkpoint=1)
      report = classifier.classify(
          root, [capsule], "layer", require_tail=True)
      self.assertEqual(
          report["classification"],
          "decode_terminal_first_difference_measured",
      )
      first = report["joins"][0]["first_difference"]
      self.assertEqual(first["checkpoint"], "raw_log_normalizer")
      self.assertEqual(first["layer"], None)
      self.assertEqual(first["max_abs"], 0.25)
      self.assertFalse(report["tail_localization_required"])

  def test_required_tail_rejects_capsule_endpoint_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root, capsule = self._write_fixture(Path(directory), mutate=False)
      self._write_tail(root, mutate_checkpoint=None)
      path = root / "p38_tail_000001.npz"
      with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True)
                  for name in archive.files}
      arrays["tail_values"][0, -1] = 1.0
      np.savez(path, **arrays)
      metadata_path = root / "p38_tail_000001.json"
      metadata = json.loads(metadata_path.read_text())
      metadata["npz_sha256"] = classifier._sha256(path)
      metadata_path.write_text(json.dumps(metadata))
      with self.assertRaisesRegex(
          classifier.SeamError, "differs from the mismatch capsule"
      ):
        classifier.classify(root, [capsule], "layer", require_tail=True)

  def test_missing_b_record_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root, capsule = self._write_fixture(Path(directory))
      (root / "p38_seam_000001.json").unlink()
      with self.assertRaisesRegex(
          classifier.SeamError, "not every red action joined"
      ):
        classifier.classify(root, [capsule], "layer")

  def test_reduction_manifest_admits_byte_preserving_sparse_indices(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      selected = root / "selected"
      capsules = root / "capsules"
      selected.mkdir()
      capsules.mkdir()
      _, capsule = self._write_fixture(selected, indices=(17, 42))
      reduced_capsule = capsules / capsule.name
      capsule.replace(reduced_capsule)
      files = []
      for path in sorted(selected.glob("p38_seam_*")):
        files.append({
            "path": f"selected/{path.name}",
            "sha256": classifier._sha256(path),
            "bytes": path.stat().st_size,
        })
      manifest = root / "REDUCTION_MANIFEST.json"
      manifest.write_text(json.dumps({
          "schema": "p38-seam-reduction-v1",
          "selection_complete": True,
          "unmatched_keys": [],
          "ambiguous_keys": [],
          "selected_directory": "selected",
          "selected_files": files,
          "capsules": [{
              "path": f"capsules/{reduced_capsule.name}",
              "sha256": classifier._sha256(reduced_capsule),
              "bytes": reduced_capsule.stat().st_size,
          }],
          "source_gcs_uri": (
              "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
              "job/attempt-0/live/000001"
          ),
          "source_snapshot_manifest_sha256": "0" * 64,
      }))
      report = classifier.classify(
          selected, [reduced_capsule], "layer", reduction_manifest=manifest)
      self.assertEqual(report["joined_red_points"], 1)
      self.assertEqual(report["selected_layer"], 1)
      self.assertIn("reduction_provenance", report)


if __name__ == "__main__":
  unittest.main()
