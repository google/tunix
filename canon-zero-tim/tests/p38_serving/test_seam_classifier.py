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

  def _write_fixture(self, root: Path, mutate: bool = True) -> tuple[Path, Path]:
    prompt = np.asarray([[11, 12]], dtype=np.int32)
    completion = np.asarray([[21, 22, 23]], dtype=np.int32)
    tokens = np.concatenate((prompt[0], completion[0]))
    source_position = 2 + 1 - 1
    prefix = classifier._prefix_sha256(tokens[:source_position + 1])
    for index, arm in enumerate(("A", "B")):
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

  def test_exact_observer_on_red_action_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root, capsule = self._write_fixture(Path(directory), mutate=False)
      with self.assertRaisesRegex(
          classifier.SeamError, "no divergent seam fingerprint"
      ):
        classifier.classify(root, [capsule], "layer")

  def test_missing_b_record_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root, capsule = self._write_fixture(Path(directory))
      (root / "p38_seam_000001.json").unlink()
      with self.assertRaisesRegex(
          classifier.SeamError, "not every red action joined"
      ):
        classifier.classify(root, [capsule], "layer")


if __name__ == "__main__":
  unittest.main()
