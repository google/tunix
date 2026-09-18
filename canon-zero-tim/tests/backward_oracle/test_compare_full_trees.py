#!/usr/bin/env python3
"""Negative controls for the P61 full-tree numerical comparator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PATH = Path(__file__).with_name("compare_full_trees.py")
SPEC = importlib.util.spec_from_file_location("p61_compare_full_trees", PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARE
SPEC.loader.exec_module(COMPARE)


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture(root: Path, name: str, arrays: list[np.ndarray]) -> None:
  directory = root / name
  directory.mkdir(parents=True)
  leaves = []
  total = 0
  for index, value in enumerate(arrays):
    value = np.ascontiguousarray(value)
    path = directory / f"leaf_{index:05d}.npy"
    with path.open("xb") as output:
      np.save(output, value, allow_pickle=False)
    total += value.nbytes
    leaves.append({
        "index": index,
        "path": f"['leaf{index}']",
        "file": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "elements": int(value.size),
        "data_bytes": int(value.nbytes),
        "data_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        "file_sha256": _sha(path),
    })
  (directory / "manifest.json").write_text(
      json.dumps({
          "schema": "canon-p61-full-tree-capture-v1",
          "capture": name,
          "leaves": leaves,
          "leaf_count": len(leaves),
          "total_data_bytes": total,
      }),
      encoding="utf-8",
  )


def _hashes(token: str = "same") -> list[dict[str, str]]:
  return [
      {key: f"{key}-{token}-{index}" for key in COMPARE.HASH_KEYS}
      for index in range(16)
  ]


class CompareFullTreesTest(unittest.TestCase):

  def _fixture(self, root: Path) -> dict[str, Path]:
    control = root / "control"
    candidate = root / "candidate"
    before = [
        np.array([1.0, -2.0, 3.0], np.float32),
        np.array([0.25, -0.5], np.float32),
    ]
    control_gradient = [
        np.array([0.5, -0.25, 0.125], np.float32),
        np.array([0.75, -1.0], np.float32),
    ]
    candidate_gradient = [value.copy() for value in control_gradient]
    control_after = [
        value - np.float32(0.01) * gradient
        for value, gradient in zip(before, control_gradient, strict=True)
    ]
    candidate_after = [value.copy() for value in control_after]
    for arm, gradient, after in (
        (control, control_gradient, control_after),
        (candidate, candidate_gradient, candidate_after),
    ):
      _capture(arm, "model_before", [value.copy() for value in before])
      _capture(arm, "gradient", gradient)
      _capture(arm, "model_after", after)

    paths = {
        "control_root": control,
        "candidate_root": candidate,
        "control_update": root / "control.update.jsonl",
        "candidate_update": root / "candidate.update.jsonl",
        "control_classification": root / "control.classification.json",
        "candidate_classification": root / "candidate.classification.json",
        "tier1_baseline": root / "tier1.json",
    }
    for arm in ("control", "candidate"):
      update = {
          "verdict": "PASS",
          "dp_size": 4,
          "tp_size": 1,
          "dp_pullback_invocations_per_transaction": 4 if arm == "control" else 1,
          "alignment_hashes": _hashes(),
          "state_fingerprints_before": {"model": "same"},
          "commit_evidence": {
              "effective_learning_rate": 2.0e-7,
              "parameter_changed_elements": 5,
          },
      }
      paths[f"{arm}_update"].write_text(json.dumps(update) + "\n", encoding="utf-8")
      paths[f"{arm}_classification"].write_text(
          json.dumps({
              "verdict": "PASS",
              "zero_tim": {
                  "expected_pass": 17,
                  "observed_pass": 17,
                  "observed_fail": 0,
              },
          }),
          encoding="utf-8",
      )
    paths["tier1_baseline"].write_text(
        json.dumps({
            "schema": "canon-p61-tier1-baseline-v1",
            "gradient": {key: 1.0e-3 for key in COMPARE.METRIC_KEYS},
            "parameter_update": {
                key: 1.0e-3 for key in COMPARE.METRIC_KEYS
            },
        }),
        encoding="utf-8",
    )
    return paths

  def test_exact_pair_passes(self):
    with tempfile.TemporaryDirectory() as directory:
      result = COMPARE.compare(**self._fixture(Path(directory)))
    self.assertEqual(result["verdict"], "NUMERICAL_KEEP_DP4_PROXY")
    self.assertEqual(result["gradient"]["rel_l2"], 0.0)
    self.assertEqual(result["parameter_update"]["rel_l2"], 0.0)
    self.assertTrue(result["model_before_array_exact"])

  def test_material_gradient_error_rejects(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      paths = self._fixture(root)
      _capture_path = root / "candidate" / "gradient"
      for path in _capture_path.iterdir():
        path.unlink()
      _capture_path.rmdir()
      _capture(
          root / "candidate",
          "gradient",
          [
              np.zeros(3, np.float32),
              np.array([0.75, -1.0], np.float32),
          ],
      )
      result = COMPARE.compare(**paths)
    self.assertEqual(result["verdict"], "NUMERICAL_REJECT")
    self.assertTrue(result["gradient"]["dead_candidate_leaves"])

  def test_input_hash_mismatch_is_inconclusive(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      paths = self._fixture(root)
      update = json.loads(paths["candidate_update"].read_text())
      update["alignment_hashes"] = _hashes("different")
      paths["candidate_update"].write_text(json.dumps(update) + "\n")
      result = COMPARE.compare(**paths)
    self.assertEqual(result["verdict"], "INCONCLUSIVE_CARRIER")
    self.assertIn("same_input_seven_hashes", result["contract_reasons"])

  def test_real_alignment_fail_is_fatal(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      paths = self._fixture(root)
      classification = json.loads(
          paths["candidate_classification"].read_text()
      )
      classification["verdict"] = "FAIL"
      classification["zero_tim"]["observed_pass"] = 16
      classification["zero_tim"]["observed_fail"] = 1
      paths["candidate_classification"].write_text(json.dumps(classification))
      result = COMPARE.compare(**paths)
    self.assertEqual(result["verdict"], "REJECT_ZERO_TIM")

  def test_tampered_leaf_is_refused(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      paths = self._fixture(root)
      leaf = root / "candidate" / "gradient" / "leaf_00000.npy"
      with leaf.open("ab") as output:
        output.write(b"tamper")
      with self.assertRaisesRegex(ValueError, "SHA mismatch"):
        COMPARE.compare(**paths)


if __name__ == "__main__":
  unittest.main()
