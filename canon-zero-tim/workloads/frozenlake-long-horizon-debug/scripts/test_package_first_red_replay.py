#!/usr/bin/env python3
"""Regression tests for bounded first-red return packaging."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).with_name("package_first_red_replay.py")
SPEC = importlib.util.spec_from_file_location("package_first_red_replay", MODULE)
assert SPEC and SPEC.loader
package_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = package_module
SPEC.loader.exec_module(package_module)


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class PackageFirstRedReplayTest(unittest.TestCase):

  def _inputs(self, root: Path, *, status: str = "FRESH_TARGET_RED_FROZEN"):
    row_arrays = {
        "prompt_ids": np.arange(12, dtype=np.int32).reshape(2, 6),
        "prompt_mask": np.ones((2, 6), dtype=np.bool_),
        "completion_ids": np.arange(16, dtype=np.int32).reshape(2, 8),
        "completion_valid_mask": np.ones((2, 8), dtype=np.bool_),
        "action_mask": np.ones((2, 8), dtype=np.bool_),
        "s_decode": np.ones((2, 8), dtype=np.float32),
        "s_prefill": np.zeros((2, 8), dtype=np.float32),
        "t_old": np.zeros((2, 8), dtype=np.float32),
        "policy_version": np.arange(2, dtype=np.int32).reshape(2, 1),
        "sampling_values": np.ones((2, 3), dtype=np.float32),
    }
    metadata = {
        "schema": "p38-frozenlake-mismatch-capsule-v1",
        "selected_rows": [10, 9],
        "row_identity": [
            {"source_row": 10, "batch_group_index": 1, "generation_index": 2},
            {"source_row": 9, "batch_group_index": 1, "generation_index": 1},
        ],
        "arrays": {},
    }
    for name, value in row_arrays.items():
      metadata["arrays"][name] = {
          "shape": list(value.shape),
          "dtype": str(value.dtype),
          "sha256": hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest(),
      }
    capsule = root / "capsule.npz"
    np.savez_compressed(
        capsule,
        selected_rows=np.array([10, 9], dtype=np.int32),
        metadata_json=np.frombuffer(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
            dtype=np.uint8,
        ),
        **row_arrays,
    )
    capsule_sha = sha256(capsule)
    joins = [
        {
            "source_row": 10, "completion_position": 2,
            "num_computed_tokens": 1300, "request_id": "late",
            "call_index": 8, "dp_rank": 3, "local_scheduler_slot": 4,
            "token_history_sha256": "a" * 64, "physical_pages": [3],
            "page_generations": [7], "scheduled_request_count": 2,
            "co_batch_request_ids": ["late", "peer"],
        },
        {
            "source_row": 9, "completion_position": 0,
            "num_computed_tokens": 1226, "request_id": "first",
            "call_index": 7, "dp_rank": 2, "local_scheduler_slot": 1,
            "token_history_sha256": "b" * 64, "physical_pages": [1, 2],
            "page_generations": [4, 5], "scheduled_request_count": 1,
            "co_batch_request_ids": ["first"],
        },
    ]
    capture = root / "capture.json"
    capture.write_text(json.dumps({
        "verdict": "PASS",
        "mismatch_capsule": {"sha256": capsule_sha},
        "incident_exact_joins": joins,
    }), encoding="utf-8")
    m15 = root / "m15.json"
    m15.write_text(json.dumps({
        "status": status,
        "arm": "on",
        "source_commit": "7" * 40,
        "artifacts": {"mismatch_capsule_sha256": capsule_sha},
    }), encoding="utf-8")
    return capsule, capture, m15

  def test_packages_earliest_incident_row_and_attests_output(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      capsule, capture, m15 = self._inputs(root)
      output = root / "bundle"
      result = package_module.package(
          capsule_path=capsule,
          capture_classification_path=capture,
          m15_classification_path=m15,
          output_dir=output,
      )
      self.assertEqual(result["status"], "FIRST_RED_ROW_FROZEN")
      self.assertEqual(result["source_row"], 9)
      with np.load(output / "first_red_capsule.npz", allow_pickle=False) as archive:
        self.assertEqual(archive["selected_rows"].tolist(), [9])
        self.assertEqual(archive["prompt_ids"].shape[0], 1)
        self.assertEqual(archive["policy_version"].shape[0], 1)
      self.assertEqual(len((output / "SHA256SUMS").read_text().splitlines()), 2)

  def test_rejects_nonred_input(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      capsule, capture, m15 = self._inputs(root, status="TARGET_NOT_REPRODUCED")
      with self.assertRaisesRegex(package_module.PackageError, "fresh APC-on target red"):
        package_module.package(
            capsule_path=capsule,
            capture_classification_path=capture,
            m15_classification_path=m15,
            output_dir=root / "bundle",
        )


if __name__ == "__main__":
  unittest.main()
