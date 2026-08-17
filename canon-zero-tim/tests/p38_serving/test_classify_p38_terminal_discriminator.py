#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "classify_p38_terminal_discriminator.py"
)
SPEC = importlib.util.spec_from_file_location("p38_terminal_classifier", SCRIPT)
assert SPEC and SPEC.loader
classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classifier)


class TerminalDiscriminatorClassifierTest(unittest.TestCase):

  def _prefix(self) -> bytes:
    return hashlib.sha256(
        np.asarray([101, 11], dtype="<i8").tobytes()).hexdigest().encode()

  def _write(
      self, root: Path, index: int, arm: str, stage: str,
      *, position: int = 1, target: int = 19,
  ) -> None:
    hidden = np.zeros((1, 8), dtype=np.float32)
    signatures = np.zeros((1, 2, 6), dtype=np.uint32)
    processed_signatures = np.zeros((1, 2, 6), dtype=np.uint32)
    block_max = np.zeros((1, 2), dtype=np.float32)
    block_sum = np.ones((1, 2), dtype=np.float32)
    row_max = np.zeros((1,), dtype=np.float32)
    block_lse = np.ones((1,), dtype=np.float32)
    processed_block_max = np.zeros((1, 2), dtype=np.float32)
    processed_block_sum = np.ones((1, 2), dtype=np.float32)
    processed_row_max = np.zeros((1,), dtype=np.float32)
    processed_block_lse = np.ones((1,), dtype=np.float32)
    tail = np.zeros((1, 6), dtype=np.float32)
    if arm == "A" and stage != "exact":
      tail[0, -1] = 0.25
      if stage == "pre_lm_head_hidden":
        hidden[0, 2] = 1
      elif stage == "lm_head_logits":
        signatures[0, 1, 2] = 1
      elif stage == "vocab_block_reduction":
        block_sum[0, 1] = 2
      elif stage == "logits_processing":
        processed_signatures[0, 1, 2] = 1
      elif stage == "processed_vocab_block_reduction":
        processed_block_sum[0, 1] = 2
    arrays = {
        "row_indices": np.asarray([17], dtype=np.int32),
        "positions": np.asarray([position], dtype=np.int32),
        "token_ids": np.asarray([11], dtype=np.int32),
        "request_ordinals": np.asarray([0], dtype=np.int32),
        "token_prefix_sha256": np.asarray([self._prefix()], dtype="S64"),
        "logit_row_indices": np.asarray([5], dtype=np.int32),
        "target_ids": np.asarray([target], dtype=np.int32),
        "final_hidden_rows": hidden,
        "raw_logit_signatures": signatures,
        "raw_block_max": block_max,
        "raw_block_exp_sum": block_sum,
        "raw_row_max": row_max,
        "raw_block_observer_log_normalizer": block_lse,
        "processed_logit_signatures": processed_signatures,
        "processed_block_max": processed_block_max,
        "processed_block_exp_sum": processed_block_sum,
        "processed_row_max": processed_row_max,
        "processed_block_observer_log_normalizer": processed_block_lse,
        "tail_values": tail,
    }
    npz = root / f"p38_terminal_{index:06d}.npz"
    np.savez(npz, **arrays)
    metadata = {
        "arm": arm,
        "array_keys": sorted(arrays),
        "diagnostic_round": 0,
        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "schema": "p38-terminal-discriminator-v1",
        "reduction_program": "shared-fixed-four-row-v1",
    }
    (root / f"p38_terminal_{index:06d}.json").write_text(
        json.dumps(metadata), encoding="utf-8")

  def _classify(self, stage: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._write(root, 0, "A", stage)
      self._write(root, 1, "B", stage)
      return classifier.classify(root)

  def test_classifies_every_registered_terminal_branch(self):
    for stage in (
        "pre_lm_head_hidden",
        "lm_head_logits",
        "vocab_block_reduction",
        "logits_processing",
        "processed_vocab_block_reduction",
        "production_tail_only",
    ):
      with self.subTest(stage=stage):
        self.assertEqual(self._classify(stage)["classification"], stage)

  def test_exact_rows_are_not_reported_as_a_repair(self):
    result = self._classify("exact")
    self.assertEqual(result["classification"], "terminal_rows_exact")
    self.assertEqual(result["red_rows"], [])

  def test_conflicting_duplicate_is_fail_closed(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._write(root, 0, "A", "production_tail_only")
      self._write(root, 1, "A", "lm_head_logits")
      self._write(root, 2, "B", "production_tail_only")
      with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
        classifier.classify(root)

  def _write_capsule(self, root: Path) -> Path:
    path = root / "capsule.npz"
    metadata = np.frombuffer(
        json.dumps({"diagnostic_round": 0}).encode(), dtype=np.uint8)
    np.savez(
        path,
        metadata_json=metadata,
        selected_rows=np.asarray([7], np.int32),
        prompt_ids=np.asarray([[101]], np.int32),
        prompt_mask=np.asarray([[True]]),
        completion_ids=np.asarray([[11, 19]], np.int32),
        completion_valid_mask=np.asarray([[True, True]]),
        action_mask=np.asarray([[False, True]]),
        s_decode=np.asarray([[0.0, 0.25]], np.float32),
        s_prefill=np.asarray([[0.0, 0.0]], np.float32),
    )
    return path

  def test_capsule_scope_requires_every_red_point(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      capsule = self._write_capsule(root)
      self._write(root, 0, "A", "production_tail_only")
      self._write(root, 1, "B", "production_tail_only")
      result = classifier.classify(
          root, capsules=[capsule], require_red_join=True)
      self.assertEqual(result["selection_scope"], "capsule_red_points")
      self.assertEqual(result["joined_red_points"], 1)
      self.assertEqual(result["classification"], "production_tail_only")

      self._write(root, 2, "A", "production_tail_only", target=23)
      with np.load(capsule, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
      arrays["completion_ids"] = np.asarray([[11, 23]], np.int32)
      np.savez(capsule, **arrays)
      with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
        classifier.classify(root, capsules=[capsule], require_red_join=True)

  def test_required_red_join_without_capsule_fails_closed(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._write(root, 0, "A", "exact")
      self._write(root, 1, "B", "exact")
      with self.assertRaisesRegex(ValueError, "requires a capsule"):
        classifier.classify(root, require_red_join=True)

  def test_old_shape_dependent_reduction_records_are_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._write(root, 0, "A", "exact")
      metadata_path = root / "p38_terminal_000000.json"
      metadata = json.loads(metadata_path.read_text())
      metadata.pop("reduction_program")
      metadata_path.write_text(json.dumps(metadata))
      with self.assertRaisesRegex(ValueError, "reduction program drifted"):
        classifier.classify(root)

  def test_legacy_tail_observer_drift_does_not_override_exact_endpoint(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._write(root, 0, "A", "exact")
      self._write(root, 1, "B", "exact")
      npz = root / "p38_terminal_000000.npz"
      with np.load(npz, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
      arrays["tail_values"] = arrays["tail_values"].copy()
      arrays["tail_values"][0, 1] = 1
      np.savez(npz, **arrays)
      metadata_path = root / "p38_terminal_000000.json"
      metadata = json.loads(metadata_path.read_text())
      metadata["npz_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
      metadata_path.write_text(json.dumps(metadata))
      result = classifier.classify(root)
      self.assertEqual(result["classification"], "terminal_rows_exact")
      self.assertEqual(
          len(result["legacy_shape_dependent_tail_drift_rows"]), 1)


if __name__ == "__main__":
  unittest.main()
