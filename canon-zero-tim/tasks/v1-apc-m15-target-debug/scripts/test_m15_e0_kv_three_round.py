#!/usr/bin/env python3
"""Host gates for the independently sealed three-round M15 E0 KV carrier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str):
  path = SCRIPT_DIR / f"{name}.py"
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec and spec.loader
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


stager = _load("stage_m15_e0_kv_round")
aggregator = _load("aggregate_m15_e0_kv_rounds")
SOURCE = "a" * 40


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_record(
    root: Path, index: int, arm: str, round_index: int, source_index: int | None
) -> None:
  token_ids = np.array([1, 2, 3], dtype=np.int32)
  arrays = {
      "aggregates": np.zeros((1, 2, 4, 4), dtype=np.uint32),
      "samples": np.zeros((1, 2, 4, 3, 2), dtype=np.uint16),
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
  record = {
      "arm": arm,
      "array_keys": sorted(arrays),
      "block_size": 4,
      "cache_dtype": "bfloat16",
      "cache_shape": [8, 4, 1, 2, 4],
      "cache_sharding": "test",
      "diagnostic_round": round_index,
      "layer_count": 1,
      "layer_indices": [0],
      "logical_pages": 1,
      "npz_sha256": _sha256(npz),
      "observer_pages": 2,
      "record_index": index,
      "request_id": f"{arm.lower()}-{round_index}-{index}",
      "schema": "p38-live-kv-prefix-table-v1",
      "source_a_record_index": source_index,
      "source_a_request_id": (
          f"a-{round_index}-{source_index}" if source_index is not None
          else f"a-{round_index}-{index}"
      ),
      "target_seq_len": 3,
      "token_history_sha256": token_sha,
  }
  Path(str(base) + ".json").write_text(
      json.dumps(record), encoding="utf-8"
  )


def _alignment(round_index: int, a_b_bytes: int = 0) -> dict:
  return {
      "boundaries": {
          "S_decode_vs_S_prefill": {
              "differing_bytes": a_b_bytes,
              "differing_elements": a_b_bytes,
          },
          "S_prefill_vs_T_old": {
              "differing_bytes": 0,
              "differing_elements": 0,
          },
      },
      "diagnostic_round": round_index,
  }


class E0KvThreeRoundTest(unittest.TestCase):

  def _stage_fixture(self, root: Path) -> dict[str, Path]:
    observer = root / "observer"
    observer.mkdir()
    alignment = root / "pre-alignment.jsonl"
    replay = root / "m15-replay-envelope.jsonl"
    alignment.write_text(
        "".join(json.dumps(_alignment(value)) + "\n" for value in range(3)),
        encoding="utf-8",
    )
    replay.write_text(
        "".join(json.dumps({
            "schema": "m15-apc-serving-envelope-v1",
            "diagnostic_round": value,
        }) + "\n" for value in range(3)),
        encoding="utf-8",
    )
    for round_index in range(3):
      start = round_index * 16
      for alias in range(8):
        _write_record(observer, start + alias, "A", round_index, None)
        _write_record(
            observer, start + 8 + alias, "B", round_index, start + alias
        )
    return {
        "alignment": alignment,
        "capsule": root / "mismatch-capsule.npz",
        "classifier": (
            SCRIPT_DIR.parents[1]
            / "p38-pathways-decode-prefill-carrier/scripts/"
            "classify_p38_kv_observer.py"
        ),
        "observer": observer,
        "replay": replay,
    }

  def test_stager_selects_only_one_contiguous_round(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      fixture = self._stage_fixture(root)
      output = root / "round-1"
      result = stager.stage(
          directory=fixture["observer"],
          alignment_report=fixture["alignment"],
          capsule_base=fixture["capsule"],
          replay_ledger=fixture["replay"],
          classifier=fixture["classifier"],
          output=output,
          round_index=1,
          arm="off",
          expected_source=SOURCE,
          runtime_source=SOURCE,
      )
      self.assertEqual(result["kv_record_index_start"], 16)
      self.assertEqual(result["kv_record_index_end"], 31)
      self.assertEqual(result["kv_records"], 16)
      self.assertEqual(
          len(list(output.glob("p38_kv_observer_*.json"))), 16
      )
      self.assertEqual(
          {json.loads(path.read_text())["diagnostic_round"]
           for path in output.glob("p38_kv_observer_*.json")},
          {1},
      )

  def _write_round(
      self, root: Path, round_index: int, arm: str, outcome: str
  ) -> None:
    directory = root / f"{round_index:06d}"
    directory.mkdir(parents=True)
    red = outcome != "observer_pairs_valid_red_join_pending"
    round_input = {
        "a_b_differing_bytes": 7 if red else 0,
        "a_b_differing_elements": 3 if red else 0,
        "arm": arm,
        "b_c_differing_bytes": 0,
        "b_c_differing_elements": 0,
        "diagnostic_round": round_index,
        "expected_source_commit": SOURCE,
        "kv_pairs": 8,
        "kv_records": 16,
        "runtime_source_commit": SOURCE,
        "schema": "m15-e0-kv-round-input-v1",
    }
    classification = {
        "classification": outcome,
        "comparisons": [
            {"diagnostic_round": round_index} for _ in range(8)
        ],
        "pairs": 8,
        "records": 16,
        "schema": "p38-live-kv-classification-v2",
        "status": "PASS",
    }
    if red:
      classification["source_request_binding"] = {
          "status": "UNIQUE_FUTURE_PREFIX_BINDING"
      }
    input_path = directory / "ROUND_INPUT.json"
    classifier_path = directory / "kv-observer-classification.json"
    checkpoint_path = directory / "CLASSIFIER_INPUT_RECEIPT.json"
    input_path.write_text(json.dumps(round_input), encoding="utf-8")
    classifier_path.write_text(json.dumps(classification), encoding="utf-8")
    checkpoint_path.write_text(json.dumps({
        "a_b_differing_bytes": round_input["a_b_differing_bytes"],
        "arm": arm,
        "diagnostic_round": round_index,
        "kv_pairs": 8,
        "kv_records": 16,
        "runtime_source_commit": SOURCE,
        "schema": "m15-e0-kv-classifier-input-receipt-v1",
        "source_commit": SOURCE,
        "status": "uploaded-readback-verified-before-classification",
    }), encoding="utf-8")
    completion = {
        "arm": arm,
        "classification_sha256": _sha256(classifier_path),
        "classifier_input_receipt_sha256": _sha256(checkpoint_path),
        "diagnostic_round": round_index,
        "round_input_sha256": _sha256(input_path),
        "runtime_source_commit": SOURCE,
        "schema": "m15-e0-kv-round-completion-v1",
        "source_commit": SOURCE,
        "status": "sealed-uploaded-readback-verified",
    }
    (directory / "ROUND_COMPLETE.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )

  def test_aggregate_requires_three_stable_rounds(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for round_index in range(3):
        self._write_round(
            root, round_index, "on", "live_kv_fingerprint_differs_on_red_row"
        )
      result = aggregator.aggregate(root, "on", 3, SOURCE)
      self.assertEqual(result["status"], "LIVE_KV_FINGERPRINT_DIFFERS_3_OF_3")
      (root / "000002/ROUND_COMPLETE.json").unlink()
      with self.assertRaisesRegex(
          aggregator.E0AggregateError, "artifact is absent"
      ):
        aggregator.aggregate(root, "on", 3, SOURCE)

  def test_aggregate_rejects_mixed_treatment_outcomes(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      outcomes = [
          "live_kv_fingerprint_differs_on_red_row",
          "live_kv_fingerprint_equal_on_red_row",
          "live_kv_fingerprint_differs_on_red_row",
      ]
      for round_index, outcome in enumerate(outcomes):
        self._write_round(root, round_index, "on", outcome)
      with self.assertRaisesRegex(
          aggregator.E0AggregateError, "treatment is unstable"
      ):
        aggregator.aggregate(root, "on", 3, SOURCE)


if __name__ == "__main__":
  unittest.main()
