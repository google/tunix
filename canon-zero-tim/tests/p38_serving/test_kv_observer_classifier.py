#!/usr/bin/env python3

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE = (
    ROOT / "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "classify_p38_kv_observer.py"
)
SPEC = importlib.util.spec_from_file_location("p38_kv_classifier", MODULE)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classifier)


class KvObserverClassifierTest(unittest.TestCase):

  def _record(
      self,
      root: Path,
      index: int,
      arm: str,
      *,
      changed=False,
      invalid_tail_changed=False,
      request_id: str | None = None,
      source_a_index: int = 0,
      layer_indices: list[int] | None = None,
  ):
    token_ids = np.array([1, 2, 3], dtype=np.int32)
    aggregates = np.zeros((1, 2, 4, 4), dtype=np.uint32)
    samples = np.zeros((1, 2, 4, 3, 2), dtype=np.uint16)
    if changed:
      aggregates[0, 0, 1, 0] = 1
    if invalid_tail_changed:
      samples[0, 0, 3, 0, 0] = 1
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
        np.ascontiguousarray(token_ids, dtype="<i8").tobytes()).hexdigest()
    resolved_request_id = request_id or (
        "decode-a" if arm == "A" else "clean-b"
    )
    resolved_source_a_request_id = (
        resolved_request_id if arm == "A"
        else ("decode-a" if request_id is None else f"decode-{source_a_index}")
    )
    record = {
        "schema": "p38-live-kv-prefix-table-v1",
        "arm": arm,
        "record_index": index,
        "request_id": resolved_request_id,
        "source_a_request_id": resolved_source_a_request_id,
        "source_a_record_index": None if arm == "A" else source_a_index,
        "diagnostic_round": 0,
        "target_seq_len": 3,
        "token_history_sha256": token_sha,
        "block_size": 4,
        "logical_pages": 1,
        "observer_pages": 2,
        "layer_count": 1,
        "cache_shape": [8, 4, 1, 2, 4],
        "cache_dtype": "bfloat16",
        "cache_sharding": "test",
        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "array_keys": sorted(arrays),
    }
    if layer_indices is not None:
      record["layer_indices"] = layer_indices
    Path(str(base) + ".json").write_text(json.dumps(record))

  def _capsule(self, root: Path) -> Path:
    path = root / "p38_frozenlake_mismatch_capsule.round-0.npz"
    metadata = json.dumps({
        "schema": "p38-frozenlake-mismatch-capsule-v1",
        "diagnostic_round": 0,
    }).encode()
    np.savez(
        path,
        metadata_json=np.frombuffer(metadata, dtype=np.uint8),
        selected_rows=np.array([255], dtype=np.int32),
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.array([[True, True]]),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_valid_mask=np.array([[True, True]]),
        action_mask=np.array([[True, True]]),
        s_decode=np.array([[0.1, 0.0]], dtype=np.float32),
        s_prefill=np.array([[0.2, 0.0]], dtype=np.float32),
    )
    return path

  def test_exact_pair_without_capsule_is_valid_but_not_a_mechanism_verdict(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._record(root, 0, "A")
      self._record(root, 1, "B")
      report = classifier.classify(root, [], False)
      self.assertEqual(
          report["classification"], "observer_pairs_valid_red_join_pending")
      self.assertTrue(report["comparisons"][0]["fingerprint_equal"])
      self.assertEqual(report["schema"], "p38-live-kv-classification-v2")
      self.assertEqual(report["comparisons"][0]["valid_tokens"], [3])
      self.assertEqual(len(report["source_inputs"]["observer_records"]), 2)
      self.assertEqual(report["source_inputs"]["capsules"], [])
      self.assertEqual(
          len(report["source_inputs"]["classifier"]["sha256"]), 64
      )

  def test_invalid_page_tail_difference_is_masked(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._record(root, 0, "A", invalid_tail_changed=True)
      self._record(root, 1, "B")
      report = classifier.classify(root, [self._capsule(root)], True)
      self.assertEqual(
          report["classification"],
          "live_kv_fingerprint_equal_on_red_row",
      )
      self.assertTrue(report["comparisons"][0]["fingerprint_equal"])
      self.assertEqual(
          report["comparisons"][0]["sample_prefix_cells_differing"], 0
      )

  def test_red_join_with_changed_prefix_cell_localizes_first_difference(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._record(root, 0, "A", changed=True)
      self._record(root, 1, "B")
      report = classifier.classify(root, [self._capsule(root)], True)
      self.assertEqual(
          report["classification"],
          "live_kv_fingerprint_differs_on_red_row",
      )
      self.assertEqual(report["red_joins"][0]["source_row"], 255)
      self.assertEqual(
          report["comparisons"][0]["first_difference"],
          {
              "layer": 0,
              "logical_page": 0,
              "page_prefix_extent": 2,
              "aggregate_diff": True,
              "sample_diff": False,
          },
      )
      self.assertEqual(
          report["source_inputs"]["capsules"][0]["path"],
          "p38_frozenlake_mismatch_capsule.round-0.npz",
      )
      self.assertEqual(
          len(report["source_inputs"]["capsules"][0]["sha256"]), 64
      )

  def test_selected_layer_reports_absolute_layer_index(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._record(root, 0, "A", changed=True, layer_indices=[7])
      self._record(root, 1, "B", layer_indices=[7])
      report = classifier.classify(root, [self._capsule(root)], True)
      self.assertEqual(
          report["comparisons"][0]["first_difference"]["layer"], 7
      )
      self.assertEqual(report["comparisons"][0]["differing_layers"], [7])

  def test_replay_ledger_selects_one_future_bound_alias(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for alias in range(8):
        a_index = alias * 2
        self._record(
            root, a_index, "A", changed=alias == 3,
            request_id=f"decode-{a_index}", source_a_index=a_index,
            layer_indices=[0],
        )
        self._record(
            root, a_index + 1, "B", request_id=f"clean-{a_index}",
            source_a_index=a_index, layer_indices=[0],
        )
      source = np.array([1, 2, 3, 4], dtype=np.int32)
      conflict = np.array([1, 2, 3, 9], dtype=np.int32)
      lines = []
      for alias in range(8):
        request_id = f"decode-{alias * 2}"
        values = source if alias == 3 else conflict
        digest = hashlib.sha256(
            np.ascontiguousarray(values, dtype="<i8").tobytes()
        ).hexdigest()
        lines.append(json.dumps({
            "schema": "m15-apc-serving-envelope-v1",
            "serving_arm": "A",
            "diagnostic_round": 0,
            "call_index": 10 + alias,
            "requests": [{
                "request_id": request_id,
                "num_tokens": 4,
                "token_history_sha256": digest,
            }],
        }))
      replay = root / "m15_replay_envelope.jsonl"
      replay.write_text("\n".join(lines) + "\n", encoding="utf-8")
      report = classifier.classify(
          root, [self._capsule(root)], True, replay
      )
      self.assertEqual(
          report["classification"],
          "live_kv_fingerprint_differs_on_red_row",
      )
      binding = report["source_request_binding"]
      self.assertEqual(binding["status"], "UNIQUE_FUTURE_PREFIX_BINDING")
      self.assertEqual(binding["selected_request_id"], "decode-6")
      self.assertEqual(binding["selected_source_a_record_index"], 6)
      self.assertEqual(binding["required_elimination_horizon"], 4)
      self.assertEqual(binding["selected_proof_prefix_tokens"], 4)
      self.assertEqual(
          report["source_inputs"]["replay_ledger"]["path"], replay.name
      )

  def test_require_red_join_rejects_an_unjoined_observer_pair(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._record(root, 0, "A")
      self._record(root, 1, "B")
      self._record(root, 2, "A")
      self._record(root, 3, "B")
      second_json = root / "p38_kv_observer_0002_a.json"
      second = json.loads(second_json.read_text())
      second["diagnostic_round"] = 1
      second_json.write_text(json.dumps(second))
      clean_json = root / "p38_kv_observer_0003_b.json"
      clean = json.loads(clean_json.read_text())
      clean["diagnostic_round"] = 1
      clean["source_a_record_index"] = 2
      clean_json.write_text(json.dumps(clean))
      with self.assertRaisesRegex(
          classifier.ObserverError, "not every observer pair joined"
      ):
        classifier.classify(root, [self._capsule(root)], True)

  def test_missing_clean_pair_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self._record(root, 0, "A")
      with self.assertRaisesRegex(classifier.ObserverError, "counts differ"):
        classifier.classify(root, [], False)


if __name__ == "__main__":
  unittest.main()
