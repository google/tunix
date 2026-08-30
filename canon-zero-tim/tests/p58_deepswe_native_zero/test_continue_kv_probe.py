#!/usr/bin/env python3
"""Regression tests for the P58.22 continue-decode KV discriminator."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts"
    / "classify_continue_kv_probe.py"
)
SOURCE_SHA = "1" * 40


def _load():
  spec = importlib.util.spec_from_file_location("continue_kv_probe", SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PROBE = _load()


def _write_kv(root: Path, target: np.ndarray, *, differs: bool) -> None:
  directory = root / "continue-kv"
  directory.mkdir()
  page_size = 16
  observer_pages = 192
  logical_pages = (target.size + page_size - 1) // page_size
  valid = np.full((logical_pages,), page_size, dtype=np.int32)
  valid[-1] = target.size - (logical_pages - 1) * page_size
  base_aggregates = np.zeros(
      (1, observer_pages, page_size, 4), dtype=np.uint64
  )
  base_samples = np.zeros(
      (1, observer_pages, page_size, 3, 2), dtype=np.uint16
  )
  token_sha = hashlib.sha256(
      np.ascontiguousarray(target, dtype="<i8").tobytes()
  ).hexdigest()
  for index, arm in enumerate(("A", "B")):
    aggregates = base_aggregates.copy()
    if differs and arm == "B":
      aggregates[0, 8, 12, 0] = 1
    arrays = {
        "aggregates": aggregates,
        "samples": base_samples.copy(),
        "token_ids": target,
        "physical_pages": np.arange(logical_pages, dtype=np.int32),
        "padded_global_pages": np.arange(observer_pages, dtype=np.int32),
        "valid_tokens": valid,
    }
    base = directory / f"p38_kv_observer_{index:04d}_{arm.lower()}"
    np.savez(str(base) + ".npz", **arrays)
    npz_path = Path(str(base) + ".npz")
    metadata = {
        "schema": "p38-live-kv-prefix-table-v1",
        "arm": arm,
        "record_index": index,
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "array_keys": sorted(arrays),
        "request_id": "live-a" if arm == "A" else "clean-b",
        "source_a_request_id": "live-a",
        "source_a_record_index": None if arm == "A" else 0,
        "diagnostic_round": 0,
        "tag_call_index": 4,
        "tag_prefix": 2280,
        "dp_rank": 0,
        "target_seq_len": int(target.size),
        "token_history_sha256": token_sha,
        "block_size": page_size,
        "logical_pages": logical_pages,
        "observer_pages": observer_pages,
        "layer_count": 1,
        "cache_shape": [32, page_size, 8, 2, 128],
        "cache_dtype": "bfloat16",
        "cache_sharding": "NamedSharding(mesh=Mesh('model': 4))",
        "cache_effective_sharding": {
            "schema": "p38-effective-device-sharding-v1",
            "global_shape": [32, page_size, 8, 2, 128],
            "devices": [{
                "platform": "tpu",
                "process_index": 0,
                "id": 0,
                "coords": [0, 0, 0],
                "core_on_chip": 0,
                "index": [
                    {"kind": "slice", "start": 0, "stop": 32, "step": 1},
                    {"kind": "slice", "start": 0, "stop": page_size, "step": 1},
                    {"kind": "slice", "start": 0, "stop": 8, "step": 1},
                    {"kind": "slice", "start": 0, "stop": 2, "step": 1},
                    {"kind": "slice", "start": 0, "stop": 128, "step": 1},
                ],
            }],
        },
        "device_read_bytes": 1,
        "elapsed_seconds": 0.1,
        "fingerprint_claim": "diagnostic_non_cryptographic",
    }
    Path(str(base) + ".json").write_text(json.dumps(metadata))


def _write_fixture(
    root: Path,
    *,
    kv_differs: bool,
    repaired_exact: bool = False,
    left_pad_prompt: bool = False,
) -> None:
  prompt = np.arange(2200, dtype=np.int32) % 101
  stored_prompt = (
      np.concatenate((np.full((128,), 151643, dtype=np.int32), prompt))
      if left_pad_prompt
      else prompt
  )
  completion = np.arange(200, dtype=np.int32) % 97
  completion[86] = 12
  target = np.concatenate((prompt, completion[:150]))
  manifest = {
      "schema": "canon.local.deepswe.run-manifest.v1",
      "source_commit": SOURCE_SHA,
      "expected_hostname": "v5p-host",
      "model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "contract_name": "local-qwen4b-dp1-tp4-zero-admission",
      "role_topology": {"dp": 1, "tp": 4, "devices": 4},
      "onehost_seam_probe": True,
      "onehost_xprof_arm": "zero-hp",
      "stage": "backward-no-commit",
      "whitelist_sha256": "2" * 64,
      "q4_tp4_zero_admission": True,
      "q4_tp4_seam_diagnostic": "",
      "q4_tp4_continue_kv_diagnostic": True,
      "alignment_precheck_only": repaired_exact,
      "alignment_controlled_exit": repaired_exact,
      "continue_decode_steps": "8",
      "sampling_contract": {
          "source": "explicit-cli",
          "temperature": 0.7,
          "top_k": 0,
          "top_p": 1.0,
      },
      "global_trajectories": 2,
  }
  (root / "run_manifest.json").write_text(json.dumps(manifest))
  rows = [{
      "schema": "canon.local.deepswe.trajectory.v1",
      "status": "SUCCEEDED",
      "compact_filtered": False,
      "trajectory": {
          "prompt_length": 2200,
          "prompt_tokens": stored_prompt.tolist(),
          "conversation_tokens": completion.tolist(),
          "conversation_masks": [1] * 200,
          "old_logprobs": [(-0.25 if index == 86 else -0.5)
                           for index in range(200)],
      },
  }, {
      "schema": "canon.local.deepswe.trajectory.v1",
      "status": "MODEL_TIMEOUT",
      "compact_filtered": True,
      "trajectory": {
          "prompt_length": None,
          "prompt_tokens": [],
          "conversation_tokens": [],
          "conversation_masks": [],
          "old_logprobs": [],
      },
  }]
  trajectory = root / "batch-000000.trajectories.jsonl.gz"
  with gzip.open(trajectory, "wt") as output:
    for row in rows:
      output.write(json.dumps(row) + "\n")
  mismatch = {
      "coordinate": [0, 86],
      "completion_position": 86,
      "completion_valid_length": 200,
      "prompt_length": 2200,
      "logical_kv_prefix_length": 2286,
      "token_id": 12,
      "a": -0.25,
      "b": -0.20,
      "action_run_start": True,
      "previous_token_is_environment": True,
  }
  a_b = {
      "valid": True,
      "finite": True,
      "differing_elements": 0 if repaired_exact else 1,
      "differing_bytes": 0 if repaired_exact else 2,
      "total_elements": 200,
      "element_fraction": 0.0 if repaired_exact else 0.005,
      "byte_fraction": 0.0 if repaired_exact else 0.0025,
      "max_abs": 0.0 if repaired_exact else 0.05,
      "first_mismatch": None if repaired_exact else mismatch,
      "mismatches": [] if repaired_exact else [mismatch],
      "mismatches_truncated": False,
  }
  b_c = {
      "valid": True,
      "finite": True,
      "differing_elements": 0,
      "differing_bytes": 0,
      "total_elements": 200,
      "element_fraction": 0.0,
      "byte_fraction": 0.0,
      "max_abs": 0.0,
      "first_mismatch": None,
      "mismatches": [],
      "mismatches_truncated": False,
  }
  (root / "pre_alignment.jsonl").write_text(json.dumps({
      "N_action": 200,
      "boundaries": {
          "S_decode_vs_S_prefill": a_b,
          "S_prefill_vs_T_old": b_c,
      },
  }) + "\n")
  (root / "batch_metrics.jsonl").write_text(json.dumps({
      "schema": "canon.local.deepswe.batch-metrics.v1",
      "trajectories": 2,
      "trajectory_path": trajectory.name,
      "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
  }) + "\n")
  (root / "probe_process_status.json").write_text(json.dumps({
      "profile": "seam",
      "training_process_status": 42 if repaired_exact else 1,
  }))
  (root / "raw.log").write_text(
      "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD "
      "rounds=1 step=0 N_action=200 verdict=PASS "
      f"a_b_differing_bytes={0 if repaired_exact else 2}\n"
      "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0\n"
      if repaired_exact
      else "strict gate stopped before backward\n"
  )
  _write_kv(root, target, differs=kv_differs)


class ContinueKvProbeTest(unittest.TestCase):

  def test_repr_drift_is_accepted_when_effective_sharding_matches(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, kv_differs=False)
      clean_path = (
          root / "continue-kv" / "p38_kv_observer_0001_b.json"
      )
      clean = json.loads(clean_path.read_text())
      clean["cache_sharding"] = "equivalent-unit-axis-repr"
      clean_path.write_text(json.dumps(clean))
      report = PROBE.classify(
          root, source_sha=SOURCE_SHA, expected_hostname="v5p-host"
      )
      self.assertTrue(report["kv_observer"]["comparison"][
          "cache_effective_sharding_equal"
      ])
      self.assertFalse(report["kv_observer"]["comparison"][
          "cache_sharding_repr_equal"
      ])

  def test_effective_sharding_drift_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, kv_differs=False)
      clean_path = (
          root / "continue-kv" / "p38_kv_observer_0001_b.json"
      )
      clean = json.loads(clean_path.read_text())
      clean["cache_effective_sharding"]["devices"][0]["id"] = 3
      clean_path.write_text(json.dumps(clean))
      with self.assertRaises(PROBE.KV.ObserverError):
        PROBE.classify(
            root, source_sha=SOURCE_SHA, expected_hostname="v5p-host"
        )

  def test_equal_fingerprint_points_to_read_program(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, kv_differs=False)
      report = PROBE.classify(
          root, source_sha=SOURCE_SHA, expected_hostname="v5p-host"
      )
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(
          report["classification"],
          "LIVE_KV_FINGERPRINT_EQUAL_READ_PROGRAM_SUSPECT",
      )

  def test_different_fingerprint_points_to_write_state(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, kv_differs=True)
      report = PROBE.classify(
          root, source_sha=SOURCE_SHA, expected_hostname="v5p-host"
      )
      self.assertEqual(
          report["classification"],
          "LIVE_KV_FINGERPRINT_DIFFERS_WRITE_STATE_SUSPECT",
      )
      self.assertFalse(
          report["kv_observer"]["comparison"]["fingerprint_equal"]
      )

  def test_repaired_exact_token_continuity_is_alignment_only_pass(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, kv_differs=False, repaired_exact=True)
      report = PROBE.classify(
          root, source_sha=SOURCE_SHA, expected_hostname="v5p-host"
      )
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(
          report["classification"], "EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS"
      )
      self.assertEqual(
          report["decode_alignment"]["outcome"],
          "ZERO_TIM_ALIGNMENT_ONLY_PASS",
      )
      self.assertIn("does not certify backward", report["claim"])

  def test_left_padded_durable_prompt_joins_live_semantic_prefix(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(
          root,
          kv_differs=False,
          repaired_exact=True,
          left_pad_prompt=True,
      )
      report = PROBE.classify(
          root, source_sha=SOURCE_SHA, expected_hostname="v5p-host"
      )
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(
          report["classification"], "EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS"
      )


if __name__ == "__main__":
  unittest.main()
