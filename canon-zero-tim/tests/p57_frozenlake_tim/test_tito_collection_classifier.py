#!/usr/bin/env python3
"""Negative controls for the P57 TiTO three-way witness classifier."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = (
    ROOT
    / "tasks/multiturn-tito-cross-workload/scripts/classify_tito_collection.py"
)
SPEC = importlib.util.spec_from_file_location(
    "p57_tito_collection_classifier", CLASSIFIER_PATH
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import TiTO collection classifier")
classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classifier)


def _write(path: Path, value: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
  path.chmod(0o600)


def _token_sha(tokens: list[int]) -> str:
  return classifier.hashlib.sha256(
      np.asarray(tokens, dtype="<i8").tobytes()
  ).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path, Path]:
  state = root / "state"
  witness = state / "p57_tito_witness"
  token_sha = _token_sha([1, 2])
  summary = {
      "schema": "canon.p57-tito-diagnostic.v1",
      "workload": "p45",
      "source_commit": "a" * 40,
      "dataset_eval_sha256": "b" * 64,
      "train_steps_before": 0,
      "train_steps_after": 0,
      "global_steps_before": 0,
      "global_steps_after": 0,
      "backward_calls": 0,
      "optimizer_commits": 0,
      "checkpoint_writes": 0,
      "collection": {
          "active": True,
          "mode": "collect-64",
          "trajectories": 1,
          "compared_trajectories": 1,
          "unexercised_single_turn_trajectories": 0,
          "equal_trajectories": 1,
          "different_trajectories": 0,
          "later_turn_comparisons": 1,
          "engine_echo_comparisons": 2,
          "engine_echo_differences": 0,
          "capsules_reserved": 0,
          "capsules_emitted": 0,
          "capsules_omitted": 0,
          "emission_failures": 0,
          "backward_transactions": 0,
          "gradient_microbatches": 0,
          "optimizer_commits": 0,
          "alignment_updates": 0,
      },
      "rollout": {
          "trajectories": 1,
          "records": [{"status": "SUCCEEDED"}],
      },
  }
  host = {
      "schema": "canon.p57-tito-host-witness.v1",
      "request_id": "request-1",
      "trajectory_id": "c" * 32,
      "workload": "p45",
      "turn": 0,
      "pair_index": "0",
      "group_id": "0",
      "submitted_tokens": 2,
      "submitted_sha256": token_sha,
      "engine_echo_tokens": 2,
      "engine_echo_sha256": token_sha,
      "submitted_equals_engine_echo": True,
  }
  runner = {
      "schema": "canon.p57-tito-runner-input.v1",
      "record_index": 1,
      "request_id": "request-1",
      "dp_rank": 0,
      "input_batch_index": 0,
      "prompt_tokens": 2,
      "prompt_sha256": token_sha,
  }
  summary_path = witness / "diagnostic-summary.json"
  host_path = witness / "host/host-request-a.json"
  runner_path = witness / "runner/runner-input-000001-a.json"
  _write(summary_path, summary)
  _write(host_path, host)
  _write(runner_path, runner)
  host_second = {**host, "request_id": "request-2", "turn": 1}
  runner_second = {
      **runner,
      "record_index": 2,
      "request_id": "request-2",
  }
  _write(witness / "host/host-request-b.json", host_second)
  _write(witness / "runner/runner-input-000002-b.json", runner_second)
  return state, host_path, runner_path


def _add_difference_capsule(state: Path) -> Path:
  summary_path = state / "p57_tito_witness/diagnostic-summary.json"
  summary = json.loads(summary_path.read_text())
  summary["collection"].update({
      "equal_trajectories": 0,
      "different_trajectories": 1,
      "capsules_reserved": 1,
      "capsules_emitted": 1,
  })
  summary["rollout"]["records"][0]["status"] = (
      "TOKEN_CONTINUITY_DIFFERENT"
  )
  _write(summary_path, summary)
  actual = np.asarray([1, 3], dtype=np.int32)
  expected = np.asarray([1, 2], dtype=np.int32)
  digest = lambda value: classifier.hashlib.sha256(  # noqa: E731
      np.ascontiguousarray(value).tobytes()
  ).hexdigest()
  capsule = {
      "schema": "p57-token-first-diff-capsule-v1",
      "header": {
          "schema": "p57-token-first-diff-v1",
          "record": "header",
          "capsule_id": "d" * 32,
          "workload": "p45",
          "trajectory_id": "c" * 32,
          "policy_step": 0,
          "turn": 1,
          "pair_index": "0",
          "group_id": "0",
          "trajectory_steps": 1,
          "actual_tokens": 2,
          "expected_tokens": 2,
          "actual_sha256": digest(actual),
          "expected_sha256": digest(expected),
          "first_mismatch": 1,
          "segments": 2,
          "token_chunk_records": 3,
          "records_metadata_sha256": "e" * 64,
      },
      "actual": {
          "stream": "actual",
          "segment_index": 0,
          "kind": "serving_prompt",
          "turn_index": 1,
          "length": 2,
          "sha256": digest(actual),
          "tokens": actual.tolist(),
      },
      "expected_segments": [
          {
              "stream": "expected",
              "segment_index": 0,
              "kind": "initial_prompt",
              "turn_index": -1,
              "length": 1,
              "sha256": digest(expected[:1]),
              "tokens": expected[:1].tolist(),
          },
          {
              "stream": "expected",
              "segment_index": 1,
              "kind": "assistant",
              "turn_index": 0,
              "done": True,
              "length": 1,
              "sha256": digest(expected[1:]),
              "tokens": expected[1:].tolist(),
          },
      ],
  }
  path = state / "token-continuity-first-diff/p57-p45-diff.json"
  _write(path, capsule)
  return path


class TitoCollectionClassifierTest(unittest.TestCase):

  def test_three_way_witness_passes(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, _, _ = _fixture(Path(tmp))
      result = classifier.classify(state)
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["witness_verdict"], "PASS")
      self.assertEqual(result["token_verdict"], "EQUAL")
      self.assertEqual(result["requests"], 2)

  def test_verified_difference_capsule_is_preserved_as_data_not_gate_failure(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, _, _ = _fixture(Path(tmp))
      capsule_path = _add_difference_capsule(state)
      result = classifier.classify(state)
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["token_verdict"], "DIFFERENT")
      self.assertEqual(result["capsules_emitted"], 1)
      capsule = json.loads(capsule_path.read_text())
      capsule["header"]["first_mismatch"] = 0
      _write(capsule_path, capsule)
      with self.assertRaisesRegex(ValueError, "verified difference"):
        classifier.classify(state)

  def test_capsule_segment_attribution_is_verified(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, _, _ = _fixture(Path(tmp))
      capsule_path = _add_difference_capsule(state)
      capsule = json.loads(capsule_path.read_text())
      capsule["expected_segments"][1]["turn_index"] = 7
      _write(capsule_path, capsule)
      with self.assertRaisesRegex(ValueError, "assistant attribution"):
        classifier.classify(state)

  def test_wrong_hash_missing_foreign_duplicate_and_order_fail(self):
    mutations = ("hash", "missing", "foreign", "duplicate", "order")
    for mutation in mutations:
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
        state, host_path, runner_path = _fixture(Path(tmp))
        if mutation == "hash":
          runner = json.loads(runner_path.read_text())
          runner["prompt_sha256"] = "f" * 64
          _write(runner_path, runner)
        elif mutation == "missing":
          runner_path.unlink()
        elif mutation == "foreign":
          runner = json.loads(runner_path.read_text())
          runner["request_id"] = "foreign"
          _write(runner_path, runner)
        elif mutation == "duplicate":
          _write(
              host_path.with_name("host-request-duplicate.json"),
              json.loads(host_path.read_text()),
          )
        else:
          runner = json.loads(runner_path.read_text())
          runner["record_index"] = 2
          _write(runner_path, runner)
        with self.assertRaises(ValueError):
          classifier.classify(state)

  def test_emission_failure_and_training_state_mutation_fail(self):
    for mutation in ("emission", "training"):
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
        state, _, _ = _fixture(Path(tmp))
        path = state / "p57_tito_witness/diagnostic-summary.json"
        summary = json.loads(path.read_text())
        if mutation == "emission":
          summary["collection"].update({
              "different_trajectories": 1,
              "equal_trajectories": 0,
              "capsules_reserved": 1,
              "emission_failures": 1,
          })
          summary["rollout"]["records"][0]["status"] = (
              "TOKEN_CONTINUITY_DIFFERENT"
          )
        else:
          summary["train_steps_after"] = 1
        _write(path, summary)
        with self.assertRaises(ValueError):
          classifier.classify(state)

  def test_host_workload_and_turn_coverage_fail(self):
    for mutation in ("workload", "turn"):
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
        state, host_path, _ = _fixture(Path(tmp))
        host = json.loads(host_path.read_text())
        if mutation == "workload":
          host["workload"] = "m15"
        else:
          host["turn"] = 1
        _write(host_path, host)
        with self.assertRaises(ValueError):
          classifier.classify(state)


if __name__ == "__main__":
  unittest.main()
