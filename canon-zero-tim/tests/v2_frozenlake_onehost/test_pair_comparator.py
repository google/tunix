"""Tests for strict adjacent-arm FrozenLake pair admission."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
COMPARATOR_PATH = (
    ROOT
    / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
    "compare_frozenlake_dp2tp2_pair.py"
)
SPEC = importlib.util.spec_from_file_location(
    "v2_frozenlake_onehost_pair", COMPARATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
comparator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparator
SPEC.loader.exec_module(comparator)


def _manifest(arm: str, label: str) -> dict:
  return {
      "schema": "canon.v2-frozenlake-onehost.run.v1",
      "source_commit": "a" * 40,
      "source_diff_sha256": "b" * 64,
      "image_id_sha256": "c" * 64,
      "model_snapshot_sha256": "d" * 64,
      "dataset_train_sha256": "e" * 64,
      "dataset_test_sha256": "f" * 64,
      "runner_sha256": "0" * 64,
      "label": label,
      "workload": "m15",
      "workload_name": "frozenlake-m15-onehost-dp2-tp2",
      "arm": arm,
      "stage": "backward-no-commit",
      "model_id": "Qwen/Qwen3-8B",
      "model_dir_name": "qwen8b_tp2",
      "topology": {"dp": 2, "tp": 2, "devices": 4},
      "global_prompts": 4,
      "num_generations": 4,
      "global_trajectories": 16,
      "gradient_groups": 8,
      "local_m": 256,
      "global_m": 512,
      "max_prompt_length": 4096,
      "max_response_length": 8192,
      "max_turns": 15,
      "data_shuffle_seed": 42,
      "vllm_global_seed": 0,
      "vllm_hbm_utilization": 0.37,
      "selectors": comparator._ARMS[arm],  # pylint: disable=protected-access
      "checked_vma": True,
      "wandb_mode": "disabled",
      "classification_mode": "certify",
  }


def _classification(arm: str) -> dict:
  return {
      "schema": "canon.v2-frozenlake-onehost.classification.v1",
      "verdict": "PASS",
      "workload": "m15",
      "arm": arm,
      "classification_mode": "certify",
      "zero_tim": {
          "input_hashes": {
              "tokens": "1" * 64,
              "action_mask": "2" * 64,
              "S_decode": "3" * 64,
              "S_prefill": "4" * 64,
              "T_old": "4" * 64,
          },
          "action_tokens": 5000,
      },
      "landed_shape": {
          "n_real": list(range(4000, 4016)),
          "group_chunks": [16] * 8,
      },
  }


def _apply_expected_length_sort(classification: dict) -> None:
  n_real = sorted(classification["landed_shape"]["n_real"], reverse=True)
  classification["landed_shape"]["n_real"] = n_real
  classification["landed_shape"]["group_chunks"] = [
      (max(n_real[index:index + 2]) + 255) // 256
      for index in range(0, len(n_real), 2)
  ]


class FrozenLakePairComparatorTest(unittest.TestCase):

  def test_selector_schema_matches_the_six_field_run_manifest(self):
    self.assertEqual(
        comparator._ARMS,  # pylint: disable=protected-access
        {
            "r0": {
                "keep_tape": "0",
                "reduce_once": "0",
                "length_sort": "0",
                "report_adjoint_buckets": "0",
                "chunk_dependency_ticket": "0",
                "chunk_backpressure": "0",
            },
            "r1": {
                "keep_tape": "stream",
                "reduce_once": "0",
                "length_sort": "0",
                "report_adjoint_buckets": "0",
                "chunk_dependency_ticket": "0",
                "chunk_backpressure": "0",
            },
            "r2": {
                "keep_tape": "stream",
                "reduce_once": "1",
                "length_sort": "0",
                "report_adjoint_buckets": "0",
                "chunk_dependency_ticket": "0",
                "chunk_backpressure": "0",
            },
            "r3": {
                "keep_tape": "stream",
                "reduce_once": "1",
                "length_sort": "1",
                "report_adjoint_buckets": "0",
                "chunk_dependency_ticket": "0",
                "chunk_backpressure": "0",
            },
        },
    )

  def _pair(self, mutate=None, *, left_arm="r1", right_arm="r2") -> dict:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      left = root / "left"
      right = root / "right"
      left.mkdir()
      right.mkdir()
      payload = {
          "left_manifest": _manifest(left_arm, "left-fresh"),
          "right_manifest": _manifest(right_arm, "right-fresh"),
          "left_class": _classification(left_arm),
          "right_class": _classification(right_arm),
      }
      if (left_arm, right_arm) == ("r2", "r3"):
        _apply_expected_length_sort(payload["right_class"])
      if mutate is not None:
        mutate(payload)
      (left / "run_manifest.json").write_text(
          json.dumps(payload["left_manifest"]), encoding="utf-8"
      )
      (right / "run_manifest.json").write_text(
          json.dumps(payload["right_manifest"]), encoding="utf-8"
      )
      (left / "classification.json").write_text(
          json.dumps(payload["left_class"]), encoding="utf-8"
      )
      (right / "classification.json").write_text(
          json.dumps(payload["right_class"]), encoding="utf-8"
      )
      return comparator.compare(left, right)

  def test_exact_adjacent_pair_is_performance_eligible(self):
    result = self._pair()
    self.assertEqual(result["verdict"], "PASS")
    self.assertTrue(result["performance_eligible"])

  def test_r2_to_r3_requires_the_exact_length_sort_schedule(self):
    result = self._pair(left_arm="r2", right_arm="r3")
    self.assertEqual(result["verdict"], "PASS")
    self.assertEqual(result["input_relation"], "stable-length-sort")
    self.assertTrue(result["matched_input"])

    def mutate_unsorted(payload):
      payload["right_class"]["landed_shape"] = copy.deepcopy(
          payload["left_class"]["landed_shape"]
      )

    result = self._pair(
        mutate_unsorted, left_arm="r2", right_arm="r3"
    )
    self.assertEqual(result["verdict"], "INCOMPARABLE_INPUT")
    self.assertIn("length_sort_n_real_order", result["input_reasons"])

  def test_r2_to_r3_rejects_length_multiset_or_derived_chunk_drift(self):
    def mutate_length(payload):
      payload["right_class"]["landed_shape"]["n_real"][-1] += 1

    result = self._pair(mutate_length, left_arm="r2", right_arm="r3")
    self.assertEqual(result["verdict"], "INCOMPARABLE_INPUT")
    self.assertIn("landed_n_real_multiset", result["input_reasons"])

    def mutate_chunks(payload):
      payload["right_class"]["landed_shape"]["group_chunks"][0] += 1

    result = self._pair(mutate_chunks, left_arm="r2", right_arm="r3")
    self.assertEqual(result["verdict"], "INCOMPARABLE_INPUT")
    self.assertIn(
        "right_group_chunks_from_n_real", result["input_reasons"]
    )

  def test_input_hash_or_landed_length_drift_is_incomparable(self):
    def mutate_hash(payload):
      payload["right_class"]["zero_tim"]["input_hashes"]["tokens"] = "9" * 64

    def mutate_length(payload):
      payload["right_class"]["landed_shape"]["n_real"][5] += 1

    for name, mutation, reason in (
        ("hash", mutate_hash, "input_hashes"),
        ("length", mutate_length, "landed_n_real"),
    ):
      with self.subTest(name=name):
        result = self._pair(mutation)
        self.assertEqual(result["verdict"], "INCOMPARABLE_INPUT")
        self.assertIn(reason, result["input_reasons"])
        self.assertFalse(result["performance_eligible"])

  def test_nonadjacent_or_source_drift_is_incomparable_configuration(self):
    result = self._pair(left_arm="r0", right_arm="r2")
    self.assertEqual(result["verdict"], "INCOMPARABLE_CONFIGURATION")

    def mutate(payload):
      payload["right_manifest"]["source_commit"] = "9" * 40

    result = self._pair(mutate)
    self.assertEqual(result["verdict"], "INCOMPARABLE_CONFIGURATION")
    self.assertIn("source_commit", result["manifest_mismatches"])

  def test_failed_arm_blocks_the_pair(self):
    def mutate(payload):
      payload["right_class"]["verdict"] = "FAIL"

    result = self._pair(mutate)
    self.assertEqual(result["verdict"], "FAIL")
    self.assertFalse(result["performance_eligible"])

  def test_selector_drift_is_incomparable_configuration(self):
    def mutate(payload):
      payload["right_manifest"]["selectors"] = copy.deepcopy(
          payload["right_manifest"]["selectors"]
      )
      payload["right_manifest"]["selectors"]["reduce_once"] = "0"

    result = self._pair(mutate)
    self.assertEqual(result["verdict"], "INCOMPARABLE_CONFIGURATION")
    self.assertIn("right_selectors", result["configuration_reasons"])

    def mutate_diagnostic_selector(payload):
      payload["right_manifest"]["selectors"] = copy.deepcopy(
          payload["right_manifest"]["selectors"]
      )
      payload["right_manifest"]["selectors"][
          "chunk_dependency_ticket"
      ] = "1"

    result = self._pair(mutate_diagnostic_selector)
    self.assertEqual(result["verdict"], "INCOMPARABLE_CONFIGURATION")
    self.assertIn("right_selectors", result["configuration_reasons"])

  def test_measurement_mode_cannot_enter_a_performance_pair(self):
    def mutate(payload):
      payload["right_manifest"]["classification_mode"] = "measure"
      payload["right_class"]["classification_mode"] = "measure"

    result = self._pair(mutate)
    self.assertEqual(result["verdict"], "INCOMPARABLE_CONFIGURATION")
    self.assertIn(
        "right_classification_not_certify", result["configuration_reasons"]
    )

  def test_missing_hash_inventory_cannot_match_by_none_equality(self):
    def mutate(payload):
      payload["left_class"]["zero_tim"].pop("input_hashes")
      payload["right_class"]["zero_tim"].pop("input_hashes")

    result = self._pair(mutate)
    self.assertEqual(result["verdict"], "INCOMPARABLE_INPUT")
    self.assertIn("left_input_hash_inventory", result["input_reasons"])


if __name__ == "__main__":
  unittest.main()
