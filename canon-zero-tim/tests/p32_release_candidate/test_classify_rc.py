#!/usr/bin/env python3
"""Unit tests for the DP16 checkpoint-forward classifier."""

from __future__ import annotations

import copy
import json
import unittest

from classify_rc import classify_text


def _record() -> dict:
  return {
      "attempt": 0,
      "stage": "checkpoint-forward",
      "topology": {
          "devices": 64,
          "dp": 16,
          "mesh_shape": [16, 4],
          "tp": 4,
          "unique_devices": 64,
      },
      "batch": {
          "global_trajectories": 256,
          "local_trajectories": 16,
          "sample_to_rank_mapping": "frozen-contiguous-16",
          "sequence_length": 16,
      },
      "scope": {
          "production_training_admitted": False,
          "rollout_engine_initialized": False,
          "zero_tim_alignment": "NOT_MEASURED",
      },
      "model": {
          "name": "qwen3-8b",
          "checkpoint_loaded": True,
          "checkpoint": {
              "files": 5,
              "bytes": 16_000_000_000,
              "manifest_sha256": "9" * 64,
          },
          "inventory": {
              "leaves": 399,
              "arrays": 399,
              "logical_bytes": 32_762_941_440,
              "dp_partitioned_leaves": 0,
              "tp_partitioned_leaves": 399,
              "memory_kinds": ["device"],
          },
      },
      "forward_repeat_exact": True,
      "forward_shape": [256, 151936],
      "parameter_sample_sha256_before": "a" * 64,
      "parameter_sample_sha256_after": "a" * 64,
      "execution": {
          "forward": 2,
          "backward": 0,
          "optimizer_updates": 0,
          "training_steps": 0,
      },
  }


def _log(record: dict) -> str:
  return "\n".join((
      "[T1.PATHWAYS] required=1 initialized=1 status=ok",
      "[P32.RC] START stage=checkpoint-forward attempt=0 dp=16 tp=4 ",
      f"[P32.RC] JSON {json.dumps(record, sort_keys=True)}",
      "[P32.RC] VERDICT PASS stage=checkpoint-forward",
      "",
  ))


class ClassifyRCTest(unittest.TestCase):

  def test_positive(self):
    self.assertEqual(classify_text(_log(_record()))["status"], "PASS")

  def test_late_fatal_marker_rejects_nominal_pass(self):
    text = _log(_record()) + "RESOURCE_EXHAUSTED\n"
    self.assertEqual(classify_text(text)["status"], "INCONCLUSIVE")

  def test_retry_rejected(self):
    record = _record()
    record["attempt"] = 1
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_parameter_mutation_rejected(self):
    record = _record()
    record["parameter_sample_sha256_after"] = "b" * 64
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_wrong_execution_count_rejected(self):
    record = copy.deepcopy(_record())
    record["execution"]["backward"] = 1
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")


if __name__ == "__main__":
  unittest.main()
