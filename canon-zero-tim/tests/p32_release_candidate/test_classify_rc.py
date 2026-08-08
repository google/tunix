#!/usr/bin/env python3
"""Unit tests for the DP16 release-candidate classifier."""

from __future__ import annotations

import copy
import json
import unittest

from classify_rc import classify_text


def _record(stage: str) -> dict:
  counts = {
      "checkpoint-forward": (2, 0, 0),
      "backward": (36, 32, 0),
      "one-update": (19, 16, 1),
      "three-update": (53, 48, 3),
  }
  forward, backward, updates = counts[stage]
  sha_a = "a" * 64
  sha_b = "b" * 64
  records = 2 if stage == "backward" else updates
  replica_check = {
      "schema_version": 2,
      "sample": {
          "algorithm": "host-prefix-sample",
          "checked_leaves": 8,
          "total_leaves": 399,
          "samples_per_shard": 8,
          "checked_physical_values": 4096,
          "replica_groups": 32,
          "replica_comparisons": 480,
          "exact": True,
      },
      "full": {
          "algorithm": "device-ring-all-elements",
          "checked_leaves": 399,
          "dp_size": 16,
          "tp_size": 4,
          "physical_flags": 64,
          "exact": True,
      },
  }
  steps = [
      {
          "step": index,
          "loss": 1.0,
          "third_program_exact": True,
          "gradient_sample_sha256": "c" * 64,
          "gradient_health": {"finite": True, "nonzero": 12, "norm": 2.0},
          "rank_local_stats_distinct": True,
          "post_reduction_replica_check": copy.deepcopy(replica_check),
          "rank_contribution_signature_sha256": [
              f"{rank:064x}" for rank in range(16)
          ],
          **(
              {
                  "parameter_sample_sha256": ("def"[index]) * 64,
                  "optimizer_sample_sha256": ("123"[index]) * 64,
              }
              if updates
              else {}
          ),
      }
      for index in range(records)
  ]
  return {
      "attempt": 0,
      "stage": stage,
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
          "attention_backend": "dense-reference",
          "checkpoint": {
              "files": 5,
              "bytes": 16_000_000_000,
              "manifest_sha256": "9" * 64,
          },
          "compute_dtype": "<class 'jax.numpy.bfloat16'>",
          "param_dtype": "<class 'jax.numpy.float32'>",
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
      "parameter_sample_sha256_before": sha_a,
      "parameter_sample_sha256_after": sha_a if not updates else sha_b,
      "third_program_exact": None if stage == "checkpoint-forward" else True,
      "gradient_repeat_exact": True if stage == "backward" else None,
      "gradient_health": (
          None
          if stage == "checkpoint-forward"
          else {"finite": True, "nonzero": 12, "norm": 2.0}
      ),
      "rank_local_stats_distinct": (
          None if stage == "checkpoint-forward" else True
      ),
      "post_reduction_replicas_exact": (
          None
      ),
      "post_reduction_replica_check": (
          None if stage == "checkpoint-forward" else copy.deepcopy(replica_check)
      ),
      "dp_reduction_transactions": (
          0 if not backward else 2 if stage == "backward" else updates
      ),
      "dp_reduction_rounds_per_transaction": 0 if not backward else 15,
      "dp_rank_pullbacks_per_transaction": 0 if not backward else 16,
      "dp_rank_ordered_additions_per_transaction": 0 if not backward else 15,
      "optimizer_state_memory_between_commits": (
          None if not updates else ["pinned_host"]
      ),
      "optimizer_state_memory_during_commit": (
          None if not updates else ["device"]
      ),
      "step_records": steps,
      "execution": {
          "forward": forward,
          "backward": backward,
          "optimizer_updates": updates,
          "training_steps": updates,
      },
  }


def _log(record: dict) -> str:
  stage = record["stage"]
  return "\n".join((
      "[T1.PATHWAYS] required=1 initialized=1 status=ok",
      f"[P32.RC] START stage={stage} attempt=0 dp=16 tp=4 ",
      f"[P32.RC] JSON {json.dumps(record, sort_keys=True)}",
      f"[P32.RC] VERDICT PASS stage={stage}",
      "",
  ))


class ClassifyRCTest(unittest.TestCase):

  def test_all_four_stages_pass(self):
    for stage in (
        "checkpoint-forward", "backward", "one-update", "three-update"
    ):
      with self.subTest(stage=stage):
        self.assertEqual(classify_text(_log(_record(stage)), stage)["status"], "PASS")

  def test_late_fatal_marker_rejects_nominal_pass(self):
    text = _log(_record("checkpoint-forward")) + "RESOURCE_EXHAUSTED\n"
    self.assertEqual(classify_text(text)["status"], "INCONCLUSIVE")

  def test_retry_rejected(self):
    record = _record("checkpoint-forward")
    record["attempt"] = 1
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_missing_reducer_visibility_rejected(self):
    record = _record("backward")
    record["rank_local_stats_distinct"] = None
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_missing_backward_rank_signature_rejected(self):
    record = _record("backward")
    record["step_records"][0]["rank_contribution_signature_sha256"] = []
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_false_third_program_observation_is_recorded_truthfully(self):
    record = _record("backward")
    record["step_records"][0]["third_program_exact"] = False
    record["third_program_exact"] = False
    result = classify_text(_log(record))
    self.assertEqual(result["status"], "PASS")
    self.assertIs(result["record"]["third_program_exact"], False)

  def test_forged_third_program_aggregate_is_rejected(self):
    record = _record("backward")
    record["step_records"][0]["third_program_exact"] = False
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_unequal_replicas_rejected(self):
    record = _record("backward")
    record["post_reduction_replica_check"]["full"]["exact"] = False
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_forged_full_replica_aggregate_is_rejected(self):
    record = _record("backward")
    record["step_records"][0]["post_reduction_replica_check"]["full"][
        "exact"
    ] = False
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_update_without_full_replica_evidence_is_rejected(self):
    record = _record("one-update")
    record["post_reduction_replica_check"] = None
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_archived_backward_sample_is_labeled_not_upgraded(self):
    record = _record("backward")
    record["post_reduction_replica_check"] = None
    record["post_reduction_replicas_exact"] = True
    for step in record["step_records"]:
      step["post_reduction_replica_check"] = None
      step["post_reduction_replicas_exact"] = True
    result = classify_text(_log(record))
    self.assertEqual(result["status"], "PASS")
    self.assertEqual(
        result["replica_evidence_scope"], "sampled-prefix-legacy"
    )

  def test_update_without_pinned_host_rejected(self):
    record = _record("one-update")
    record["optimizer_state_memory_between_commits"] = ["device"]
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_non_monotonic_steps_rejected(self):
    record = _record("three-update")
    record["step_records"][2]["step"] = 7
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_no_commit_parameter_mutation_rejected(self):
    record = _record("backward")
    record["parameter_sample_sha256_after"] = "b" * 64
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_wrong_execution_count_rejected(self):
    record = copy.deepcopy(_record("one-update"))
    record["execution"]["optimizer_updates"] = 2
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_missing_attention_backend_scope_rejected(self):
    record = _record("checkpoint-forward")
    del record["model"]["attention_backend"]
    self.assertEqual(classify_text(_log(record))["status"], "INCONCLUSIVE")


if __name__ == "__main__":
  unittest.main()
