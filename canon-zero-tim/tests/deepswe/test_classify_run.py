"""Positive and one-fault-at-a-time tests for the P34 classifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


PATH = Path(__file__).with_name("classify_run.py")
SPEC = importlib.util.spec_from_file_location("p34_classifier", PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P34 classifier")
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)

ARTIFACT_SPEC = importlib.util.spec_from_file_location(
    "p34_production_artifacts",
    PATH.parents[3] / "tunix/rl/deepswe_debug.py",
)
if ARTIFACT_SPEC is None or ARTIFACT_SPEC.loader is None:
  raise RuntimeError("cannot import DeepSWE artifact writer")
artifacts = importlib.util.module_from_spec(ARTIFACT_SPEC)
sys.modules[ARTIFACT_SPEC.name] = artifacts
ARTIFACT_SPEC.loader.exec_module(artifacts)


def _alignment():
  return {
      "verdict": "PASS",
      "boundaries": {
          name: {"differing_bytes": 0}
          for name in (
              "S_decode_vs_S_prefill",
              "S_prefill_vs_T_old",
              "T_old_vs_T_current",
          )
      },
      "exact": {
          "w_all_exactly_1": True,
          "r_all_exactly_1": True,
          "wr_all_exactly_1": True,
      },
      "clip_hits": 0,
      "tis_hits": 0,
  }


def _pre_alignment():
  return {
      "verdict": "PASS",
      "N_action": 8,
      "boundaries": {
          name: {"differing_bytes": 0}
          for name in (
              "S_decode_vs_S_prefill",
              "S_prefill_vs_T_old",
          )
      },
  }


def _weight_attestation(step=0):
  return {
      "schema": "canon.p34.deepswe.weight-attestation.v1",
      "step": step,
      "verdict": "PASS",
      "equal": True,
      "mapped_leaves": 706,
      "live_leaves": 706,
      "total_elements": 1_000_000,
      "mismatch_indices": [],
      "mesh_shape": {"dp": 16, "tp": 8},
      "mesh_device_ids": list(range(128)),
  }


def _log(updates: int):
  weight_attestations = max(1, updates)
  return "\n".join((
      "[entrypoint] JOBSET_ATTEMPT 0 (first attempt)",
      "[P34.PATHWAYS] initialized_once=1 before_jax=1",
      "[P34.CLI] PASS",
      "[sync] provenance ok",
      "[env] P34 whitelist SHA256 OK: " + "a" * 64,
      "[P34.TOPOLOGY] PASS",
      "[P34.DATASET] GOLD_FILTER_PASS rows=10->8 images=8",
      "[P34.R2E] BOUNDED_KUBERNETES_PATCH_PASS",
      "[CANON_P34_WANDB] ONLINE_RUN_PASS",
      "Prepared token paddings: [4096]",
      "Precompile worker0 backbone --> {'num_tokens': 4096, 'num_reqs': 64}",
      "CANON_FIXED_AR=1 fixed-order tree",
      "CANON_FIXED_AR_EMBED=1 fixed-order embed gather",
      "CANON_LOGPROB_M on",
      *(["[P34.WEIGHTS] EXACT"] * weight_attestations),
      *(["[P28.G6] weight_sync_committed count=1"] * updates),
  ))


class ClassifierTest(unittest.TestCase):

  def test_production_artifacts_accept_zero_signal_and_count_eight_of_eight(self):
    trajectories = []
    for index in range(64):
      group, pair = divmod(index, 8)
      trajectories.append(types.SimpleNamespace(
          group_id=group,
          pair_index=pair,
          metadata={
              "task_identity": {
                  "instance_id": group,
                  "docker_image": f"repo/image-{group}:latest",
              }
          },
          traj={
              "status": "SUCCEEDED",
              "trajectory_reward": 1.0,
              "conversation_text": [
                  {"role": "user", "content": f"prompt-{group}"},
                  {"role": "assistant", "content": f"answer-{pair}"},
              ],
          },
      ))
    values = {
        "CANON_P34_TRAJECTORY_CAPTURE": "1",
        "CANON_P43_DEEPSWE_DEBUG": "0",
        "CANON_P44_DEEPSWE_PARITY": "0",
        "CANON_DEEPSWE_ONEHOST_SMOKE": "0",
        "CANON_P34_RUN_STAGE": "full",
        "CANON_EXPECT_COMMIT": "1" * 40,
        "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
        "CANON_RUN_ID": "production-artifact",
        "CANON_P34_DATASET_NAME": "R2E-Gym/R2E-Gym-Subset",
        "CANON_P34_DATASET_REVISION": classifier._DATASET_REVISION,
        "CANON_P34_DATASET_SPLIT": "train",
        "CANON_P34_DATASET_ROWS": "4578",
        "CANON_P34_CLEAN_ROWS": "1851",
        "CANON_P34_WHITELIST_SHA256": classifier._WHITELIST_SHA256,
    }
    with tempfile.TemporaryDirectory() as text:
      root = Path(text).resolve()
      metrics = artifacts.persist_batch(
          trajectories,
          [1.0] * 64,
          [0.0] * 64,
          expected_step=0,
          output_dir=root,
          model_id="Qwen/Qwen3-32B",
          values=values,
      )
      checks, rows = classifier._artifact_checks(root, expected_batches=1)
      manifest = json.loads((root / "run_manifest.json").read_text())
    self.assertTrue(all(checks.values()), checks)
    self.assertEqual(manifest["schema"], artifacts.P34_MANIFEST_SCHEMA)
    self.assertEqual(metrics["all_solved_prompt_groups"], 8)
    self.assertEqual(metrics["effective_prompt_groups"], 0)
    self.assertEqual(rows[0]["effective_prompt_groups"], 0)

  def test_three_update_positive(self):
    updates = [{
        "verdict": "PASS",
        "commits": 1,
        "gradient_activity": [True] * 4,
        "gradient_finite": True,
        "gradient_deterministic": True,
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 4,
        "dp_reduction_rounds_per_transaction": 8,
        "dp_rank_pullbacks_per_transaction": 16,
        "optimizer_placement": "device-resident",
        "optimizer_memory_kinds_before": ["device"],
        "optimizer_memory_kinds_after": ["device"],
        "train_steps_after": step,
    } for step in (1, 2, 3)]
    report = classifier.classify(
        log_text=_log(3),
        weight_attestations=[_weight_attestation(step) for step in range(3)],
        pre_alignment=[_pre_alignment() for _ in range(3)],
        alignment=[_alignment() for _ in range(12)],
        updates=updates,
        stage="three-update",
    )
    self.assertEqual(report["verdict"], "PASS")

  def test_one_bit_boundary_is_rejected(self):
    records = [_alignment() for _ in range(4)]
    records[2]["boundaries"]["T_old_vs_T_current"]["differing_bytes"] = 1
    update = {
        "verdict": "PASS",
        "commits": 1,
        "gradient_activity": [True] * 4,
        "gradient_finite": True,
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 4,
        "dp_reduction_rounds_per_transaction": 8,
        "dp_rank_pullbacks_per_transaction": 16,
        "optimizer_placement": "device-resident",
        "optimizer_memory_kinds_before": ["device"],
        "optimizer_memory_kinds_after": ["device"],
        "train_steps_after": 1,
    }
    report = classifier.classify(
        log_text=_log(1),
        weight_attestations=[_weight_attestation()],
        pre_alignment=[_pre_alignment()],
        alignment=records,
        updates=[update],
        stage="one-update",
    )
    self.assertEqual(report["verdict"], "FAIL")
    self.assertIn("four_boundaries_exact", report["failed"])

  def test_retry_is_rejected(self):
    update = {
        "verdict": "PASS",
        "commits": 0,
        "gradient_activity": [True] * 4,
        "gradient_finite": True,
        "gradient_deterministic": True,
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 4,
        "dp_reduction_rounds_per_transaction": 8,
        "dp_rank_pullbacks_per_transaction": 16,
        "optimizer_placement": "device-resident",
        "optimizer_memory_kinds_before": ["device"],
    }
    report = classifier.classify(
        log_text=_log(0).replace("ATTEMPT 0", "ATTEMPT 1"),
        weight_attestations=[_weight_attestation()],
        pre_alignment=[_pre_alignment()],
        alignment=[_alignment() for _ in range(4)],
        updates=[update],
        stage="backward-no-commit",
    )
    self.assertEqual(report["verdict"], "FAIL")
    self.assertIn("attempt_zero", report["failed"])

  def test_extra_scheduler_bucket_is_rejected(self):
    update = {
        "verdict": "PASS",
        "commits": 1,
        "gradient_activity": [True] * 4,
        "gradient_finite": True,
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 4,
        "dp_reduction_rounds_per_transaction": 8,
        "dp_rank_pullbacks_per_transaction": 16,
        "optimizer_placement": "device-resident",
        "optimizer_memory_kinds_before": ["device"],
        "optimizer_memory_kinds_after": ["device"],
        "train_steps_after": 1,
    }
    log_text = _log(1).replace(
        "Prepared token paddings: [4096]",
        "Prepared token paddings: [4096, 8192]",
    )
    report = classifier.classify(
        log_text=log_text,
        weight_attestations=[_weight_attestation()],
        pre_alignment=[_pre_alignment()],
        alignment=[_alignment() for _ in range(4)],
        updates=[update],
        stage="one-update",
    )
    self.assertEqual(report["verdict"], "FAIL")
    self.assertIn("scheduler_bucket_exact", report["failed"])

  def test_global_as_local_request_capacity_is_rejected(self):
    update = {
        "verdict": "PASS",
        "commits": 1,
        "gradient_activity": [True] * 4,
        "gradient_finite": True,
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 4,
        "dp_reduction_rounds_per_transaction": 8,
        "dp_rank_pullbacks_per_transaction": 16,
        "optimizer_placement": "device-resident",
        "optimizer_memory_kinds_before": ["device"],
        "optimizer_memory_kinds_after": ["device"],
        "train_steps_after": 1,
    }
    log_text = _log(1).replace("'num_reqs': 64", "'num_reqs': 1024")
    report = classifier.classify(
        log_text=log_text,
        weight_attestations=[_weight_attestation()],
        pre_alignment=[_pre_alignment()],
        alignment=[_alignment() for _ in range(4)],
        updates=[update],
        stage="one-update",
    )
    self.assertEqual(report["verdict"], "FAIL")
    self.assertIn("scheduler_precompile_exact", report["failed"])

  def test_missing_full_gradient_repeat_is_rejected(self):
    update = {
        "verdict": "PASS",
        "commits": 0,
        "gradient_activity": [True] * 4,
        "gradient_finite": True,
        "gradient_deterministic": False,
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 4,
        "dp_reduction_rounds_per_transaction": 8,
        "dp_rank_pullbacks_per_transaction": 16,
        "optimizer_placement": "device-resident",
        "optimizer_memory_kinds_before": ["device"],
    }
    report = classifier.classify(
        log_text=_log(0),
        weight_attestations=[_weight_attestation()],
        pre_alignment=[_pre_alignment()],
        alignment=[_alignment() for _ in range(4)],
        updates=[update],
        stage="backward-no-commit",
    )
    self.assertEqual(report["verdict"], "FAIL")
    self.assertIn("gradient_deterministic_repeat", report["failed"])

  def test_nonzero_pre_backward_boundary_is_rejected(self):
    pre_alignment = _pre_alignment()
    pre_alignment["boundaries"]["S_decode_vs_S_prefill"][
        "differing_bytes"
    ] = 1
    update = {
        "verdict": "PASS",
        "commits": 0,
        "gradient_activity": [True] * 4,
        "gradient_finite": True,
        "gradient_deterministic": True,
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 4,
        "dp_reduction_rounds_per_transaction": 8,
        "dp_rank_pullbacks_per_transaction": 16,
        "optimizer_placement": "device-resident",
        "optimizer_memory_kinds_before": ["device"],
    }
    report = classifier.classify(
        log_text=_log(0),
        weight_attestations=[_weight_attestation()],
        pre_alignment=[pre_alignment],
        alignment=[_alignment() for _ in range(4)],
        updates=[update],
        stage="backward-no-commit",
    )
    self.assertEqual(report["verdict"], "FAIL")
    self.assertIn("pre_backward_boundaries_exact", report["failed"])

  def test_weight_mismatch_is_rejected(self):
    weight = _weight_attestation()
    weight["equal"] = False
    weight["mismatch_indices"] = [8]
    update = {
        "verdict": "PASS",
        "commits": 0,
        "gradient_activity": [True] * 4,
        "gradient_finite": True,
        "gradient_deterministic": True,
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 4,
        "dp_reduction_rounds_per_transaction": 8,
        "dp_rank_pullbacks_per_transaction": 16,
        "optimizer_placement": "device-resident",
        "optimizer_memory_kinds_before": ["device"],
    }
    report = classifier.classify(
        log_text=_log(0),
        weight_attestations=[weight],
        pre_alignment=[_pre_alignment()],
        alignment=[_alignment() for _ in range(4)],
        updates=[update],
        stage="backward-no-commit",
    )
    self.assertEqual(report["verdict"], "FAIL")
    self.assertIn("weight_attestation_exact", report["failed"])


if __name__ == "__main__":
  unittest.main()
