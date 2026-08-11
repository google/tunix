"""Positive and one-fault-at-a-time tests for the P34 classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).with_name("classify_run.py")
SPEC = importlib.util.spec_from_file_location("p34_classifier", PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P34 classifier")
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


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
        "optimizer_memory_kinds_before": ["pinned_host"],
        "optimizer_memory_kinds_after": ["pinned_host"],
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
        "optimizer_memory_kinds_before": ["pinned_host"],
        "optimizer_memory_kinds_after": ["pinned_host"],
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
        "optimizer_memory_kinds_before": ["pinned_host"],
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
        "optimizer_memory_kinds_before": ["pinned_host"],
        "optimizer_memory_kinds_after": ["pinned_host"],
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
        "optimizer_memory_kinds_before": ["pinned_host"],
        "optimizer_memory_kinds_after": ["pinned_host"],
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
        "optimizer_memory_kinds_before": ["pinned_host"],
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
        "optimizer_memory_kinds_before": ["pinned_host"],
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
        "optimizer_memory_kinds_before": ["pinned_host"],
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
