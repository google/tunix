"""Positive and negative controls for the P39 pilot classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p39_pilot_classifier",
    ROOT / "canon-zero-tim/tests/p39_deepswe_pilot/classify_run.py",
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P39 classifier")
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _log():
  return "\n".join((
      "[entrypoint] JOBSET_ATTEMPT 0 (first attempt)",
      "[P34.PATHWAYS] initialized_once=1 before_jax=1",
      "[P34.CLI] PASS",
      "[sync] provenance ok",
      "[P34.TOPOLOGY] PASS",
      "[CANON_P34_WANDB] ONLINE_RUN_PASS",
      "Prepared token paddings: [1024]",
      "Precompile worker0 backbone --> {'num_tokens': 1024, 'num_reqs': 64}",
  ))


def _policy():
  return {
      "id": "deepswe-pilot-alignment-warning-v1",
      "claim_level": "convergence-only",
  }


def _weight():
  return {
      "verdict": "PASS",
      "equal": True,
      "mesh_shape": {"dp": 4, "tp": 8},
      "mesh_device_ids": list(range(32)),
  }


def _pre():
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS",
      "blocking_reds": [],
      "N_action": 10,
      "admission_policy": _policy(),
  }


def _alignment():
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS",
      "blocking_reds": [],
      "ratio_finite": True,
      "gradient": {"finite": True},
      "admission_policy": _policy(),
  }


def _update(free_bytes=10 * 1024**3, device_count=32):
  limit = 100 * 1024**3
  snapshot = [
      {
          "device": index,
          "bytes_in_use": limit - free_bytes,
          "peak_bytes_in_use": limit - free_bytes,
          "bytes_limit": limit,
      }
      for index in range(device_count)
  ]
  return {
      "contract_name": "p39-64chip-pilot",
      "dp_size": 4,
      "tp_size": 8,
      "global_m": 1024,
      "verdict": "PASS",
      "commits": 1,
      "gradient_finite": True,
      "gradient_activity": [True] * 16,
      "dp_replicas_exact": True,
      "dp_reduction_transactions": 16,
      "dp_reduction_rounds_per_transaction": 4,
      "dp_rank_pullbacks_per_transaction": 4,
      "optimizer_placement": "device-resident",
      "optimizer_memory_kinds_before": ["device"],
      "optimizer_memory_kinds_after": ["device"],
      "optimizer_transaction_valid": True,
      "hbm_before": snapshot,
      "hbm_after_accumulation": snapshot,
      "hbm_after_commit": snapshot,
  }


def _classify(update=None, pre=None):
  return classifier.classify(
      log_text=_log(),
      weight_attestations=[_weight()],
      pre_alignment=[_pre() if pre is None else pre],
      alignment=[_alignment() for _ in range(16)],
      updates=[_update() if update is None else update],
      stage="one-update",
  )


class P39ClassifierTest(unittest.TestCase):

  def test_positive_fixture_passes(self):
    report = _classify()
    self.assertEqual(report["verdict"], "PASS")
    self.assertEqual(report["minimum_hbm_free_bytes"], 10 * 1024**3)

  def test_host_offload_is_rejected(self):
    update = _update()
    update["optimizer_placement"] = "pinned-host-offload"
    report = _classify(update=update)
    self.assertIn("optimizer_device_resident", report["failed"])

  def test_small_hbm_margin_is_rejected(self):
    report = _classify(update=_update(free_bytes=4 * 1024**3))
    self.assertIn("hbm_margin", report["failed"])

  def test_proxy_wide_hbm_telemetry_is_accepted(self):
    report = _classify(update=_update(device_count=64))
    self.assertTrue(report["checks"]["hbm_telemetry_complete"])

  def test_incomplete_hbm_telemetry_is_rejected(self):
    report = _classify(update=_update(device_count=31))
    self.assertIn("hbm_telemetry_complete", report["failed"])

  def test_blocking_alignment_red_is_rejected(self):
    pre = _pre()
    pre["blocking_reds"] = ["ratio_nonfinite"]
    report = _classify(pre=pre)
    self.assertIn("pre_alignment_nonblocking", report["failed"])


if __name__ == "__main__":
  unittest.main()
