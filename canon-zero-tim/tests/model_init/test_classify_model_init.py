"""Negative controls for the P32 model-init classifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_model_init  # pylint: disable=g-import-not-at-top


def _record() -> dict:
  return {
      "attempt": 0,
      "topology": {
          "devices": 64,
          "dp": 16,
          "full_slice": True,
          "mesh_shape": [16, 4],
          "tp": 4,
          "unique_devices": 64,
      },
      "model": {
          "name": "qwen3-8b",
          "layers": 36,
          "vocab": 151936,
          "embed": 4096,
          "hidden": 12288,
          "heads": 32,
          "kv_heads": 8,
          "head_dim": 128,
          "compute_dtype": "<class 'jax.numpy.bfloat16'>",
          "param_dtype": "<class 'jax.numpy.float32'>",
          "checkpoint_loaded": False,
          "state_kind": "zero-structural",
      },
      "inventory": {
          "model": {
              "leaves": 399,
              "arrays": 399,
              "logical_bytes": 32762941440,
              "dp_partitioned_leaves": 0,
              "tp_partitioned_leaves": 326,
              "memory_kinds": ["device"],
          },
          "optimizer": {
              "leaves": 799,
              "arrays": 799,
              "logical_bytes": 65525882884,
              "dp_partitioned_leaves": 0,
              "tp_partitioned_leaves": 652,
              "memory_kinds": ["pinned_host"],
          },
          "accumulator": {
              "leaves": 399,
              "arrays": 399,
              "logical_bytes": 32762941440,
              "dp_partitioned_leaves": 0,
              "tp_partitioned_leaves": 326,
              "memory_kinds": ["device"],
          },
      },
      "physical_bytes_per_device": {
          "model": 8190735360,
          "optimizer": 16381470724,
          "accumulator": 8190735360,
      },
      "optimizer": {
          "name": "adamw",
          "learning_rate": 1e-6,
          "b1": 0.9,
          "b2": 0.95,
          "weight_decay": 0.0,
          "memory_kind": "pinned_host",
          "commits": 0,
      },
      "execution": {
          "backward": 0,
          "forward": 0,
          "optimizer_updates": 0,
          "training_steps": 0,
      },
      "wandb": {
          "project": "canon-zero-tim",
          "group": "p32-model-init",
          "run_name": "attempt-0",
          "network_initialized": False,
      },
  }


def _log(record: dict) -> str:
  return "\n".join((
      "[T1.PATHWAYS] required=1 initialized=1 status=ok",
      "[P32.INIT] START dp=16 tp=4 devices=64 checkpoint_loaded=0 "
      "forward=0 backward=0 update=0",
      "[P32.INIT] MESH shape=(16, 4) unique=64 full_slice=1",
      f"[P32.INIT] JSON {json.dumps(record, sort_keys=True)}",
      "[P32.INIT] VERDICT PASS",
  ))


class ModelInitClassifierTest(unittest.TestCase):

  def test_positive(self):
    self.assertEqual(classify_model_init.classify_text(_log(_record()))["status"], "PASS")

  def test_missing_json_is_rejected(self):
    text = _log(_record()).replace("[P32.INIT] JSON", "[P32.INIT] MISSING")
    self.assertEqual(classify_model_init.classify_text(text)["status"], "INCONCLUSIVE")

  def test_retry_is_rejected(self):
    record = _record()
    record["attempt"] = 1
    self.assertEqual(classify_model_init.classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_dp_sharded_state_is_rejected(self):
    record = _record()
    record["inventory"]["model"]["dp_partitioned_leaves"] = 1
    self.assertEqual(classify_model_init.classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_optimizer_commit_is_rejected(self):
    record = _record()
    record["optimizer"]["commits"] = 1
    self.assertEqual(classify_model_init.classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_optimizer_hyperparameter_drift_is_rejected(self):
    record = _record()
    record["optimizer"]["b2"] = 0.999
    self.assertEqual(classify_model_init.classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_forward_is_rejected(self):
    record = _record()
    record["execution"]["forward"] = 1
    self.assertEqual(classify_model_init.classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_wrong_memory_kind_is_rejected(self):
    record = copy.deepcopy(_record())
    record["optimizer"]["memory_kind"] = "device"
    record["inventory"]["optimizer"]["memory_kinds"] = ["device"]
    self.assertEqual(classify_model_init.classify_text(_log(record))["status"], "INCONCLUSIVE")

  def test_forbidden_traceback_is_rejected(self):
    text = _log(_record()) + "\nTraceback (most recent call last):\n"
    self.assertEqual(classify_model_init.classify_text(text)["status"], "INCONCLUSIVE")

  def test_late_host_buffer_failure_is_rejected(self):
    text = _log(_record()) + "\nCheck failed: pthread_create() failed\n"
    self.assertEqual(
        classify_model_init.classify_text(text)["status"], "INCONCLUSIVE"
    )


if __name__ == "__main__":
  unittest.main()
