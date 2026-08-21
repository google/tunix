"""Pinned-image gate for the P57 stock-fast resolved environment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from tunix.rl import dp_workloads


ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_PATH = (
    ROOT
    / "canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_stock_discovery.py"
)
SPEC = importlib.util.spec_from_file_location(
    "p57_stock_fast_classifier_contract", CLASSIFIER_PATH
)
assert SPEC and SPEC.loader
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


class P57StockFastContractTest(unittest.TestCase):

  def _environment(self):
    workload = dp_workloads.get_workload("frozenlake-dp8-tp8")
    values = {
        "CANON_P57_RUN_KIND": "calibration",
        "CANON_P57_TIM_ARM": "mismatch",
        "CANON_P57_INFERENCE_REGIME": "stock-fast",
        "CANON_P32_WORKLOAD": workload.name,
        "CANON_DP_SIZE": str(workload.dp_size),
        "CANON_TP_SIZE": str(workload.tp_size),
        "CANON_TOTAL_DEVICES": str(workload.total_devices),
        "CANON_ENGINE_DP_SIZE": str(workload.dp_size),
        "CANON_QWEN3_TP_SIZE": str(workload.tp_size),
        "CANON_GLOBAL_PROMPTS": str(workload.global_prompts),
        "CANON_LOCAL_PROMPTS": str(workload.local_prompts),
        "CANON_NUM_GENERATIONS": str(workload.num_generations),
        "CANON_LOCAL_TRAJECTORIES": str(workload.local_trajectories),
        "CANON_GLOBAL_TRAJECTORIES": str(workload.global_trajectories),
        "CANON_TARGET_M": str(workload.local_m),
        "MIN_TOKEN_BUCKET": str(workload.global_m),
        "XLA_FLAGS": "--xla_cpu_max_isa=AVX2",
    }
    values.update({
        name: "0" for name in dp_workloads.P57_STOCK_FAST_ZERO_SWITCHES
    })
    return workload, values

  def test_complete_bundle_is_accepted(self):
    workload, values = self._environment()
    attestation = dp_workloads.validate_p57_stock_fast_environment(
        workload, values
    )
    self.assertEqual(attestation["regime"], "stock-fast")
    self.assertEqual(len(attestation["absent_switches"]), 12)
    self.assertEqual(len(attestation["zero_switches"]), 25)

  def test_runtime_and_offline_classifier_attest_the_same_switches(self):
    self.assertEqual(
        tuple(classifier._ABSENT_SWITCHES),
        dp_workloads.P57_STOCK_FAST_ABSENT_SWITCHES,
    )
    self.assertEqual(
        tuple(classifier._ZERO_SWITCHES),
        dp_workloads.P57_STOCK_FAST_ZERO_SWITCHES,
    )

  def test_each_switch_class_has_a_working_negative(self):
    workload, values = self._environment()
    for name, value in (
        ("CANON_FIXED_AR", "1"),
        ("CANON_ENGINE_MODULE_C", "1"),
        ("XLA_FLAGS", "--xla_allow_excess_precision=false"),
    ):
      with self.subTest(name=name):
        with self.assertRaisesRegex(ValueError, "stock-fast environment"):
          dp_workloads.validate_p57_stock_fast_environment(
              workload, {**values, name: value}
          )


if __name__ == "__main__":
  unittest.main()
