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

  def test_stock_train_and_eval_bundles_are_accepted(self):
    workload, base = self._environment()
    shared = {
        **base,
        "CANON_P57_WORKLOAD_CANDIDATE": "m15",
        "CANON_P57_DATA_SPLIT": "selection",
        "CANON_P57_EXPECTED_UPDATES": "200",
        "CANON_P30_OPT_STATE_OFFLOAD": "0",
    }
    train = {
        **shared,
        "CANON_P57_RUN_KIND": "train",
        **{name: "0" for name in dp_workloads.P57_STOCK_TRAIN_ZERO_SWITCHES},
        **{name: "1" for name in dp_workloads.P57_STOCK_TRAIN_ONE_SWITCHES},
    }
    eval_values = {
        **shared,
        "CANON_P57_RUN_KIND": "eval",
        **{name: "0" for name in dp_workloads.P57_STOCK_EVAL_ZERO_SWITCHES},
        **{name: "1" for name in dp_workloads.P57_STOCK_EVAL_ONE_SWITCHES},
    }
    train_attestation = dp_workloads.validate_p57_stock_train_environment(
        workload, train
    )
    eval_attestation = dp_workloads.validate_p57_stock_eval_environment(
        workload, eval_values
    )
    self.assertEqual(train_attestation["arm"], "mismatch")
    self.assertEqual(eval_attestation["arm"], "mismatch")
    self.assertIn(
        "CANON_PROMPT_PROCESSED_LOGPROBS", train_attestation["one_switches"]
    )
    self.assertNotIn(
        "CANON_PROMPT_PROCESSED_LOGPROBS", train_attestation["zero_switches"]
    )
    self.assertIn(
        "CANON_PROMPT_PROCESSED_LOGPROBS", eval_attestation["zero_switches"]
    )

    with self.assertRaisesRegex(ValueError, "stock-train environment"):
      dp_workloads.validate_p57_stock_train_environment(
          workload, {**train, "CANON_P28_SEGMENTED_TRAIN": "1"}
      )
    with self.assertRaisesRegex(ValueError, "stock-train environment"):
      dp_workloads.validate_p57_stock_train_environment(
          workload, {**train, "CANON_PROMPT_PROCESSED_LOGPROBS": "0"}
      )
    with self.assertRaisesRegex(ValueError, "stock-eval environment"):
      dp_workloads.validate_p57_stock_eval_environment(
          workload, {**eval_values, "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0"}
      )
    with self.assertRaisesRegex(ValueError, "stock-eval environment"):
      dp_workloads.validate_p57_stock_eval_environment(
          workload,
          {**eval_values, "CANON_PROMPT_PROCESSED_LOGPROBS": "1"},
      )

  def test_stock_observer_is_train_only_and_does_not_enable_fixed_m(self):
    entrypoint = (ROOT / "canon-zero-tim/cluster/entrypoint.sh").read_text()
    installer = (
        ROOT
        / "canon-zero-tim/cluster/steps/39_install_p57_stock_observer.sh"
    ).read_text()
    runner_patch = (
        ROOT
        / "canon-zero-tim/patches/p57_stock_observer/01-tpu-runner.patch"
    ).read_text()
    self.assertIn("if p57_is_stock_fast_training; then", entrypoint)
    self.assertIn("step 39_install_p57_stock_observer.sh", entrypoint)
    self.assertIn("observer_overlay=$p57_observer_overlay", entrypoint)
    self.assertIn("p57_is_stock_fast_training", installer)
    self.assertIn("stock_runner_verified=1 treatment=observer-only", installer)
    self.assertIn("compute_processed_prompt_logprobs", runner_patch)
    self.assertIn("CANON_PROMPT_PROCESSED_LOGPROBS", runner_patch)
    self.assertNotIn("CANON_LOGPROB_M", runner_patch)
    postflight = (ROOT / "canon-zero-tim/cluster/steps/90_run.sh").read_text()
    self.assertIn(
        "observer=warning-only processed_b=observer-only$", postflight
    )
    self.assertIn('p57_stock_observer" -ne 1', postflight)

  def test_stock_observer_manifest_is_exactly_runner_plus_helper(self):
    manifest = (
        ROOT / "canon-zero-tim/P57_STOCK_OBSERVER_MANIFEST.sha256"
    ).read_text().splitlines()
    self.assertEqual(len(manifest), 2)
    self.assertEqual(
        {line.split(maxsplit=1)[1] for line in manifest},
        {
            "runner/tpu_runner.py",
            "runner/p57_stock_prompt_observer.py",
        },
    )


if __name__ == "__main__":
  unittest.main()
