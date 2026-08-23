#!/usr/bin/env python3
"""Negative controls for the P59 timing and zero-TIM classifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


CLASSIFIER = _load(
    "test_p59_classifier", Path(__file__).with_name("classify_and_analyze.py")
)
P33_TEST = _load(
    "test_p59_p33_fixtures",
    ROOT / "canon-zero-tim" / "tests" / "p33_workloads" / "test_classify_run.py",
)


class ClassifyAndAnalyzeTest(unittest.TestCase):

  def _fixture(
      self,
      root: Path,
      *,
      invocations: int = 1,
      profile: bool = False,
      workload: str = "frozenlake",
      dp_size: int = 16,
      tp_size: int = 4,
      steps: int = 3,
      numerical: bool = False,
  ):
    paths = {
        name: root / name
        for name in ("run.log", "pre.jsonl", "updates.jsonl", "align.jsonl")
    }
    signed_dp4 = workload == "gsm8k-p59-dp4-tp1"
    log = (
        "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
        + (
            "[CANON_P33_EVAL] DISABLED workload=frozenlake\n"
            if workload.startswith("frozenlake")
            else ""
        )
        + "[CANON_P31_METRICS] monotonic_direct "
        + f"last_step={steps if signed_dp4 else steps - 1} "
        + f"events={steps} regressions=0\n"
        + f"[CANON_P33_DP{dp_size}] update_step_committed\n" * steps
        + "[CANON_ALIGN_PRE] scope=test verdict=PASS\n" * steps
        + "[CANON_ALIGN] scope=test verdict=PASS\n" * (steps * 16)
    )
    for index in range(steps):
      log += (
          f"[PERF] stage=p32_vag_forward seconds={4 + index}.0 groups=16\n"
          f"[PERF] stage=p32_vag_reverse seconds={10 + index}.0 groups=16\n"
          f"[PERF] stage=segmented_value_and_grad seconds={15 + index}.0\n"
          "[PERF] stage=optimizer_transaction seconds=1.0\n"
          f"Global step {index + 1} completed in {30 + index}.0 seconds.\n"
          f"[PERF] step={index + 1} stage=weight_sync seconds=3.0\n"
      )
    if profile:
      log += (
          "[P59.XPROF] phase=backward_group started update=1 groups=1\n"
          "[P59.XPROF] phase=backward_group stopped update=1 groups=1 "
          "anchor=gradient_ready\n"
      )
    if numerical:
      for name in ("model_before", "gradient", "model_after"):
        log += (
            "[P61.NUMERICAL] capture_complete "
            f"name={name} leaves=2 bytes=20 manifest=/tmp/{name}.json\n"
        )
    paths["run.log"].write_text(log, encoding="utf-8")
    paths["pre.jsonl"].write_text(
        "".join(
            json.dumps(P33_TEST._pre_alignment(index, policy_workload=workload)) + "\n"
            for index in range(steps)
        ),
        encoding="utf-8",
    )
    paths["align.jsonl"].write_text(
        "".join(
            json.dumps(P33_TEST._alignment(
                index, optimizer_skipped=False, policy_workload=workload
            )) + "\n"
            for index in range(steps * 16)
        ),
        encoding="utf-8",
    )
    updates = []
    for index in range(steps):
      update = P33_TEST._update(
          index,
          placement=(
              "device-resident"
              if workload == "gsm8k-p59-dp4-tp1"
              else "pinned-host-offload"
          ),
          dp_size=dp_size,
          tp_size=tp_size,
      )
      local_groups = 16
      update.update({
          "contract_name": workload,
          "dp_axis": "dp" if signed_dp4 else "data",
          "microsteps": local_groups,
          "gradient_activity": [True] * local_groups,
          "alignment_hashes": [{"T_current": "a"}] * local_groups,
          "micro_gradient_norms": [1.0] * local_groups,
          "dp_reduction_transactions": local_groups,
          "dp_reduction_rounds_per_transaction": 2 * (dp_size.bit_length() - 1),
          "dp_rank_pullbacks_per_transaction": dp_size,
          "dp_pullback_invocations_per_transaction": invocations,
          "dp_replicas_exact": True,
          "elapsed_seconds": 20.0 + index,
      })
      if numerical:
        update["commit_evidence"]["effective_learning_rate"] = 2.0e-7
        update["commit_evidence"]["parameter_changed_elements"] = 1
        update["commit_evidence"]["parameter_delta_max_abs"] = 1.0e-8
      updates.append(update)
    paths["updates.jsonl"].write_text(
        "".join(json.dumps(row) + "\n" for row in updates), encoding="utf-8"
    )
    return paths

  def _classify(
      self,
      paths,
      kind="candidate",
      *,
      workload="frozenlake",
      dp_size=16,
      tp_size=4,
  ):
    return CLASSIFIER.classify(
        kind=kind,
        run_log=paths["run.log"],
        pre_alignment_report=paths["pre.jsonl"],
        update_report=paths["updates.jsonl"],
        alignment_report=paths["align.jsonl"],
        workload=workload,
        dp_size=dp_size,
        tp_size=tp_size,
    )

  def test_candidate_passes_and_splits_wall(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self._classify(self._fixture(Path(directory)))
      self.assertEqual(result["verdict"], "PASS")
      stable = result["timing"]["stable_steps2_plus_mean"]
      self.assertEqual(stable["wall_seconds"], 32.0)
      self.assertEqual(stable["cycle_seconds"], 35.0)
      self.assertEqual(stable["training_seconds"], 22.0)
      self.assertEqual(stable["system_seconds"], 10.0)
      self.assertEqual(stable["system_including_sync_seconds"], 13.0)
      self.assertEqual(stable["p32_reverse_seconds"], 12.0)

  def test_control_requires_sixteen_invocations(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self._classify(
          self._fixture(Path(directory), invocations=16), kind="control"
      )
      self.assertEqual(result["verdict"], "PASS")

  def test_one_alignment_fail_is_fatal(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(Path(directory))
      text = paths["run.log"].read_text(encoding="utf-8")
      paths["run.log"].write_text(
          text.replace("verdict=PASS", "verdict=FAIL", 1), encoding="utf-8"
      )
      result = self._classify(paths)
      self.assertEqual(result["verdict"], "FAIL")
      self.assertEqual(result["zero_tim"]["observed_fail"], 1)

  def test_profile_requires_exact_capture_markers(self):
    with tempfile.TemporaryDirectory() as directory:
      fail = self._classify(self._fixture(Path(directory)), kind="profile")
      self.assertEqual(fail["verdict"], "FAIL")
    with tempfile.TemporaryDirectory() as directory:
      passed = self._classify(
          self._fixture(Path(directory), profile=True), kind="profile"
      )
      self.assertEqual(passed["verdict"], "PASS")

  def test_dp4_proxy_requires_one_parallel_invocation_and_fixed_tree(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(
          Path(directory),
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      result = self._classify(
          paths,
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["zero_tim"], {
          "expected_pass": 51,
          "observed_pass": 51,
          "observed_fail": 0,
      })
      self.assertEqual(result["topology"], {"dp": 4, "tp": 1})

  def test_v1_dp4_proxy_uses_the_same_strict_hardware_gate(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(
          Path(directory),
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      result = self._classify(
          paths,
          kind="v1",
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["zero_tim"], {
          "expected_pass": 51,
          "observed_pass": 51,
          "observed_fail": 0,
      })

  def test_dp4_tail_requires_136_passes_and_reports_six_stable_steps(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(
          Path(directory),
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
          steps=8,
      )
      result = self._classify(
          paths,
          kind="tail",
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["zero_tim"], {
          "expected_pass": 136,
          "observed_pass": 136,
          "observed_fail": 0,
      })
      self.assertEqual(result["p33"]["stage"], "p59-eight-update")
      self.assertEqual(result["timing"]["stable_sample_count"], 6)
      self.assertEqual(
          result["timing"]["stable_steps2_plus_mean"]["wall_seconds"],
          34.5,
      )

  def test_dp4_numerical_candidate_requires_one_positive_update(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(
          Path(directory),
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
          steps=1,
          numerical=True,
      )
      result = self._classify(
          paths,
          kind="numerical-candidate",
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["zero_tim"], {
          "expected_pass": 17,
          "observed_pass": 17,
          "observed_fail": 0,
      })
      self.assertEqual(result["p33"]["stage"], "one-update")
      self.assertEqual(result["timing"]["stable_sample_count"], 0)
      self.assertNotIn("stable_steps2_plus_mean", result["timing"])

  def test_dp4_numerical_control_requires_four_pullbacks(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(
          Path(directory),
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
          steps=1,
          invocations=4,
          numerical=True,
      )
      result = self._classify(
          paths,
          kind="numerical-control",
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      self.assertEqual(result["verdict"], "PASS")

  def test_dp4_numerical_rejects_zero_lr_and_missing_capture(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(
          Path(directory),
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
          steps=1,
      )
      result = self._classify(
          paths,
          kind="numerical-candidate",
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("update[0].effective_learning_rate=0.0", result["reasons"])
      self.assertIn("p61_capture.gradient=0 expected=1", result["reasons"])

  def test_dp4_tail_rejects_one_missing_alignment_pass(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(
          Path(directory),
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
          steps=8,
      )
      lines = paths["run.log"].read_text(encoding="utf-8").splitlines()
      del lines[next(
          index
          for index, line in enumerate(lines)
          if line.startswith("[CANON_ALIGN] ")
      )]
      paths["run.log"].write_text("\n".join(lines) + "\n", encoding="utf-8")
      result = self._classify(
          paths,
          kind="tail",
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertEqual(result["zero_tim"]["observed_pass"], 135)
      self.assertIn(
          "canon_align_pass=135 expected=136", result["reasons"]
      )

  def test_dp4_proxy_rejects_legacy_data_axis(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(
          Path(directory),
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      updates = [
          json.loads(line)
          for line in paths["updates.jsonl"].read_text(
              encoding="utf-8"
          ).splitlines()
      ]
      updates[0]["dp_axis"] = "data"
      paths["updates.jsonl"].write_text(
          "".join(json.dumps(row) + "\n" for row in updates),
          encoding="utf-8",
      )
      result = self._classify(
          paths,
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("p33:update[0].dp_axis", result["reasons"])

  def test_dp4_proxy_rejects_zero_based_final_metric_step(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = self._fixture(
          Path(directory),
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      text = paths["run.log"].read_text(encoding="utf-8")
      paths["run.log"].write_text(
          text.replace("last_step=3", "last_step=2"), encoding="utf-8"
      )
      result = self._classify(
          paths,
          workload="gsm8k-p59-dp4-tp1",
          dp_size=4,
          tp_size=1,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertTrue(
          any(
              reason.startswith("p33:monotonic_last_step=")
              for reason in result["reasons"]
          )
      )


if __name__ == "__main__":
  unittest.main()
