#!/usr/bin/env python3
"""Negative controls for the P66 full-depth TP4 classifiers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


ARM = _load("p66_tp4_arm", PKG / "tests/p66_backward/classify_tp4_arm.py")
CAMPAIGN = _load(
    "p66_tp4_campaign", PKG / "tests/p66_backward/classify_tp4_campaign.py"
)
ORACLE_PAIR = _load(
    "p66_tp4_oracle_pair",
    PKG / "tests/p66_backward/classify_tp4_oracle_pair.py",
)


def _components(scale: float = 1.0) -> dict[str, float]:
  return {
      "embed": 2.0 * scale,
      **{f"layer_{index}": (1.0 + index / 100.0) * scale for index in range(28)},
      "norm": 0.5 * scale,
      "head": 0.75 * scale,
  }


def _sample() -> dict:
  return {
      "eligible_leaves": 1,
      "sampled_leaves": 1,
      "sampled_bytes": 4,
      "leaves": {"['x']": {"sha256": "a" * 64}},
  }


def _row_summary(arm: str, *, padding_nonzero: int = 0) -> dict:
  return {
      "schema": "canon-p66-row-cotangent-summary-v1",
      "arm": arm,
      "records": 56,
      "chunks": 2,
      "layers": list(range(28)),
      "padding_row_layer_nonzero": padding_nonzero,
      "padding_row_layer_nonfinite": 0,
      "padding_hidden_rms_min": 0.01,
      "first_nonzero_padding_cotangent": (
          None if not padding_nonzero else {
              "chunk": 1, "layer": 27, "rows": 1, "max_abs": 0.25
          }
      ),
  }


def _oracle_summary(*, rel_l2: float = 1.0e-3) -> dict:
  endpoints = ["embed", "head", "layer_0", "layer_14", "layer_27", "norm"]
  caps = dict(ARM.ORACLE_CAPS)
  records = [
      {
          "schema": "canon-p66-same-point-vjp-oracle-v1",
          "endpoint": endpoint,
          "verdict": "PASS",
          "leaf_count": 2,
          "elements": 32,
          "finite": True,
          "array_exact": False,
          "live_reference_leaves": 2,
          "dead_candidate_leaves": 0,
          "reference_nonzero_elements": 30,
          "sign_mismatch_elements": 0,
          "reference_norm": 1.0,
          "candidate_norm": 1.0,
          "difference_norm": rel_l2,
          "worst_leaf_index": 0,
          "worst_leaf_scaled_max_error": rel_l2,
          "metrics": {
              "rel_l2": rel_l2,
              "one_minus_cos": 1.0e-5,
              "norm_ratio_error": 1.0e-4,
              "sign_mismatch_rate": 0.0,
          },
          "caps": caps,
      }
      for endpoint in endpoints
  ]
  return {
      "schema": "canon-p66-same-point-vjp-oracle-summary-v1",
      "arm": "tp4-vma-oracle",
      "negative_control_detected": True,
      "expected_endpoints": endpoints,
      "observed_endpoints": endpoints,
      "records": records,
      "verdict": "PASS",
  }


class P66Tp4ClassifierTest(unittest.TestCase):

  def _safe_arm(self, root: Path, arm: str, *, scale: float = 1.0):
    arm_root = root / arm
    arm_root.mkdir()
    hashes = {"A": "same", "B": "same"}
    pre = arm_root / "pre.jsonl"
    pre.write_text(json.dumps({"verdict": "PASS", "hashes": hashes}) + "\n")
    align = arm_root / "align.jsonl"
    align.write_text(
        "".join(json.dumps({"verdict": "PASS"}) + "\n" for _ in range(16))
    )
    update = arm_root / "update.json"
    update.write_text(json.dumps({
        "schema": "canon-p66-backward-gate-v1",
        "arm": arm,
        "verdict": "PASS",
        "commits": 0,
        "dp_size": 1,
        "tp_size": 4,
        "global_trajectories": 16,
        "gradient_groups": 16,
        "gradient": {
            "all_finite": True,
            "any_nonzero": True,
            "stable_norm": 2.0 * scale,
        },
        "engine_vjp": {
            "all_finite": True,
            "any_nonzero": True,
            "stable_norm": 32.0 * scale,
        },
        "layerwise_profile": {
            "schema": "canon-p66-full-depth-profile-v1",
            "arm": arm,
            "components": _components(scale),
        },
        "row_cotangent_summary": _row_summary(arm),
        "model_before_sample": _sample(),
        "gradient_sample": _sample(),
        "alignment_hashes": [{"A": "same", "B": "same"}] * 16,
        "alignment_verdicts": ["PASS"] * 16,
        "state_changed_paths": {
            "model": [], "optimizer": [], "accumulator": [], "reference": []
        },
        "train_steps_before": 0,
        "train_steps_after": 0,
        "vjp_oracle": _oracle_summary() if arm == "tp4-vma-oracle" else None,
    }))
    raw = arm_root / "raw.log"
    extras = ""
    if arm in ("tp4-p59", "tp4-gather-off", "tp4-vma-oracle"):
      extras = (
          "[P66.VMA] outer_check_enabled\n"
          "tp_input_reduction=vma_autodiff_psum\n"
      )
    if arm == "tp4-vma-oracle":
      extras += (
          "[P66.ORACLE.NEGATIVE] detected=1 perturbation=normal_value\n"
          + "".join(
              "[P66.ORACLE.ENDPOINT] {}\n"
              for _ in range(len(ARM.ORACLE_ENDPOINTS))
          )
          + "[P66.ORACLE.SUMMARY] {}\n"
      )
    raw.write_text(
        extras
        + "[P66.TP4.ROWS.SUMMARY] "
        + json.dumps(_row_summary(arm))
        + "\n"
        + f"[P66.BACKWARD] arm={arm} verdict=PASS commits=0\n"
    )
    result = ARM.classify(
        arm=arm,
        run_log=raw,
        pre_alignment_report=pre,
        alignment_report=align,
        update_report=update,
        docker_exit=0,
    )
    classification = arm_root / "classification.json"
    classification.write_text(json.dumps(result))
    return result, classification, update, pre

  def _unsafe_arm(self, root: Path):
    arm = "tp4-p59-old"
    arm_root = root / arm
    arm_root.mkdir()
    pre = arm_root / "pre.jsonl"
    pre.write_text(
        json.dumps({"verdict": "PASS", "hashes": {"A": "same", "B": "same"}})
        + "\n"
    )
    align = arm_root / "align.jsonl"
    raw = arm_root / "raw.log"
    raw.write_text(
        "[P66.TP4.NUMERIC] "
        + json.dumps({"all_finite": True, "stable_norm": 1.0e21})
        + "\n[P66.TP4.PROFILE] "
        + json.dumps({"components": _components(1.0e20)})
        + "\n[P66.TP4.ROWS.SUMMARY] "
        + json.dumps(_row_summary(arm, padding_nonzero=4))
        + "\n"
    )
    result = ARM.classify(
        arm=arm,
        run_log=raw,
        pre_alignment_report=pre,
        alignment_report=align,
        update_report=arm_root / "missing.json",
        docker_exit=1,
    )
    classification = arm_root / "classification.json"
    classification.write_text(json.dumps(result))
    return result, classification, pre

  def test_expected_red_and_repaired_arms_support_h1(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      records = {
          arm: self._safe_arm(root, arm)
          for arm in ("tp4-serial", "tp4-p59", "tp4-gather-off")
      }
      unsafe, unsafe_classification, unsafe_pre = self._unsafe_arm(root)
      self.assertEqual(unsafe["verdict"], "EXPECTED_RED")
      for value in records.values():
        self.assertEqual(value[0]["verdict"], "PASS", value[0])
      result = CAMPAIGN.classify(
          classifications={
              **{arm: value[1] for arm, value in records.items()},
              "tp4-p59-old": unsafe_classification,
          },
          updates={arm: value[2] for arm, value in records.items()},
          pre_alignments={
              **{arm: value[3] for arm, value in records.items()},
              "tp4-p59-old": unsafe_pre,
          },
      )
      self.assertEqual(result["verdict"], "H1_VMA_SUPPORTED", result)

  def test_wrong_vma_marker_and_huge_safe_profile_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      result, _, update_path, _ = self._safe_arm(
          root, "tp4-p59", scale=1.0e7
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("mapped_gradient", result["reasons"])
      update = json.loads(update_path.read_text())
      update["gradient"]["stable_norm"] = 1.0
      update["engine_vjp"]["stable_norm"] = 1.0
      update["layerwise_profile"]["components"] = _components()
      update_path.write_text(json.dumps(update))
      raw = root / "tp4-p59/raw.log"
      raw.write_text("[P66.BACKWARD] arm=tp4-p59 verdict=PASS commits=0\n")
      second = ARM.classify(
          arm="tp4-p59",
          run_log=raw,
          pre_alignment_report=root / "tp4-p59/pre.jsonl",
          alignment_report=root / "tp4-p59/align.jsonl",
          update_report=update_path,
          docker_exit=0,
      )
      self.assertIn("vma_marker", second["reasons"])

  def test_missing_run_artifacts_are_classified_not_raised(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      raw = root / "raw.log"
      raw.write_text("admission failed before pre-alignment\n")
      result = ARM.classify(
          arm="tp4-serial",
          run_log=raw,
          pre_alignment_report=root / "missing-pre.jsonl",
          alignment_report=root / "missing-align.jsonl",
          update_report=root / "missing-update.json",
          docker_exit=1,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("docker_exit", result["reasons"])
      self.assertIn("pre_alignment", result["reasons"])

  def test_oracle_requires_all_endpoints_and_live_negative_control(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      result, _, update_path, _ = self._safe_arm(
          root, "tp4-vma-oracle"
      )
      self.assertEqual(result["verdict"], "PASS", result)
      update = json.loads(update_path.read_text())
      update["vjp_oracle"]["records"][0]["metrics"]["rel_l2"] = 0.5
      update_path.write_text(json.dumps(update))
      raw = root / "tp4-vma-oracle/raw.log"
      raw.write_text(raw.read_text().replace(
          "[P66.ORACLE.NEGATIVE] detected=1",
          "[P66.ORACLE.NEGATIVE] detected=0",
      ))
      rejected = ARM.classify(
          arm="tp4-vma-oracle",
          run_log=raw,
          pre_alignment_report=root / "tp4-vma-oracle/pre.jsonl",
          alignment_report=root / "tp4-vma-oracle/align.jsonl",
          update_report=update_path,
          docker_exit=0,
      )
      self.assertEqual(rejected["verdict"], "FAIL")
      self.assertIn("vjp_oracle", rejected["reasons"])

  def test_oracle_pair_requires_exact_candidate_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _, reference_class, reference_update, reference_pre = self._safe_arm(
          root, "tp4-p59"
      )
      _, oracle_class, oracle_update, oracle_pre = self._safe_arm(
          root, "tp4-vma-oracle"
      )
      result = ORACLE_PAIR.classify(
          reference_classification=reference_class,
          reference_pre_alignment=reference_pre,
          reference_update=reference_update,
          oracle_classification=oracle_class,
          oracle_pre_alignment=oracle_pre,
          oracle_update=oracle_update,
      )
      self.assertEqual(result["verdict"], "PASS", result)
      changed = json.loads(oracle_update.read_text())
      changed["engine_vjp"]["stable_norm"] = 3.0
      oracle_update.write_text(json.dumps(changed))
      rejected = ORACLE_PAIR.classify(
          reference_classification=reference_class,
          reference_pre_alignment=reference_pre,
          reference_update=reference_update,
          oracle_classification=oracle_class,
          oracle_pre_alignment=oracle_pre,
          oracle_update=oracle_update,
      )
      self.assertEqual(rejected["verdict"], "FAIL_OBSERVER_RED")
      self.assertIn("engine_vjp", rejected["observer_reasons"])

  def test_oracle_pair_reports_frozen_input_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _, reference_class, reference_update, reference_pre = self._safe_arm(
          root, "tp4-p59"
      )
      _, oracle_class, oracle_update, oracle_pre = self._safe_arm(
          root, "tp4-vma-oracle"
      )
      rows = [json.loads(line) for line in oracle_pre.read_text().splitlines()]
      rows[0]["hashes"]["A"] = "different"
      oracle_pre.write_text("".join(json.dumps(row) + "\n" for row in rows))
      result = ORACLE_PAIR.classify(
          reference_classification=reference_class,
          reference_pre_alignment=reference_pre,
          reference_update=reference_update,
          oracle_classification=oracle_class,
          oracle_pre_alignment=oracle_pre,
          oracle_update=oracle_update,
      )
      self.assertEqual(result["verdict"], "INCONCLUSIVE_INPUT_MISMATCH")
      self.assertIn("pre_alignment_hashes", result["input_reasons"])


if __name__ == "__main__":
  unittest.main()
