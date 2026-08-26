#!/usr/bin/env python3
"""Host gates for the P66 ordinary/segmented backward carrier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
PROFILE = PKG / "cluster/profiles/qwen3-1p7b-dp4-tp1-gsm8k-p59.env"
TP4_PROFILE = PKG / "cluster/profiles/qwen3-1p7b-dp1-tp4-gsm8k-p66.env"
RUNNER = PKG / "tasks/p59-dp16-parallel-backward/scripts/run_onehost_dp4.sh"
INNER = PKG / "tasks/p59-dp16-parallel-backward/scripts/run_dp4_inner.sh"
PAIR = PKG / "tasks/p66-onehost-gsm8k-convergence/scripts/run_backward_ab.sh"
TP4_RUNNER = PKG / "tasks/p66-onehost-gsm8k-convergence/scripts/run_onehost_tp4_arm.sh"
TP4_CAMPAIGN = PKG / "tasks/p66-onehost-gsm8k-convergence/scripts/run_tp4_campaign.sh"
LEARNER = ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
GSM8K_DEMO = ROOT / "examples/math_gsm8k/qwen3_grpo_demo.py"
ADAPTER = ROOT / "tunix/rl/canonical_qwen3_adapter.py"
LINEAR_SHIM = PKG / "src/engine_shims/linear_p22xf.py"
ATTENTION_PATCH = (
    PKG / "patches/tpu_inference/25-attention-p59-local-kv.patch"
)
EMBED_PATCH = PKG / "patches/tpu_inference/02-embed.patch"
PALLAS_VMA_SHIMS = tuple(
    PKG / "src/engine_shims" / name
    for name in (
        "p22_pallas_matmul.py",
        "p22_pallas_rmsnorm.py",
        "p22_pallas_swiglu.py",
        "p56_pallas_norm_matmul.py",
    )
)


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


CLASSIFIER = _load(
    "p66_classify_arm", PKG / "tests/p66_backward/classify_arm.py"
)
COMPARATOR = _load(
    "p66_compare_arms", PKG / "tests/p66_backward/compare_arms.py"
)


def _capture(root: Path, name: str, values: list[np.ndarray]) -> None:
  capture = root / name
  capture.mkdir(parents=True)
  leaves = []
  total = 0
  for index, value in enumerate(values):
    value = np.ascontiguousarray(value)
    path = capture / f"leaf_{index:05d}.npy"
    with path.open("wb") as output:
      np.save(output, value, allow_pickle=False)
    data = value.tobytes(order="C")
    leaves.append({
        "index": index,
        "path": f"['leaf{index}']",
        "file": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "elements": int(value.size),
        "data_bytes": int(value.nbytes),
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    total += int(value.nbytes)
  (capture / "manifest.json").write_text(
      json.dumps({
          "schema": "canon-p61-full-tree-capture-v1",
          "capture": name,
          "leaves": leaves,
          "leaf_count": len(leaves),
          "total_data_bytes": total,
      }),
      encoding="utf-8",
  )


def _alignment_hashes() -> list[dict[str, str]]:
  return [
      {key: f"{index:02d}-{key}" for key in COMPARATOR._P61.HASH_KEYS}
      for index in range(16)
  ]


class P66ContractTest(unittest.TestCase):

  def test_profile_admits_only_exact_p66_no_commit(self):
    base = {
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P33_RUN_STAGE": "backward-no-commit",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_P59_DP4_TAIL8": "0",
        "CANON_P60_DETERMINISTIC_AB": "1",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "0",
        "CANON_P66_BACKWARD_ARM": "ordinary",
        "CANON_P66_BACKWARD_CAPTURE_DIR": "/tmp/p66-capture",
    }
    accepted = subprocess.run(
        ["bash", "-c", f"source {PROFILE}; echo PASS"],
        cwd=ROOT,
        env={**os.environ, **base},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    self.assertEqual(accepted.returncode, 0, accepted.stdout)
    self.assertIn("PASS", accepted.stdout)
    for change, marker in (
        ({"CANON_P66_BACKWARD_ARM": "bad"}, "must be empty, ordinary, or segmented"),
        ({"CANON_P66_BACKWARD_CAPTURE_DIR": "relative"}, "absolute capture"),
        ({"CANON_P59_RANK_PARALLEL_BACKWARD": "1"}, "reserved for the exact P66"),
        ({"CANON_P33_NO_COMMIT": "0"}, "no-commit selection changed"),
    ):
      rejected = subprocess.run(
          ["bash", "-c", f"source {PROFILE}"],
          cwd=ROOT,
          env={**os.environ, **base, **change},
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )
      self.assertNotEqual(rejected.returncode, 0)
      self.assertIn(marker, rejected.stdout)

  def test_runtime_gate_is_no_commit_and_two_arm_only(self):
    learner = LEARNER.read_text(encoding="utf-8")
    start = learner.index("  def _run_p66_backward_gate(")
    end = learner.index("  def _run_p28_g6_update(", start)
    gate = learner[start:end]
    self.assertIn('"tp4-p59-old"', gate)
    self.assertIn('"tp4-gather-off"', gate)
    self.assertIn("nnx.value_and_grad(", gate)
    self.assertIn("segmented_dp_grpo_value_and_grad(", gate)
    self.assertIn('gradient_microbatch_sink=None', gate)
    self.assertNotIn("commit_precomputed_gradients", gate)
    self.assertNotIn("optimizer.update", gate)
    runner = RUNNER.read_text(encoding="utf-8")
    self.assertIn("p66-ordinary)", runner)
    self.assertIn("p66-segmented)", runner)
    self.assertIn('-e CANON_P33_NO_COMMIT="$no_commit"', runner)
    self.assertIn('-e CANON_P66_BACKWARD_CAPTURE_DIR=', runner)
    self.assertIn("backward-no-commit:0) max_steps=1", INNER.read_text())
    pair = PAIR.read_text(encoding="utf-8")
    self.assertLess(pair.index("p66-ordinary"), pair.index("p66-segmented"))
    self.assertNotIn("git push", pair)
    self.assertNotIn("rm -rf", pair)

  def test_tp4_profile_and_runner_admit_only_registered_arms(self):
    base = {
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P33_RUN_STAGE": "backward-no-commit",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_P60_DETERMINISTIC_AB": "1",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_P66_P59_CHECK_VMA": "1",
        "CANON_FIXED_AR_GATHER": "1",
        "CANON_P66_BACKWARD_ARM": "tp4-p59",
        "CANON_P66_BACKWARD_CAPTURE_DIR": "/tmp/p66-tp4",
    }
    accepted = subprocess.run(
        ["bash", "-c", f"source {TP4_PROFILE}; echo PASS"],
        cwd=ROOT,
        env={**os.environ, **base},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    self.assertEqual(accepted.returncode, 0, accepted.stdout)
    self.assertIn("PASS", accepted.stdout)
    rejected = subprocess.run(
        ["bash", "-c", f"source {TP4_PROFILE}"],
        cwd=ROOT,
        env={**os.environ, **base, "CANON_P66_P59_CHECK_VMA": "0"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    self.assertNotEqual(rejected.returncode, 0)
    self.assertIn("not registered", rejected.stdout)
    runner = TP4_RUNNER.read_text(encoding="utf-8")
    for arm in (
        "tp4-serial",
        "tp4-p59-old",
        "tp4-p59",
        "tp4-gather-off",
        "tp4-vma-oracle",
    ):
      self.assertIn(arm, runner)
    self.assertIn("CANON_P33_NO_COMMIT=1", runner)
    self.assertIn("CANON_P63_OVERFLOW_SAFE_CLIP=0", runner)
    self.assertNotIn("git push", runner)
    campaign = TP4_CAMPAIGN.read_text(encoding="utf-8")
    self.assertIn(
        "order=tp4-serial,tp4-p59-old,tp4-p59,tp4-gather-off", campaign
    )
    self.assertNotIn("rm -rf", campaign)
    demo = GSM8K_DEMO.read_text(encoding="utf-8")
    deterministic_admission = demo[
        demo.index("if (\n    CANON_P60_DETERMINISTIC_AB"):
        demo.index("if CANON_P61_BACKWARD_NUMERICAL_DIR")
    ]
    self.assertIn('"gsm8k-p66-dp1-tp4"', deterministic_admission)
    adapter = ADAPTER.read_text(encoding="utf-8")
    group_spec = adapter[
        adapter.index("  def _p32_group_spec("):
        adapter.index("  def _p32_group_chunk_inputs(")
    ]
    self.assertIn("p66_tp4_proxy", group_spec)
    self.assertIn('== "gsm8k-p66-dp1-tp4"', group_spec)
    self.assertIn("and bool(_p66_tp4_arm())", group_spec)
    rank_start = adapter.index("  def assemble_full_state_gradient(")
    rank_assembly = adapter[
        rank_start:adapter.index("  def __init__(", rank_start)
    ]
    self.assertIn("p66_unit_rank", rank_assembly)
    self.assertIn('"tp4-vma-oracle"', rank_assembly)
    attention_patch = ATTENTION_PATCH.read_text(encoding="utf-8")
    self.assertIn('== "gsm8k-p66-dp1-tp4"', attention_patch)
    self.assertIn('"tp4-vma-oracle"', attention_patch)
    for shim in PALLAS_VMA_SHIMS:
      source = shim.read_text(encoding="utf-8")
      self.assertIn("p66_vma_output_manual_axis_type", source, shim)
      self.assertIn("manual_axis_type=", source, shim)
    self.assertIn("nested_engine_body_reuses_outer_map", adapter)
    self.assertIn("return localized_fun", adapter)
    self.assertIn("def mark_data_varying(leaf):", adapter)
    self.assertIn("if data_axis in manual_axis_type.varying:", adapter)
    self.assertIn("manual_axis_type.unreduced", adapter)
    self.assertIn("manual_axis_type.reduced", adapter)
    linear_shim = LINEAR_SHIM.read_text(encoding="utf-8")
    self.assertIn("def _p66_replicated_tp_value(value):", linear_shim)
    self.assertIn(
        "base.jax.lax.pmean(value, base._CANON_TP_AXIS)", linear_shim
    )
    self.assertIn("return _p66_replicated_tp_value(acc)", linear_shim)
    self.assertIn("return _p66_replicated_tp_value(acc[0])", linear_shim)
    embed_patch = EMBED_PATCH.read_text(encoding="utf-8")
    self.assertIn('CANON_P66_P59_CHECK_VMA", "0"', embed_patch)
    self.assertIn("jax.lax.pmean(result, _ax)", embed_patch)
    rpa_vma_patch = (
        PKG / "patches/tpu_inference/29-rpa-p66-vma-output.patch"
    ).read_text(encoding="utf-8")
    self.assertIn("manual_axis_type", rpa_vma_patch)
    self.assertIn("jax.typeof(q).mat", rpa_vma_patch)
    self.assertIn("jax.typeof(kv_cache).mat", rpa_vma_patch)
    self.assertIn("CANON_P66_P59_CHECK_VMA", rpa_vma_patch)
    install = (PKG / "install.sh").read_text(encoding="utf-8")
    self.assertIn("rpa_kernel_p66.py", install)
    self.assertIn("29-rpa-p66-vma-output.patch", install)
    self.assertIn("rpa_kernel_p66.py", runner)
    self.assertIn(
        'os.environ.get("CANON_P66_P59_CHECK_VMA", "0") == "1"',
        adapter,
    )
    self.assertIn("p66_vjp_oracle.compare(", adapter)
    self.assertIn("p66_vjp_oracle.negative_control()", adapter)
    self.assertIn("layer_index in (27, 14, 0)", adapter)
    learner = LEARNER.read_text(encoding="utf-8")
    self.assertIn('"tp4-vma-oracle": ("1", "1", "1")', learner)

  def test_classifier_and_comparator_keep_exact_pair(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      arm_roots = {}
      updates = {}
      classifications = {}
      for arm in ("ordinary", "segmented"):
        arm_root = root / arm
        capture_root = arm_root / "capture"
        _capture(capture_root, "model_before", [np.array([1, 2], np.float32)])
        _capture(capture_root, "gradient", [np.array([0.5, -0.25], np.float32)])
        raw = arm_root / "raw.log"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(
            f"[P66.BACKWARD] arm={arm} verdict=PASS commits=0 "
            "alignments=16/16 gradient_norm=1.0 seconds=1.0\n",
            encoding="utf-8",
        )
        pre = arm_root / "pre.jsonl"
        pre.write_text(json.dumps({"verdict": "PASS"}) + "\n", encoding="utf-8")
        align = arm_root / "align.jsonl"
        align.write_text(
            "".join(json.dumps({"verdict": "PASS"}) + "\n" for _ in range(16)),
            encoding="utf-8",
        )
        update = arm_root / "update.json"
        update.write_text(json.dumps({
            "schema": "canon-p66-backward-gate-v1",
            "arm": arm,
            "verdict": "PASS",
            "commits": 0,
            "dp_size": 4,
            "tp_size": 1,
            "global_trajectories": 64,
            "gradient_groups": 16,
            "gradient": {
                "all_finite": True,
                "any_nonzero": True,
                "stable_norm": 1.0,
            },
            "alignment_hashes": _alignment_hashes(),
            "alignment_verdicts": ["PASS"] * 16,
            "state_changed_paths": {
                "model": [], "optimizer": [], "accumulator": [], "reference": []
            },
            "train_steps_before": 0,
            "train_steps_after": 0,
        }), encoding="utf-8")
        result = CLASSIFIER.classify(
            arm=arm,
            run_log=raw,
            pre_alignment_report=pre,
            alignment_report=align,
            update_report=update,
            capture_root=capture_root,
        )
        self.assertEqual(result["verdict"], "PASS", result)
        classification = arm_root / "classification.json"
        classification.write_text(json.dumps(result), encoding="utf-8")
        arm_roots[arm] = capture_root
        updates[arm] = update
        classifications[arm] = classification
      baseline = root / "tier1.json"
      baseline.write_text(json.dumps({
          "schema": "canon-p61-tier1-baseline-v1",
          "gradient": {
              "rel_l2": 0.01,
              "one_minus_cos": 0.0001,
              "norm_ratio_error": 0.01,
              "sign_mismatch_rate": 0.01,
          },
      }), encoding="utf-8")
      result = COMPARATOR.compare(
          ordinary_root=arm_roots["ordinary"],
          segmented_root=arm_roots["segmented"],
          ordinary_update=updates["ordinary"],
          segmented_update=updates["segmented"],
          ordinary_classification=classifications["ordinary"],
          segmented_classification=classifications["segmented"],
          tier1_baseline=baseline,
      )
      self.assertEqual(result["verdict"], "P66_GRADIENT_KEEP", result)
      self.assertTrue(result["model_before_array_exact"])
      self.assertTrue(result["same_input_seven_hashes"])


if __name__ == "__main__":
  unittest.main()
