#!/usr/bin/env python3
"""Static and shell gates for the direct-attached P59 DP4 carrier."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
PROFILE = PKG / "cluster/profiles/qwen3-1p7b-dp4-tp1-gsm8k-p59.env"
V1_PROFILE = (
    PKG / "cluster/profiles/qwen3-1p7b-dp4-tp1-gsm8k-v1-hp.env"
)
RUNNER = PKG / "tasks/p59-dp16-parallel-backward/scripts/run_onehost_dp4.sh"
INNER = PKG / "tasks/p59-dp16-parallel-backward/scripts/run_dp4_inner.sh"
P61_PAIR = (
    PKG
    / "tasks/p61-backward-numerical-oracle/scripts/"
    "run_onehost_dp4_numerical_ab.sh"
)
MODEL_CONTRACT = (
    PKG / "src/engine_shims/models/qwen1p7b_tp1/p22xf_contract.py"
)
ADAPTER = ROOT / "tunix/rl/canonical_qwen3_adapter.py"


class DP4CarrierTest(unittest.TestCase):

  def test_profile_resolves_exact_proxy_and_p56_recipe(self):
    keys = (
        "CANON_PROFILE",
        "CANON_MODEL_DIR_NAME",
        "CANON_P32_WORKLOAD",
        "CANON_DP_SIZE",
        "CANON_TP_SIZE",
        "CANON_TOTAL_DEVICES",
        "CANON_ENGINE_DP_SIZE",
        "CANON_GLOBAL_PROMPTS",
        "CANON_LOCAL_PROMPTS",
        "CANON_GLOBAL_TRAJECTORIES",
        "CANON_LOCAL_TRAJECTORIES",
        "MIN_TOKEN_BUCKET",
        "CANON_QWEN3_TP_SIZE",
        "FL_SHARED_MESH",
        "CANON_P28_BATCHED_REVERSE",
        "CANON_FUSED_TREE_OPS",
        "CANON_PALLAS_GATHERED_LOGPROBS",
        "CANON_FIXED_AR_GATHER",
        "CANON_PALLAS_NORM_MATMUL",
        "CANON_LOGPROB_STEP_FUSION",
        "CANON_CONTINUE_DECODE",
        "CANON_P59_DP4_SERIAL_MESH_BRIDGE",
        "CANON_P59_DP4_TAIL8",
    )
    python = (
        "import json,os; print(json.dumps({key: os.environ.get(key) "
        f"for key in {keys!r}}}))"
    )
    command = f"source {PROFILE}; python3 -c {json.dumps(python)}"
    environ = os.environ.copy()
    environ.update({
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P33_RUN_STAGE": "three-update",
        "CANON_P33_NO_COMMIT": "0",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_P59_DP4_SERIAL_MESH_BRIDGE": "1",
        "CANON_P59_DP4_TAIL8": "0",
        "CANON_OPT_STATE_RESIDENT": "1",
        "CANON_P30_OPT_STATE_OFFLOAD": "0",
        "CANON_WANDB_RUN_NAME": "p59-test",
    })
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=environ,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    self.assertEqual(result.returncode, 0, result.stdout)
    resolved = json.loads(result.stdout.splitlines()[-1])
    expected = {
        "CANON_PROFILE": "qwen3-1p7b-dp4-tp1-gsm8k-p59",
        "CANON_MODEL_DIR_NAME": "qwen1p7b_tp1",
        "CANON_P32_WORKLOAD": "gsm8k-p59-dp4-tp1",
        "CANON_DP_SIZE": "4",
        "CANON_TP_SIZE": "1",
        "CANON_TOTAL_DEVICES": "4",
        "CANON_ENGINE_DP_SIZE": "4",
        "CANON_GLOBAL_PROMPTS": "8",
        "CANON_LOCAL_PROMPTS": "2",
        "CANON_GLOBAL_TRAJECTORIES": "64",
        "CANON_LOCAL_TRAJECTORIES": "16",
        "MIN_TOKEN_BUCKET": "1024",
        "CANON_QWEN3_TP_SIZE": "1",
        "FL_SHARED_MESH": "4,1",
        "CANON_P28_BATCHED_REVERSE": "1",
        "CANON_FUSED_TREE_OPS": "1",
        "CANON_PALLAS_GATHERED_LOGPROBS": "0",
        "CANON_FIXED_AR_GATHER": "1",
        "CANON_PALLAS_NORM_MATMUL": "1",
        "CANON_LOGPROB_STEP_FUSION": "1",
        "CANON_CONTINUE_DECODE": "8",
        "CANON_P59_DP4_SERIAL_MESH_BRIDGE": "1",
        "CANON_P59_DP4_TAIL8": "0",
    }
    self.assertEqual(resolved, expected)

  def test_profile_rejects_partial_admission_and_wrong_stage(self):
    for exports, marker in (
        (
            "CANON_P32_TRAIN_ADMITTED=1 "
            "CANON_P32_DP_REDUCTION_ADMITTED=0 "
            "CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1 "
            "CANON_P33_RUN_STAGE=three-update CANON_P33_NO_COMMIT=0",
            "admissions must be all zero or all one",
        ),
        (
            "CANON_P32_TRAIN_ADMITTED=1 "
            "CANON_P32_DP_REDUCTION_ADMITTED=1 "
            "CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1 "
            "CANON_P33_RUN_STAGE=full CANON_P33_NO_COMMIT=0",
            "requires P61 one-update, three-update, or p59-eight-update",
        ),
        (
            "CANON_P59_DP4_SERIAL_MESH_BRIDGE=bad",
            "CANON_P59_DP4_SERIAL_MESH_BRIDGE must be exactly 0 or 1",
        ),
        (
            "CANON_P33_RUN_STAGE=p59-eight-update "
            "CANON_P59_DP4_TAIL8=0",
            "p59-eight-update requires CANON_P59_DP4_TAIL8=1",
        ),
        (
            "CANON_P33_RUN_STAGE=three-update CANON_P59_DP4_TAIL8=1",
            "three-update requires CANON_P59_DP4_TAIL8=0",
        ),
    ):
      result = subprocess.run(
          ["bash", "-c", f"{exports}; export {exports}; source {PROFILE}"],
          cwd=ROOT,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn(marker, result.stdout)

  def test_v1_profile_resolves_final_supported_bundle(self):
    keys = (
        "CANON_PROFILE",
        "CANON_CONTINUE_DECODE",
        "CANON_FIXED_AR_GATHER",
        "CANON_PALLAS_GATHERED_LOGPROBS",
        "CANON_LOGPROB_STEP_FUSION",
        "CANON_VLLM_ENABLE_PREFIX_CACHING",
        "CANON_P59_RANK_PARALLEL_BACKWARD",
        "CANON_P28_BATCHED_REPORT",
        "CANON_BATCHED_EVIDENCE",
        "CANON_P28_BATCHED_REVERSE",
        "CANON_FUSED_TREE_OPS",
        "CANON_PALLAS_NORM_MATMUL",
        "CANON_P38_FIXED_LM_HEAD",
    )
    python = (
        "import json,os; print(json.dumps({key: os.environ.get(key) "
        f"for key in {keys!r}}}))"
    )
    command = f"source {V1_PROFILE}; python3 -c {json.dumps(python)}"
    environ = os.environ.copy()
    environ.update({
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P33_RUN_STAGE": "three-update",
        "CANON_P33_NO_COMMIT": "0",
        "CANON_P59_KIND": "v1",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_P59_DP4_SERIAL_MESH_BRIDGE": "1",
        "CANON_P59_DP4_TAIL8": "0",
        "CANON_OPT_STATE_RESIDENT": "1",
        "CANON_P30_OPT_STATE_OFFLOAD": "0",
        "CANON_WANDB_RUN_NAME": "v1-onehost-test",
    })
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=environ,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertEqual(json.loads(result.stdout.splitlines()[-1]), {
        "CANON_PROFILE": "qwen3-1p7b-dp4-tp1-gsm8k-v1-hp",
        "CANON_CONTINUE_DECODE": "8",
        "CANON_FIXED_AR_GATHER": "1",
        "CANON_PALLAS_GATHERED_LOGPROBS": "1",
        "CANON_LOGPROB_STEP_FUSION": "1",
        "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_P28_BATCHED_REPORT": "1",
        "CANON_BATCHED_EVIDENCE": "1",
        "CANON_P28_BATCHED_REVERSE": "0",
        "CANON_FUSED_TREE_OPS": "0",
        "CANON_PALLAS_NORM_MATMUL": "0",
        "CANON_P38_FIXED_LM_HEAD": "0",
    })

  def test_runner_freezes_immutable_hard_gate_and_serial_lane(self):
    text = RUNNER.read_text(encoding="utf-8")
    for token in (
        'root="$evidence/p59_dp4_${kind}_${label}"',
        'if [ -e "$root" ]',
        "grep -E '^(p51_|p59_)'",
        'align_fail" -eq 0',
        "--workload gsm8k-p59-dp4-tp1",
        "--dp-size 4 --tp-size 1",
        "CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3",
        "CANON_EXPECT_TRAIN_MESH_IDS=0,2,1,3",
        '-e WANDB_API_KEY="$WANDB_API_KEY"',
        "--model qwen1p7b_tp1",
        "CANON_P59_DP4_SERIAL_MESH_BRIDGE=1",
        "CANON_P59_XPROF_BACKWARD_DIR",
        'tail) rank_parallel=1; capture=0; numerical=0; tail8=1; run_stage=p59-eight-update; expected_steps=8; expected_align=136',
        'numerical-control) rank_parallel=0; capture=0; numerical=1; tail8=0; run_stage=one-update; expected_steps=1; expected_align=17',
        'numerical-candidate) rank_parallel=1; capture=0; numerical=1; tail8=0; run_stage=one-update; expected_steps=1; expected_align=17',
        'v1) rank_parallel=1; capture=0; numerical=0; tail8=0; run_stage=three-update; expected_steps=3; expected_align=51; recipe=v1-phase4-current-bundle',
        "qwen3-1p7b-dp4-tp1-gsm8k-v1-hp.env",
        '-e CANON_P59_DP4_TAIL8="$tail8"',
        '-e CANON_P61_BACKWARD_NUMERICAL_DIR=',
        'steps_done" -eq "$expected_steps"',
        'align_pass" -eq "$expected_align"',
    ):
      self.assertIn(token, text)
    self.assertNotIn("git push", text)
    self.assertNotIn("rm -rf", text)

  def test_serial_mesh_bridge_is_wired_only_after_the_rank_adjoint(self):
    text = ADAPTER.read_text(encoding="utf-8")
    function = text[text.index("  def segmented_dp_grpo_value_and_grad("):]
    parallel_start = function.index("      if rank_parallel_backward:")
    serial_start = function.index(
        "      for rank in range(contract.dp_size):", parallel_start
    )
    bridge_call = function.index(
        "_p59_align_serial_gradient_to_trainer_state(", serial_start
    )
    serial_reducer = function.index("          reducer = reducer_factory(", bridge_call)
    self.assertNotIn(
        "_p59_align_serial_gradient_to_trainer_state(",
        function[parallel_start:serial_start],
    )
    self.assertLess(serial_start, bridge_call)
    self.assertLess(bridge_call, serial_reducer)

  def test_tp1_model_contract_has_exact_local_shapes_and_negative(self):
    spec = importlib.util.spec_from_file_location(
        "p59_qwen1p7b_tp1_contract", MODEL_CONTRACT
    )
    assert spec is not None and spec.loader is not None
    contract = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = contract
    try:
      spec.loader.exec_module(contract)
    finally:
      sys.modules.pop(spec.name, None)
    self.assertEqual(contract.TP_SIZE, 1)
    self.assertEqual(contract.MATMUL_K_PADDING, {})
    self.assertEqual(contract.MATMUL_N_PADDING, {151936: 152064})
    self.assertEqual(
        {site.family: (site.k_local, site.n_local) for site in contract.SITES},
        {
            "q_proj": (2048, 2048),
            "k_proj": (2048, 1024),
            "v_proj": (2048, 1024),
            "o_proj": (2048, 2048),
            "gate_proj": (2048, 6144),
            "up_proj": (2048, 6144),
            "down_proj": (6144, 2048),
        },
    )
    contract.validate_manifest(contract.SITES)
    enabled = {**contract._MODEL_ENV, contract.ENV: "1", "CANON_FIXED_AR": "1"}
    with mock.patch.dict(os.environ, enabled, clear=True):
      contract.preflight(require_enabled=True)
      os.environ["CANON_QWEN3_TP_SIZE"] = "4"
      with self.assertRaisesRegex(RuntimeError, "model contract mismatch"):
        contract.preflight(require_enabled=True)

  def test_inner_command_selects_only_frozen_three_or_eight_update_proxy(self):
    text = INNER.read_text(encoding="utf-8")
    for token in (
        "--mesh_dp=4 --mesh_tp=1",
        "--batch_size=8 --mini_batch_size=8",
        "--train_trajectory_micro_batch_size=4",
        "three-update:0) max_steps=3",
        "one-update:0) max_steps=1",
        "p59-eight-update:1) max_steps=8",
        '--max_steps="$max_steps" --num_generations=8',
        "max_concurrency=64",
        "max_response_length=1024",
        "max_concurrency=1",
        "max_response_length=256",
        'CANON_P60_DETERMINISTIC_AB:-0',
        '--max_concurrency="$max_concurrency"',
        'v1) p59_profile=qwen3-1p7b-dp4-tp1-gsm8k-v1-hp.env',
        "--rollout_vllm_max_num_seqs=16",
        "--rollout_vllm_max_num_batched_tokens=256",
    ):
      self.assertIn(token, text)

  def test_profile_reserves_one_update_for_exact_p61_carrier(self):
    base = (
        "CANON_P32_TRAIN_ADMITTED=1 "
        "CANON_P32_DP_REDUCTION_ADMITTED=1 "
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1 "
        "CANON_P33_RUN_STAGE=one-update CANON_P33_NO_COMMIT=0 "
        "CANON_P59_DP4_TAIL8=0 CANON_P60_DETERMINISTIC_AB=1 "
    )
    rejected = subprocess.run(
        ["bash", "-c", f"export {base}; source {PROFILE}"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    self.assertNotEqual(rejected.returncode, 0)
    self.assertIn("one-update is reserved", rejected.stdout)
    accepted = subprocess.run(
        [
            "bash",
            "-c",
            f"export {base} CANON_P61_BACKWARD_NUMERICAL_DIR=/tmp/p61; "
            f"source {PROFILE}; echo PASS",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    self.assertEqual(accepted.returncode, 0, accepted.stdout)
    self.assertIn("PASS", accepted.stdout)

  def test_profile_uses_only_the_one_group_backward_window(self):
    text = RUNNER.read_text(encoding="utf-8")
    self.assertIn("-e CANON_XPROF_DIR=", text)
    self.assertIn("-e CANON_P59_XPROF_BACKWARD_DIR=", text)
    self.assertIn("inspect_xprof_capture.py", text)
    self.assertIn("xprof_inspection_exit", text)

  def test_p61_pair_is_serial_and_requires_external_tier1_baseline(self):
    text = P61_PAIR.read_text(encoding="utf-8")
    serial = text.index('numerical-control "$serial_label"')
    parallel = text.index('numerical-candidate "$parallel_label"')
    compare = text.index('python3 "$comparator"')
    self.assertLess(serial, parallel)
    self.assertLess(parallel, compare)
    for token in (
        'if [ ! -s "$tier1_baseline" ]',
        'if [ -e "$serial_root" ] || [ -e "$parallel_root" ] || [ -e "$ab_root" ]',
        "--control-root",
        "--candidate-root",
        "--tier1-baseline",
        "performance_eligible=0",
    ):
      self.assertIn(token, text)
    self.assertNotIn("git push", text)
    self.assertNotIn("rm -rf", text)


if __name__ == "__main__":
  unittest.main()
