#!/usr/bin/env bash
# Container-side full-depth P66 DP1xTP4 group-0 backward entrypoint.
set -euo pipefail

: "${P66_REPO:?P66_REPO unset}"
: "${P66_XLA_FLAGS:?P66_XLA_FLAGS unset}"
: "${CANON_P66_BACKWARD_ARM:?CANON_P66_BACKWARD_ARM unset}"

# shellcheck disable=SC1091
source "$P66_REPO/canon-zero-tim/cluster/profiles/_canonical_engine.env"
# shellcheck disable=SC1091
source "$P66_REPO/canon-zero-tim/cluster/profiles/qwen3-1p7b-dp1-tp4-gsm8k-p66.env"
export XLA_FLAGS="$P66_XLA_FLAGS"

python3 - <<'PY'
import os
import jax
import numpy as np
from tunix.rl import dp_workloads

devices = tuple(jax.devices())
if len(devices) != 4 or jax.default_backend() != "tpu":
  raise SystemExit("P66 TP4 proxy requires exactly four direct-attached TPU devices")
actual_ids = [int(device.id) for device in devices]
expected_ids = [
    int(value)
    for value in os.environ["CANON_EXPECT_TRAIN_MESH_IDS"].split(",")
]
if actual_ids != expected_ids:
  raise SystemExit(
      f"P66 TP4 device order mismatch: expected={expected_ids} actual={actual_ids}"
  )
workload = dp_workloads.get_workload("gsm8k-p66-dp1-tp4")
dp_workloads.validate_environment(
    workload, os.environ, require_reduction_admission=True
)
print(
    "[P66.TP4] PREFLIGHT_PASS "
    f"arm={os.environ['CANON_P66_BACKWARD_ARM']} topology=DP1xTP4 "
    f"device_ids={actual_ids} global_trajectories=16 groups=16 "
    "local_m=256 global_m=256 reverse_groups=1/16 optimizer_commits=0",
    flush=True,
)
PY

exec python3 -u examples/math_gsm8k/qwen3_grpo_demo.py \
  --mesh_dp=1 --mesh_tp=4 --batch_size=2 --mini_batch_size=2 \
  --train_micro_batch_size=2 --compute_logps_micro_batch_size=2 \
  --train_trajectory_micro_batch_size=1 \
  --max_steps=1 --num_generations=8 --max_prompt_length=1024 \
  --max_response_length=256 --max_concurrency=1 \
  --rollout_vllm_hbm_utilization=0.20 \
  --rollout_vllm_max_num_seqs=16 \
  --rollout_vllm_max_num_batched_tokens=256
