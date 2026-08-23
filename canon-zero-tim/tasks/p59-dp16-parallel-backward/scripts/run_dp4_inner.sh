#!/usr/bin/env bash
# Container-side frozen P59 DP4xTP1 workload entrypoint.
set -euo pipefail

: "${P59_REPO:?P59_REPO unset}"
: "${P59_XLA_FLAGS:?P59_XLA_FLAGS unset}"
: "${CANON_P59_KIND:?CANON_P59_KIND unset}"
: "${CANON_PRE_ALIGN_REPORT:?CANON_PRE_ALIGN_REPORT unset}"
: "${CANON_ALIGN_REPORT:?CANON_ALIGN_REPORT unset}"
: "${CANON_UPDATE_REPORT:?CANON_UPDATE_REPORT unset}"

case "${CANON_P33_RUN_STAGE:-}:${CANON_P59_DP4_TAIL8:-0}" in
  one-update:0) max_steps=1 ;;
  three-update:0) max_steps=3 ;;
  p59-eight-update:1) max_steps=8 ;;
  *)
    echo "[P59.DP4] invalid stage/tail pairing: "
    echo "${CANON_P33_RUN_STAGE:-unset}:${CANON_P59_DP4_TAIL8:-unset}"
    exit 2
    ;;
esac

case "${CANON_P60_DETERMINISTIC_AB:-0}" in
  0)
    max_concurrency=64
    max_response_length=1024
    ;;
  1)
    # Match the proven P41/P48 deterministic carrier: requests enter vLLM one
    # at a time, so identical seeds consume RNG in an identical request order.
    max_concurrency=1
    # 1024 + 256 is divisible by the model's exact 256-token Splash block;
    # this is the already-proven P35 bounded GSM8K envelope.
    max_response_length=256
    ;;
  *)
    echo "[P60.HASH_AB] CANON_P60_DETERMINISTIC_AB must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

# shellcheck disable=SC1091
source "$P59_REPO/canon-zero-tim/cluster/profiles/_canonical_engine.env"
case "$CANON_P59_KIND" in
  v1) p59_profile=qwen3-1p7b-dp4-tp1-gsm8k-v1-hp.env ;;
  *) p59_profile=qwen3-1p7b-dp4-tp1-gsm8k-p59.env ;;
esac
# shellcheck disable=SC1091
source "$P59_REPO/canon-zero-tim/cluster/profiles/$p59_profile"
unset p59_profile
export XLA_FLAGS="$P59_XLA_FLAGS"

python3 - <<'PY'
import os
import numpy as np
import jax
from jax.experimental import mesh_utils
from tunix.rl import dp_workloads

devices = tuple(jax.devices())
if len(devices) != 4 or jax.default_backend() != "tpu":
  raise SystemExit(
      "P59 DP4 proxy requires exactly four direct-attached TPU devices"
  )
arranged = mesh_utils.create_device_mesh(
    (4, 1), devices, allow_split_physical_axes=True
)
actual_ids = [int(device.id) for device in np.asarray(arranged).reshape(-1)]
expected_ids = [
    int(value)
    for value in os.environ["CANON_EXPECT_TRAIN_MESH_IDS"].split(",")
]
if actual_ids != expected_ids:
  raise SystemExit(
      f"P59 DP4 training mesh mismatch: expected={expected_ids} "
      f"actual={actual_ids}"
  )
workload = dp_workloads.get_workload("gsm8k-p59-dp4-tp1")
dp_workloads.validate_environment(
    workload, os.environ, require_reduction_admission=True
)
print(
    "[P59.DP4] PREFLIGHT_PASS "
    f"kind={os.environ['CANON_P59_KIND']} profile={os.environ['CANON_PROFILE']} "
    "topology=DP4xTP1 "
    f"mesh_ids={actual_ids} global_trajectories=64 local_trajectories=16 "
    "groups=16 local_m=256 global_m=1024",
    flush=True,
)
PY

exec python3 -u examples/math_gsm8k/qwen3_grpo_demo.py \
  --mesh_dp=4 --mesh_tp=1 --batch_size=8 --mini_batch_size=8 \
  --train_micro_batch_size=8 --compute_logps_micro_batch_size=8 \
  --train_trajectory_micro_batch_size=4 \
  --max_steps="$max_steps" --num_generations=8 --max_prompt_length=1024 \
  --max_response_length="$max_response_length" \
  --max_concurrency="$max_concurrency" \
  --rollout_vllm_hbm_utilization=0.20 \
  --rollout_vllm_max_num_seqs=16 \
  --rollout_vllm_max_num_batched_tokens=256
