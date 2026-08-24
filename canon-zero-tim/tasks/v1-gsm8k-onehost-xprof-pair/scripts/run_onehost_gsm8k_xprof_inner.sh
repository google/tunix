#!/usr/bin/env bash
# Container-side entrypoint shared by the matched GSM8K XProf arms.
set -euo pipefail

: "${V1_GSM8K_XPROF_REPO:?V1_GSM8K_XPROF_REPO unset}"
: "${V1_GSM8K_XPROF_ARM:?V1_GSM8K_XPROF_ARM unset}"
: "${V1_GSM8K_XPROF_XLA_FLAGS:?V1_GSM8K_XPROF_XLA_FLAGS unset}"

case "$V1_GSM8K_XPROF_ARM" in
  native)
    if [ "${CANON_GSM8K_VANILLA:-}" != "1" ] || \
       [ -n "${CANON_P32_WORKLOAD:-}" ] || \
       [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" != "0" ] || \
       [ "${CANON_P28_G6_UPDATE:-0}" != "0" ]; then
      echo "[V1.GSM8K.XPROF] native treatment isolation failed" >&2
      exit 2
    fi
    ;;
  zero-hp)
    # shellcheck disable=SC1091
    source "$V1_GSM8K_XPROF_REPO/canon-zero-tim/cluster/profiles/_canonical_engine.env"
    # shellcheck disable=SC1091
    source "$V1_GSM8K_XPROF_REPO/canon-zero-tim/cluster/profiles/qwen3-1p7b-dp4-tp1-gsm8k-v1-hp.env"
    if [ -n "${CANON_GSM8K_VANILLA:-}" ]; then
      echo "[V1.GSM8K.XPROF] zero-hp inherited the vanilla selector" >&2
      exit 2
    fi
    ;;
  *)
    echo "[V1.GSM8K.XPROF] invalid arm: $V1_GSM8K_XPROF_ARM" >&2
    exit 2
    ;;
esac

export XLA_FLAGS="$V1_GSM8K_XPROF_XLA_FLAGS"

python3 - <<'PY'
import os
import numpy as np
import jax
from jax.experimental import mesh_utils
from tunix.rl import gsm8k_xprof

devices = tuple(jax.devices())
if len(devices) != 4 or jax.default_backend() != "tpu":
  raise SystemExit(
      "V1 GSM8K one-host XProf requires four direct-attached TPU devices"
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
      f"V1 GSM8K training mesh mismatch: expected={expected_ids} "
      f"actual={actual_ids}"
  )
selected = gsm8k_xprof.arm()
print(
    "[V1.GSM8K.XPROF] PREFLIGHT_PASS "
    f"arm={selected} topology=DP4xTP1 mesh_ids={actual_ids} "
    "prompts=8 generations=8 trajectories=64 groups=16 "
    "capture=update:1->2",
    flush=True,
)
PY

set +e
python3 -u examples/math_gsm8k/qwen3_grpo_demo.py \
  --mesh_dp=4 --mesh_tp=1 --batch_size=8 --mini_batch_size=8 \
  --train_micro_batch_size=8 --compute_logps_micro_batch_size=8 \
  --train_trajectory_micro_batch_size=4 \
  --max_steps=3 --num_generations=8 --max_prompt_length=1024 \
  --max_response_length=256 --max_concurrency=1 \
  --rollout_vllm_hbm_utilization=0.20 \
  --rollout_vllm_max_num_seqs=16 \
  --rollout_vllm_max_num_batched_tokens=256 \
  --wandb_project=v1-gsm8k-onehost-xprof \
  --wandb_run_name="v1-gsm8k-xprof-${V1_GSM8K_XPROF_ARM}-${V1_GSM8K_XPROF_LABEL}"
demo_rc=$?
set -e

# Docker runs as root so XProf, Perfetto, W&B and trajectory outputs otherwise
# become unreadable to the host-side classifiers.  Normalize only this fresh
# run's train tree; never mutate a shared parent or an unresolved path.
train_root="$(dirname "${GSM8K_LOG_DIR:?GSM8K_LOG_DIR unset}")"
case "$train_root" in
  /mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_*/train) ;;
  *) echo "[V1.GSM8K.XPROF] refusing unsafe artifact chmod: $train_root" >&2; exit 2 ;;
esac
chmod -R a+rX "$train_root"
exit "$demo_rc"
