#!/usr/bin/env bash
# Container-side entrypoint shared by the matched GSM8K XProf arms.
set -euo pipefail
trap '' HUP

: "${V1_GSM8K_XPROF_REPO:?V1_GSM8K_XPROF_REPO unset}"
: "${V1_GSM8K_XPROF_ARM:?V1_GSM8K_XPROF_ARM unset}"
: "${V1_GSM8K_XPROF_XLA_FLAGS:?V1_GSM8K_XPROF_XLA_FLAGS unset}"

# The registered carrier geometries on the same four chips.  Absent means
# the byte-identical DP4xTP1 default; the launcher forwards dp2-tp2
# explicitly.  Every derived value (mesh axes, trajectory microbatch,
# rollout seq budget, gradient groups) follows the registered workload
# arithmetic: 64 global trajectories / dp ranks.
geometry="${V1_GSM8K_XPROF_GEOMETRY:-dp4-tp1}"
case "$geometry" in
  dp4-tp1)
    mesh_dp=4; mesh_tp=1; trajectory_micro=4; vllm_max_seqs=16
    zero_profile=qwen3-1p7b-dp4-tp1-gsm8k-v1-hp.env
    ;;
  dp2-tp2)
    mesh_dp=2; mesh_tp=2; trajectory_micro=2; vllm_max_seqs=32
    zero_profile=qwen3-1p7b-dp2-tp2-gsm8k-v1-hp.env
    ;;
  *)
    echo "[V1.GSM8K.XPROF] unsupported V1_GSM8K_XPROF_GEOMETRY: $geometry" >&2
    exit 2
    ;;
esac
# The Python contract reads the CANON_-prefixed twin; a disagreement between
# the two means a hand-rolled container environment, not a launcher run.
if [ "${CANON_V1_GSM8K_XPROF_GEOMETRY:-dp4-tp1}" != "$geometry" ]; then
  echo "[V1.GSM8K.XPROF] geometry disagreement: V1_GSM8K_XPROF_GEOMETRY=$geometry CANON_V1_GSM8K_XPROF_GEOMETRY=${CANON_V1_GSM8K_XPROF_GEOMETRY:-unset}" >&2
  exit 2
fi

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
    source "$V1_GSM8K_XPROF_REPO/canon-zero-tim/cluster/profiles/$zero_profile"
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

# The update horizon comes from the P33 stage, never from a literal.  The
# zero-hp arm additionally receives CANON_P33_RUN_STAGE, and the recipe
# derives its own budget from tunix.rl.dp_workloads, so a disagreement
# between the two derivations is a fatal --max_steps mismatch rather than
# a silently shortened run.
run_stage="${V1_GSM8K_XPROF_RUN_STAGE:-three-update}"
case "$run_stage" in
  three-update) max_steps=3 ;;
  six-update) max_steps=6 ;;
  *)
    echo "[V1.GSM8K.XPROF] unsupported run stage: $run_stage" >&2
    exit 2
    ;;
esac
if [ "$V1_GSM8K_XPROF_ARM" = zero-hp ] && \
   [ "${CANON_P33_RUN_STAGE:-}" != "$run_stage" ]; then
  echo "[V1.GSM8K.XPROF] stage disagreement: carrier=$run_stage container=${CANON_P33_RUN_STAGE:-unset}" >&2
  exit 2
fi

export XLA_FLAGS="$V1_GSM8K_XPROF_XLA_FLAGS"

python3 - <<'PY'
import os
import numpy as np
import jax
from jax.experimental import mesh_utils
from tunix.rl import gsm8k_xprof

geometry = gsm8k_xprof.geometry()
mesh_shape = {"dp4-tp1": (4, 1), "dp2-tp2": (2, 2)}[geometry]
groups = gsm8k_xprof.geometry_groups()
topology = f"DP{mesh_shape[0]}xTP{mesh_shape[1]}"
devices = tuple(jax.devices())
if len(devices) != 4 or jax.default_backend() != "tpu":
  raise SystemExit(
      "V1 GSM8K one-host XProf requires four direct-attached TPU devices"
  )
arranged = mesh_utils.create_device_mesh(
    mesh_shape, devices, allow_split_physical_axes=True
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
    f"arm={selected} topology={topology} mesh_ids={actual_ids} "
    f"prompts=8 generations=8 trajectories=64 groups={groups} "
    "capture=update:2->3",
    flush=True,
)
PY

set +e
python3 -u examples/math_gsm8k/qwen3_grpo_demo.py \
  --mesh_dp="$mesh_dp" --mesh_tp="$mesh_tp" --batch_size=8 --mini_batch_size=8 \
  --train_micro_batch_size=8 --compute_logps_micro_batch_size=8 \
  --train_trajectory_micro_batch_size="$trajectory_micro" \
  --max_steps="$max_steps" --num_generations=8 --max_prompt_length=1024 \
  --max_response_length=256 --max_concurrency=1 \
  --rollout_vllm_hbm_utilization=0.20 \
  --rollout_vllm_max_num_seqs="$vllm_max_seqs" \
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
