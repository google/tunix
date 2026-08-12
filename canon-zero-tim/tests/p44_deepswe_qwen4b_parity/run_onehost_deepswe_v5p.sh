#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

STAGE="${1:?usage: run_onehost_deepswe_v5p.sh <rollout-only|backward-no-commit>}"
case "$STAGE" in
  rollout-only)
    ROLLOUT_ONLY=1
    NO_COMMIT=0
    ;;
  backward-no-commit)
    ROLLOUT_ONLY=0
    NO_COMMIT=1
    ;;
  *)
    echo "one-host runner admits rollout-only or backward-no-commit" >&2
    exit 2
    ;;
esac

PYTHON="${DEEPSWE_TRAIN_PYTHON:-/mnt/disks/tunix-data/venvs/train/bin/python}"
MODEL_PATH="${DEEPSWE_QWEN4B_MODEL_PATH:-/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
DATASET_CACHE="${DEEPSWE_DATASET_CACHE:-/mnt/disks/tunix-data/dataset_cache}"
R2EGYM_ROOT="${DEEPSWE_R2EGYM_ROOT:-/home/yuxuan/tunix/submodules/R2E-Gym}"
GOLD_WHITELIST="${DEEPSWE_ONEHOST_WHITELIST:-/home/yuxuan/code_rl_repro/google_dev/tunix/experimental/smoke_test_whitelist.jsonl}"
TASK_IMAGE="${DEEPSWE_ONEHOST_TASK_IMAGE:-namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d}"
RUN_ID="${CANON_RUN_ID:-onehost-${STAGE}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
ARTIFACT_DIR="${DEEPSWE_ONEHOST_ARTIFACT_DIR:-/mnt/disks/tunix-data/deepswe-onehost-evidence/${RUN_ID}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing DeepSWE training interpreter: $PYTHON" >&2
  exit 2
fi
for path in "$MODEL_PATH" "$DATASET_CACHE" "$R2EGYM_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "missing one-host prerequisite directory: $path" >&2
    exit 2
  fi
done
if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "Qwen3-4B snapshot is incomplete: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$GOLD_WHITELIST" ]]; then
  echo "missing reviewed DeepSWE whitelist: $GOLD_WHITELIST" >&2
  exit 2
fi
if [[ "$ARTIFACT_DIR" != /* ]] || [[ -e "$ARTIFACT_DIR" ]]; then
  echo "artifact directory must be a new absolute path: $ARTIFACT_DIR" >&2
  exit 2
fi

SOURCE_SHA="$(git rev-parse HEAD)"
SOURCE_BRANCH="${CANON_SOURCE_BRANCH:-$(git branch --show-current)}"
SOURCE_BRANCH="${SOURCE_BRANCH:-detached}"
SOURCE_DIRTY="$(git status --porcelain --untracked-files=no)"
if [[ -n "$SOURCE_DIRTY" ]] && \
   [[ "${DEEPSWE_ONEHOST_ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "one-host evidence requires a clean tracked worktree" >&2
  echo "set DEEPSWE_ONEHOST_ALLOW_DIRTY=1 only for development evidence" >&2
  exit 2
fi
R2EGYM_SHA="$(git -C "$R2EGYM_ROOT" rev-parse HEAD)"
if [[ "$R2EGYM_SHA" != "0d94c4eb9431cd195c55a7ea3abd54006c9a1735" ]]; then
  echo "R2E-Gym source changed: $R2EGYM_SHA" >&2
  exit 2
fi

export CANON_DEEPSWE_ONEHOST_SMOKE=1
export CANON_DEEPSWE_ONEHOST_STAGE="$STAGE"
export CANON_DEEPSWE_ONEHOST_ROLLOUT_ONLY="$ROLLOUT_ONLY"
export CANON_DEEPSWE_ONEHOST_NO_COMMIT="$NO_COMMIT"
export CANON_DEEPSWE_ONEHOST_DEBUG_DIR="$ARTIFACT_DIR"
export CANON_DEEPSWE_ONEHOST_REPORT="$ARTIFACT_DIR/backward_no_commit.json"
export CANON_DEEPSWE_ONEHOST_TASK_IMAGE="$TASK_IMAGE"
export CANON_EXPECT_COMMIT="$SOURCE_SHA"
export CANON_SOURCE_BRANCH="$SOURCE_BRANCH"
export CANON_RUN_ID="$RUN_ID"
export CANON_P34_DEEPSWE=0
export CANON_P39_64CHIP_PILOT=0
export CANON_P43_DEEPSWE_DEBUG=0
export CANON_P44_DEEPSWE_PARITY=0
export DATASET_CACHE
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
# TPU must remain the default backend, while vLLM's TPU weight loader needs one
# visible CPU device for its host-side staging mesh.
export JAX_PLATFORMS=tpu,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SKIP_JAX_PRECOMPILE=true
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH="$ROOT:$R2EGYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset DOCKER_HOST JAX_BACKEND_TARGET PATHWAYS_HEAD
unset CANON_ALIGNMENT_GATE CANON_ALIGNMENT_GATE_ONLY
unset CANON_ALIGNMENT_UPDATE_CANARY CANON_ALIGNMENT_TRAIN
unset CANON_P28_SEGMENTED_TRAIN CANON_P28_G6_UPDATE

echo "[DEEPSWE.ONEHOST.INVENTORY] source_sha=$SOURCE_SHA source_branch=$SOURCE_BRANCH tracked_dirty=$([[ -n "$SOURCE_DIRTY" ]] && echo 1 || echo 0) r2egym_sha=$R2EGYM_SHA stage=$STAGE artifact_dir=$ARTIFACT_DIR"
"$PYTHON" -c '
import jax
devices = jax.devices()
assert len(devices) == 4, devices
assert all(device.platform == "tpu" for device in devices), devices
print(
    "[DEEPSWE.ONEHOST.DEVICES] PASS count=4 kinds="
    + ",".join(str(device.device_kind) for device in devices),
    flush=True,
)
'
"$PYTHON" -c '
import docker
import r2egym
client = docker.from_env()
assert client.ping() is True
print("[DEEPSWE.ONEHOST.R2E] PASS docker=1 import=1", flush=True)
'

"$PYTHON" examples/deepswe/train_deepswe_nb.py \
  --model_version Qwen3-4B-Instruct-2507 \
  --model_absolute_path "$MODEL_PATH" \
  --gold_whitelist "$GOLD_WHITELIST" \
  --batch_size 1 \
  --mini_batch_size 1 \
  --train_micro_batch_size 1 \
  --compute_logps_micro_batch_size 1 \
  --rollout_micro_batch_size 1 \
  --num_generations 2 \
  --num_iterations 1 \
  --max_prompt_length 3584 \
  --max_response_length 512 \
  --max_turns 2 \
  --max_steps 1 \
  --num_epochs 1 \
  --eval_every_n_steps 10 \
  --max_concurrency 2 \
  --temperature 0.7 \
  --rollout_engine vllm \
  --vllm_utilization 0.3 \
  --rollout_vllm_max_num_seqs 2 \
  --max_num_batched_tokens 512 \
  --rollout_mesh_dp 1 \
  --rollout_mesh_tp 4 \
  --train_mesh_dp 1 \
  --train_mesh_tp 4 \
  --ckpt_dir none \
  --dtype bfloat16 \
  --param_dtype bfloat16 \
  --use_rollout_logps \
  --logging_level INFO

if [[ ! -s "$ARTIFACT_DIR/run_manifest.json" ]] || \
   [[ ! -s "$ARTIFACT_DIR/batch-000000.trajectories.jsonl.gz" ]] || \
   [[ ! -s "$ARTIFACT_DIR/batch_metrics.jsonl" ]]; then
  echo "one-host trajectory evidence is incomplete: $ARTIFACT_DIR" >&2
  exit 1
fi

if [[ "$STAGE" == "rollout-only" ]]; then
  echo "DEEPSWE_ONEHOST_ROLLOUT_PASS model=qwen3-4b-instruct-2507 devices=4 trajectories=2"
  exit 0
fi

VERDICT="$("$PYTHON" -c '
import json
import os
path = os.environ["CANON_DEEPSWE_ONEHOST_REPORT"]
with open(path, encoding="utf-8") as source:
  report = json.load(source)
assert report["commits"] == 0, report
assert report["gradient_finite"] is True, report
assert not report["model_changed_paths"], report
assert not report["optimizer_changed_paths"], report
assert not report["accumulator_changed_paths"], report
assert not report["reference_changed_paths"], report
print(report["verdict"])
')"
if [[ "$VERDICT" == "PASS" ]]; then
  echo "DEEPSWE_ONEHOST_BACKWARD_NO_COMMIT_PASS model=qwen3-4b-instruct-2507 devices=4"
  exit 0
fi
if [[ "$VERDICT" == "INCONCLUSIVE_NO_SIGNAL" ]]; then
  echo "DEEPSWE_ONEHOST_BACKWARD_INCONCLUSIVE_NO_SIGNAL model=qwen3-4b-instruct-2507 devices=4" >&2
  exit 3
fi
echo "one-host backward report failed with verdict=$VERDICT" >&2
exit 1
