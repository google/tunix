#!/usr/bin/env bash
# Real direct-attached v5p L1/L2 gate for P46 reward-only evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${DEEPSWE_TRAIN_PYTHON:-/mnt/disks/tunix-data/venvs/train/bin/python}"
MODEL_PATH="${DEEPSWE_QWEN4B_MODEL_PATH:-/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
DATASET_CACHE="${DEEPSWE_DATASET_CACHE:-/mnt/disks/tunix-data/dataset_cache}"
R2EGYM_ROOT="${DEEPSWE_R2EGYM_ROOT:-/home/yuxuan/tunix/submodules/R2E-Gym}"
GOLD_JSONL="${DEEPSWE_REWARD_ONLY_WHITELIST:-$ROOT/canon-zero-tim/clean_data/final_filter_result/task_report_good_qwen3_128_retry_20260713_090141.jsonl}"
RUN_ID="${CANON_RUN_ID:-reward-only-onehost-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
ARTIFACT_DIR="${DEEPSWE_REWARD_ONLY_ARTIFACT_DIR:-/mnt/disks/tunix-data/deepswe-reward-only-evidence/$RUN_ID}"

for path in "$MODEL_PATH" "$DATASET_CACHE" "$R2EGYM_ROOT"; do
  [[ -d "$path" ]] || { echo "missing one-host prerequisite: $path" >&2; exit 2; }
done
[[ -x "$PYTHON" ]] || { echo "missing one-host interpreter: $PYTHON" >&2; exit 2; }
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || {
  echo "Qwen3-4B snapshot is incomplete: $MODEL_PATH" >&2; exit 2;
}
[[ -f "$GOLD_JSONL" ]] || { echo "missing clean whitelist: $GOLD_JSONL" >&2; exit 2; }
[[ "$ARTIFACT_DIR" = /* && ! -e "$ARTIFACT_DIR" ]] || {
  echo "artifact directory must be a new absolute path: $ARTIFACT_DIR" >&2; exit 2;
}

SOURCE_SHA="$(git rev-parse HEAD)"
SOURCE_DIRTY="$(git status --porcelain --untracked-files=no)"
if [[ -n "$SOURCE_DIRTY" && "${DEEPSWE_ONEHOST_ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "one-host evidence requires a clean tracked worktree" >&2
  echo "set DEEPSWE_ONEHOST_ALLOW_DIRTY=1 only for development evidence" >&2
  exit 2
fi
R2EGYM_SHA="$(git -C "$R2EGYM_ROOT" rev-parse HEAD)"
[[ "$R2EGYM_SHA" = "0d94c4eb9431cd195c55a7ea3abd54006c9a1735" ]] || {
  echo "R2E-Gym source changed: $R2EGYM_SHA" >&2; exit 2;
}
WHITELIST_SHA="$(sha256sum "$GOLD_JSONL" | cut -d' ' -f1)"
[[ "$WHITELIST_SHA" = "2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7" ]] || {
  echo "clean whitelist digest changed: $WHITELIST_SHA" >&2; exit 2;
}
PYTHON_SHA="$(sha256sum "$PYTHON" | cut -d' ' -f1)"

mkdir -p "$ARTIFACT_DIR"
export CANON_EXPECT_COMMIT="$SOURCE_SHA"
export CANON_CLIENT_IMAGE="local-v5p@sha256:$PYTHON_SHA"
export CANON_P46_TOPOLOGY=4
export CANON_P46_EVALUATION_MODE=reward_only
export CANON_P46_RESUME_TAG="${CANON_P46_RESUME_TAG:-onehost-reward-only}"
export CANON_P46_ONEHOST_PROBE=1
export CANON_P46_PARITY_CANARY=0
export CANON_P46_LOGICAL_SHARD_INDEX=0
export CANON_P46_PHYSICAL_SHARD_INDEX=0
export MODEL_VERSION=Qwen/Qwen3-4B-Instruct-2507
export MODEL_ABSOLUTE_PATH="$MODEL_PATH"
export GOLD_JSONL
export GOLD_JSONL_SHA256="$WHITELIST_SHA"
export OUTPUT_DIR="$ARTIFACT_DIR/eval"
export DATASET_CACHE
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export JAX_PLATFORMS=tpu,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SKIP_JAX_PRECOMPILE=true
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH="$ROOT:$R2EGYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset DOCKER_HOST JAX_BACKEND_TARGET PATHWAYS_HEAD

echo "[P46.REWARD_ONLY.ONEHOST.INVENTORY] source=$SOURCE_SHA tracked_dirty=$([[ -n "$SOURCE_DIRTY" ]] && echo 1 || echo 0) r2egym=$R2EGYM_SHA model=$MODEL_PATH artifact_dir=$ARTIFACT_DIR"
"$PYTHON" -c '
import jax
devices = jax.devices()
assert len(devices) == 4, devices
assert all(device.platform == "tpu" for device in devices), devices
print("[P46.REWARD_ONLY.ONEHOST.DEVICES] PASS count=4", flush=True)
'
"$PYTHON" examples/deepswe/probe_reward_only_v5p.py \
  --output "$ARTIFACT_DIR/report.json" \
  --repeats "${DEEPSWE_REWARD_ONLY_REPEATS:-3}" \
  2>&1 | tee "$ARTIFACT_DIR/run.log"

[[ -s "$ARTIFACT_DIR/report.json" ]] || {
  echo "one-host reward-only report is missing" >&2; exit 1;
}
echo "P46_REWARD_ONLY_ONEHOST_ARTIFACT path=$ARTIFACT_DIR/report.json sha256=$(sha256sum "$ARTIFACT_DIR/report.json" | cut -d' ' -f1)"
