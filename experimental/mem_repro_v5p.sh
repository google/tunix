#!/bin/bash
# HBM / per-step-op regression repro: run the pure-SFT Gemma4-E2B harness under
# xprof for three arms, so the memory profile (HBM peak) and the device op view
# (optimizer-update fusion: multiply-add vs add-convert) can be compared.
#
# The three arms isolate the fp32 optimizer-state promotion:
#   optax             -> optax path (no promote, no accumulator)   [~12GB baseline]
#   stream            -> stream path (promote ON, accumulator)     [~16GB "now"]
#   stream_nopromote  -> stream path, PROMOTE_FP32=0               [isolation arm]
# stream_nopromote ~= optax  => the promote is the cause.
#
# Usage (on the TPU VM, inside the tunix_base_image container, repo root):
#   bash experimental/mem_repro_v5p.sh
#
# Each arm writes an xprof trace to $XPROF_DIR/<arm>/ ; load them in the
# TensorBoard/xprof memory_viewer (HBM peak) and trace_viewer (op fusions).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/google/gemma-4-e2b-it}"
MESH_FSDP="${MESH_FSDP:-2}"
MESH_TP="${MESH_TP:-2}"
MAX_STEPS="${MAX_STEPS:-5}"           # a few steps so the peak stabilizes
XPROF_DIR="${XPROF_DIR:-/mnt/workspace/mem_repro_xprof}"

mkdir -p "$XPROF_DIR"
cd "$TUNIX_DIR"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
  echo "  hf download google/gemma-4-e2b-it --local-dir $MODEL_PATH"
  exit 1
fi

# arm = "name GRAD_ACCUM PROMOTE_FP32"
run_arm() {
  local name="$1" grad_accum="$2" promote="$3"
  local out="$XPROF_DIR/$name"
  echo
  echo "===== [$name] grad_accum=$grad_accum promote=$promote -> $out ====="
  # PREALLOCATE=false so the allocator sizes on demand and the xprof memory
  # profile reflects real peak HBM instead of the preallocated pool.
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PROMOTE_FP32="$promote" \
  PROFILE_XPROF="$out" \
  PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
  PYTHONUNBUFFERED=1 \
  python3 -X faulthandler -u experimental/compile_repro_sft.py \
    --model_path "$MODEL_PATH" \
    --mesh_fsdp "$MESH_FSDP" --mesh_tp "$MESH_TP" \
    --max_steps "$MAX_STEPS" \
    --grad_accum "$grad_accum" \
    2>&1 | tee "$XPROF_DIR/${name}.log"
  echo "===== [$name] done (exit=${PIPESTATUS[0]}) ====="
}

run_arm optax            optax  1
run_arm stream           stream 1
run_arm stream_nopromote stream 0

echo
echo "################ SUMMARY ################"
echo "xprof traces:"
for name in optax stream stream_nopromote; do
  echo "  $name -> $XPROF_DIR/$name"
done
echo "Load in memory_viewer (HBM peak) and trace_viewer (update-op fusion)."
echo "Expect: stream peak >> optax; if stream_nopromote ~= optax, the fp32"
echo "optimizer-state promote is the cause of both the HBM and the op change."
