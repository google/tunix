#!/bin/bash
# HBM / performance verification for yuxzhang/fix_accum_fp32:
# 4 arms comparing depth-1 vs depth-4 and float32 vs bfloat16 accumulator_dtype.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/tianshu/models/Gemma4-E2B-it}"
MESH_FSDP="${MESH_FSDP:-2}"
MESH_TP="${MESH_TP:-2}"
MAX_STEPS="${MAX_STEPS:-5}"
XPROF_DIR="${XPROF_DIR:-/mnt/workspace/mem_repro_xprof}"

mkdir -p "$XPROF_DIR"
cd "$TUNIX_DIR"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
  echo "  hf download google/gemma-4-e2b-it --local-dir $MODEL_PATH"
  exit 1
fi

run_arm() {
  local name="$1" grad_accum="$2" steps="$3" dtype="$4"
  local out="$XPROF_DIR/$name"
  echo
  echo "===== [$name] grad_accum=$grad_accum steps=$steps accum_dtype=$dtype -> $out ====="
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GRAD_ACCUM="$grad_accum" \
  GRAD_ACCUM_STEPS="$steps" \
  GRAD_ACCUM_DTYPE="$dtype" \
  PROFILE_XPROF="$out" \
  PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
  PYTHONUNBUFFERED=1 \
  python3 -X faulthandler -u experimental/compile_repro_sft.py \
    --model_path "$MODEL_PATH" \
    --mesh_fsdp "$MESH_FSDP" --mesh_tp "$MESH_TP" \
    --max_steps "$MAX_STEPS" \
    --grad_accum "$grad_accum" \
    --grad_accum_steps "$steps" \
    --gradient_accumulator_dtype "$dtype" \
    2>&1 | tee "$XPROF_DIR/${name}.log"
  echo "===== [$name] done (exit=${PIPESTATUS[0]}) ====="
}

#         name             grad_accum  steps  dtype
run_arm   optax_d1         optax       1      float32
run_arm   stream_d1        stream      1      float32
run_arm   stream_d4_fp32   stream      4      float32
run_arm   stream_d4_bf16   stream      4      bfloat16

echo
echo "################ SUMMARY ################"
echo "xprof traces:"
for name in optax_d1 stream_d1 stream_d4_fp32 stream_d4_bf16; do
  echo "  $name -> $XPROF_DIR/$name"
done
echo "Load in memory_viewer (HBM peak) and trace_viewer (update-op fusion)."
