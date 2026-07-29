#!/bin/bash
# optax.MultiSteps native-accumulation parity: 4 arms mirroring
# mem_repro_fix_accum.sh, but with optax's own gradient accumulation.
# Compare the [[MEM]] peaks to the custom-accumulator arms of the same names.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/tianshu/models/Gemma4-E2B-it}"
MESH_FSDP="${MESH_FSDP:-2}"
MESH_TP="${MESH_TP:-2}"
OUT_DIR="${OUT_DIR:-/mnt/workspace/mem_repro_xprof/optax_4arm}"

mkdir -p "$OUT_DIR"
cd "$TUNIX_DIR"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
  echo "  hf download google/gemma-4-e2b-it --local-dir $MODEL_PATH"
  exit 1
fi

run_arm() {
  local name="$1" steps="$2"
  echo
  echo "===== [$name] steps=$steps -> $OUT_DIR/${name}.log ====="
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
  PYTHONUNBUFFERED=1 \
  python3 -X faulthandler -u experimental/optax_multistep_bench.py \
    --arm_name "$name" \
    --grad_accum_steps "$steps" \
    --model gemma4 \
    --model_path "$MODEL_PATH" \
    --mesh_fsdp "$MESH_FSDP" --mesh_tp "$MESH_TP" \
    2>&1 | tee "$OUT_DIR/${name}.log"
  echo "===== [$name] done (exit=${PIPESTATUS[0]}) ====="
}

#         name                   steps
run_arm   optax_d1               1
run_arm   optax_d4_fp32_accum    4
run_arm   optax_d4_bf16_accum    4
run_arm   optax_d4_fp32_moments  4

echo
echo "################ SUMMARY (compare to mem_repro_fix_accum arms) ################"
grep -h "\[\[MEM\]\]" "$OUT_DIR"/*.log | grep "device=0" || true
echo
echo "Parity check: optax_d4_fp32_accum peak should ~= custom d4_fp32_accum (15.46);"
echo "optax_d1 should be HIGHER than custom d1_default (MultiSteps allocates an"
echo "accumulator even at k=1, while our fast path skips it)."
