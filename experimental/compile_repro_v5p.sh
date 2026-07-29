#!/bin/bash
# Compile-time repro: pure SFT train_step, optax vs stream grad accum, on a
# single-host TPU VM (e.g. v5p, 4 chips). No vLLM / no RL / no profiler -
# each variant is one fresh-process run of experimental/compile_repro_sft.py;
# the [[COMPILE_REPRO]] train_wall line (plus the trainer's first-step
# elapsed and JAX_LOG_COMPILES output in the per-variant logs) is the signal.
#
# Usage (on the TPU VM, from the tunix repo root, deps present - e.g. inside
# the tunix_base_image container):
#   bash experimental/compile_repro_v5p.sh
#
# Everything is overridable via env vars. GQA fallback if the default (2,2)
# mesh fails to shard E2B's num_kv_heads=1 over tp=2:
#   MESH_FSDP=4 MESH_TP=1 bash experimental/compile_repro_v5p.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/google/gemma-4-e2b-it}"
MESH_FSDP="${MESH_FSDP:-2}"
MESH_TP="${MESH_TP:-2}"
MAX_STEPS="${MAX_STEPS:-3}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"   # empty => None (notebook parity)
VARIANTS="${VARIANTS:-optax stream}"
LOG_DIR="${LOG_DIR:-/tmp/compile_repro_logs}"

mkdir -p "$LOG_DIR"
cd "$TUNIX_DIR"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
  echo "Provision once with:"
  echo "  hf download google/gemma-4-e2b-it --local-dir $MODEL_PATH"
  exit 1
fi

# ---------------------------------------------------------------------------
# Run both variants sequentially; the ONLY difference is --grad_accum.
# ---------------------------------------------------------------------------
for v in $VARIANTS; do
  log="$LOG_DIR/compile_repro_${v}.log"
  echo
  echo "===== [$v] compile repro: mesh ${MESH_FSDP}x${MESH_TP} max_steps=$MAX_STEPS (log: $log) ====="
  extra_args=()
  if [ -n "$GRAD_ACCUM_STEPS" ]; then
    extra_args+=(--grad_accum_steps "$GRAD_ACCUM_STEPS")
  fi
  JAX_LOG_COMPILES=1 \
  PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
  PYTHONUNBUFFERED=1 \
  python3 -X faulthandler -u experimental/compile_repro_sft.py \
    --model_path "$MODEL_PATH" \
    --mesh_fsdp "$MESH_FSDP" --mesh_tp "$MESH_TP" \
    --max_steps "$MAX_STEPS" \
    --grad_accum "$v" \
    "${extra_args[@]}" \
    2>&1 | tee "$log"
  echo "===== [$v] done (exit=${PIPESTATUS[0]}) ====="
done

# ---------------------------------------------------------------------------
# Summary: [[COMPILE_REPRO]] wall clock per variant + compile-related lines.
# ---------------------------------------------------------------------------
echo
echo "################ SUMMARY ################"
for v in $VARIANTS; do
  log="$LOG_DIR/compile_repro_${v}.log"
  echo "--- [$v] ---"
  [ -f "$log" ] || { echo "  (no log)"; continue; }
  grep -h "\[\[COMPILE_REPRO\]\]" "$log" || echo "  (no [[COMPILE_REPRO]] line - run failed? see $log)"
  grep -in "Compiled train_step cache size" "$log" | tail -2
  # Best-effort JAX_LOG_COMPILES evidence (grep is loose; exact format TBD
  # from the first real log).
  grep -inE "took [0-9.]+ ?s|Compiling|Finished tracing|Train loop finished" "$log" | tail -6
done

optax_wall="$(sed -n 's/.*\[\[COMPILE_REPRO\]\].*train_wall_s=\([0-9.]*\).*/\1/p' "$LOG_DIR/compile_repro_optax.log" 2>/dev/null | tail -1)"
stream_wall="$(sed -n 's/.*\[\[COMPILE_REPRO\]\].*train_wall_s=\([0-9.]*\).*/\1/p' "$LOG_DIR/compile_repro_stream.log" 2>/dev/null | tail -1)"
if [ -n "$optax_wall" ] && [ -n "$stream_wall" ]; then
  awk -v o="$optax_wall" -v s="$stream_wall" 'BEGIN {
    printf "DELTA: stream=%.1fs optax=%.1fs  stream-optax=%.1fs  stream/optax=%.2fx\n", s, o, s - o, s / o
  }'
else
  echo "DELTA: unavailable (need both variants' [[COMPILE_REPRO]] lines)"
fi
echo "logs: $LOG_DIR/"
