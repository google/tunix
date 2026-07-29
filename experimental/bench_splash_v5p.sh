#!/bin/bash
# Sequence-packing cost model microbench on a single-host TPU VM (e.g. v5p).
#
# Answers, with numbers, why packing left the e2e gsm8k train time flat while
# flash-attention backward went 6ms -> 16ms: splash schedules blocks from a
# STATIC mask, so a packed row costs row_len^2/2 no matter how many sequences
# it holds (segment_ids only zeroes blocks that were already computed). That
# implies a BUDGET LAW -- at a fixed token total, attention grows LINEARLY with
# the budget while the MLP saving is budget-independent -- which this bench
# measures directly by running the same data at 8192 and 4096.
#
# Runs the production code path: data from rl_utils.pack_sequences, attention
# from model_lib.Attention. No model download, no vLLM, no RL -- random
# weights, a few seconds per case.
#
# Usage (on the TPU VM, from the tunix repo root, deps present - e.g. inside
# the tunix_base_image container):
#   bash experimental/bench_splash_v5p.sh
#
# Everything is overridable via env vars, e.g. a quick kernel-only read:
#   SKIP_MODULE=1 TRACE_DEST= bash experimental/bench_splash_v5p.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

# GLOBAL geometry of the e2e run being explained: 32 sequences over fsdp 4,
# which is what the production module executes.
NUM_SEQS="${NUM_SEQS:-32}"
# Production mesh: gsm8k trains on (fsdp 4, tp 1).
MESH_FSDP="${MESH_FSDP:-4}"
MESH_TP="${MESH_TP:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
# Real-token range: 700-950 out of 2048 padded lands at ~20% dummy_ratio, the
# ratio the e2e packed run reported.
# Uniform real tokens per sequence: the arms then hold the SAME sequences and
# therefore the same effective attention work, so the only variable is the row
# geometry. Set SEQ_TOKENS=0 to sample MIN_TOKENS..MAX_TOKENS (ragged) instead.
SEQ_TOKENS="${SEQ_TOKENS:-1024}"
MIN_TOKENS="${MIN_TOKENS:-700}"
MAX_TOKENS="${MAX_TOKENS:-950}"
BUDGETS="${BUDGETS:-4096,8192}"
# Segment counts swept at a FIXED row shape (SWEEP_BUDGET). This is the
# decisive test: same shape, same tokens, only the segment structure changes.
SEGMENT_SWEEP="${SEGMENT_SWEEP:-1,2,4,8,16,32}"
SWEEP_BUDGET="${SWEEP_BUDGET:-8192}"
# qwen3_1p7b = the gsm8k run this explains; qwen3_8b = the FrozenLake geometry.
MODEL_CONFIG="${MODEL_CONFIG:-qwen3_1p7b}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-3}"
TRACE_DEST="${TRACE_DEST:-gs://yuxzhang-tunix-models/xprof/splash_bench}"
TRACE_ITERS="${TRACE_ITERS:-3}"
# The production attention module (kernel + projections) and the per-chip raw
# kernel both run by default; either can be skipped to halve the runtime.
SKIP_MODULE="${SKIP_MODULE:-0}"
SKIP_KERNEL="${SKIP_KERNEL:-0}"
LOG_DIR="${LOG_DIR:-/tmp/bench_splash_logs}"
RUN_TAG="${RUN_TAG:-splash_bench}"

mkdir -p "$LOG_DIR"
case "$TRACE_DEST" in gs://*|"") ;; *) mkdir -p "$TRACE_DEST" ;; esac
cd "$TUNIX_DIR"

extra_args=()
[ "${SKIP_MODULE}" != "0" ] && extra_args+=(--skip_module)
[ "${SKIP_KERNEL}" != "0" ] && extra_args+=(--skip_kernel)

log="$LOG_DIR/${RUN_TAG}.log"
echo "===== SPLASH BENCH seqs=${NUM_SEQS}x${SEQ_LEN} real=${MIN_TOKENS}-${MAX_TOKENS} "\
"budgets=${BUDGETS} iters=${ITERS} (log: $log) ====="

PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
python3 -X faulthandler -u experimental/bench_splash_packed.py \
  --num_seqs "$NUM_SEQS" \
  --seq_len "$SEQ_LEN" \
  --mesh_fsdp "$MESH_FSDP" --mesh_tp "$MESH_TP" \
  --seq_tokens "$SEQ_TOKENS" \
  --min_tokens "$MIN_TOKENS" \
  --max_tokens "$MAX_TOKENS" \
  --budgets "$BUDGETS" \
  --segment_sweep "$SEGMENT_SWEEP" \
  --sweep_budget "$SWEEP_BUDGET" \
  --model_config "$MODEL_CONFIG" \
  --iters "$ITERS" \
  --warmup "$WARMUP" \
  --trace_dest "$TRACE_DEST" \
  --trace_iters "$TRACE_ITERS" \
  "${extra_args[@]}" \
  2>&1 | tee "$log"

rc=${PIPESTATUS[0]}
echo "===== done (exit=$rc) ====="

echo
echo "################ SUMMARY ################"
if [ -f "$log" ]; then
  sed -n '/^VERDICTS/,$p' "$log"
  [ -n "${TRACE_DEST}" ] && \
    echo "--- xprof traces: $TRACE_DEST/splash_bench_*  (match the kernel names"\
"against the e2e trace: packed = *_segmented_{fwd,dq,dkv}) ---"
fi
exit "$rc"
