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
# from model_lib.Attention / DecoderLayer. No model download, no vLLM, no RL --
# random weights, a few seconds per case.
#
# Usage (on the TPU VM, from the tunix repo root, deps present - e.g. inside
# the tunix_base_image container):
#   bash experimental/bench_splash_v5p.sh
#
# Everything is overridable via env vars, e.g. skip the xprof traces and the
# full-layer case for a quick attention-only read:
#   WITH_MODULE=1 WITH_LAYER=1 bash experimental/bench_splash_v5p.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

# Per-chip geometry of the e2e run being explained: 32 sequences over fsdp 4.
# The kernel bench is single-device, so this is exactly one chip's share.
NUM_SEQS="${NUM_SEQS:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
# Real-token range: 700-950 out of 2048 padded lands at ~20% dummy_ratio, the
# ratio the e2e packed run reported.
MIN_TOKENS="${MIN_TOKENS:-700}"
MAX_TOKENS="${MAX_TOKENS:-950}"
BUDGETS="${BUDGETS:-2048,4096,8192,16384}"
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
# The raw kernel always runs. These add context at the cost of runtime:
# the attention module (kernel + projections) and a full decoder layer.
WITH_MODULE="${WITH_MODULE:-0}"
WITH_LAYER="${WITH_LAYER:-0}"
LOG_DIR="${LOG_DIR:-/tmp/bench_splash_logs}"
RUN_TAG="${RUN_TAG:-splash_bench}"

mkdir -p "$LOG_DIR"
case "$TRACE_DEST" in gs://*|"") ;; *) mkdir -p "$TRACE_DEST" ;; esac
cd "$TUNIX_DIR"

extra_args=()
[ "${WITH_MODULE}" != "0" ] && extra_args+=(--with_module)
[ "${WITH_LAYER}" != "0" ] && extra_args+=(--with_layer)

log="$LOG_DIR/${RUN_TAG}.log"
echo "===== SPLASH BENCH seqs=${NUM_SEQS}x${SEQ_LEN} real=${MIN_TOKENS}-${MAX_TOKENS} "\
"budgets=${BUDGETS} iters=${ITERS} (log: $log) ====="

PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
python3 -X faulthandler -u experimental/bench_splash_packed.py \
  --num_seqs "$NUM_SEQS" \
  --seq_len "$SEQ_LEN" \
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
