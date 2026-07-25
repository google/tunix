#!/bin/bash
# Agentic FrozenLake GRPO on a single-host TPU VM (v5p-4), driving
# examples/frozenlake/train_frozenlake.py -- the Gemma4-E2B recipe whose
# docstring targets exactly this host class ("Designed for v5p-4 / v6e-8") and
# which runs the rollout as a vLLM SERVER rather than an in-process sampler,
# "avoiding the trace-context issues of running the in-process sampler under
# REMAT" -- that is what makes its log-probs trustworthy. The separation is
# process-level, not hardware: both roles are colocated on the same 4 chips,
# time-shared, under two logical mesh shapes (rollout (dp 2, tp 2), trainer
# (fsdp 4, tp 1)). Colocation is why ROLLOUT_HBM bounds TOTAL HBM, trainer
# weights included.
#
# The script's own hyperparameters are left alone -- they were tuned for this
# host. This wrapper only supplies what the script cannot know: the pinned
# JAX/vLLM versions, a reachable data directory, and the packing switch.
#
# THE TWO RUNS (pack vs unpack; the script packs only when told to, so the
# plain run IS the unpacked baseline):
#   # unpacked baseline / convergence reference
#   RUN_TAG=fl_unpack bash experimental/train_frozenlake_v5p_1host.sh
#   # packed
#   MAX_TOKEN_PER_TPU=8192 RUN_TAG=fl_pack \
#     bash experimental/train_frozenlake_v5p_1host.sh
#
# Budget 8192 = 2 maximal sequences (max_prompt 2048 + max_response 2048) per
# row. Note the measured trade-off (tracing_logs/splash_microbench_RUN1_results
# .log): splash schedules a row's full causal area whatever it holds, so
# attention cost tracks rows*len^2 -- 4096 (one maximal sequence, the unpacked
# row length) minimises it, and each doubling above that doubles the attention
# work at the same token count.
#
# Inspect the command without launching (no TPU needed):
#   DRY_RUN=1 bash experimental/train_frozenlake_v5p_1host.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

# ---- pinned environment ---------------------------------------------------
# The recipe is only known to converge on these versions; a mismatch changes
# the vLLM sampler's log-probs, which GRPO consumes as old_per_token_logps.
JAX_VERSION="${JAX_VERSION:-0.10.1}"
VLLM_VERSION="${VLLM_VERSION:-0.25.0}"
SKIP_PIP="${SKIP_PIP:-0}"          # 1 once the versions are already installed

# ---- knobs ----------------------------------------------------------------
ENGINE="${ROLLOUT_ENGINE:-vllm}"           # vllm | vanilla
ROLLOUT_HBM="${ROLLOUT_HBM:-0.2}"          # script default; raise if vLLM OOMs
MAX_TOKEN_PER_TPU="${MAX_TOKEN_PER_TPU:-0}"       # 0 = packing OFF (baseline); 8192 for the packed arm
MAX_SEGMENTS_PER_ROW="${MAX_SEGMENTS_PER_ROW:-}"  # empty = budget-derived
# Script defaults, tuned for this host -- override only deliberately.
BATCH="${BATCH:-64}"
MINI="${MINI:-64}"
NUM_GEN="${NUM_GEN:-8}"
NUM_BATCHES="${NUM_BATCHES:-150}"          # x NUM_EPOCHS(3) = 450 updates
DATA_DIR="${DATA_DIR:-/tmp/data/frozenlake}"
TB_LOG_DIR="${TB_LOG_DIR:-/tmp/tunix-tb/frozenlake}"
HF_HOME="${HF_HOME:-/mnt/workspace/hf_cache}"
LOG_DIR="${LOG_DIR:-/tmp/train_frozenlake_logs}"
RUN_TAG="${RUN_TAG:-fl_unpack}"

mkdir -p "$LOG_DIR" "$DATA_DIR" "$TB_LOG_DIR"
cd "$TUNIX_DIR"

# ---------------------------------------------------------------------------
# Pinned versions, verified rather than assumed.
# ---------------------------------------------------------------------------
if [ -z "${DRY_RUN:-}" ]; then
  if [ "$SKIP_PIP" = "0" ]; then
    echo "--- installing jax[tpu]==$JAX_VERSION vllm-tpu==$VLLM_VERSION ---"
    pip install -q "jax[tpu]==$JAX_VERSION" || { echo "ERROR: jax install failed"; exit 1; }
    pip install -q --no-deps "vllm-tpu==$VLLM_VERSION" || { echo "ERROR: vllm install failed"; exit 1; }
  fi
  python3 - "$JAX_VERSION" "$VLLM_VERSION" <<'EOF' || exit 1
import sys
want_jax, want_vllm = sys.argv[1], sys.argv[2]
import jax
ok = True
if jax.__version__ != want_jax:
    print(f"ERROR: jax {jax.__version__} != {want_jax}"); ok = False
try:
    import vllm
    if not vllm.__version__.startswith(want_vllm):
        print(f"ERROR: vllm {vllm.__version__} != {want_vllm}"); ok = False
except ImportError as exc:
    print(f"ERROR: vllm not importable ({exc})"); ok = False
if not ok:
    print("The recipe is only known to converge on the pinned versions; "
          "install them or set SKIP_PIP=0 to let this script do it.")
    sys.exit(1)
print(f"versions OK: jax {jax.__version__}, vllm {vllm.__version__}")
EOF
fi

# ---------------------------------------------------------------------------
# Data. The script reads $DATA_DIR/{train,test}.parquet through fsspec; its
# default bucket is not reachable here, so generate the same schema locally
# (idempotent -- data.py loads existing files instead of regenerating).
# ---------------------------------------------------------------------------
if [ -z "${DRY_RUN:-}" ]; then
  PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" python3 - "$DATA_DIR" <<'EOF'
import sys
from examples.frozenlake import data
d = sys.argv[1]
for split in ("train", "test"):
    data.create_dataset(split=split, data_dir=d)
print(f"frozenlake data ready under {d}")
EOF
fi

# ---------------------------------------------------------------------------
# Assemble args. Packing is added only when a budget is set.
# ---------------------------------------------------------------------------
pack_args=()
if [ "${MAX_TOKEN_PER_TPU}" != "0" ]; then
  pack_args+=(--max_seq_token_per_tpu "$MAX_TOKEN_PER_TPU")
  [ -n "${MAX_SEGMENTS_PER_ROW}" ] && \
    pack_args+=(--max_segments_per_packed_row "$MAX_SEGMENTS_PER_ROW")
fi

log="$LOG_DIR/${RUN_TAG}.log"
echo "===== FROZENLAKE[gemma4-e2b] packing=${MAX_TOKEN_PER_TPU} "\
"batch=${BATCH}/${MINI} num_gen=${NUM_GEN} num_batches=${NUM_BATCHES} "\
"(log: $log) ====="

cmd=(python3 -X faulthandler -u examples/frozenlake/train_frozenlake.py
     --batch_size "$BATCH" --mini_batch_size "$MINI"
     --num_generations "$NUM_GEN" --num_batches "$NUM_BATCHES"
     --rollout_vllm_hbm_utilization "$ROLLOUT_HBM"
     "${pack_args[@]}")

if [ -n "${DRY_RUN:-}" ]; then
  echo "--- DRY_RUN ---"
  echo "  DATA_DIR=$DATA_DIR TB_LOG_DIR=$TB_LOG_DIR HF_HOME=$HF_HOME"
  echo "  ROLLOUT_ENGINE=$ENGINE WANDB_RUN_NAME=$RUN_TAG"
  printf '  %s\n' "${cmd[@]}"
  exit 0
fi

DATA_DIR="$DATA_DIR" \
TB_LOG_DIR="$TB_LOG_DIR" \
HF_HOME="$HF_HOME" \
ROLLOUT_ENGINE="$ENGINE" \
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_TAG}" \
WANDB_MODE="${WANDB_MODE:-online}" \
PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
"${cmd[@]}" 2>&1 | tee "$log"

rc=${PIPESTATUS[0]}
echo "===== done (exit=$rc) ====="

echo
echo "################ SUMMARY ################"
if [ -f "$log" ]; then
  echo "--- convergence (want solve_ratio climbing) ---"
  grep -inE "solve_ratio" "$log" | tail -8
  echo "--- packing (only in the packed arm) ---"
  grep -inE "dummy_ratio|pack_sequences|seqs_per_pack" "$log" | tail -6
  echo "--- step timing / HBM ---"
  grep -inE "steps?/sec|sec/step|step_time|HBM" "$log" | tail -6
fi
exit "$rc"
