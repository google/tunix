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
# The script's hyperparameters are left alone with ONE exception, the episode
# length -- see MAX_RESPONSE below for the measurement that forced it. Otherwise
# this wrapper only supplies what the script cannot know: the environment check,
# a reachable data directory, and the packing switch.
#
# THE TWO RUNS (pack vs unpack; the script packs only when told to, so the
# plain run IS the unpacked baseline):
#   # unpacked baseline / convergence reference
#   RUN_TAG=fl_unpack bash experimental/train_frozenlake_v5p_1host.sh
#   # packed
#   MAX_TOKEN_PER_TPU=8192 RUN_TAG=fl_pack \
#     bash experimental/train_frozenlake_v5p_1host.sh
#
# The packing budget must be at least one maximal sequence, MAX_PROMPT +
# MAX_RESPONSE, and that is also the best value: splash schedules a row's full
# causal area whatever the row holds, so attention cost tracks rows*len^2 and
# every doubling above the longest sequence doubles the attention work at the
# same token count (tracing_logs/splash_microbench_RUN1_results.log). At the
# 4096+4096 default that lower bound is 8192 -- so if you change the lengths,
# change this budget with them.
#
# Inspect the command without launching (no TPU needed):
#   DRY_RUN=1 bash experimental/train_frozenlake_v5p_1host.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

# ---- pinned environment ---------------------------------------------------
# The stack lives in the image (experimental/Dockerfile.frozenlake), not here.
# vllm-tpu pins tpu-inference, which pins jax, jaxlib and libtpu exactly, so
# installing any of them from this script is what broke the ABI before. This
# only checks that the run landed in the right environment.
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.25.0}"
EXPECT_DEVICES="${EXPECT_DEVICES:-4}"      # v5p-8 = 4 chips; the recipe's mesh
                                           # shapes assume this

# ---- knobs ----------------------------------------------------------------
ENGINE="${ROLLOUT_ENGINE:-vllm}"           # vllm | vanilla
ROLLOUT_HBM="${ROLLOUT_HBM:-0.2}"          # script default; raise if vLLM OOMs
MAX_TOKEN_PER_TPU="${MAX_TOKEN_PER_TPU:-0}"       # 0 = packing OFF (baseline); 8192 for the packed arm
MAX_SEGMENTS_PER_ROW="${MAX_SEGMENTS_PER_ROW:-}"  # empty = budget-derived
# Script defaults, tuned for this host -- override only deliberately.
# Episode length. max_response_length is the budget for the WHOLE episode, not
# for one turn -- the collector ends a trajectory once its cumulative response
# tokens reach it, so it is shared across ENV_MAX_STEPS turns. The script's own
# 2048/8 left ~256 tokens per turn, and a measured 2-batch run ended 3872
# trajectories on that budget instead of at the goal: every group scored 0, and
# a GRPO group whose rewards are all equal has zero advantage, so the run had no
# gradient signal at all. 4096+4096 gives ~512 tokens per turn; lowering
# ENV_MAX_STEPS is the other lever on the same ratio.
MAX_PROMPT="${MAX_PROMPT:-4096}"
MAX_RESPONSE="${MAX_RESPONSE:-4096}"
ENV_MAX_STEPS="${ENV_MAX_STEPS:-8}"
BATCH="${BATCH:-64}"
MINI="${MINI:-64}"
NUM_GEN="${NUM_GEN:-8}"
NUM_BATCHES="${NUM_BATCHES:-150}"          # x NUM_EPOCHS(3) = 450 updates
DATA_DIR="${DATA_DIR:-/tmp/data/frozenlake}"
TB_LOG_DIR="${TB_LOG_DIR:-/tmp/tunix-tb/frozenlake}"
# Huggingface's own default location, so this works on a machine that has no
# particular disk mounted. Point it at a persistent disk to keep the ~5GB model
# across runs; the docker wrapper mounts whatever this resolves to.
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-/tmp/train_frozenlake_logs}"
RUN_TAG="${RUN_TAG:-fl_unpack}"

mkdir -p "$LOG_DIR" "$DATA_DIR" "$TB_LOG_DIR"
cd "$TUNIX_DIR"

# ---------------------------------------------------------------------------
# The environment is checked, never installed. Initialising the TPU here is the
# point: an ABI mismatch between libtpu and jax only surfaces on device init, so
# failing here costs seconds instead of failing inside a training run.
# ---------------------------------------------------------------------------
if [ -z "${DRY_RUN:-}" ]; then
  python3 "$TUNIX_DIR/experimental/verify_tpu_stack.py" \
      --vllm-tpu "$VLLM_TPU_VERSION" --devices "$EXPECT_DEVICES" || {
    echo
    echo "This is not the FrozenLake environment. Build and run it with:"
    echo "  bash experimental/build_frozenlake_image.sh"
    echo "  IMAGE=<the tag it prints> bash experimental/train_frozenlake_v5p_1host_docker.sh"
    exit 1
  }
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
"len=${MAX_PROMPT}+${MAX_RESPONSE} turns=${ENV_MAX_STEPS} "\
"batch=${BATCH}/${MINI} num_gen=${NUM_GEN} num_batches=${NUM_BATCHES} "\
"(log: $log) ====="

cmd=(python3 -X faulthandler -u examples/frozenlake/train_frozenlake.py
     --batch_size "$BATCH" --mini_batch_size "$MINI"
     --num_generations "$NUM_GEN" --num_batches "$NUM_BATCHES"
     --max_prompt_length "$MAX_PROMPT" --max_response_length "$MAX_RESPONSE"
     --env_max_steps "$ENV_MAX_STEPS"
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
