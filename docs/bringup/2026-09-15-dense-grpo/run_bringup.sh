#!/bin/bash
# Reproduce the dense GRPO bring-up runs in docs/bringup/2026-09-15-dense-grpo.
#
# One v5p-8 (4 chips): trainer on chips 0-1, rollout on chips 2-3, both tp=2.
# Qwen3-1.7B dense on GSM8K with the GRPO recipe gates enabled.
#
#   TRELLIS_ROOT=/path/to/trellis RUN_ID=tis4 MAX_STEPS=6 ./run_bringup.sh
#
# TRELLIS_ROOT must contain `tunix/` (this repo) and `trellis_env/` (the venv).
# Everything else has a default matching the committed logs.
#
# The four runs behind the report:
#   RUN_ID=tis4 MAX_STEPS=6 TRAIN_MICRO_BATCH_SIZE=4    # training signal
#   RUN_ID=mb1  MAX_STEPS=1 TRAIN_MICRO_BATCH_SIZE=1    # batch-shape sweep
#   RUN_ID=mb2  MAX_STEPS=1 TRAIN_MICRO_BATCH_SIZE=2
#   RUN_ID=mb4  MAX_STEPS=1 TRAIN_MICRO_BATCH_SIZE=4
# tok1/tok4 are reruns of mb1/mb4 and should reproduce them exactly.
set -u

R=${TRELLIS_ROOT:?set TRELLIS_ROOT to the directory holding tunix/ and trellis_env/}
cd "$R/tunix" || exit 1

ID=${RUN_ID:-tis4}
LOGD=$R/logs_$ID

# A stale checkpoint directory makes the orchestrator silently resume and run
# zero steps, reporting only "Resuming from checkpoint: step=N". Always start
# a run from a directory nothing has written to.
rm -rf "$LOGD" "$R/ckpt_$ID"
mkdir -p "$LOGD"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$R/jax_cache}

BATCH_SIZE=${BATCH_SIZE:-2} \
NUM_GENERATIONS=${NUM_GENERATIONS:-8} \
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-4} \
MAX_STEPS=${MAX_STEPS:-6} \
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512} \
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768} \
TRAINER_TP=2 ROLLOUT_TP=2 TRAINER_TPU_CHIPS=0,1 ROLLOUT_TPU_CHIPS=2,3 \
TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 \
SAMPLER=${SAMPLER:-vanilla} BETA=0 WEIGHT_SYNC_MODE=raiden \
CHECKPOINT_SAVE_INTERVAL_STEPS=1000 CHECKPOINT_ROOT_DIRECTORY="$R/ckpt_$ID" \
REWARD_MODE=exact \
EPSILON_HIGH=${EPSILON_HIGH:-0.28} \
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean} \
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-grpo-loo} \
SEQ_LOGPROB_ERROR_THRESHOLD=${SEQ_LOGPROB_ERROR_THRESHOLD:-2.0} \
TIS_TYPE=${TIS_TYPE:-seq-mask-tis} \
TIS_RATIO_MIN=${TIS_RATIO_MIN:-0.999} \
TIS_RATIO=${TIS_RATIO:-1.002} \
PYTHON_BIN=${PYTHON_BIN:-$R/trellis_env/bin/python} \
LOG_ROOT="$LOGD" \
bash tunix/experimental/examples/math_gsm8k_dist/launcher.sh

echo "LAUNCHER_EXIT=$? logs in $LOGD"
