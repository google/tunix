#!/usr/bin/env bash
# Agentic GRPO counterpart of the distributed Qwen3-4B clean-data recipe.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-/home/haoyugao_google_com/miniconda3/envs/deepswe/bin/python}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/mnt/disks/persist/haoyu-deepswe-agentic-q4-clean}
MODEL_DIR=${MODEL_DIR:-/mnt/disks/persist/haoyu-deepswe-dist-quality/models/Qwen3-4B-Instruct-2507}
DATASET_CACHE=${DATASET_CACHE:-"$ARTIFACT_ROOT/dataset_cache"}
GOLD_WHITELIST=${GOLD_WHITELIST:-"$REPO_ROOT/canon-zero-tim/clean_data/p46_q4_learnable/p46q4census02_qwen3_4b_instruct_2507_n16_learnable_tasks.jsonl"}
WANDB_PROJECT=${WANDB_PROJECT:-trellis-deepswe}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"deepswe-agentic-q4-clean-$(date -u +%Y%m%dT%H%M%SZ)"}
export DATASET_CACHE WANDB_PROJECT WANDB_RUN_NAME
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! mountpoint -q /mnt/disks/persist; then
  echo "Persistent training disk is not mounted at /mnt/disks/persist" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model files are missing from MODEL_DIR=$MODEL_DIR" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Training Python is missing from PYTHON_BIN=$PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" examples/deepswe/train_deepswe_nb.py \
  --model_version Qwen/Qwen3-4B-Instruct-2507 \
  --model_absolute_path "$MODEL_DIR" \
  --dataset_name R2E-Gym/R2E-Gym-Subset \
  --dataset_revision 2e8108ff942f24fcb5686badfaf7f9a8808566d5 \
  --dataset_split train \
  --gold_whitelist "$GOLD_WHITELIST" \
  --seed 42 \
  --batch_size 8 \
  --mini_batch_size 8 \
  --num_generations 16 \
  --num_epochs 2 \
  --max_steps 200 \
  --max_prompt_length 4096 \
  --max_response_length 16384 \
  --max_turns 50 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 0 \
  --vllm_utilization 0.6 \
  --max_num_batched_tokens 20480 \
  --max_concurrency 32 \
  --train_micro_batch_size 2 \
  --rollout_micro_batch_size 1 \
  --compute_logps_micro_batch_size 1 \
  --rollout_mesh_fsdp 1 \
  --rollout_mesh_tp 2 \
  --train_mesh_fsdp 1 \
  --train_mesh_tp 2 \
  --learning_rate 1e-6 \
  --b1 0.9 \
  --b2 0.99 \
  --weight_decay 0.01 \
  --max_grad_norm 1 \
  --beta 0 \
  --epsilon 0.2 \
  --epsilon_high 0.28 \
  --advantage_estimator rloo \
  --score_centering true \
  --score_centering_top_k 32 \
  --score_centering_eps 1e-6 \
  --sampler_is token \
  --sampler_is_threshold 2.0 \
  --loss_agg_mode sequence-mean-token-scale \
  --env_backend docker \
  --scaffold r2egym \
  --action_compat_mode q4_r2egym_xml_v2 \
  --episode_timeout_secs 4800 \
  --step_timeout_secs 1800 \
  --reward_timeout_secs 1800 \
  --eval_every_n_steps 1000000 \
  --ckpt_dir "$ARTIFACT_ROOT/checkpoints" \
  --save_interval_steps 1 \
  --max_to_keep 2 \
  --metric_logger_dir "$ARTIFACT_ROOT/events" \
  "$@"
