#!/bin/bash
# Real (full) agentic FrozenLake GRPO training on a single-host TPU VM
# (v5p, 4 chips), Gemma4-E2B via the grpo_main CLI. The FrozenLake analogue of
# experimental/train_v5p_1host_pack.sh (which is gsm8k / Qwen3-1.7B demo).
#
# ONE run yields BOTH:
#   * convergence   -- wandb loss/reward (CLI auto-attaches a Wandb backend on
#                      non-internal envs; we just set run_name + WANDB_* env).
#   * performance    -- full-run Perfetto (v1 aggregate spans + v2 trace file)
#                      plus a short xprof kernel window. Same as the gsm8k run.
#
# Unlike the gsm8k demo (which takes --flags), the FrozenLake recipe is the
# grpo_main CLI, so every knob is passed as a `key=value` YAML override that
# joins the flattened gemma4_e2b.yaml in ONE OmegaConf.from_cli (verified in
# scratchpad/test_frozenlake_override.py: nested overrides merge with the
# override-file keys, rl_training_config siblings are preserved).
#
# Usage (on the TPU VM, tunix repo root, deps present -- e.g. inside the
# tunix_base_image container):
#   # packed (segment-aware CL3 + weighted stream)
#   RUN_TAG=cl3_frozenlake_pack bash experimental/train_frozenlake_v5p_1host.sh
#   # unpack parity (same stream+weighted accumulation, packing OFF)
#   MAX_TOKEN_PER_TPU=0 RUN_TAG=cl3_frozenlake_unpack \
#     bash experimental/train_frozenlake_v5p_1host.sh
#
# Inspect the assembled override list WITHOUT launching (no TPU needed):
#   DRY_RUN=1 bash experimental/train_frozenlake_v5p_1host.sh
#
# 4-chip sizing rationale (v5p ~95GB/chip; E2B ~2B params):
#   mesh (2,2) = 2 fsdp x 2 tp -> pack_size = fsdp = 2. TP shards the longer
#     FrozenLake activations (single seq ~4096 = 2x gsm8k's 2048). E2B
#     num_heads=8 -> 4/rank; num_kv_heads=1 -> KV replicated across tp (MQA).
#   vLLM colocated dp=2 tp=2 (aligned with the training mesh).
#   batch 16 / mini 16 / micro 2 / num_gen 8 -> 128 rollouts/step; packing at a
#     16384-token budget (single seq ~4096 -> ~4 seqs/row, same density as
#     gsm8k at 8192). Effective accumulation depth = packed_rows / micro; with a
#     large budget this can collapse to 1 (depth-1 fast path) -- P8.2d logs the
#     real row count/depth, do NOT assume depth>1 here.
#   Memory is not the constraint: model+opt ~8GB/chip, vLLM(hbm 0.2) ~19GB/chip,
#     activation ~1.7GB/chip -> ~66GB/chip free.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

# ---- knobs (same ENV interface as train_v5p_1host_pack.sh) ----------------
ENGINE="${ROLLOUT_ENGINE:-vllm}"           # vllm | vanilla
ROLLOUT_HBM="${ROLLOUT_HBM:-0.2}"          # vLLM HBM fraction (gemma default)
MESH_FSDP="${MESH_FSDP:-2}"                # mesh (2,2) = 2 fsdp x 2 tp
MESH_TP="${MESH_TP:-2}"
VLLM_DP="${VLLM_DP:-2}"                     # colocated rollout dp=2 tp=2
VLLM_TP="${VLLM_TP:-2}"
BATCH="${BATCH:-16}"                        # prompts/step (gsm8k-style light)
MINI="${MINI:-16}"
MICRO="${MICRO:-2}"
NUM_GEN="${NUM_GEN:-8}"                     # RLOO baseline samples (keep 8)
MAX_TOKEN_PER_TPU="${MAX_TOKEN_PER_TPU:-16384}"   # packing budget; 0 = DISABLE
MAX_SEGMENTS_PER_ROW="${MAX_SEGMENTS_PER_ROW:-}"  # segment/row cap (loss num_segments); empty = None = budget-derived
MAX_STEPS="${MAX_STEPS:-200}"              # REAL run length (training UPDATES;
                                           # env multi-turn steps is env_kwargs.
                                           # max_steps=8, untouched)
# The CLI clamps max_steps <= num_batches * num_train_epochs * train_fraction
# (base_rl_pipeline.py:663, raises if exceeded). gemma yaml pins num_batches=5,
# so we must lift it: with batch==mini (1 update/batch) num_batches = MAX_STEPS.
NUM_BATCHES="${NUM_BATCHES:-$MAX_STEPS}"
LOG_DIR="${LOG_DIR:-/tmp/train_frozenlake_logs}"
RUN_TAG="${RUN_TAG:-frozenlake_v5p_pack}"

# xprof: short kernel window then KEEPS TRAINING. Set PROFILER_STEPS=0 to skip.
TRACE_DEST="${TRACE_DEST:-gs://yuxzhang-tunix-models/xprof}"
PROFILER_SKIP="${PROFILER_SKIP:-10}"
PROFILER_STEPS="${PROFILER_STEPS:-5}"

# Perfetto RL perf tracing: whole-run, low overhead.
ENABLE_PERF_V1="${ENABLE_PERF_V1:-1}"
ENABLE_PERF_V2="${ENABLE_PERF_V2:-1}"
PERF_TRACE_DIR="${PERF_TRACE_DIR:-gs://yuxzhang-tunix-models/perfetto/${RUN_TAG}}"

BASE_CONFIG="tunix/cli/base_agentic_config.yaml"
OVERRIDE_CONFIG="examples/frozenlake/configs/gemma4_e2b.yaml"

mkdir -p "$LOG_DIR"
case "$TRACE_DEST" in gs://*) ;; *) mkdir -p "$TRACE_DEST" ;; esac
cd "$TUNIX_DIR"

# ---------------------------------------------------------------------------
# Assemble the CLI overrides. Every override joins the flattened
# override_config_file in ONE OmegaConf.from_cli, so nested keys under
# rl_training_config merge with (do not wipe) the gemma4 config.
# ---------------------------------------------------------------------------
overrides=(
  "$BASE_CONFIG"
  "override_config_file=$OVERRIDE_CONFIG"
  "batch_size=$BATCH"
  "num_batches=$NUM_BATCHES"
  "model_config.mesh.shape=($MESH_FSDP,$MESH_TP)"
  "actor_model_config.mesh.shape=($MESH_FSDP,$MESH_TP)"
  "vllm_config.data_parallel_size=$VLLM_DP"
  "vllm_config.tensor_parallel_size=$VLLM_TP"
  "vllm_config.hbm_utilization=$ROLLOUT_HBM"
  "rl_training_config.mini_batch_size=$MINI"
  "rl_training_config.train_micro_batch_size=$MICRO"
  "rl_training_config.max_steps=$MAX_STEPS"
  # The gemma yaml pins decay_steps: 9 (tuned for its 5-step smoke run) and the
  # CLI only auto-scales decay_steps to max_steps when UNSET -- pinned values
  # survive. At 200 steps that means LR ~0 after step ~9 (dead run). Tie the
  # cosine decay to the real run length. (warmup_steps: 0 is falsy -> the CLI
  # auto-sets it to 0.1 * max_steps = 20, which is fine.)
  "rl_training_config.actor_optimizer_config.decay_steps=$MAX_STEPS"
  "agentic_grpo_config.num_generations=$NUM_GEN"
  "rl_training_config.metrics_logging_options.run_name=$RUN_TAG"
)

# packing (omit entirely when disabled -> TrainingConfig default None -> unpack)
if [ "${MAX_TOKEN_PER_TPU}" != "0" ]; then
  overrides+=("rl_training_config.max_seq_token_per_tpu=$MAX_TOKEN_PER_TPU")
  [ -n "${MAX_SEGMENTS_PER_ROW}" ] && \
    overrides+=("rl_training_config.max_segments_per_packed_row=$MAX_SEGMENTS_PER_ROW")
fi

# xprof kernel window (into gemma's empty profiler_options={})
trace_dir="$TRACE_DEST/${RUN_TAG}"
if [ "${PROFILER_STEPS}" != "0" ]; then
  overrides+=(
    "rl_training_config.profiler_options.log_dir=$trace_dir"
    "rl_training_config.profiler_options.skip_first_n_steps=$PROFILER_SKIP"
    "rl_training_config.profiler_options.profiler_steps=$PROFILER_STEPS"
  )
fi

# Perfetto v1/v2 (perf_metrics_options created fresh; grpo_main-only)
[ "${ENABLE_PERF_V1}" != "0" ] && \
  overrides+=("rl_training_config.perf_metrics_options.enable_perf_v1=true")
if [ "${ENABLE_PERF_V2}" != "0" ]; then
  overrides+=(
    "rl_training_config.perf_metrics_options.enable_perf_v2=true"
    "rl_training_config.perf_metrics_options.trace_dir=$PERF_TRACE_DIR"
  )
fi

log="$LOG_DIR/${RUN_TAG}.log"
echo "===== FROZENLAKE packing=${MAX_TOKEN_PER_TPU} mesh=${MESH_FSDP}x${MESH_TP} "\
"vllm=dp${VLLM_DP}tp${VLLM_TP} batch=${BATCH}/${MINI}/${MICRO} num_gen=${NUM_GEN} "\
"steps=${MAX_STEPS} (log: $log) ====="
printf '  override: %s\n' "${overrides[@]}"

# DRY_RUN: print the assembled command and exit (no TPU / deps needed).
if [ -n "${DRY_RUN:-}" ]; then
  echo "--- DRY_RUN: python3 -m tunix.cli.grpo_main \\"
  printf '      %s \\\n' "${overrides[@]}"
  echo "  (would tee to $log)"
  exit 0
fi

ROLLOUT_ENGINE="$ENGINE" \
PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
WANDB_MODE="${WANDB_MODE:-online}" \
python3 -X faulthandler -u -m tunix.cli.grpo_main \
  "${overrides[@]}" \
  2>&1 | tee "$log"

rc=${PIPESTATUS[0]}
echo "===== done (exit=$rc) ====="

echo
echo "################ SUMMARY ################"
if [ -f "$log" ]; then
  echo "--- step timing / HBM ---"
  grep -inE "steps?/sec|sec/step|step_time|train_step|HBM" "$log" | tail -12
  echo "--- packing efficiency (dummy_ratio; want << 1) + row count/depth ---"
  grep -inE "dummy_ratio|pack_sequences|pack_size|max_seq_token_per_tpu" "$log" | tail -8
  echo "--- single-seq length (confirm budget is >= longest) ---"
  grep -inE "prompts/max_length|completions/max_length|max_length" "$log" | tail -6
  [ "${PROFILER_STEPS}" != "0" ] && echo "--- xprof (kernel) trace: $trace_dir ---"
  [ "${ENABLE_PERF_V2}" != "0" ] && \
    echo "--- Perfetto (full-run) trace: $PERF_TRACE_DIR  (open at ui.perfetto.dev) ---"
fi
exit "$rc"
