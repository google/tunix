#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Single-host multi-step convergence & quantization parity smoke test for MLPerf RL.
#
# Splits a single TPU host between a MaxText Trainer worker and a vLLM Rollout
# worker (e.g. chips 0,1 Trainer + chips 2,3 Rollout on v5p-8 / v7x-8, or chips
# 0..3 Trainer + chips 4..7 Rollout on 8-chip hosts) and runs a CPU GRPO
# orchestrator with MLPerf TIS and sequence-level stability gates enabled.
#
# Usage examples:
#   # 1. Baseline bf16 Sampler + bf16 Trainer on Qwen3.5-35B-A3B:
#   bash tunix/experimental/examples/recipes/mlperf_quant_convergence_smoke.sh
#
#   # 2. FP8 MoE Sampler (with FP8 KV cache) + FP8 MoE & KV Fake-Quant Trainer:
#   SAMPLER_QUANT=fp8_moe TRAINER_QUANT=fp8_moe_kv VLLM_KV_CACHE_DTYPE=fp8 GSM8K_TURNS=10 \
#     bash tunix/experimental/examples/recipes/mlperf_quant_convergence_smoke.sh
#
#   # 3. Analyze an existing orchestrator.log without launching TPU workers:
#   bash tunix/experimental/examples/recipes/mlperf_quant_convergence_smoke.sh \
#     --analyze-only /path/to/orchestrator.log

set -Eeuo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '16,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RECIPE_DIR}/../../../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/opt/venv/bin/python3" ]]; then
    PYTHON_BIN="/opt/venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi
export PATH="$(dirname "$PYTHON_BIN"):${PATH}"
export HF_TOKEN=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}

# ==============================================================================
# 1. User-Configurable Parameters
# ==============================================================================

# --- 1a. Quantization & KV Cache ---
export SAMPLER_QUANT=${SAMPLER_QUANT:-bf16}
export TRAINER_QUANT=${TRAINER_QUANT:-bf16}
export VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-bfloat16}
export QWIX_SKIP_GDN_PROJ=${QWIX_SKIP_GDN_PROJ:-0}

# --- 1b. Task, Model & Sequence Lengths ---
export TASK_MODE=${TASK_MODE:-gsm8k}
export MODEL_NAME=${MODEL_NAME:-Qwen3.5-35B-A3B}
export TRAINER_BACKEND=${TRAINER_BACKEND:-maxtext}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-raiden}
export MAX_STEPS=${MAX_STEPS:-${NUM_STEPS:-5}}
export GSM8K_TURNS=${GSM8K_TURNS:-1}
export GSM8K_MAX_TOKENS_PER_TURN=${GSM8K_MAX_TOKENS_PER_TURN:-512}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
if (( GSM8K_TURNS >= 8 )); then
  export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7680}
elif (( GSM8K_TURNS > 1 )); then
  export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3584}
else
  export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
fi
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-16}
export COMPUTE_LOGPS_CHUNK_SIZE=${COMPUTE_LOGPS_CHUNK_SIZE:-512}

# --- 1c. Convergence & TIS Numerical Stability Gates ---
export MAX_OOB_RATIO=${MAX_OOB_RATIO:-0.50}
export MIN_KEPT_FRAC=${MIN_KEPT_FRAC:-0.50}
export MAX_GEOMEAN_DRIFT=${MAX_GEOMEAN_DRIFT:-0.005}
export SEQ_LOGPROB_ERROR_THRESHOLD=${SEQ_LOGPROB_ERROR_THRESHOLD:-2.0}
export TRUNCATED_IMPORTANCE_SAMPLING_TYPE=${TRUNCATED_IMPORTANCE_SAMPLING_TYPE:-seq-mask-tis}
if (( MAX_RESPONSE_LENGTH <= 1024 )); then
  export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN:-0.997}
  export TRUNCATED_IMPORTANCE_SAMPLING_RATIO=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO:-1.006}
  export SAMPLER_IS_LENGTH_BUCKETS=${SAMPLER_IS_LENGTH_BUCKETS:-512,1024}
else
  export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN:-0.999}
  export TRUNCATED_IMPORTANCE_SAMPLING_RATIO=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO:-1.002}
  export SAMPLER_IS_LENGTH_BUCKETS=${SAMPLER_IS_LENGTH_BUCKETS:-1024,2048,4096}
fi

# --- 1d. GRPO & Optimizer Hyperparameters ---
export BATCH_SIZE=${BATCH_SIZE:-}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-}
export NUM_GENERATIONS=${NUM_GENERATIONS:-}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-}
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export SCHEDULE_TYPE=${SCHEDULE_TYPE:-constant}
export LR_INIT_VALUE=${LR_INIT_VALUE:-0.0}
export LR_PEAK_VALUE=${LR_PEAK_VALUE:-$LEARNING_RATE}
export LR_END_VALUE=${LR_END_VALUE:-$LEARNING_RATE}
export WARMUP_STEPS=${WARMUP_STEPS:-0}
export LR_DECAY_STEPS=${LR_DECAY_STEPS:-100}
export ADAM_B1=${ADAM_B1:-0.9}
export ADAM_B2=${ADAM_B2:-0.999}
export ADAM_EPS=${ADAM_EPS:-1e-8}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
export OPT_CHAIN_TYPE=${OPT_CHAIN_TYPE:-clip_by_global_norm}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.125}
export BETA=${BETA:-0.0}
export EPSILON=${EPSILON:-0.2}
export EPSILON_HIGH=${EPSILON_HIGH:-0.28}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TOP_P=${TOP_P:-1.0}
export TOP_K=${TOP_K:--1}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-grpo-loo}
export OVERLONG_LOSS_MASKING=${OVERLONG_LOSS_MASKING:-1}
export USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-false}

# --- 1e. vLLM Rollout Engine Settings ---
if (( GSM8K_TURNS > 1 )); then
  export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-true}
  export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16}
else
  export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-false}
  export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-4}
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  export VLLM_MAMBA_CACHE_MODE=${VLLM_MAMBA_CACHE_MODE:-align}
else
  export VLLM_MAMBA_CACHE_MODE=${VLLM_MAMBA_CACHE_MODE:-none}
fi
export ROLLOUT_FREE_KV_CACHE=${ROLLOUT_FREE_KV_CACHE:-false}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-$(( VLLM_MAX_MODEL_LEN < 2048 ? VLLM_MAX_MODEL_LEN : 2048 ))}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
export VLLM_BLOCK_SIZE=${VLLM_BLOCK_SIZE:-256}
export VLLM_ASYNC_SCHEDULING=${VLLM_ASYNC_SCHEDULING:-true}
export VLLM_ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL:-true}
export VLLM_LANGUAGE_MODEL_ONLY=${VLLM_LANGUAGE_MODEL_ONLY:-true}
export VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER:-qwen3}
export VLLM_LIMIT_MM_PER_PROMPT=${VLLM_LIMIT_MM_PER_PROMPT:-'{"image": 0, "video": 0}'}

# --- 1f. TPU Slice, Mesh & Checkpoint Overrides (auto-detected if empty) ---
export TRAINER_TPU_CHIPS=${TRAINER_TPU_CHIPS:-}
export ROLLOUT_TPU_CHIPS=${ROLLOUT_TPU_CHIPS:-}
export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-}
export TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS:-1,1,1}
export TRAINER_FSDP=${TRAINER_FSDP:-}
export TRAINER_TP=${TRAINER_TP:-}
export TRAINER_EXPERT=${TRAINER_EXPERT:-1}
export TRAINER_CONTEXT=${TRAINER_CONTEXT:-1}
export REMAT_POLICY=${REMAT_POLICY:-full}
export ROLLOUT_FSDP=${ROLLOUT_FSDP:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-}
export ROLLOUT_EXPERT=${ROLLOUT_EXPERT:-}
export MODEL_ID=${MODEL_ID:-}
export MODEL_DIR=${MODEL_DIR:-}
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
export SAMPLER=${SAMPLER:-}
export TRAINABLE_PARAMETERS_MASK=${TRAINABLE_PARAMETERS_MASK:-}
export MAXTEXT_EXTRA_FLAGS=${MAXTEXT_EXTRA_FLAGS:-}
export ROLLOUT_MAXTEXT_EXTRA_FLAGS=${ROLLOUT_MAXTEXT_EXTRA_FLAGS:-}

# --- 1g. Ports, Timeouts & Logging ---
export ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-orchestrator}
export ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-30000}
export TRAINER_PORT=${TRAINER_PORT:-20000}
export ROLLOUT_PORT=${ROLLOUT_PORT:-20001}
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
export CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-1}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-100000}
export WAIT_TIMEOUT_SECS=${WAIT_TIMEOUT_SECS:-1800}
export WAIT_POLL_SECS=${WAIT_POLL_SECS:-5}

if [[ "${VLLM_KV_CACHE_DTYPE:-bfloat16}" != "bfloat16" ]]; then
  KV_TAG="_kv-${VLLM_KV_CACHE_DTYPE}"
else
  KV_TAG=""
fi
if (( GSM8K_TURNS > 1 )); then
  TURNS_TAG="_t${GSM8K_TURNS}"
else
  TURNS_TAG=""
fi
RUN_TAG="smoke_${MODEL_NAME}_sq-${SAMPLER_QUANT}_tq-${TRAINER_QUANT}${KV_TAG}${TURNS_TAG}_$(date +%Y%m%d_%H%M%S)"
export LOG_ROOT=${LOG_ROOT:-/tmp/mlperf_quant_smoke/${RUN_TAG}}
export LOG_DIR=${LOG_ROOT}/tb
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-${LOG_ROOT}/maxtext_out}
export CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-${LOG_ROOT}/checkpoints}
TRAINER_LOG="${LOG_ROOT}/trainer.log"
ROLLOUT_LOG="${LOG_ROOT}/rollout.log"
ORCHESTRATOR_LOG="${LOG_ROOT}/orchestrator.log"

# ==============================================================================
# 2. Built-in Log Analyzer
# ==============================================================================
analyze_orchestrator_log() {
  "$PYTHON_BIN" - "$@" <<'PY'
import math
import os
import re
import sys

log_path = sys.argv[1]
max_oob = float(sys.argv[2])
min_kept = float(sys.argv[3])
sampler_q = sys.argv[4]
trainer_q = sys.argv[5]
max_drift = float(sys.argv[6])

NUM = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf|-inf)"
step_metrics_re = re.compile(r"\[StepMetrics step=(\d+)\]\s+trainer_metrics=(\{.*\})")
train_step_re = re.compile(
    rf"Train step\s+(\d+)\s+-\s+loss:\s+{NUM}\s+-\s+reward_mean:\s+{NUM}"
    rf"\s+-\s+advantage_mean:\s+{NUM}\s+-\s+grad_norm:\s+{NUM}"
    rf"\s+-\s+perplexity:\s+{NUM}\s+-\s+step_time:\s+{NUM}s"
)
step_end_re = re.compile(r"<<< Step\s+(\d+)\s+finished\s+\|\s+Advanced to Policy Version:\s+(\d+)")

steps = {}
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
  for line in f:
    if m := step_metrics_re.search(line):
      try:
        parsed = eval(m.group(2), {"__builtins__": {}}, {"nan": float("nan"), "inf": float("inf")})
        if isinstance(parsed, dict):
          steps.setdefault(int(m.group(1)), {})["trainer_metrics"] = parsed
      except Exception:
        pass
    elif m := train_step_re.search(line):
      keys = ("loss", "reward_mean", "advantage_mean", "grad_norm", "perplexity", "step_time")
      steps.setdefault(int(m.group(1)), {}).update({k: float(m.group(i + 2)) for i, k in enumerate(keys)})
    elif m := step_end_re.search(line):
      steps.setdefault(int(m.group(1)), {})["next_policy_version"] = int(m.group(2))

if not steps:
  sys.exit(f"ERROR: No training steps found in {log_path}")

kv_dtype = os.environ.get("VLLM_KV_CACHE_DTYPE", "bfloat16")
bar = "=" * 117
print(f"\n{bar}")
print(
    f" MLPerf RL Quantization & Convergence Summary "
    f"(SAMPLER_QUANT={sampler_q}, TRAINER_QUANT={trainer_q}, KV_CACHE={kv_dtype})"
)
print(f" Log: {log_path}\n{bar}")
header = (
    f"{'Step':>4} | {'PolVer':>6} | {'Reward':>8} | {'Loss':>10} | {'GradNorm':>9} | "
    f"{'Geomean':>8} | {'TIS OOB':>8} | {'KeptFrac':>8} | {'MultProbErr':>11} | {'TokLogDiff':>10} | {'Time(s)':>7}"
)
print(f"{header}\n{'-' * len(header)}")

failures = []
kept_fracs = []
ordered = sorted(steps)
seq_err_thresh = float(os.environ.get("SEQ_LOGPROB_ERROR_THRESHOLD", "2.0"))

for s in ordered:
  info = steps[s]
  tm = info.get("trainer_metrics", {})
  pol_ver = info.get("next_policy_version", s + 1)
  reward = info.get("reward_mean", float("nan"))
  loss = info.get("loss", float(tm.get("loss", float("nan"))))
  grad_norm = info.get("grad_norm", float(tm.get("grad_norm", float("nan"))))
  geomean = float(tm.get("sampler_is/seq_geomean_mean", float("nan")))
  tis_oob = float(tm.get("tis/is_oob_ratio", float("nan")))
  kept_frac = float(tm.get("sample_mask/kept_frac", float("nan")))
  mult_max = float(tm.get("sample_mask/mult_prob_error_max", float("nan")))
  mult_err = mult_max if math.isfinite(mult_max) else float(tm.get("sample_mask/mult_prob_error_mean", float("nan")))
  tok_diff = float(tm.get("sampler_is/token_logdiff_absmean", float("nan")))
  step_time = info.get("step_time", float("nan"))

  if math.isfinite(kept_frac):
    kept_fracs.append(kept_frac)
  print(
      f"{s:>4d} | {pol_ver:>6d} | {reward:>8.4f} | {loss:>10.5f} | {grad_norm:>9.4f} | "
      f"{geomean:>8.5f} | {tis_oob:>8.4f} | {kept_frac:>8.4f} | {mult_err:>11.5f} | {tok_diff:>10.5f} | {step_time:>7.1f}"
  )

  if not math.isfinite(loss):
    failures.append(f"Step {s}: non-finite loss ({loss})")
  if not math.isfinite(grad_norm):
    failures.append(f"Step {s}: non-finite grad_norm ({grad_norm})")
  if math.isfinite(geomean) and abs(geomean - 1.0) > max_drift:
    failures.append(f"Step {s}: |seq_geomean_mean - 1.0|={abs(geomean - 1.0):.5f} > MAX_GEOMEAN_DRIFT={max_drift:.5f}")
  if math.isfinite(tis_oob) and tis_oob > max_oob:
    failures.append(f"Step {s}: tis/is_oob_ratio={tis_oob:.4f} > MAX_OOB_RATIO={max_oob:.4f}")
  if math.isfinite(mult_max) and mult_max > seq_err_thresh:
    failures.append(f"Step {s}: sample_mask/mult_prob_error_max={mult_max:.5f} > SEQ_LOGPROB_ERROR_THRESHOLD={seq_err_thresh:.4f}")
  if math.isfinite(kept_frac) and (kept_frac <= 0.0 or (len(ordered) == 1 and kept_frac < min_kept)):
    failures.append(f"Step {s}: sample_mask/kept_frac={kept_frac:.4f} below threshold")

mean_kept = sum(kept_fracs) / len(kept_fracs) if kept_fracs else float("nan")
if math.isfinite(mean_kept) and len(ordered) > 1 and mean_kept < min_kept:
  failures.append(f"Run mean sample_mask/kept_frac={mean_kept:.4f} < MIN_KEPT_FRAC={min_kept:.4f}")

r0 = steps[ordered[0]].get("reward_mean", float("nan"))
r1 = steps[ordered[-1]].get("reward_mean", float("nan"))
r_mean = sum(steps[s].get("reward_mean", 0.0) for s in ordered) / len(ordered)
print(bar)
print(
    f"Steps completed: {len(ordered)} | "
    f"Reward: {r0:.4f} (step {ordered[0]}) -> {r1:.4f} (step {ordered[-1]}), mean={r_mean:.4f} | "
    f"Mean KeptFrac: {mean_kept:.4f}"
)

if failures:
  print("\nSTATUS: FAIL - Quantization/convergence gate violations detected:")
  for msg in failures:
    print(f"  - {msg}")
  sys.exit(1)
print(
    f"\nSTATUS: PASS - All {len(ordered)} step(s) satisfied TIS & numerical health gates "
    f"(|geomean - 1.0| <= {max_drift}, tis/is_oob_ratio <= {max_oob}, mean kept_frac={mean_kept:.4f} >= {min_kept})."
)
PY
}

if [[ "${1:-}" == "--analyze-only" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 --analyze-only /path/to/orchestrator.log" >&2
    exit 1
  fi
  analyze_orchestrator_log \
    "$2" \
    "$MAX_OOB_RATIO" \
    "$MIN_KEPT_FRAC" \
    "$SAMPLER_QUANT" \
    "$TRAINER_QUANT" \
    "$MAX_GEOMEAN_DRIFT"
  exit 0
fi

mkdir -p "$LOG_ROOT"

# ==============================================================================
# 3. Quantization Flag Resolution & Hardware Topology Auto-Detection
# ==============================================================================
ROLLOUT_QUANT_FLAGS=""
ROLLOUT_QWIX_MOE_ONLY_QTYPE=""
ROLLOUT_QWIX_SKIP_GDN_PROJ="$QWIX_SKIP_GDN_PROJ"
case "$SAMPLER_QUANT" in
  bf16|none)
    ROLLOUT_QUANT_FLAGS="quantization= use_qwix_quantization=false"
    ;;
  fp8_moe)
    ROLLOUT_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    ROLLOUT_QWIX_MOE_ONLY_QTYPE="float8_e4m3fn"
    ;;
  int8_moe)
    ROLLOUT_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    ROLLOUT_QWIX_MOE_ONLY_QTYPE="int8"
    ;;
  fp8_no_gdn)
    ROLLOUT_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    ROLLOUT_QWIX_SKIP_GDN_PROJ="1"
    ;;
  fp8|fp8_full)
    ROLLOUT_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    ;;
  int8)
    ROLLOUT_QUANT_FLAGS="quantization=int8 use_qwix_quantization=true quantize_router_proj=false"
    ;;
  *)
    echo "Error: Unsupported SAMPLER_QUANT='$SAMPLER_QUANT' (expected bf16, fp8_moe, int8_moe, fp8_no_gdn, fp8_full, fp8, int8)." >&2
    exit 1
    ;;
esac

TRAINER_QUANT_FLAGS=""
TRAINER_QWIX_MOE_ONLY_QTYPE=""
TRAINER_QWIX_SKIP_GDN_PROJ="$QWIX_SKIP_GDN_PROJ"
case "$TRAINER_QUANT" in
  bf16|none)
    TRAINER_QUANT_FLAGS="quantization= use_qwix_quantization=false"
    ;;
  fp8_kv|fp8_kv_fake_quant)
    TRAINER_QUANT_FLAGS="quantization= use_qwix_quantization=false fp8_kv_fake_quant=true"
    ;;
  fp8_moe|fp8_moe_fake_quant)
    TRAINER_QUANT_FLAGS="quantization= use_qwix_quantization=false fp8_moe_fake_quant=true"
    ;;
  fp8_moe_kv|fp8_moe_kv_fake_quant)
    TRAINER_QUANT_FLAGS="quantization= use_qwix_quantization=false fp8_moe_fake_quant=true fp8_kv_fake_quant=true"
    ;;
  fp8_moe_qwix)
    TRAINER_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    TRAINER_QWIX_MOE_ONLY_QTYPE="float8_e4m3fn"
    ;;
  int8_moe)
    TRAINER_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    TRAINER_QWIX_MOE_ONLY_QTYPE="int8"
    ;;
  fp8_no_gdn)
    TRAINER_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    TRAINER_QWIX_SKIP_GDN_PROJ="1"
    ;;
  fp8|fp8_full)
    TRAINER_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    ;;
  int8)
    TRAINER_QUANT_FLAGS="quantization=int8 use_qwix_quantization=true quantize_router_proj=false"
    ;;
  *)
    echo "Error: Unsupported TRAINER_QUANT='$TRAINER_QUANT' (expected bf16, fp8_kv, fp8_moe, fp8_moe_kv, fp8_moe_qwix, int8_moe, fp8_no_gdn, fp8_full, fp8, int8)." >&2
    exit 1
    ;;
esac

rm -f /tmp/libtpu_lockfile 2>/dev/null || true

# Probe host TPU topology and per-worker JAX device count in a single helper.
IFS='|' read -r \
  HOST_PHYSICAL_TPU_CHIPS \
  DETECTED_TPU_KIND \
  TRAINER_TPU_CHIPS \
  ROLLOUT_TPU_CHIPS \
  TPU_CHIPS_PER_HOST_BOUNDS \
  ACTUAL_TRAINER_DEVICES \
  EFFECTIVE_MEGACORE_ARG < <(
  "$PYTHON_BIN" - \
    "${TRAINER_TPU_CHIPS:-}" \
    "${ROLLOUT_TPU_CHIPS:-}" \
    "${TPU_CHIPS_PER_HOST_BOUNDS:-}" \
    "${TPU_HOST_BOUNDS:-1,1,1}" <<'PY'
import os
import subprocess
import sys

u_t_chips, u_r_chips, u_bounds, host_bounds = sys.argv[1:5]

_SLICE_KEYS = (
    "TPU_VISIBLE_DEVICES",
    "TPU_VISIBLE_CHIPS",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_MEGACORE",
    "LIBTPU_INIT_ARGS",
)

def _probe(extra_env):
  env = os.environ.copy()
  for k in _SLICE_KEYS:
    env.pop(k, None)
  env["JAX_PLATFORMS"] = "tpu,cpu"
  env.update(extra_env)
  code = (
      "import jax; devs = jax.devices(); "
      "chips = len({tuple(getattr(d, 'coords', (i,))) for i, d in enumerate(devs)}); "
      "print(f'{chips}|{len(devs)}|{devs[0].device_kind if devs else \"unknown\"}')"
  )
  res = subprocess.run(
      [sys.executable, "-c", code],
      env=env,
      capture_output=True,
      text=True,
      timeout=60,
      check=False,
  )
  if res.returncode == 0 and res.stdout.strip():
    last = res.stdout.strip().splitlines()[-1]
    if last.count("|") == 2:
      c, d, k = last.split("|")
      return int(c), int(d), k.strip()
  return None

host_chips, _, host_kind = _probe({}) or (4, 8, "unknown")
if host_chips >= 8:
  t_chips = u_t_chips or "0,1,2,3"
  r_chips = u_r_chips or "4,5,6,7"
  bounds = u_bounds or "1,4,1"
  try_mega = ""
else:
  t_chips = u_t_chips or "0,1"
  r_chips = u_r_chips or "2,3"
  bounds = u_bounds or "1,2,1"
  try_mega = "" if any(x in host_kind for x in ("7x", "v5e", "v6e")) else "--deepsea_chip_config_name=megacore"

worker_info = None
mega_used = ""
for flag in ([try_mega, ""] if try_mega else [""]):
  w_env = {
      "TPU_VISIBLE_DEVICES": t_chips,
      "TPU_VISIBLE_CHIPS": t_chips,
      "TPU_CHIPS_PER_HOST_BOUNDS": bounds,
      "TPU_CHIPS_PER_PROCESS_BOUNDS": bounds,
      "TPU_HOST_BOUNDS": host_bounds,
      "LIBTPU_INIT_ARGS": f"{flag} --deepsea_chips_per_host_bounds={bounds} --deepsea_host_bounds={host_bounds}".strip(),
  }
  if flag:
    w_env["TPU_MEGACORE"] = "megacore"
  if (worker_info := _probe(w_env)) is not None:
    mega_used = flag
    break

w_devs = worker_info[1] if worker_info else len([c for c in t_chips.split(",") if c.strip()])
w_kind = worker_info[2] if worker_info else host_kind
print(f"{host_chips}|{w_kind}|{t_chips}|{r_chips}|{bounds}|{w_devs}|{mega_used}")
PY
)

export TRAINER_TPU_CHIPS
export ROLLOUT_TPU_CHIPS
export TPU_CHIPS_PER_HOST_BOUNDS
export TPU_CHIPS_PER_PROCESS_BOUNDS=${TPU_CHIPS_PER_PROCESS_BOUNDS:-$TPU_CHIPS_PER_HOST_BOUNDS}
ACTUAL_ROLLOUT_DEVICES=${ACTUAL_ROLLOUT_DEVICES:-$ACTUAL_TRAINER_DEVICES}

ROLLOUT_XLA_TPU_FLAGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false"
if [[ "$DETECTED_TPU_KIND" == *"7x"* ]]; then
  EXTRA_CHIP_XLA_FLAGS="--xla_tpu_scoped_vmem_limit_kib=65472"
else
  EXTRA_CHIP_XLA_FLAGS=""
fi

export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:-${EFFECTIVE_MEGACORE_ARG:+$EFFECTIVE_MEGACORE_ARG }--deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${TPU_HOST_BOUNDS}}"
export ROLLOUT_LIBTPU_INIT_ARGS="${ROLLOUT_LIBTPU_INIT_ARGS:-${TRAINER_LIBTPU_INIT_ARGS} ${ROLLOUT_XLA_TPU_FLAGS}${EXTRA_CHIP_XLA_FLAGS:+ $EXTRA_CHIP_XLA_FLAGS}}"
export RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST:-$ACTUAL_TRAINER_DEVICES}

if [[ "$MODEL_NAME" == "Qwen3.5-35B-A3B" ]]; then
  export SAMPLER=${SAMPLER:-vllm}
  export MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}
  export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3.5-35b-a3b}
  export MODEL_DIR=${MODEL_DIR:-/tmp/models/Qwen_Qwen3.5-35B-A3B}
  export MAXTEXT_CKPT=${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}
  export TRAINER_FSDP=${TRAINER_FSDP:-$(( ACTUAL_TRAINER_DEVICES <= 2 ? 1 : ACTUAL_TRAINER_DEVICES / 2 ))}
  export TRAINER_TP=${TRAINER_TP:-$(( ACTUAL_TRAINER_DEVICES <= 2 ? ACTUAL_TRAINER_DEVICES : 2 ))}
  export TRAINER_MAXTEXT_ATTENTION=${TRAINER_MAXTEXT_ATTENTION:-flash}
  export TRAINER_BASE_NUM_KV_HEADS=${TRAINER_BASE_NUM_KV_HEADS:-2}
  export USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER:-true}
  export TRAINER_PREFUSE_MOE_WEIGHTS=${TRAINER_PREFUSE_MOE_WEIGHTS:-true}
  export ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS:-true}
  export RETURN_ROUTED_EXPERTS=${RETURN_ROUTED_EXPERTS:-true}
  export EOS_TOKENS=${EOS_TOKENS:-248046,248044}
  export ROLLOUT_TP=${ROLLOUT_TP:-1}
  export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-1}
  export ROLLOUT_EXPERT=${ROLLOUT_EXPERT:-$ACTUAL_ROLLOUT_DEVICES}
  export VLLM_ENABLE_EXPERT_PARALLEL=${VLLM_ENABLE_EXPERT_PARALLEL:-true}
  export VLLM_DATA_PARALLEL_SIZE=${VLLM_DATA_PARALLEL_SIZE:-1}

  if (( ACTUAL_TRAINER_DEVICES <= 4 )); then
    export TRAINABLE_PARAMETERS_MASK=${TRAINABLE_PARAMETERS_MASK:-'^(?!.*routed_experts/gate/kernel)(?!.*mlp/routed_experts/(wi|wi_0|wi_1|wo)).*'}
    LOW_MEM_FLAGS="grad_dtype=bfloat16 mu_dtype=bfloat16 optimizer_memory_host_offload=true"
  else
    export TRAINABLE_PARAMETERS_MASK=${TRAINABLE_PARAMETERS_MASK:-'^(?!.*routed_experts/gate/kernel).*'}
    LOW_MEM_FLAGS=""
  fi
  BASE_MAXTEXT_FLAGS="float32_gate_logits=true float32_logits=true${LOW_MEM_FLAGS:+ $LOW_MEM_FLAGS}"
  ROLLOUT_BASE_MAXTEXT_FLAGS="float32_gate_logits=true float32_logits=true enable_dp_attention=true"
  export MAMBA_CACHE_SPLIT=${MAMBA_CACHE_SPLIT:-$(( GSM8K_TURNS > 1 ? 16 : 2 ))}
  export ROLLOUT_SHARDING_JSON=${ROLLOUT_SHARDING_JSON:-"{\"additional_config\":{\"sharding\":{\"sharding_strategy\":{\"expert_parallelism\":${ROLLOUT_EXPERT},\"tensor_parallelism\":1,\"enable_dp_attention\":true}},\"custom_mamba_cache_multiplier\":${MAMBA_CACHE_SPLIT},\"maxtext_config\":{\"scan_layers\":false,\"attention\":\"vllm_rpa\",\"allow_split_physical_axes\":true,\"use_multimodal\":false,\"prefuse_moe_weights\":true}}}"}
else
  export SAMPLER=${SAMPLER:-inprocess_vllm}
  export MODEL_ID=${MODEL_ID:-Qwen/${MODEL_NAME}}
  export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')}
  export MODEL_DIR=${MODEL_DIR:-/tmp/models/Qwen_${MODEL_NAME}}
  export MAXTEXT_CKPT=${MAXTEXT_CKPT:-${MODEL_DIR}_maxtext/0/items}
  export TRAINER_FSDP=${TRAINER_FSDP:-$(( ACTUAL_TRAINER_DEVICES <= 2 ? 1 : 2 ))}
  export TRAINER_TP=${TRAINER_TP:-$(( ACTUAL_TRAINER_DEVICES <= 2 ? ACTUAL_TRAINER_DEVICES : ACTUAL_TRAINER_DEVICES / 2 ))}
  export TRAINER_MAXTEXT_ATTENTION=${TRAINER_MAXTEXT_ATTENTION:-autoselected}
  export TRAINER_BASE_NUM_KV_HEADS=${TRAINER_BASE_NUM_KV_HEADS:-}
  export USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER:-false}
  export TRAINER_PREFUSE_MOE_WEIGHTS=${TRAINER_PREFUSE_MOE_WEIGHTS:-false}
  export ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS:-false}
  export RETURN_ROUTED_EXPERTS=${RETURN_ROUTED_EXPERTS:-false}
  export EOS_TOKENS=${EOS_TOKENS:-}
  export ROLLOUT_TP=${ROLLOUT_TP:-$ACTUAL_ROLLOUT_DEVICES}
  export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-$ACTUAL_ROLLOUT_DEVICES}
  export ROLLOUT_EXPERT=${ROLLOUT_EXPERT:-1}
  export TRAINABLE_PARAMETERS_MASK=${TRAINABLE_PARAMETERS_MASK:-}
  BASE_MAXTEXT_FLAGS="float32_logits=true"
  ROLLOUT_BASE_MAXTEXT_FLAGS="float32_logits=true"
  export ROLLOUT_SHARDING_JSON=${ROLLOUT_SHARDING_JSON:-"{}"}
fi
export TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_DIR}
export MAXTEXT_EXTRA_FLAGS="${BASE_MAXTEXT_FLAGS} ${TRAINER_QUANT_FLAGS}${MAXTEXT_EXTRA_FLAGS:+ ${MAXTEXT_EXTRA_FLAGS}}"
export ROLLOUT_MAXTEXT_EXTRA_FLAGS="${ROLLOUT_BASE_MAXTEXT_FLAGS} ${ROLLOUT_QUANT_FLAGS}${ROLLOUT_MAXTEXT_EXTRA_FLAGS:+ ${ROLLOUT_MAXTEXT_EXTRA_FLAGS}}"

if (( ACTUAL_TRAINER_DEVICES <= 2 )) && [[ "$MODEL_NAME" == "Qwen3.5-35B-A3B" ]]; then
  export BATCH_SIZE=${BATCH_SIZE:-1}
  export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
else
  export BATCH_SIZE=${BATCH_SIZE:-2}
  export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
fi
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$BATCH_SIZE}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-$TRAINER_FSDP}

# ==============================================================================
# 4. Runtime Hook (sitecustomize.py) & vLLM Config JSON Builder
# ==============================================================================
# Lazy MetaPathFinder (avoids importing JAX before raiden_preload.import_raiden):
#   1. Qwix MoE-only quantization rule when QWIX_MOE_ONLY_QTYPE is set.
#   2. Mask-aware FP32 master optimizer so frozen parameters in
#      TRAINABLE_PARAMETERS_MASK do not allocate duplicate FP32 master copies.
#   3. Multi-turn GSM8K prompt/env wrapper when GSM8K_TURNS > 1.
PY_HOOKS_DIR="${LOG_ROOT}/py_hooks"
mkdir -p "$PY_HOOKS_DIR"
cat << 'PY' > "${PY_HOOKS_DIR}/sitecustomize.py"
import importlib.abc
import importlib.machinery
import os
import sys


def _patch_quantizations(quantizations_mod):
  orig_get_rule = quantizations_mod.get_quantization_rule
  qtype_name = os.environ.get("QWIX_MOE_ONLY_QTYPE", "").strip().lower()
  bwd_quant_env = os.environ.get("QWIX_MOE_BWD_QUANT", "").strip().lower()
  skip_gdn_proj = os.environ.get("QWIX_SKIP_GDN_PROJ", "").strip().lower() in ("1", "true", "yes")

  def _custom_get_quantization_rule(config):
    import jax.numpy as jnp
    import qwix

    qmap = {
        "float8_e4m3fn": jnp.float8_e4m3fn,
        "fp8": jnp.float8_e4m3fn,
        "fp8_e4m3": jnp.float8_e4m3fn,
        "fp8_full": jnp.float8_e4m3fn,
        "int8": jnp.int8,
    }
    quant_mode = getattr(config, "quantization", "") or ""
    if qtype_name:
      qtype = qmap[qtype_name]
      moe_only = True
    elif quant_mode in qmap:
      qtype = qmap[quant_mode]
      moe_only = False
    else:
      return orig_get_rule(config)

    if bwd_quant_env in ("1", "true", "yes"):
      enable_moe_bwd_q = True
    elif bwd_quant_env in ("0", "false", "no"):
      enable_moe_bwd_q = False
    else:
      enable_moe_bwd_q = not bool(getattr(config, "use_gmm_v2", False))

    dense_bwd_q = jnp.float8_e5m2 if qtype == jnp.float8_e4m3fn else qtype
    w_cal = getattr(config, "weight_quantization_calibration_method", "absmax")
    a_cal = getattr(config, "act_quantization_calibration_method", "absmax")
    b_cal = getattr(config, "bwd_quantization_calibration_method", "absmax")

    rules = []
    if not moe_only:
      if not getattr(config, "quantize_router_proj", False):
        rules.append(
            qwix.QtRule(
                module_path=r".*/(gate|shared_expert_gate)$",
                weight_qtype=None,
                act_qtype=None,
                bwd_qtype=None,
                op_names=("dot_general",),
            )
        )
      gdn_projs = "" if skip_gdn_proj else "in_proj_qkvz|out_proj|"
      rules.append(
          qwix.QtRule(
              module_path=(
                  r"decoder/.*layers.*/"
                  rf"(query|key|value|qkv_proj|out|wq_a|wq_b|wkv_a|wkv_b|"
                  rf"{gdn_projs}wi|wi_0|wi_1|wo)$"
              ),
              weight_qtype=qtype,
              act_qtype=qtype,
              bwd_qtype=dense_bwd_q,
              weight_calibration_method=w_cal,
              act_calibration_method=a_cal,
              bwd_calibration_method=b_cal,
              op_names=("dot_general",),
          )
      )
    rules.append(
        qwix.QtRule(
            module_path="decoder/.*layers.*" if not moe_only else ".*",
            weight_qtype=qtype,
            act_qtype=qtype if enable_moe_bwd_q else None,
            bwd_qtype=dense_bwd_q if enable_moe_bwd_q else None,
            weight_calibration_method=w_cal,
            act_calibration_method=a_cal,
            bwd_calibration_method=b_cal,
            op_names=("gmm", "ragged_dot"),
        )
    )
    return rules

  quantizations_mod.get_quantization_rule = _custom_get_quantization_rule


def _patch_maxtext_utils(maxtext_utils_mod):
  def _build_mask_aware_fp32_master_optimizer_cls(nnx_mod=None):
    from flax.nnx.training import optimizer as nnx_opt
    import jax
    import jax.numpy as jnp
    from maxtext.optimizers import optimizers as maxtext_optimizers
    import optax

    if nnx_mod is None:
      from flax import nnx as nnx_mod
    opt_var_cls = getattr(nnx_opt, "OptState", getattr(nnx_mod, "Variable", None))
    to_opt_state = getattr(nnx_opt, "to_opt_state", lambda x: x)

    class Fp32MasterOptimizer(nnx_mod.Optimizer):
      """FP32 master-weight optimizer that skips allocating master copies for frozen params."""

      def __init__(self, model, tx, *, wrt=nnx_mod.Param):
        self.step = opt_var_cls(jnp.array(0, dtype=jnp.uint32))
        self.tx = tx
        self.wrt = wrt
        params = nnx_mod.state(model, wrt)
        cfg = getattr(model, "config", None) or getattr(getattr(model, "base_model", None), "config", None)
        patterns = getattr(cfg, "trainable_parameters_mask", None)
        if not patterns and (raw := os.environ.get("TRAINABLE_PARAMETERS_MASK", "").strip()):
          patterns = [raw]
        freeze_fn = maxtext_optimizers.get_path_mask_fn(patterns, match_returns_true=False) if patterns else None
        frozen = freeze_fn(params) if freeze_fn else jax.tree.map(lambda _: False, params)
        master = jax.tree.map(
            lambda x, f: None if (f or x.dtype == jnp.float32) else x.astype(jnp.float32),
            params,
            frozen,
        )
        inner = tx.init(jax.tree.map(lambda p, m: p if m is None else m, params, master))
        state = {"master_params": master, "inner": inner}
        if isinstance(inner, dict) and "is_skipped" in inner:
          state["is_skipped"] = inner["is_skipped"]
        self.opt_state = nnx_mod.data(to_opt_state(state))

      def update(self, model, grads, /, **kwargs):
        p_arr = nnx_mod.as_pure(nnx_mod.state(model, self.wrt))
        g_arr = nnx_mod.as_pure(nnx_mod.state(grads, self.wrt))
        s_arr = nnx_mod.as_pure(self.opt_state)
        master = s_arr["master_params"]
        inner = s_arr["inner"]
        full_f32 = jax.tree.map(lambda p, m: p if m is None else m, p_arr, master)
        g_f32 = jax.tree.map(
            lambda g, p, m: g.astype(jnp.float32) if (m is not None or p.dtype == jnp.float32) else g,
            g_arr,
            p_arr,
            master,
        )
        updates, new_inner = self.tx.update(g_f32, inner, full_f32, **nnx_mod.as_pure(kwargs))
        up_f32 = optax.apply_updates(full_f32, updates)
        new_master = jax.tree.map(lambda _p, m, u: u if m is not None else None, p_arr, master, up_f32)
        new_state = {"master_params": new_master, "inner": new_inner}
        if isinstance(new_inner, dict) and "is_skipped" in new_inner:
          new_state["is_skipped"] = new_inner["is_skipped"]
        nnx_mod.update(model, jax.tree.map(lambda p, u: u.astype(p.dtype), p_arr, up_f32))
        nnx_mod.update(self.opt_state, nnx_mod.state(new_state))
        self.step[...] += 1
        return updates

    return Fp32MasterOptimizer

  maxtext_utils_mod._build_fp32_master_optimizer_cls = _build_mask_aware_fp32_master_optimizer_cls


def _patch_gsm8k(gsm8k_mod):
  env_cls = gsm8k_mod.GSM8KEnv
  orig_init = env_cls.__init__
  orig_obs = env_cls._initial_observation
  orig_step = env_cls._step_impl

  def _init(self, *args, followup_turns=None, **kwargs):
    orig_init(self, *args, **kwargs)
    self._followup_turns = list(followup_turns or [])
    self._init_task_snapshot = dict(self.task)
    self._turn_idx = 0

  def _initial_obs(self):
    if getattr(self, "_init_task_snapshot", None):
      self.task.update(self._init_task_snapshot)
    self._turn_idx = 0
    return orig_obs(self)

  def _step_impl(self, action):
    followups = getattr(self, "_followup_turns", None)
    if not followups:
      return orig_step(self, action)
    reward, info = gsm8k_mod.gsm8k_env_reward(self.task, action)
    info["correct"] = bool(info["answer_correct"])
    info["turn_idx"] = self._turn_idx
    scaled_reward = float(reward) / float(1 + len(followups))
    if self._turn_idx < len(followups):
      nxt = followups[self._turn_idx]
      self._turn_idx += 1
      p_txt = str(nxt.get("prompts", ""))
      self.task.update({
          "prompts": p_txt,
          "question": str(nxt.get("question", p_txt)),
          "answer": str(nxt.get("answer", "")),
          "gold_answer": str(nxt.get("answer", "")),
      })
      return gsm8k_mod.base_environment.EnvStepResult(
          observation={"prompts": p_txt},
          reward=scaled_reward,
          done=False,
          info=info,
      )
    self._turn_idx += 1
    comp = action.action if hasattr(action, "action") else str(action)
    return gsm8k_mod.base_environment.EnvStepResult(
        observation={
            "answer": str(comp),
            "gold_answer": str(self.task.get("gold_answer", "")),
        },
        reward=scaled_reward,
        done=True,
        info=info,
    )

  env_cls.__init__ = _init
  env_cls._initial_observation = _initial_obs
  env_cls._step_impl = _step_impl


def _patch_gsm8k_runner(runner_mod):
  orig_iter = runner_mod._iter_prompt_items
  orig_build = runner_mod._build_prompt_item

  def _multi_turn_iter(args):
    num_turns = int(os.environ.get("GSM8K_TURNS", "1") or "1")
    if num_turns <= 1:
      yield from orig_iter(args)
      return
    max_toks = int(os.environ.get("GSM8K_MAX_TOKENS_PER_TURN", "512") or "512")
    g_mod = runner_mod.gsm8k
    ds = g_mod.load_gsm8k_dataset(
        split=args.tfds_split,
        data_dir=args.tfds_data_dir,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    for idx in range(args.max_steps * args.batch_size):
      item = orig_build(example=ds[(idx * num_turns) % len(ds)], prompt_idx=idx)
      followups = []
      for t in range(1, num_turns):
        ex = ds[(idx * num_turns + t) % len(ds)]
        ans = g_mod.normalize_example_value(ex["answer"])
        followups.append({
            "prompts": g_mod.as_text(ex["prompts"]),
            "question": g_mod.as_text(ex["question"]),
            "answer": ans,
            "gold_answer": ans,
        })
      item["max_turns"] = num_turns
      item["generation_kwargs"] = {"max_tokens": max_toks}
      item["metadata"]["env_config"].update({
          "max_steps": num_turns,
          "followup_turns": followups,
      })
      yield item

  runner_mod._iter_prompt_items = _multi_turn_iter


_PATCH_TARGETS = {
    "maxtext.layers.quantizations": _patch_quantizations,
    "tunix.utils.maxtext_utils": _patch_maxtext_utils,
    "tunix.experimental.examples.math_gsm8k_dist.gsm8k": _patch_gsm8k,
    "tunix.experimental.examples.math_gsm8k_dist.run_gsm8k_dist_grpo": _patch_gsm8k_runner,
}


class _SmokeTestPatchFinder(importlib.abc.MetaPathFinder):

  def __init__(self):
    self._active = set()

  def find_spec(self, fullname, path, target=None):
    if fullname not in _PATCH_TARGETS or fullname in self._active:
      return None
    self._active.add(fullname)
    try:
      spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
    finally:
      self._active.discard(fullname)
    if spec and spec.loader and hasattr(spec.loader, "exec_module"):
      orig_exec = spec.loader.exec_module

      def _exec(module):
        orig_exec(module)
        _PATCH_TARGETS[fullname](module)

      spec.loader.exec_module = _exec
    return spec


sys.meta_path.insert(0, _SmokeTestPatchFinder())
PY

ROLLOUT_VLLM_CONFIG_JSON=$(
  "$PYTHON_BIN" - \
    "$ROLLOUT_SHARDING_JSON" \
    "${ROLLOUT_VLLM_CONFIG_JSON:-}" \
    "$ROLLOUT_MAXTEXT_EXTRA_FLAGS" <<'PY'
import json
import os
import sys

base_json, user_json, extra_flags = sys.argv[1:4]
cfg = json.loads(base_json) if base_json.strip() else {}
if user_json.strip():
  cfg.update(json.loads(user_json))

def _parse_val(raw: str):
  low = raw.lower()
  if low in ("true", "false"):
    return low == "true"
  if low == "none":
    return None
  for fn in (int, float):
    try:
      return fn(raw)
    except ValueError:
      pass
  return raw

as_bool = lambda v: v.lower() in ("true", "1")
env_mapping = {
    "VLLM_MAX_MODEL_LEN": ("max_model_len", int),
    "VLLM_MAX_NUM_BATCHED_TOKENS": ("max_num_batched_tokens", int),
    "VLLM_MAX_NUM_SEQS": ("max_num_seqs", int),
    "VLLM_GPU_MEMORY_UTILIZATION": ("gpu_memory_utilization", float),
    "VLLM_DATA_PARALLEL_SIZE": ("data_parallel_size", int),
    "VLLM_ENABLE_EXPERT_PARALLEL": ("enable_expert_parallel", as_bool),
    "VLLM_MAMBA_CACHE_MODE": ("mamba_cache_mode", str),
    "VLLM_KV_CACHE_DTYPE": ("kv_cache_dtype", str),
    "VLLM_BLOCK_SIZE": ("block_size", int),
    "VLLM_ASYNC_SCHEDULING": ("async_scheduling", as_bool),
    "VLLM_ENABLE_CHUNKED_PREFILL": ("enable_chunked_prefill", as_bool),
    "VLLM_LANGUAGE_MODEL_ONLY": ("language_model_only", as_bool),
    "VLLM_REASONING_PARSER": ("reasoning_parser", str),
}
for env_k, (cfg_k, parse_fn) in env_mapping.items():
  if (val := os.getenv(env_k)) and cfg_k not in cfg:
    try:
      cfg[cfg_k] = parse_fn(val)
    except Exception:
      cfg[cfg_k] = val

if (limit_mm := os.getenv("VLLM_LIMIT_MM_PER_PROMPT")) and "limit_mm_per_prompt" not in cfg:
  try:
    cfg["limit_mm_per_prompt"] = json.loads(limit_mm)
  except Exception:
    pass

mt_cfg = cfg.setdefault("additional_config", {}).setdefault("maxtext_config", {})
for tok in extra_flags.split():
  if "=" in tok:
    k, v = tok.split("=", 1)
    mt_cfg[k] = _parse_val(v)

print(json.dumps(cfg))
PY
)
export ROLLOUT_VLLM_CONFIG_JSON

# ==============================================================================
# 5. Model & MaxText Checkpoint Preparation
# ==============================================================================
download_hf_repo() {
  local config_only="${1:-0}"
  echo "Downloading $MODEL_ID to $MODEL_DIR (config_only=$config_only)..."
  "$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" "$config_only" <<'PY'
import os
import sys
import huggingface_hub as hf

model_id = sys.argv[1]
model_dir = sys.argv[2]
config_only = sys.argv[3] == "1"
token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
ignore = ["original/*", "*.safetensors", "*.bin", "*.pt"] if config_only else ["original/*"]
hf.snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    token=token,
    ignore_patterns=ignore,
)
PY
}

prepare_model_and_ckpt() {
  if [[ "$TRAINER_BACKEND" == "maxtext" ]] && { [[ "$MAXTEXT_CKPT" =~ ^gs:// ]] || [[ -d "$MAXTEXT_CKPT" ]]; }; then
    echo "Using existing MaxText checkpoint: $MAXTEXT_CKPT"
    if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
      download_hf_repo 1
    fi
    return
  fi
  if [[ ! -d "$MODEL_DIR" ]] || [[ -z "$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print -quit 2>/dev/null)" ]]; then
    download_hf_repo 0
  fi
  if [[ "$TRAINER_BACKEND" == "maxtext" && ! -d "$MAXTEXT_CKPT" ]]; then
    local ckpt_base
    ckpt_base="$(dirname "$(dirname "$MAXTEXT_CKPT")")"
    echo "Converting HF checkpoint $MODEL_DIR to MaxText Orbax checkpoint at $ckpt_base..."
    mkdir -p "$ckpt_base"
    JAX_PLATFORMS=cpu PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" \
      -m maxtext.checkpoint_conversion.to_maxtext \
      model_name="${MAXTEXT_MODEL_NAME}" \
      --hf_model_path="${MODEL_DIR}" \
      base_output_directory="${ckpt_base}" \
      scan_layers=True \
      skip_jax_distributed_system=True \
      checkpoint_storage_use_zarr3=false \
      checkpoint_storage_use_ocdbt=false
  fi
}

# ==============================================================================
# 6. Process Management & Worker Launch
# ==============================================================================
wait_for_port() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local elapsed=0
  while true; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Error: $name process (pid=$pid) exited before gRPC port $port became ready." >&2
      tail -n 80 "$log_file" >&2 2>/dev/null || true
      exit 1
    fi
    if "$PYTHON_BIN" -c "import socket, sys; socket.create_connection(('localhost', int(sys.argv[1])), timeout=1).close()" "$port" 2>/dev/null; then
      echo "$name port $port is ready after ${elapsed}s."
      return
    fi
    if (( elapsed >= WAIT_TIMEOUT_SECS )); then
      echo "Error: timed out waiting ${WAIT_TIMEOUT_SECS}s for $name port $port." >&2
      tail -n 80 "$log_file" >&2 2>/dev/null || true
      exit 1
    fi
    echo "Waiting for $name port $port... elapsed=${elapsed}s (log: $log_file)"
    sleep "$WAIT_POLL_SECS"
    elapsed=$((elapsed + WAIT_POLL_SECS))
  done
}

kill_tree() {
  local sig="$1"
  local p="$2"
  local c
  for c in $(cat "/proc/$p/task/$p/children" 2>/dev/null); do
    kill_tree "$sig" "$c"
  done
  kill "$sig" "$p" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  local pids=()
  for pid in "${TRAINER_PID:-}" "${ROLLOUT_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      pids+=("$pid")
    fi
  done
  if (( ${#pids[@]} > 0 )); then
    echo "Stopping worker processes: ${pids[*]}"
    for pid in "${pids[@]}"; do
      kill_tree -TERM "$pid"
    done
    sleep 3 || true
    for pid in "${pids[@]}"; do
      kill_tree -KILL "$pid"
    done
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ensure_protos_compiled() {
  local proto_dir="${REPO_ROOT}/tunix/experimental/distributed/runtime/discovery"
  if [[ ! -f "${proto_dir}/discovery_service_pb2.py" || ! -f "${proto_dir}/discovery_service_pb2_grpc.py" ]]; then
    echo "Compiling gRPC protobufs (${proto_dir}/discovery_service.proto)..."
    "$PYTHON_BIN" -m grpc_tools.protoc \
      -I"${REPO_ROOT}" \
      --python_out="${REPO_ROOT}" \
      --grpc_python_out="${REPO_ROOT}" \
      "${proto_dir}/discovery_service.proto"
  fi
}

echo "=========================================================================="
echo "MLPerf Single-Host Quantization & Convergence Smoke Test"
echo "  Task mode:                  $TASK_MODE (gsm8k_turns=$GSM8K_TURNS, max_prompt=$MAX_PROMPT_LENGTH, max_resp=$MAX_RESPONSE_LENGTH, seq_len=$MAX_SEQ_TOKEN_PER_TPU)"
echo "  Model:                      $MODEL_NAME ($MAXTEXT_MODEL_NAME)"
echo "  Host TPU topology:          ${HOST_PHYSICAL_TPU_CHIPS} physical chips (${DETECTED_TPU_KIND}), ${ACTUAL_TRAINER_DEVICES} JAX devices/worker (${EFFECTIVE_MEGACORE_ARG:-default})"
echo "  Sampler quantization:       $SAMPLER_QUANT (qwix_moe_only=${ROLLOUT_QWIX_MOE_ONLY_QTYPE:-off}, kv_cache=${VLLM_KV_CACHE_DTYPE}, prefix_cache=${ENABLE_PREFIX_CACHING}, mamba_cache=${VLLM_MAMBA_CACHE_MODE})"
echo "  Trainer quantization:       $TRAINER_QUANT (qwix_moe_only=${TRAINER_QWIX_MOE_ONLY_QTYPE:-off}, flags=${TRAINER_QUANT_FLAGS})"
echo "  ROLLOUT_MAXTEXT_EXTRA_FLAGS: $ROLLOUT_MAXTEXT_EXTRA_FLAGS"
echo "  MAXTEXT_EXTRA_FLAGS:         $MAXTEXT_EXTRA_FLAGS"
echo "  Trainer chips / mesh:       chips=$TRAINER_TPU_CHIPS (FSDP=$TRAINER_FSDP, TP=$TRAINER_TP, EP=$TRAINER_EXPERT, bounds=$TPU_CHIPS_PER_HOST_BOUNDS)"
echo "  Rollout chips / mesh:       chips=$ROLLOUT_TPU_CHIPS (sampler=$SAMPLER, FSDP=$ROLLOUT_FSDP, TP=$ROLLOUT_TP, EP=${ROLLOUT_EXPERT:-1}, bounds=$TPU_CHIPS_PER_HOST_BOUNDS)"
echo "  Steps / Batch / Gens:       steps=$MAX_STEPS, batch=$BATCH_SIZE, gens=$NUM_GENERATIONS, micro_batch=$TRAIN_MICRO_BATCH_SIZE"
echo "  TIS gate:                   type=$TRUNCATED_IMPORTANCE_SAMPLING_TYPE [$TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN, $TRUNCATED_IMPORTANCE_SAMPLING_RATIO]"
echo "  Log directory:              $LOG_ROOT"
echo "=========================================================================="

ensure_protos_compiled
prepare_model_and_ckpt

: > "$TRAINER_LOG"
: > "$ROLLOUT_LOG"
: > "$ORCHESTRATOR_LOG"

export PYTHONPATH="${PY_HOOKS_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export USE_RAIDEN_FFI=false
export RAIDEN_USE_FFI=0
export PYTHONUNBUFFERED=1

echo "Launching Trainer worker on TPU chips $TRAINER_TPU_CHIPS..."
(
  TRAINER_CMD=(
    "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
    --discovery_addrs="${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT}"
    --process_main=tunix.experimental.examples.common.run_trainer_node.main
    --port="$TRAINER_PORT"
    --mesh_fsdp="$TRAINER_FSDP"
    --mesh_tp="$TRAINER_TP"
    --mesh_expert="$TRAINER_EXPERT"
    --mesh_context="$TRAINER_CONTEXT"
    --prefuse_moe_weights="$TRAINER_PREFUSE_MOE_WEIGHTS"
    --model_id="$MODEL_ID"
    --model_dir="$MODEL_DIR"
    --model_name="$MODEL_NAME"
    --sampler_type="$SAMPLER"
    --tokenizer_path="$TOKENIZER_PATH"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --mini_batch_size="$MINI_BATCH_SIZE"
    --num_generations="$NUM_GENERATIONS"
    --train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
    --compute_logps_chunk_size="${COMPUTE_LOGPS_CHUNK_SIZE:-0}"
    --eval_every_n_steps="$EVAL_EVERY_N_STEPS"
    --optimizer_b1="$ADAM_B1"
    --optimizer_b2="$ADAM_B2"
    --optimizer_eps="$ADAM_EPS"
    --optimizer_weight_decay="$WEIGHT_DECAY"
    --optimizer_learning_rate="$LEARNING_RATE"
    --optimizer_schedule_type="$SCHEDULE_TYPE"
    --optimizer_init_value="$LR_INIT_VALUE"
    --optimizer_peak_value="$LR_PEAK_VALUE"
    --optimizer_end_value="$LR_END_VALUE"
    --optimizer_warmup_steps="$WARMUP_STEPS"
    --optimizer_decay_steps="$LR_DECAY_STEPS"
    --optimizer_opt_chain_type="$OPT_CHAIN_TYPE"
    --optimizer_chain_kwargs="{'max_norm': $MAX_GRAD_NORM}"
    --trainer_backend="$TRAINER_BACKEND"
    --checkpoint_save_interval_steps="$CHECKPOINT_SAVE_INTERVAL_STEPS"
    --checkpoint_max_to_keep="$CHECKPOINT_MAX_TO_KEEP"
    --checkpoint_root_directory="$CHECKPOINT_ROOT_DIRECTORY"
    --maxtext_model_name="$MAXTEXT_MODEL_NAME"
    --maxtext_ckpt_path="$MAXTEXT_CKPT"
    --maxtext_output_directory="$MAXTEXT_OUTPUT_DIR"
    --maxtext_attention="$TRAINER_MAXTEXT_ATTENTION"
    --remat_policy="$REMAT_POLICY"
    --rollout_mesh_tp="$ROLLOUT_MESH_TP"
    --use_weight_converter="${USE_WEIGHT_CONVERTER:-false}"
    --max_seq_token_per_tpu="$MAX_SEQ_TOKEN_PER_TPU"
  )
  if [[ -n "${TRAINER_BASE_NUM_KV_HEADS:-}" ]]; then
    TRAINER_CMD+=(--base_num_kv_heads="$TRAINER_BASE_NUM_KV_HEADS")
  fi
  if [[ -n "${TRAINABLE_PARAMETERS_MASK:-}" ]]; then
    TRAINER_CMD+=(--trainable_parameters_mask="$TRAINABLE_PARAMETERS_MASK")
  fi

  export QWIX_MOE_ONLY_QTYPE="$TRAINER_QWIX_MOE_ONLY_QTYPE"
  export QWIX_SKIP_GDN_PROJ="$TRAINER_QWIX_SKIP_GDN_PROJ"
  export RAIDEN_DEVICES_PER_HOST="$ACTUAL_TRAINER_DEVICES"
  export JAX_PLATFORMS=tpu,cpu
  export TPU_VISIBLE_DEVICES="$TRAINER_TPU_CHIPS"
  export TPU_VISIBLE_CHIPS="$TRAINER_TPU_CHIPS"
  if [[ -n "${EFFECTIVE_MEGACORE_ARG:-}" ]]; then
    export TPU_MEGACORE=megacore
  fi
  export LIBTPU_INIT_ARGS="$TRAINER_LIBTPU_INIT_ARGS"
  exec "${TRAINER_CMD[@]}" > "$TRAINER_LOG" 2>&1
) &
TRAINER_PID=$!

echo "Launching Rollout worker on TPU chips $ROLLOUT_TPU_CHIPS..."
(
  ROLLOUT_CMD=(
    "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
    --discovery_addrs="${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT}"
    --process_main=tunix.experimental.examples.common.run_rollout_node.main
    --port="$ROLLOUT_PORT"
    --model_id="$MODEL_ID"
    --model_dir="$MODEL_DIR"
    --model_name="$MODEL_NAME"
    --sampler="$SAMPLER"
    --mesh_fsdp="$ROLLOUT_FSDP"
    --mesh_tp="$ROLLOUT_TP"
    --tensor_parallel_size="$ROLLOUT_MESH_TP"
    --tokenizer_path="$TOKENIZER_PATH"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --weight_sync_mode="$WEIGHT_SYNC_MODE"
    --maxtext_model_name="$MAXTEXT_MODEL_NAME"
    --prefuse_moe_weights="$ROLLOUT_PREFUSE_MOE_WEIGHTS"
    --enable_prefix_caching="$ENABLE_PREFIX_CACHING"
    --return_routed_experts="$RETURN_ROUTED_EXPERTS"
    --free_kv_cache_during_weight_sync="$ROLLOUT_FREE_KV_CACHE"
    --vllm_config_json="$ROLLOUT_VLLM_CONFIG_JSON"
  )
  if [[ -n "${EOS_TOKENS:-}" ]]; then
    ROLLOUT_CMD+=(--eos_tokens="$EOS_TOKENS")
  fi
  if [[ -n "${ROLLOUT_MAXTEXT_ATTENTION:-}" ]]; then
    ROLLOUT_CMD+=(--maxtext_attention="$ROLLOUT_MAXTEXT_ATTENTION")
  fi
  if [[ "$TASK_MODE" == "deepswe" ]]; then
    ROLLOUT_CMD+=(
      --registry_module=tunix.experimental.examples.deepswe_dist.deepswe
      --env_name=deepswe_env
      --agent_name=deepswe_agent
      --max_concurrency="${ROLLOUT_MAX_CONCURRENCY:-16}"
    )
  else
    ROLLOUT_CMD+=(--chat_parser="${CHAT_PARSER:-auto}")
  fi

  export QWIX_MOE_ONLY_QTYPE="$ROLLOUT_QWIX_MOE_ONLY_QTYPE"
  export QWIX_SKIP_GDN_PROJ="$ROLLOUT_QWIX_SKIP_GDN_PROJ"
  export RAIDEN_DEVICES_PER_HOST="$ACTUAL_ROLLOUT_DEVICES"
  export NEW_MODEL_DESIGN=1
  export ATTN_BUCKETIZED_NUM_REQS=true
  export ATTN_CUSTOM_NUM_REQS_BUCKETS=4
  export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
  export VLLM_MOE_CHUNK_SIZE=256
  export SLICE_ROPE_CACHE=1
  export DP_SCHED_BATCH_PREFILL=false
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  export JAX_PLATFORMS=tpu,cpu
  export SKIP_JAX_PRECOMPILE=1
  export TPU_VISIBLE_DEVICES="$ROLLOUT_TPU_CHIPS"
  export TPU_VISIBLE_CHIPS="$ROLLOUT_TPU_CHIPS"
  if [[ -n "${EFFECTIVE_MEGACORE_ARG:-}" ]]; then
    export TPU_MEGACORE=megacore
  fi
  export LIBTPU_INIT_ARGS="$ROLLOUT_LIBTPU_INIT_ARGS"
  exec "${ROLLOUT_CMD[@]}" > "$ROLLOUT_LOG" 2>&1
) &
ROLLOUT_PID=$!

wait_for_port "trainer" "$TRAINER_PORT" "$TRAINER_PID" "$TRAINER_LOG"
wait_for_port "rollout" "$ROLLOUT_PORT" "$ROLLOUT_PID" "$ROLLOUT_LOG"

echo "Launching CPU Orchestrator ($TASK_MODE)..."
(
  if [[ "$TASK_MODE" == "gsm8k" ]]; then
    PROCESS_MAIN="tunix.experimental.examples.math_gsm8k_dist.run_gsm8k_dist_grpo.main"
  elif [[ "$TASK_MODE" == "deepswe" ]]; then
    PROCESS_MAIN="tunix.experimental.examples.deepswe_dist.run_deepswe_dist.main"
  else
    echo "Error: Unsupported TASK_MODE='$TASK_MODE' (expected 'gsm8k' or 'deepswe')." >&2
    exit 1
  fi

  ORCHESTRATOR_CMD=(
    "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
    --discovery_id="$ORCHESTRATOR_ID"
    --discovery_port="$ORCHESTRATOR_PORT"
    --process_main="$PROCESS_MAIN"
    --model_id="$MODEL_ID"
    --tokenizer_path="$TOKENIZER_PATH"
    --batch_size="$BATCH_SIZE"
    --mini_batch_size="$MINI_BATCH_SIZE"
    --num_generations="$NUM_GENERATIONS"
    --temperature="$TEMPERATURE"
    --top_p="$TOP_P"
    --top_k="$TOP_K"
    --max_steps="$MAX_STEPS"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
    --beta="$BETA"
    --epsilon="$EPSILON"
    --epsilon_high="$EPSILON_HIGH"
    --loss_agg_mode="$LOSS_AGG_MODE"
    --advantage_estimator="$ADVANTAGE_ESTIMATOR"
    --seq_logprob_error_threshold="$SEQ_LOGPROB_ERROR_THRESHOLD"
    --truncated_importance_sampling_type="$TRUNCATED_IMPORTANCE_SAMPLING_TYPE"
    --truncated_importance_sampling_ratio_min="$TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN"
    --truncated_importance_sampling_ratio="$TRUNCATED_IMPORTANCE_SAMPLING_RATIO"
    --sampler_is_length_buckets="$SAMPLER_IS_LENGTH_BUCKETS"
    --max_seq_token_per_tpu="$MAX_SEQ_TOKEN_PER_TPU"
    --max_segments_per_packed_row="$MAX_SEGMENTS_PER_PACKED_ROW"
    --trainer_fsdp="$TRAINER_FSDP"
    --trainer_expert="$TRAINER_EXPERT"
    --weight_sync_mode="$WEIGHT_SYNC_MODE"
    --log_dir="$LOG_DIR"
    --no-shuffle
    --stop_workers_on_exit
  )
  if [[ "$OVERLONG_LOSS_MASKING" == "1" || "$OVERLONG_LOSS_MASKING" == "true" ]]; then
    ORCHESTRATOR_CMD+=(--overlong_loss_masking)
  fi
  if [[ "$USE_ROLLOUT_LOGPS" == "false" || "$USE_ROLLOUT_LOGPS" == "0" ]]; then
    ORCHESTRATOR_CMD+=(--no-use_rollout_logps)
  else
    ORCHESTRATOR_CMD+=(--use_rollout_logps)
  fi
  if [[ "$TASK_MODE" == "gsm8k" ]]; then
    DEFAULT_TFDS_EXAMPLES=$(( MAX_STEPS * BATCH_SIZE * GSM8K_TURNS > 4 ? MAX_STEPS * BATCH_SIZE * GSM8K_TURNS : 4 ))
    ORCHESTRATOR_CMD+=(
      --tfds_split="${TFDS_SPLIT:-train[:${DEFAULT_TFDS_EXAMPLES}]}"
      --reward_mode="${REWARD_MODE:-env}"
    )
  else
    ORCHESTRATOR_CMD+=(
      --max_turns="${MAX_TURNS:-5}"
      --env_backend="${ENV_BACKEND:-docker}"
    )
  fi

  export JAX_PLATFORMS=cpu
  "${ORCHESTRATOR_CMD[@]}" > "$ORCHESTRATOR_LOG" 2>&1
) || {
  exit_code="$?"
  echo "Error: CPU orchestrator failed with exit code $exit_code." >&2
  tail -n 80 "$ORCHESTRATOR_LOG" >&2 2>/dev/null || true
  exit "$exit_code"
}

analyze_orchestrator_log \
  "$ORCHESTRATOR_LOG" \
  "$MAX_OOB_RATIO" \
  "$MIN_KEPT_FRAC" \
  "$SAMPLER_QUANT" \
  "$TRAINER_QUANT" \
  "$MAX_GEOMEAN_DRIFT"
