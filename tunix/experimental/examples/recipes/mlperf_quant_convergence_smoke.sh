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
# Standalone, single-host multi-step convergence & quantization parity test for
# mlperf_35b_128_v5p.sh + mlperf_base.sh.
#
# Self-contained: requires NO modifications to any existing scripts or Python
# modules in tunix. Launches the Trainer worker, vLLM Rollout worker, and CPU
# Orchestrator directly on a single 8-chip TPU host (chips 0..3 Trainer,
# chips 4..7 Rollout) with all MLPerf GRPO/TIS/router-replay stability gates
# enabled, and supports independent quantization modes for Sampler and Trainer.
#
# Usage examples:
#   # 1. Baseline bf16 Sampler + bf16 Trainer on Qwen3.5-35B-A3B (single 8-chip host):
#   bash tunix/experimental/examples/recipes/mlperf_quant_convergence_smoke.sh
#
#   # 2. Test FP8 MoE Sampler + BF16 Trainer:
#   SAMPLER_QUANT=fp8_moe TRAINER_QUANT=bf16 \
#     bash tunix/experimental/examples/recipes/mlperf_quant_convergence_smoke.sh
#
#   # 3. Test INT8 MoE Sampler + BF16 Trainer:
#   SAMPLER_QUANT=int8_moe TRAINER_QUANT=bf16 \
#     bash tunix/experimental/examples/recipes/mlperf_quant_convergence_smoke.sh
#
#   # 4. Test FP8 MoE on both Sampler and Trainer:
#   SAMPLER_QUANT=fp8_moe TRAINER_QUANT=fp8_moe \
#     bash tunix/experimental/examples/recipes/mlperf_quant_convergence_smoke.sh
#
#   # 5. Use an existing GCS MaxText checkpoint (skips 70GB HF safetensors download):
#   MAXTEXT_CKPT=gs://tunix-MaxText-Prod/qwen3.5/35b/bf16/0/items \
#   SAMPLER_QUANT=fp8_moe TRAINER_QUANT=bf16 \
#     bash tunix/experimental/examples/recipes/mlperf_quant_convergence_smoke.sh
#
#   # 6. Analyze an existing orchestrator.log without launching TPU workers:
#   bash tunix/experimental/examples/recipes/mlperf_quant_convergence_smoke.sh \
#     --analyze-only /path/to/orchestrator.log

set -Eeuo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RECIPE_DIR}/../../../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}
export HF_TOKEN=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}

MAX_OOB_RATIO=${MAX_OOB_RATIO:-0.50}
MIN_KEPT_FRAC=${MIN_KEPT_FRAC:-0.50}
MAX_GEOMEAN_DRIFT=${MAX_GEOMEAN_DRIFT:-0.005}

# ==============================================================================
# Built-in Log Analyzer
# ==============================================================================
analyze_orchestrator_log() {
  local log_file="$1"
  local max_oob="$2"
  local min_kept="$3"
  local sampler_q="${4:-unknown}"
  local trainer_q="${5:-unknown}"
  local max_geomean_drift="${6:-0.005}"

  "$PYTHON_BIN" - "$log_file" "$max_oob" "$min_kept" "$sampler_q" "$trainer_q" "$max_geomean_drift" <<'PY'
import ast
import math
import re
import sys

log_path = sys.argv[1]
max_oob_ratio = float(sys.argv[2])
min_kept_frac = float(sys.argv[3])
sampler_quant = sys.argv[4]
trainer_quant = sys.argv[5]
max_geomean_drift = float(sys.argv[6])

step_metrics_re = re.compile(
    r"\[StepMetrics step=(\d+)\]\s+trainer_metrics=(\{.*\})"
)
train_step_re = re.compile(
    r"Train step\s+(\d+)\s+-\s+loss:\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf|-inf)"
    r"\s+-\s+reward_mean:\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf|-inf)"
    r"\s+-\s+advantage_mean:\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf|-inf)"
    r"\s+-\s+grad_norm:\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf|-inf)"
    r"\s+-\s+perplexity:\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf|-inf)"
    r"\s+-\s+step_time:\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)s"
)
step_end_re = re.compile(
    r"<<< Step\s+(\d+)\s+finished\s+\|\s+Advanced to Policy Version:\s+(\d+)"
)

steps = {}

with open(log_path, "r", encoding="utf-8", errors="replace") as f:
  for line in f:
    m_metrics = step_metrics_re.search(line)
    if m_metrics:
      step = int(m_metrics.group(1))
      entry = steps.setdefault(step, {})
      try:
        parsed = eval(
            m_metrics.group(2),
            {"__builtins__": {}},
            {"nan": float("nan"), "inf": float("inf")},
        )
        if isinstance(parsed, dict):
          entry["trainer_metrics"] = parsed
      except Exception:
        pass
      continue

    m_train = train_step_re.search(line)
    if m_train:
      step = int(m_train.group(1))
      entry = steps.setdefault(step, {})
      entry["loss"] = float(m_train.group(2))
      entry["reward_mean"] = float(m_train.group(3))
      entry["advantage_mean"] = float(m_train.group(4))
      entry["grad_norm"] = float(m_train.group(5))
      entry["perplexity"] = float(m_train.group(6))
      entry["step_time"] = float(m_train.group(7))
      continue

    m_end = step_end_re.search(line)
    if m_end:
      step = int(m_end.group(1))
      entry = steps.setdefault(step, {})
      entry["next_policy_version"] = int(m_end.group(2))

if not steps:
  print(f"ERROR: No training steps found in {log_path}", file=sys.stderr)
  sys.exit(2)

print()
print("=====================================================================================================================")
print(f" MLPerf RL Quantization & Convergence Summary (SAMPLER_QUANT={sampler_quant}, TRAINER_QUANT={trainer_quant})")
print(f" Log: {log_path}")
print("=====================================================================================================================")
header = (
    f"{'Step':>4} | {'PolVer':>6} | {'Reward':>8} | {'Loss':>10} | "
    f"{'GradNorm':>9} | {'Geomean':>8} | {'TIS OOB':>8} | {'KeptFrac':>8} | "
    f"{'MultProbErr':>11} | {'TokLogDiff':>10} | {'Time(s)':>7}"
)
print(header)
print("-" * len(header))

failures = []
ordered_steps = sorted(steps.keys())

for step in ordered_steps:
  info = steps[step]
  tm = info.get("trainer_metrics", {})
  pol_ver = info.get("next_policy_version", step + 1)
  reward = info.get("reward_mean", float("nan"))
  loss = info.get("loss", float(tm.get("loss", float("nan"))))
  grad_norm = info.get("grad_norm", float(tm.get("grad_norm", float("nan"))))
  geomean = float(tm.get("sampler_is/seq_geomean_mean", float("nan")))
  tis_oob = float(tm.get("tis/is_oob_ratio", float("nan")))
  kept_frac = float(tm.get("sample_mask/kept_frac", float("nan")))
  mult_err = float(tm.get("sample_mask/mult_prob_error_mean", float("nan")))
  tok_diff = float(tm.get("sampler_is/token_logdiff_absmean", float("nan")))
  step_time = info.get("step_time", float("nan"))

  print(
      f"{step:>4d} | {pol_ver:>6d} | {reward:>8.4f} | {loss:>10.5f} | "
      f"{grad_norm:>9.4f} | {geomean:>8.5f} | {tis_oob:>8.4f} | {kept_frac:>8.4f} | "
      f"{mult_err:>11.5f} | {tok_diff:>10.5f} | {step_time:>7.1f}"
  )

  if not math.isfinite(loss):
    failures.append(f"Step {step}: non-finite loss ({loss})")
  if not math.isfinite(grad_norm):
    failures.append(f"Step {step}: non-finite grad_norm ({grad_norm})")
  if math.isfinite(geomean) and abs(geomean - 1.0) > max_geomean_drift:
    failures.append(
        f"Step {step}: |seq_geomean_mean - 1.0|={abs(geomean - 1.0):.5f} > MAX_GEOMEAN_DRIFT={max_geomean_drift:.5f} "
        f"(systematic sampler/trainer sequence ratio bias: {geomean:.5f})"
    )
  if math.isfinite(tis_oob) and tis_oob > max_oob_ratio:
    failures.append(
        f"Step {step}: tis/is_oob_ratio={tis_oob:.4f} > MAX_OOB_RATIO={max_oob_ratio:.4f} "
        "(sampler/trainer sequence drift exceeds TIS [0.999, 1.002] gate)"
    )
  if math.isfinite(kept_frac) and kept_frac < min_kept_frac:
    failures.append(
        f"Step {step}: sample_mask/kept_frac={kept_frac:.4f} < MIN_KEPT_FRAC={min_kept_frac:.4f} "
        "(too many trajectories rejected by TIS / seq_logprob_error_threshold)"
    )

print("=====================================================================================================================")
first_reward = steps[ordered_steps[0]].get("reward_mean", float("nan"))
last_reward = steps[ordered_steps[-1]].get("reward_mean", float("nan"))
print(
    f"Steps completed: {len(ordered_steps)} | "
    f"Reward: {first_reward:.4f} (step {ordered_steps[0]}) -> {last_reward:.4f} (step {ordered_steps[-1]})"
)

if failures:
  print("\nSTATUS: FAIL - Quantization/convergence gate violations detected:")
  for msg in failures:
    print(f"  - {msg}")
  sys.exit(1)
else:
  print(
      f"\nSTATUS: PASS - All {len(ordered_steps)} step(s) satisfied TIS & numerical health gates "
      f"(|geomean - 1.0| <= {max_geomean_drift}, tis/is_oob_ratio <= {max_oob_ratio}, sample_mask/kept_frac >= {min_kept_frac})."
  )
PY
}

if [[ "${1:-}" == "--analyze-only" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 --analyze-only /path/to/orchestrator.log" >&2
    exit 1
  fi
  analyze_orchestrator_log "$2" "$MAX_OOB_RATIO" "$MIN_KEPT_FRAC" "${SAMPLER_QUANT:-unknown}" "${TRAINER_QUANT:-unknown}" "$MAX_GEOMEAN_DRIFT"
  exit 0
fi

# ==============================================================================
# 1. Quantization Configuration (SAMPLER_QUANT / TRAINER_QUANT)
# ==============================================================================
# Supported modes:
#   bf16     : Unquantized bfloat16 (MLPerf baseline)
#   fp8_moe  : Qwix dynamic float8_e4m3fn on MoE (gmm/ragged_dot) only; dense/attn/router stay bf16
#   int8_moe : Qwix dynamic int8 on MoE (gmm/ragged_dot) only; dense/attn/router stay bf16
#   fp8_full : Qwix dynamic float8_e4m3fn on all Linear/Einsum/MoE ops
#   fp8      : Qwix/AQT fp8 rule
#   int8     : Qwix/AQT int8 rule
#   fp8_ckpt : Pre-quantized block-wise FP8 checkpoint (use_qwix_quantization=false);
#              requires both SAMPLER_QUANT=fp8_ckpt and TRAINER_QUANT=fp8_ckpt.
export SAMPLER_QUANT=${SAMPLER_QUANT:-bf16}
export TRAINER_QUANT=${TRAINER_QUANT:-bf16}

if [[ "$SAMPLER_QUANT" == "fp8_ckpt" || "$SAMPLER_QUANT" == "fp8_serve" || \
      "$TRAINER_QUANT" == "fp8_ckpt" || "$TRAINER_QUANT" == "fp8_serve" ]]; then
  if [[ "$SAMPLER_QUANT" != "$TRAINER_QUANT" ]]; then
    echo "Error: fp8_ckpt/fp8_serve stores pre-quantized 1-byte FP8 weights in the model state," >&2
    echo "so Raiden weight sync requires BOTH SAMPLER_QUANT and TRAINER_QUANT to be fp8_ckpt." >&2
    echo "To test mixed sampler/trainer quantization (e.g. FP8 sampler + BF16 trainer)," >&2
    echo "use SAMPLER_QUANT=fp8_moe or SAMPLER_QUANT=fp8_full with TRAINER_QUANT=bf16." >&2
    exit 1
  fi
fi

ROLLOUT_QUANT_FLAGS=""
ROLLOUT_QWIX_MOE_ONLY_QTYPE=""
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
  fp8_full)
    ROLLOUT_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    ;;
  fp8)
    ROLLOUT_QUANT_FLAGS="quantization=fp8 use_qwix_quantization=true quantize_router_proj=false"
    ;;
  int8)
    ROLLOUT_QUANT_FLAGS="quantization=int8 use_qwix_quantization=true quantize_router_proj=false"
    ;;
  fp8_ckpt|fp8_serve)
    ROLLOUT_QUANT_FLAGS="quantization=fp8 use_qwix_quantization=false quantize_router_proj=false"
    ;;
  *)
    echo "Error: Unsupported SAMPLER_QUANT='$SAMPLER_QUANT' (expected bf16, fp8_moe, int8_moe, fp8_full, fp8, int8, fp8_ckpt)." >&2
    exit 1
    ;;
esac

TRAINER_QUANT_FLAGS=""
TRAINER_QWIX_MOE_ONLY_QTYPE=""
case "$TRAINER_QUANT" in
  bf16|none)
    TRAINER_QUANT_FLAGS="quantization= use_qwix_quantization=false"
    ;;
  fp8_moe)
    TRAINER_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    TRAINER_QWIX_MOE_ONLY_QTYPE="float8_e4m3fn"
    ;;
  int8_moe)
    TRAINER_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    TRAINER_QWIX_MOE_ONLY_QTYPE="int8"
    ;;
  fp8_full)
    TRAINER_QUANT_FLAGS="quantization=fp8_full use_qwix_quantization=true quantize_router_proj=false"
    ;;
  fp8)
    TRAINER_QUANT_FLAGS="quantization=fp8 use_qwix_quantization=true quantize_router_proj=false"
    ;;
  int8)
    TRAINER_QUANT_FLAGS="quantization=int8 use_qwix_quantization=true quantize_router_proj=false"
    ;;
  fp8_ckpt|fp8_serve)
    TRAINER_QUANT_FLAGS="quantization=fp8 use_qwix_quantization=false quantize_router_proj=false"
    ;;
  *)
    echo "Error: Unsupported TRAINER_QUANT='$TRAINER_QUANT' (expected bf16, fp8_moe, int8_moe, fp8_full, fp8, int8, fp8_ckpt)." >&2
    exit 1
    ;;
esac

# ==============================================================================
# 2. Single-Host TPU Topology Auto-Detection & Model Preset
# ==============================================================================
# Note on TPU VM topologies:
#   - v5p-8 and v4-8 VMs have 4 physical TPU chips (0,1,2,3), each with 2
#     TensorCores fused into 1 Megacore (96 GB HBM per chip on v5p). Splitting
#     a single host between Trainer and Rollout assigns 2 chips (0,1) to Trainer
#     and 2 chips (2,3) to Rollout with TPU_CHIPS_PER_HOST_BOUNDS=1,2,1.
#     Critically, when TPU_CHIPS_PER_HOST_BOUNDS is set in the environment,
#     libtpu skips its automatic GetChipConfigNameToOverride() check unless
#     --deepsea_chip_config_name=megacore is explicitly passed in LIBTPU_INIT_ARGS.
#   - v5e-8 and v6e-8 VMs have 8 physical TPU chips (0..7), 1 core per chip.
#     Splitting a single host assigns 4 chips (0,1,2,3) to Trainer and 4 chips
#     (4,5,6,7) to Rollout with TPU_CHIPS_PER_HOST_BOUNDS=1,4,1.
export TASK_MODE=${TASK_MODE:-gsm8k}   # gsm8k (no K8s/Docker needed) or deepswe
export MODEL_NAME=${MODEL_NAME:-Qwen3.5-35B-A3B}
export TRAINER_BACKEND=${TRAINER_BACKEND:-maxtext}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-raiden}

ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-orchestrator}
ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-30000}
TRAINER_PORT=${TRAINER_PORT:-20000}
ROLLOUT_PORT=${ROLLOUT_PORT:-20001}

detect_host_tpu_topology() {
  "$PYTHON_BIN" - <<'PY'
import os
import subprocess
import sys

code = (
    "import jax; "
    "devs = jax.devices(); "
    "chips = len({getattr(d, 'coords', i) for i, d in enumerate(devs)}); "
    "kind = devs[0].device_kind if devs else 'unknown'; "
    "print(f'{chips}|{len(devs)}|{kind}')"
)
env = os.environ.copy()
env["JAX_PLATFORMS"] = "tpu,cpu"
for k in (
    "TPU_VISIBLE_DEVICES",
    "TPU_VISIBLE_CHIPS",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
    "TPU_HOST_BOUNDS",
    "LIBTPU_INIT_ARGS",
):
  env.pop(k, None)
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
  if "|" in last:
    print(last)
    sys.exit(0)
print("4|8|unknown")
PY
}

rm -f /tmp/libtpu_lockfile 2>/dev/null || true
IFS='|' read -r HOST_PHYSICAL_TPU_CHIPS HOST_TOTAL_JAX_DEVICES HOST_TPU_KIND < <(detect_host_tpu_topology)

if (( HOST_PHYSICAL_TPU_CHIPS >= 8 )); then
  export TRAINER_TPU_CHIPS=${TRAINER_TPU_CHIPS:-0,1,2,3}
  export ROLLOUT_TPU_CHIPS=${ROLLOUT_TPU_CHIPS:-4,5,6,7}
  export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-1,4,1}
  export TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS:-1,1,1}
  DEFAULT_MEGACORE_ARG=""
else
  # 4-chip host (e.g. TPU7x / v7x-8, v5p-8, v4-8):
  #   - Trainer gets physical chips 0,1; Rollout gets physical chips 2,3
  #   - TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 (must NOT be 2,2,1 or libtpu locks /tmp/libtpu_lockfile)
  export TRAINER_TPU_CHIPS=${TRAINER_TPU_CHIPS:-0,1}
  export ROLLOUT_TPU_CHIPS=${ROLLOUT_TPU_CHIPS:-2,3}
  export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-1,2,1}
  export TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS:-1,1,1}
  if [[ "$HOST_TPU_KIND" == *"7x"* || "$HOST_TPU_KIND" == *"v5e"* || "$HOST_TPU_KIND" == *"v6e"* ]]; then
    DEFAULT_MEGACORE_ARG=""
  else
    DEFAULT_MEGACORE_ARG="--deepsea_chip_config_name=megacore"
  fi
fi
export TPU_CHIPS_PER_PROCESS_BOUNDS=${TPU_CHIPS_PER_PROCESS_BOUNDS:-$TPU_CHIPS_PER_HOST_BOUNDS}

BASE_XLA_TPU_FLAGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false"

# Probe the actual JAX device count visible to the Trainer worker with these
# LIBTPU_INIT_ARGS so that mesh dimensions are guaranteed to match jax.device_count().
probe_worker_tpu_env() {
  local chips="$1"
  local bounds="$2"
  local host_bounds="$3"
  local try_megacore="$4"
  "$PYTHON_BIN" - "$chips" "$bounds" "$host_bounds" "$try_megacore" "$BASE_XLA_TPU_FLAGS" <<'PY'
import os
import subprocess
import sys

chips, bounds, host_bounds, try_megacore, base_xla = sys.argv[1:6]

def _run_probe(megacore_flag: str):
  env = os.environ.copy()
  env["JAX_PLATFORMS"] = "tpu,cpu"
  env["TPU_VISIBLE_DEVICES"] = chips
  env["TPU_VISIBLE_CHIPS"] = chips
  env["TPU_CHIPS_PER_HOST_BOUNDS"] = bounds
  env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = bounds
  env["TPU_HOST_BOUNDS"] = host_bounds
  if megacore_flag:
    env["TPU_MEGACORE"] = "megacore"
  init_args = (
      f"{megacore_flag} "
      f"--deepsea_chips_per_host_bounds={bounds} "
      f"--deepsea_host_bounds={host_bounds} "
      f"{base_xla}"
  ).strip()
  env["LIBTPU_INIT_ARGS"] = init_args
  code = (
      "import jax; "
      "devs = jax.devices(); "
      "print(f'{len(devs)}|{devs[0].device_kind if devs else \"unknown\"}')"
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
    last_line = res.stdout.strip().splitlines()[-1]
    if "|" in last_line:
      cnt_str, kind = last_line.split("|", 1)
      return int(cnt_str), kind.strip(), megacore_flag
  return None

out = None
if try_megacore:
  out = _run_probe(try_megacore)
if out is None:
  out = _run_probe("")
if out is None:
  num_chips = len([c for c in chips.split(",") if c.strip()])
  print(f"{num_chips}|unknown|{try_megacore}")
else:
  print(f"{out[0]}|{out[1]}|{out[2]}")
PY
}

IFS='|' read -r ACTUAL_TRAINER_DEVICES DETECTED_TPU_KIND EFFECTIVE_MEGACORE_ARG < <(
  probe_worker_tpu_env "$TRAINER_TPU_CHIPS" "$TPU_CHIPS_PER_HOST_BOUNDS" "$TPU_HOST_BOUNDS" "$DEFAULT_MEGACORE_ARG"
)
ACTUAL_ROLLOUT_DEVICES=${ACTUAL_ROLLOUT_DEVICES:-$ACTUAL_TRAINER_DEVICES}
EXTRA_CHIP_XLA_FLAGS=""
if [[ "$DETECTED_TPU_KIND" == *"7x"* ]]; then
  EXTRA_CHIP_XLA_FLAGS="--xla_tpu_scoped_vmem_limit_kib=65472"
fi
export WORKER_LIBTPU_INIT_ARGS="${EFFECTIVE_MEGACORE_ARG:+$EFFECTIVE_MEGACORE_ARG }--deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${TPU_HOST_BOUNDS} ${BASE_XLA_TPU_FLAGS}${EXTRA_CHIP_XLA_FLAGS:+ $EXTRA_CHIP_XLA_FLAGS}"
export RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST:-$ACTUAL_TRAINER_DEVICES}

if [[ "$MODEL_NAME" == "Qwen3.5-35B-A3B" ]]; then
  export SAMPLER=${SAMPLER:-vllm}
  export MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}
  export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3.5-35b-a3b}
  export MODEL_DIR=${MODEL_DIR:-/tmp/models/Qwen_Qwen3.5-35B-A3B}
  export TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_DIR}
  export MAXTEXT_CKPT=${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}

  # Match Trainer mesh (FSDP * TP * EP * CONTEXT) to ACTUAL_TRAINER_DEVICES
  if (( ACTUAL_TRAINER_DEVICES <= 2 )); then
    export TRAINER_FSDP=${TRAINER_FSDP:-1}
    export TRAINER_TP=${TRAINER_TP:-$ACTUAL_TRAINER_DEVICES}
  else
    export TRAINER_FSDP=${TRAINER_FSDP:-$((ACTUAL_TRAINER_DEVICES / 2))}
    export TRAINER_TP=${TRAINER_TP:-2}
  fi
  export TRAINER_EXPERT=${TRAINER_EXPERT:-1}
  export TRAINER_CONTEXT=${TRAINER_CONTEXT:-1}
  export REMAT_POLICY=${REMAT_POLICY:-full}
  export TRAINER_MAXTEXT_ATTENTION=${TRAINER_MAXTEXT_ATTENTION:-flash}
  export TRAINER_BASE_NUM_KV_HEADS=${TRAINER_BASE_NUM_KV_HEADS:-2}
  export USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER:-true}
  export TRAINER_PREFUSE_MOE_WEIGHTS=${TRAINER_PREFUSE_MOE_WEIGHTS:-true}
  export EOS_TOKENS=${EOS_TOKENS:-248046,248044}

  # Match Rollout EP mesh to ACTUAL_ROLLOUT_DEVICES (matching mlperf_35b_128_v5p.sh's EP-only rollout)
  export ROLLOUT_FSDP=${ROLLOUT_FSDP:-1}
  export ROLLOUT_TP=${ROLLOUT_TP:-1}
  export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-1}
  export ROLLOUT_EXPERT=${ROLLOUT_EXPERT:-$ACTUAL_ROLLOUT_DEVICES}
  export VLLM_ENABLE_EXPERT_PARALLEL=${VLLM_ENABLE_EXPERT_PARALLEL:-true}
  export VLLM_DATA_PARALLEL_SIZE=${VLLM_DATA_PARALLEL_SIZE:-1}
  export MAMBA_CACHE_SPLIT=${MAMBA_CACHE_SPLIT:-2}
  export ROLLOUT_SHARDING_JSON=${ROLLOUT_SHARDING_JSON:-"{\"additional_config\":{\"sharding\":{\"sharding_strategy\":{\"expert_parallelism\":${ROLLOUT_EXPERT},\"tensor_parallelism\":1,\"enable_dp_attention\":true}},\"custom_mamba_cache_multiplier\":${MAMBA_CACHE_SPLIT},\"maxtext_config\":{\"scan_layers\":false,\"attention\":\"vllm_rpa\",\"allow_split_physical_axes\":true,\"use_multimodal\":false,\"prefuse_moe_weights\":true}}}"}
  export ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS:-true}
  export RETURN_ROUTED_EXPERTS=${RETURN_ROUTED_EXPERTS:-true}

  # On a single host (<=4 Trainer devices), all 40 layers execute in forward/backward
  # (including Qwix MoE quantization and router replay on all 256 routed experts),
  # while freezing the massive 32.2B routed_experts tensors in AdamW so optimizer
  # state + gradients fit comfortably in single-host HBM. Set
  # TRAINABLE_PARAMETERS_MASK='^(?!.*routed_experts/gate/kernel).*' on >=8 devices.
  if (( ACTUAL_TRAINER_DEVICES <= 4 )); then
    export TRAINABLE_PARAMETERS_MASK=${TRAINABLE_PARAMETERS_MASK:-'^(?!.*routed_experts/gate/kernel)(?!.*mlp/routed_experts/(wi|wi_0|wi_1|wo)).*'}
    LOW_MEM_TRAINER_FLAGS="grad_dtype=bfloat16 mu_dtype=bfloat16 optimizer_memory_host_offload=true"
  else
    export TRAINABLE_PARAMETERS_MASK=${TRAINABLE_PARAMETERS_MASK:-'^(?!.*routed_experts/gate/kernel).*'}
    LOW_MEM_TRAINER_FLAGS=""
  fi

  EFFECTIVE_SEQ_LEN=${MAX_SEQ_TOKEN_PER_TPU:-1024}
  SA_BLOCK_KV=$(( EFFECTIVE_SEQ_LEN < 4096 ? EFFECTIVE_SEQ_LEN : 4096 ))
  SA_BLOCK_Q=$(( SA_BLOCK_KV < 1024 ? SA_BLOCK_KV / 2 : 1024 ))
  SA_BLOCK_COMPUTE=$(( SA_BLOCK_Q < 512 ? SA_BLOCK_Q : 512 ))

  # Qwen3.5 GDN & Tokamax Splash Attention numerical stability flags (with use_gdn_kernel=false so remat_policy=full rematerializes GDN intermediates on single host)
  BASE_MAXTEXT_FLAGS="float32_gate_logits=true float32_logits=true use_gdn_kernel=false megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true use_gmm_v2_heuristic_tiling=true merge_gating_gmm=false use_custom_sort_vjp=false use_tokamax_splash=true use_splash_scheduler=true sa_block_q=${SA_BLOCK_Q} sa_block_kv=${SA_BLOCK_KV} sa_block_kv_compute=${SA_BLOCK_COMPUTE} sa_block_q_dkv=${SA_BLOCK_Q} sa_block_kv_dkv=${SA_BLOCK_KV} sa_block_kv_dkv_compute=${SA_BLOCK_COMPUTE} sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3 allow_split_physical_axes=false num_vocab_tiling=8 use_iota_embed=false${LOW_MEM_TRAINER_FLAGS:+ $LOW_MEM_TRAINER_FLAGS}"
  ROLLOUT_BASE_MAXTEXT_FLAGS="float32_gate_logits=true float32_logits=true megablox=False sparse_matmul=False"
else
  # Dense proxy model (e.g. MODEL_NAME=Qwen3-0.6B or Qwen3-4B)
  export SAMPLER=${SAMPLER:-inprocess_vllm}
  export MODEL_ID=${MODEL_ID:-Qwen/${MODEL_NAME}}
  export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')}
  export MODEL_DIR=${MODEL_DIR:-/tmp/models/Qwen_${MODEL_NAME}}
  export TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_DIR}
  export MAXTEXT_CKPT=${MAXTEXT_CKPT:-${MODEL_DIR}_maxtext/0/items}

  if (( ACTUAL_TRAINER_DEVICES <= 2 )); then
    export TRAINER_FSDP=${TRAINER_FSDP:-1}
    export TRAINER_TP=${TRAINER_TP:-$ACTUAL_TRAINER_DEVICES}
  else
    export TRAINER_FSDP=${TRAINER_FSDP:-2}
    export TRAINER_TP=${TRAINER_TP:-$((ACTUAL_TRAINER_DEVICES / 2))}
  fi
  export TRAINER_EXPERT=${TRAINER_EXPERT:-1}
  export TRAINER_CONTEXT=${TRAINER_CONTEXT:-1}
  export REMAT_POLICY=${REMAT_POLICY:-full}
  export TRAINER_MAXTEXT_ATTENTION=${TRAINER_MAXTEXT_ATTENTION:-autoselected}
  export TRAINER_BASE_NUM_KV_HEADS=${TRAINER_BASE_NUM_KV_HEADS:-}
  export USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER:-false}
  export TRAINER_PREFUSE_MOE_WEIGHTS=${TRAINER_PREFUSE_MOE_WEIGHTS:-false}
  export EOS_TOKENS=${EOS_TOKENS:-}

  export ROLLOUT_FSDP=${ROLLOUT_FSDP:-1}
  export ROLLOUT_TP=${ROLLOUT_TP:-$ACTUAL_ROLLOUT_DEVICES}
  export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-$ACTUAL_ROLLOUT_DEVICES}
  export ROLLOUT_EXPERT=${ROLLOUT_EXPERT:-1}
  export ROLLOUT_SHARDING_JSON=${ROLLOUT_SHARDING_JSON:-"{}"}
  export ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS:-false}
  export RETURN_ROUTED_EXPERTS=${RETURN_ROUTED_EXPERTS:-false}
  export TRAINABLE_PARAMETERS_MASK=${TRAINABLE_PARAMETERS_MASK:-}

  BASE_MAXTEXT_FLAGS="float32_logits=true"
  ROLLOUT_BASE_MAXTEXT_FLAGS="float32_logits=true"
fi

export MAXTEXT_EXTRA_FLAGS="${BASE_MAXTEXT_FLAGS} ${TRAINER_QUANT_FLAGS}${MAXTEXT_EXTRA_FLAGS:+ ${MAXTEXT_EXTRA_FLAGS}}"
export ROLLOUT_MAXTEXT_EXTRA_FLAGS="${ROLLOUT_BASE_MAXTEXT_FLAGS} ${ROLLOUT_QUANT_FLAGS}${ROLLOUT_MAXTEXT_EXTRA_FLAGS:+ ${ROLLOUT_MAXTEXT_EXTRA_FLAGS}}"

# ==============================================================================
# 3. MLPerf GRPO + TIS + Sequence Mask Hyperparameters (from mlperf_base.sh)
# ==============================================================================
if (( ACTUAL_TRAINER_DEVICES <= 2 )) && [[ "$MODEL_NAME" == "Qwen3.5-35B-A3B" ]]; then
  export BATCH_SIZE=${BATCH_SIZE:-1}
  export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
else
  export BATCH_SIZE=${BATCH_SIZE:-2}
  export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
fi
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$BATCH_SIZE}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-$TRAINER_FSDP}
export MAX_STEPS=${MAX_STEPS:-${NUM_STEPS:-5}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-1024}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-2}
export COMPUTE_LOGPS_CHUNK_SIZE=${COMPUTE_LOGPS_CHUNK_SIZE:-256}

export LEARNING_RATE=${LEARNING_RATE:-3e-5}
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
export SEQ_LOGPROB_ERROR_THRESHOLD=${SEQ_LOGPROB_ERROR_THRESHOLD:-2.0}
export TRUNCATED_IMPORTANCE_SAMPLING_TYPE=${TRUNCATED_IMPORTANCE_SAMPLING_TYPE:-seq-mask-tis}
if (( MAX_RESPONSE_LENGTH <= 1024 )); then
  # Per algo_core.py (sampler_is_length_bucket_sums), IID per-token logprob noise
  # averages down as 1/sqrt(T). mlperf_base.sh's [0.999, 1.002] band is calibrated
  # for T~4096..32768 tokens; at smoke-test length T<=512 (~9x shorter), 1/sqrt(T)
  # spread is ~3x wider, so [0.997, 1.006] preserves the exact same 3-sigma gate.
  export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN:-0.997}
  export TRUNCATED_IMPORTANCE_SAMPLING_RATIO=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO:-1.006}
else
  export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN:-0.999}
  export TRUNCATED_IMPORTANCE_SAMPLING_RATIO=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO:-1.002}
fi
export SAMPLER_IS_LENGTH_BUCKETS=${SAMPLER_IS_LENGTH_BUCKETS:-512,1024}

# vLLM Rollout Engine settings (matched to mlperf_base.sh, scaled to smoke seq len)
export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-false}
export ROLLOUT_FREE_KV_CACHE=${ROLLOUT_FREE_KV_CACHE:-false}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-$VLLM_MAX_MODEL_LEN}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-4}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.75}
export VLLM_MAMBA_CACHE_MODE=${VLLM_MAMBA_CACHE_MODE:-none}
export VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-bfloat16}
export VLLM_BLOCK_SIZE=${VLLM_BLOCK_SIZE:-256}
export VLLM_ASYNC_SCHEDULING=${VLLM_ASYNC_SCHEDULING:-true}
export VLLM_ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL:-true}
export VLLM_LANGUAGE_MODEL_ONLY=${VLLM_LANGUAGE_MODEL_ONLY:-true}
export VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER:-qwen3}
export VLLM_LIMIT_MM_PER_PROMPT=${VLLM_LIMIT_MM_PER_PROMPT:-'{"image": 0, "video": 0}'}

export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
export CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-1}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-100000}
export WAIT_TIMEOUT_SECS=${WAIT_TIMEOUT_SECS:-1800}
export WAIT_POLL_SECS=${WAIT_POLL_SECS:-5}

RUN_TAG="smoke_${MODEL_NAME}_sq-${SAMPLER_QUANT}_tq-${TRAINER_QUANT}_$(date +%Y%m%d_%H%M%S)"
export LOG_ROOT=${LOG_ROOT:-/tmp/mlperf_quant_smoke/${RUN_TAG}}
export LOG_DIR=${LOG_ROOT}/tb
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-${LOG_ROOT}/maxtext_out}
export CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-${LOG_ROOT}/checkpoints}
mkdir -p "$LOG_ROOT"

TRAINER_LOG="${LOG_ROOT}/trainer.log"
ROLLOUT_LOG="${LOG_ROOT}/rollout.log"
ORCHESTRATOR_LOG="${LOG_ROOT}/orchestrator.log"

# ==============================================================================
# 4. Self-Contained Runtime Hook (sitecustomize.py) & Rollout JSON Builder
# ==============================================================================
# Uses a lazy MetaPathFinder so JAX is NEVER imported before raiden_preload.import_raiden():
#   1. Injects Qwix MoE-only quantization rules (fp8_moe / int8_moe) when
#      maxtext.layers.quantizations is imported.
#   2. Patches tunix.utils.maxtext_utils._build_fp32_master_optimizer_cls so
#      frozen parameters in TRAINABLE_PARAMETERS_MASK do not allocate FP32
#      master_params copies in HBM.
PY_HOOKS_DIR="${LOG_ROOT}/py_hooks"
mkdir -p "$PY_HOOKS_DIR"
cat << 'PY' > "${PY_HOOKS_DIR}/sitecustomize.py"
import importlib.abc
import importlib.machinery
import os
import sys


def _patch_quantizations(quantizations_mod):
  qtype_name = os.environ.get("QWIX_MOE_ONLY_QTYPE", "").strip().lower()
  if not qtype_name:
    return
  import jax.numpy as jnp
  import qwix

  qtype_map = {
      "float8_e4m3fn": jnp.float8_e4m3fn,
      "fp8": jnp.float8_e4m3fn,
      "fp8_e4m3": jnp.float8_e4m3fn,
      "int8": jnp.int8,
  }
  if qtype_name not in qtype_map:
    raise ValueError(f"Unsupported QWIX_MOE_ONLY_QTYPE={qtype_name!r}")
  qtype = qtype_map[qtype_name]
  bwd_quant_env = os.environ.get("QWIX_MOE_BWD_QUANT", "").strip().lower()

  def _moe_only_rule(config):
    use_gmm_v2 = bool(getattr(config, "use_gmm_v2", False))
    if bwd_quant_env in ("1", "true", "yes"):
      enable_bwd_q = True
    elif bwd_quant_env in ("0", "false", "no"):
      enable_bwd_q = False
    else:
      # When use_gmm_v2=True, gmm_v2 forward dynamically quantizes lhs inside the
      # Pallas kernel whenever rhs is quantized (rhs_scale is not None), while
      # keeping act_qtype=None and bwd_qtype=None in _gmm_bwd avoids the 2.2x
      # higher dlhs error and tgmm_v2 sublane alignment constraints.
      enable_bwd_q = not use_gmm_v2

    act_q = qtype if enable_bwd_q else None
    bwd_q = (jnp.float8_e5m2 if qtype == jnp.float8_e4m3fn else qtype) if enable_bwd_q else None
    return [
        qwix.QtRule(
            module_path=".*",
            weight_qtype=qtype,
            act_qtype=act_q,
            bwd_qtype=bwd_q,
            weight_calibration_method=getattr(
                config, "weight_quantization_calibration_method", "absmax"
            ),
            act_calibration_method=getattr(
                config, "act_quantization_calibration_method", "absmax"
            ),
            bwd_calibration_method=getattr(
                config, "bwd_quantization_calibration_method", "absmax"
            ),
            op_names=("gmm", "ragged_dot"),
        )
    ]

  quantizations_mod.get_quantization_rule = _moe_only_rule


def _patch_maxtext_utils(maxtext_utils_mod):
  def _build_mask_aware_fp32_master_optimizer_cls(nnx_mod=None):
    from flax.nnx.training import optimizer as nnx_opt
    import jax
    import jax.numpy as jnp
    from maxtext.optimizers import optimizers as maxtext_optimizers
    import optax

    if nnx_mod is None:
      from flax import nnx as nnx_mod

    opt_state_var_cls = getattr(
        nnx_opt, "OptState", getattr(nnx_mod, "Variable", None)
    )
    to_opt_state_fn = getattr(nnx_opt, "to_opt_state", lambda x: x)

    class Fp32MasterOptimizer(nnx_mod.Optimizer):
      """FP32 master-weight optimizer that skips allocating master copies for frozen params."""

      def __init__(self, model, tx, *, wrt=nnx_mod.Param):
        self.step = opt_state_var_cls(jnp.array(0, dtype=jnp.uint32))
        self.tx = tx
        self.wrt = wrt
        params_state = nnx_mod.state(model, wrt)

        cfg = getattr(model, "config", None) or getattr(
            getattr(model, "base_model", None), "config", None
        )
        patterns = getattr(cfg, "trainable_parameters_mask", None)
        if not patterns:
          raw_mask = os.environ.get("TRAINABLE_PARAMETERS_MASK", "").strip()
          if raw_mask:
            patterns = [raw_mask]

        freeze_mask_fn = (
            maxtext_optimizers.get_path_mask_fn(
                patterns, match_returns_true=False
            )
            if patterns
            else None
        )
        freeze_mask = (
            freeze_mask_fn(params_state)
            if freeze_mask_fn is not None
            else jax.tree.map(lambda _: False, params_state)
        )
        master_params = jax.tree.map(
            lambda x, is_frozen: (
                None
                if (is_frozen or x.dtype == jnp.float32)
                else x.astype(jnp.float32)
            ),
            params_state,
            freeze_mask,
        )
        full_f32_params = jax.tree.map(
            lambda p, mp: p if mp is None else mp,
            params_state,
            master_params,
        )
        inner_state = tx.init(full_f32_params)
        opt_state = {
            "master_params": master_params,
            "inner": inner_state,
        }
        if isinstance(inner_state, dict) and "is_skipped" in inner_state:
          opt_state["is_skipped"] = inner_state["is_skipped"]
        self.opt_state = nnx_mod.data(to_opt_state_fn(opt_state))

      def update(self, model, grads, /, **kwargs):
        param_arrays = nnx_mod.as_pure(nnx_mod.state(model, self.wrt))
        grad_arrays = nnx_mod.as_pure(nnx_mod.state(grads, self.wrt))
        opt_state_arrays = nnx_mod.as_pure(self.opt_state)
        kwargs_arrays = nnx_mod.as_pure(kwargs)

        master_params = opt_state_arrays["master_params"]
        inner_state = opt_state_arrays["inner"]
        full_f32_params = jax.tree.map(
            lambda p, mp: p if mp is None else mp,
            param_arrays,
            master_params,
        )
        grads_f32 = jax.tree.map(
            lambda g, p, mp: (
                g.astype(jnp.float32)
                if (mp is not None or p.dtype == jnp.float32)
                else g
            ),
            grad_arrays,
            param_arrays,
            master_params,
        )
        updates, new_inner = self.tx.update(
            grads_f32, inner_state, full_f32_params, **kwargs_arrays
        )
        updated_f32_params = optax.apply_updates(full_f32_params, updates)
        new_master = jax.tree.map(
            lambda _p, mp, up: up if mp is not None else None,
            param_arrays,
            master_params,
            updated_f32_params,
        )
        new_params = jax.tree.map(
            lambda p, up: up.astype(p.dtype),
            param_arrays,
            updated_f32_params,
        )
        new_opt_state = {
            "master_params": new_master,
            "inner": new_inner,
        }
        if isinstance(new_inner, dict) and "is_skipped" in new_inner:
          new_opt_state["is_skipped"] = new_inner["is_skipped"]

        nnx_mod.update(model, new_params)
        nnx_mod.update(self.opt_state, nnx_mod.state(new_opt_state))
        self.step[...] += 1
        return updates

    return Fp32MasterOptimizer

  maxtext_utils_mod._build_fp32_master_optimizer_cls = (
      _build_mask_aware_fp32_master_optimizer_cls
  )

  orig_create_maxtext_engine = maxtext_utils_mod.create_maxtext_engine

  def _wrapped_create_maxtext_engine(*args, **kwargs):
    engine = orig_create_maxtext_engine(*args, **kwargs)
    orig_record_metrics = engine.record_metrics

    def _checked_record_metrics(key, value, *r_args, **r_kwargs):
      if key == "gradient_norm" and value is not None:
        try:
          import jax
          import numpy as np

          val_host = float(jax.device_get(value))
          is_bad = not (val_host == val_host and abs(val_host) != float("inf"))
          if is_bad or val_host > 100.0:
            acc = getattr(engine, "_accumulated_grads", None)
            fmask = getattr(engine, "_freeze_mask", None)
            if (
                fmask is None
                and getattr(engine, "_freeze_mask_fn", None) is not None
                and acc is not None
            ):
              fmask = engine._freeze_mask_fn(acc)
            if acc is not None:
              bad_trainable = []
              bad_frozen = []
              top_trainable = []
              flat_acc, _ = jax.tree_util.tree_flatten_with_path(acc)
              flat_mask = (
                  jax.tree_util.tree_leaves(fmask)
                  if fmask is not None
                  else [False] * len(flat_acc)
              )
              for (path, leaf), is_frozen in zip(flat_acc, flat_mask):
                path_str = "/".join(
                    str(getattr(p, "key", getattr(p, "name", p)))
                    for p in path
                )
                shards = getattr(leaf, "addressable_shards", None)
                if shards:
                  bad_devs = []
                  max_shard_norm = 0.0
                  for s in shards:
                    arr = np.asarray(s.data)
                    if not np.all(np.isfinite(arr)):
                      n_nan = int(np.isnan(arr).sum())
                      n_inf = int(np.isinf(arr).sum())
                      bad_devs.append(
                          f"d{s.device.id}(nan={n_nan},inf={n_inf}/{arr.size})"
                      )
                    elif not is_frozen:
                      snorm = float(np.linalg.norm(arr.astype(np.float32)))
                      if snorm > max_shard_norm:
                        max_shard_norm = snorm
                  if not is_frozen and max_shard_norm > 0.0:
                    top_trainable.append((max_shard_norm, f"{path_str}({leaf.dtype})"))
                  if bad_devs:
                    entry = f"{path_str}({leaf.dtype})[{','.join(bad_devs)}]"
                    if is_frozen:
                      bad_frozen.append(entry)
                    else:
                      bad_trainable.append(entry)
              top_trainable.sort(reverse=True)
              top_str = [f"{k}:{v:.3e}" for v, k in top_trainable[:10]]
              print(
                  f"[GradInspector step={engine.train_step}]"
                  f" grad_norm={val_host}"
                  f" top_trainable={top_str}"
                  f" bad_trainable({len(bad_trainable)})={bad_trainable[:20]}"
                  f" bad_frozen({len(bad_frozen)})={bad_frozen[:10]}",
                  flush=True,
              )
        except Exception as e:
          print(f"[GradInspector] error inspecting grads: {e}", flush=True)
      return orig_record_metrics(key, value, *r_args, **r_kwargs)

    engine.record_metrics = _checked_record_metrics
    return engine

  maxtext_utils_mod.create_maxtext_engine = _wrapped_create_maxtext_engine


class _SmokeTestPatchFinder(importlib.abc.MetaPathFinder):

  def __init__(self):
    self._active = set()

  def find_spec(self, fullname, path, target=None):
    if fullname not in (
        "maxtext.layers.quantizations",
        "tunix.utils.maxtext_utils",
    ):
      return None
    if fullname in self._active:
      return None
    self._active.add(fullname)
    try:
      spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
    finally:
      self._active.discard(fullname)
    if spec is None or spec.loader is None:
      return None
    orig_loader = spec.loader

    class _Loader(importlib.abc.Loader):

      def create_module(self, s):
        if hasattr(orig_loader, "create_module"):
          return orig_loader.create_module(s)
        return None

      def exec_module(self, module):
        orig_loader.exec_module(module)
        if fullname == "maxtext.layers.quantizations":
          _patch_quantizations(module)
        elif fullname == "tunix.utils.maxtext_utils":
          _patch_maxtext_utils(module)

    spec.loader = _Loader()
    return spec


sys.meta_path.insert(0, _SmokeTestPatchFinder())
PY

# Build ROLLOUT_VLLM_CONFIG_JSON with vLLM engine kwargs + additional_config.maxtext_config
ROLLOUT_VLLM_CONFIG_JSON=$("$PYTHON_BIN" - "$ROLLOUT_SHARDING_JSON" "${ROLLOUT_VLLM_CONFIG_JSON:-}" "$ROLLOUT_MAXTEXT_EXTRA_FLAGS" <<'PY'
import json
import os
import sys

base_json, user_json, extra_flags = sys.argv[1], sys.argv[2], sys.argv[3]

def _parse_val(raw: str):
  low = raw.lower()
  if low == "true":
    return True
  if low == "false":
    return False
  if low == "none":
    return None
  try:
    return int(raw)
  except ValueError:
    pass
  try:
    return float(raw)
  except ValueError:
    pass
  return raw

cfg = json.loads(base_json) if base_json.strip() else {}
if user_json.strip():
  user_cfg = json.loads(user_json)
  cfg.update(user_cfg)

env_mapping = {
    "VLLM_MAX_MODEL_LEN": ("max_model_len", int),
    "VLLM_MAX_NUM_BATCHED_TOKENS": ("max_num_batched_tokens", int),
    "VLLM_MAX_NUM_SEQS": ("max_num_seqs", int),
    "VLLM_GPU_MEMORY_UTILIZATION": ("gpu_memory_utilization", float),
    "VLLM_DATA_PARALLEL_SIZE": ("data_parallel_size", int),
    "VLLM_ENABLE_EXPERT_PARALLEL": ("enable_expert_parallel", lambda v: v.lower() in ("true", "1")),
    "VLLM_MAMBA_CACHE_MODE": ("mamba_cache_mode", str),
    "VLLM_KV_CACHE_DTYPE": ("kv_cache_dtype", str),
    "VLLM_BLOCK_SIZE": ("block_size", int),
    "VLLM_ASYNC_SCHEDULING": ("async_scheduling", lambda v: v.lower() in ("true", "1")),
    "VLLM_ENABLE_CHUNKED_PREFILL": ("enable_chunked_prefill", lambda v: v.lower() in ("true", "1")),
    "VLLM_LANGUAGE_MODEL_ONLY": ("language_model_only", lambda v: v.lower() in ("true", "1")),
    "VLLM_REASONING_PARSER": ("reasoning_parser", str),
}
for env_k, (cfg_k, parse_fn) in env_mapping.items():
  val = os.getenv(env_k)
  if val is not None and val != "" and cfg_k not in cfg:
    try:
      cfg[cfg_k] = parse_fn(val)
    except Exception:
      cfg[cfg_k] = val

limit_mm = os.getenv("VLLM_LIMIT_MM_PER_PROMPT")
if limit_mm and "limit_mm_per_prompt" not in cfg:
  try:
    cfg["limit_mm_per_prompt"] = json.loads(limit_mm)
  except Exception:
    pass

add_cfg = cfg.setdefault("additional_config", {})
mt_cfg = add_cfg.setdefault("maxtext_config", {})
for tok in extra_flags.split():
  if "=" in tok:
    k, v = tok.split("=", 1)
    mt_cfg[k] = _parse_val(v)

print(json.dumps(cfg))
PY
)
export ROLLOUT_VLLM_CONFIG_JSON

# ==============================================================================
# 5. Model & MaxText Checkpoint Preparation (uses HF_TOKEN from env directly)
# ==============================================================================
has_direct_safetensors() {
  [[ -d "$MODEL_DIR" ]] && [[ -n "$(
    find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print -quit 2>/dev/null || true
  )" ]]
}

download_hf_repo() {
  local config_only="${1:-0}"
  echo "Downloading $MODEL_ID to $MODEL_DIR (config_only=$config_only, using HF_TOKEN from env if set)..."
  "$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" "$config_only" <<'PY'
import fnmatch
import os
import shutil
import sys
import huggingface_hub as hf

model_id, model_dir, config_only = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
os.makedirs(model_dir, exist_ok=True)

token = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or ""
).strip() or None

all_files = hf.list_repo_files(model_id, token=token)
filtered_files = [f for f in all_files if not f.startswith("original/")]
if config_only:
  ignore_pats = ("*.safetensors", "*.bin", "*.pt")
  filtered_files = [
      f for f in filtered_files
      if not any(fnmatch.fnmatch(f, pat) for pat in ignore_pats)
  ]

for filename in filtered_files:
  dst = os.path.join(model_dir, filename)
  if os.path.exists(dst) and os.path.getsize(dst) > 0:
    continue
  cached_path = hf.hf_hub_download(
      repo_id=model_id,
      filename=filename,
      token=token,
  )
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  try:
    os.remove(dst)
  except FileNotFoundError:
    pass
  try:
    os.link(os.path.realpath(cached_path), dst)
  except FileExistsError:
    pass
  except OSError:
    shutil.copy2(cached_path, dst)
print(f"Downloaded {len(filtered_files)} file(s) from {model_id} to {model_dir}.")
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

  if ! has_direct_safetensors; then
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
      echo "=== Last 80 lines of $log_file ===" >&2
      tail -n 80 "$log_file" 2>/dev/null || true
      exit 1
    fi
    if "$PYTHON_BIN" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
try:
  socket.create_connection(("localhost", port), timeout=1).close()
except OSError:
  sys.exit(1)
PY
    then
      echo "$name port $port is ready after ${elapsed}s."
      return
    fi
    if (( elapsed >= WAIT_TIMEOUT_SECS )); then
      echo "Error: timed out waiting ${WAIT_TIMEOUT_SECS}s for $name port $port." >&2
      tail -n 80 "$log_file" 2>/dev/null || true
      exit 1
    fi
    echo "Waiting for $name port $port... elapsed=${elapsed}s (log: $log_file)"
    sleep "$WAIT_POLL_SECS"
    elapsed=$((elapsed + WAIT_POLL_SECS))
  done
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
    kill "${pids[@]}" 2>/dev/null || true
    sleep 3 || true
    kill -9 "${pids[@]}" 2>/dev/null || true
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
echo "  Task mode:                  $TASK_MODE"
echo "  Model:                      $MODEL_NAME ($MAXTEXT_MODEL_NAME)"
echo "  Host TPU topology:          ${HOST_PHYSICAL_TPU_CHIPS} physical chips (${DETECTED_TPU_KIND}), ${ACTUAL_TRAINER_DEVICES} JAX devices/worker (${EFFECTIVE_MEGACORE_ARG:-default})"
echo "  Sampler quantization:       $SAMPLER_QUANT (qwix_moe_only=${ROLLOUT_QWIX_MOE_ONLY_QTYPE:-off})"
echo "  Trainer quantization:       $TRAINER_QUANT (qwix_moe_only=${TRAINER_QWIX_MOE_ONLY_QTYPE:-off})"
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

  export PYTHONPATH="${PY_HOOKS_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export QWIX_MOE_ONLY_QTYPE="$TRAINER_QWIX_MOE_ONLY_QTYPE"
  export USE_RAIDEN_FFI=false
  export RAIDEN_USE_FFI=0
  export RAIDEN_DEVICES_PER_HOST="$ACTUAL_TRAINER_DEVICES"
  export JAX_PLATFORMS=tpu,cpu
  export TPU_VISIBLE_DEVICES="$TRAINER_TPU_CHIPS"
  export TPU_VISIBLE_CHIPS="$TRAINER_TPU_CHIPS"
  export TPU_CHIPS_PER_HOST_BOUNDS="$TPU_CHIPS_PER_HOST_BOUNDS"
  export TPU_CHIPS_PER_PROCESS_BOUNDS="$TPU_CHIPS_PER_HOST_BOUNDS"
  export TPU_HOST_BOUNDS="$TPU_HOST_BOUNDS"
  if [[ -n "${EFFECTIVE_MEGACORE_ARG:-}" ]]; then
    export TPU_MEGACORE=megacore
  fi
  export LIBTPU_INIT_ARGS="$WORKER_LIBTPU_INIT_ARGS"
  export PYTHONUNBUFFERED=1
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

  export PYTHONPATH="${PY_HOOKS_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export QWIX_MOE_ONLY_QTYPE="$ROLLOUT_QWIX_MOE_ONLY_QTYPE"
  export USE_RAIDEN_FFI=false
  export RAIDEN_USE_FFI=0
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
  export TPU_CHIPS_PER_HOST_BOUNDS="$TPU_CHIPS_PER_HOST_BOUNDS"
  export TPU_CHIPS_PER_PROCESS_BOUNDS="$TPU_CHIPS_PER_HOST_BOUNDS"
  export TPU_HOST_BOUNDS="$TPU_HOST_BOUNDS"
  if [[ -n "${EFFECTIVE_MEGACORE_ARG:-}" ]]; then
    export TPU_MEGACORE=megacore
  fi
  export LIBTPU_INIT_ARGS="$WORKER_LIBTPU_INIT_ARGS"
  export PYTHONUNBUFFERED=1
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
    ORCHESTRATOR_CMD+=(
      --tfds_split="${TFDS_SPLIT:-train[:4]}"
      --reward_mode="${REWARD_MODE:-env}"
    )
  else
    ORCHESTRATOR_CMD+=(
      --max_turns="${MAX_TURNS:-5}"
      --env_backend="${ENV_BACKEND:-docker}"
    )
  fi

  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export JAX_PLATFORMS=cpu
  export PYTHONUNBUFFERED=1
  "${ORCHESTRATOR_CMD[@]}" > "$ORCHESTRATOR_LOG" 2>&1
) || {
  exit_code="$?"
  echo "Error: CPU orchestrator failed with exit code $exit_code." >&2
  echo "=== Last 80 lines of $ORCHESTRATOR_LOG ===" >&2
  tail -n 80 "$ORCHESTRATOR_LOG" 2>/dev/null || true
  exit "$exit_code"
}

analyze_orchestrator_log \
  "$ORCHESTRATOR_LOG" \
  "$MAX_OOB_RATIO" \
  "$MIN_KEPT_FRAC" \
  "$SAMPLER_QUANT" \
  "$TRAINER_QUANT" \
  "$MAX_GEOMEAN_DRIFT"
