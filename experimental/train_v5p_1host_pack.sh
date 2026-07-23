#!/bin/bash
# Real (full) sequence-packing training on a single-host TPU VM (v5p, 4 chips).
#
# Adapted from experimental/profile_v5p_grad_accum.sh, but this REALLY TRAINS
# instead of profiling a few steps and stopping:
#   * ONE run (no optax/stream ablation loop) -- pick the variant via GRAD_ACCUM.
#   * MAX_STEPS is a real run length (default 200), so training CONTINUES after
#     the profiler window closes -- the profiler only traces steps SKIP..SKIP+
#     PSTEPS, it never stops training.
#   * Sequence packing is ON (--max_seq_token_per_tpu), to exercise the CL1/CL2
#     packing framework.
#   * WANDB defaults to online so you can watch the loss curve.
#   * Perfetto RL perf tracing is ON for the WHOLE run (v1 aggregate span
#     metrics + v2 Perfetto trace), so one converge run yields BOTH convergence
#     (wandb) and full-run performance (ui.perfetto.dev) -- unlike xprof which
#     is capped at ~2GB / a few steps. The short xprof window still runs too.
#
# Usage (on the TPU VM, tunix repo root, deps present -- e.g. inside the
# tunix_base_image container):
#   bash experimental/train_v5p_1host_pack.sh
#
# Everything is overridable via env vars, e.g. a quick smoke run:
#   MAX_STEPS=20 WANDB_MODE=offline TRACE_DEST=/tmp/xprof \
#     bash experimental/train_v5p_1host_pack.sh
#
# 4-chip sizing rationale (v5p ~95GB/chip):
#   mesh fsdp=4, tp=1  -> pack_size = fsdp*dp = 4 (1.7B needs no tensor
#     parallelism; params shard cleanly over 4).
#   batch 16 / mini 16 / micro 4 / logps 4:
#     - accumulation depth mini/micro = 4 (= the 64-chip run's 64/16)
#     - micro/fsdp = 1 (each fsdp shard gets 1 packed row per micro-step;
#       micro >= fsdp is required for divisibility)
#   packing: 16 prompts * NUM_GENERATIONS(8) = 128 sequences; at ~500 tokens
#     each and a 4096-token budget that is ~8 seqs/row, so 128/(4*8) ~= 4 chunks
#     per mini-batch -- exercises multi-chunk gradient accumulation.
#   ROLLOUT_HBM=0.3 (~29GB): the trainer needs ~13GB/chip on 4 chips; the demo
#     default 0.6 (~58GB) + the 0.9 JAX pool overflows during compile (OOM).
#   OOM fallback (accumulation depth stays 4; fsdp=1 lifts micro>=fsdp so
#   micro=1 is legal):
#     MESH_FSDP=1 MESH_TP=4 BATCH=4 MINI=4 MICRO=1 LOGPS=1 \
#       bash experimental/train_v5p_1host_pack.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

# ---- knobs ----------------------------------------------------------------
ENGINE="${ROLLOUT_ENGINE:-vllm}"           # vllm | vanilla
GRAD_ACCUM="${GRAD_ACCUM:-stream}"         # stream | optax
ROLLOUT_HBM="${ROLLOUT_HBM:-0.3}"          # vLLM HBM fraction (see rationale)
MESH_FSDP="${MESH_FSDP:-4}"
MESH_TP="${MESH_TP:-1}"
BATCH="${BATCH:-16}"
MINI="${MINI:-16}"
MICRO="${MICRO:-4}"
LOGPS="${LOGPS:-4}"
MAX_SEQ_TOKEN="${MAX_SEQ_TOKEN:-4096}"     # packing budget; set 0 to DISABLE packing
MAX_STEPS="${MAX_STEPS:-200}"              # REAL run length (training continues
                                           # past the profiler window)
LOG_DIR="${LOG_DIR:-/tmp/train_v5p_logs}"
RUN_TAG="${RUN_TAG:-v5p_1host_pack}"

# xprof: detailed kernel trace of a short window then KEEPS TRAINING (2GB cap,
# so only a few steps). Set PROFILER_STEPS=0 to skip.
TRACE_DEST="${TRACE_DEST:-gs://yuxzhang-tunix-models/xprof}"
PROFILER_SKIP="${PROFILER_SKIP:-10}"
PROFILER_STEPS="${PROFILER_STEPS:-5}"

# Perfetto RL perf tracing: low overhead, runs the WHOLE training (no 2GB cap).
# v1 = aggregate span metrics (rollout/wait time) into the metrics stream;
# v2 = a Perfetto trace file at PERF_TRACE_DIR (view at ui.perfetto.dev). Both
# on by default so one converge run gives loss (wandb) + full-run performance.
ENABLE_PERF_V1="${ENABLE_PERF_V1:-1}"
ENABLE_PERF_V2="${ENABLE_PERF_V2:-1}"
PERF_TRACE_DIR="${PERF_TRACE_DIR:-gs://yuxzhang-tunix-models/perfetto/${RUN_TAG}}"

mkdir -p "$LOG_DIR"
case "$TRACE_DEST" in gs://*) ;; *) mkdir -p "$TRACE_DEST" ;; esac
cd "$TUNIX_DIR"

# ---------------------------------------------------------------------------
# vLLM/tpu_inference needs a small rope-scaling shim on qwen3 (same as the
# profiling script). No-op on the vanilla engine.
# ---------------------------------------------------------------------------
apply_rope_patch() {
  python3 - <<'EOF'
import sys
try:
    import tpu_inference.models.jax.qwen3 as m
except Exception as exc:
    print(f"ROPE-PATCH: tpu_inference not importable ({exc})")
    sys.exit(3)

file_path = m.__file__
with open(file_path, "r") as f:
    code = f.read()

if "def normalize_rope_scaling" in code:
    print(f"ROPE-PATCH: already applied -> {file_path}")
    sys.exit(0)

injected_functions = '''
from typing import Any, Dict, Optional

def normalize_rope_scaling(rope_scaling: Any) -> Optional[Dict[str, Any]]:
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        if (rope_scaling.get("rope_type", "default") == "default"
                and "factor" not in rope_scaling
                and "scale_factor" not in rope_scaling
                and "mrope_section" not in rope_scaling):
            rope_scaling = None
        elif "factor" in rope_scaling and "scale_factor" not in rope_scaling:
            rope_scaling["scale_factor"] = rope_scaling.pop("factor")
    return rope_scaling

def get_rope_scaling(config: Any) -> Optional[Dict[str, Any]]:
    rope_scaling = getattr(config, "rope_parameters", None) or getattr(
        config, "rope_scaling", None)
    return normalize_rope_scaling(rope_scaling)

def get_rope_theta(config: Any, default: float = 10000.0) -> float:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None and "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    return float(getattr(config, "rope_theta", default))

'''

code = injected_functions + code
code = code.replace(
    'self.rope_theta = config.rope_parameters["rope_theta"]',
    'self.rope_theta = get_rope_theta(config, default=1000000.0)')
code = code.replace(
    'self.rope_scaling = getattr(config, "rope_scaling", None)',
    'self.rope_scaling = get_rope_scaling(config)')

with open(file_path, "w") as f:
    f.write(code)
print(f"ROPE-PATCH: applied -> {file_path}")
EOF
}

if [ "$ENGINE" = "vllm" ]; then
  apply_rope_patch
  rc=$?
  if [ "$rc" = 3 ]; then
    echo "ERROR: ROLLOUT_ENGINE=vllm but tpu_inference is missing."
    echo "Run inside the tunix_base_image container, or fall back with:"
    echo "  ROLLOUT_ENGINE=vanilla bash $0"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Assemble args. Packing and profiler are added conditionally.
# ---------------------------------------------------------------------------
pack_args=()
if [ "${MAX_SEQ_TOKEN}" != "0" ]; then
  pack_args+=(--max_seq_token_per_tpu "$MAX_SEQ_TOKEN")
fi

prof_args=()
trace_dir="$TRACE_DEST/${RUN_TAG}_${GRAD_ACCUM}"
if [ "${PROFILER_STEPS}" != "0" ]; then
  prof_args+=(--profiler_log_dir "$trace_dir"
              --profiler_skip_steps "$PROFILER_SKIP"
              --profiler_steps "$PROFILER_STEPS")
fi

perf_args=()
[ "${ENABLE_PERF_V1}" != "0" ] && perf_args+=(--enable_perf_v1)
if [ "${ENABLE_PERF_V2}" != "0" ]; then
  perf_args+=(--enable_perf_v2 --perf_trace_dir "$PERF_TRACE_DIR")
fi

log="$LOG_DIR/${RUN_TAG}_${GRAD_ACCUM}.log"
echo "===== TRAIN [$GRAD_ACCUM] packing=${MAX_SEQ_TOKEN} mesh=${MESH_FSDP}x${MESH_TP} "\
"batch=${BATCH}/${MINI}/${MICRO}/${LOGPS} steps=${MAX_STEPS} (log: $log) ====="

ROLLOUT_ENGINE="$ENGINE" \
PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
WANDB_MODE="${WANDB_MODE:-online}" \
SKIP_JAX_PRECOMPILE=True \
XLA_PYTHON_CLIENT_PREALLOCATE=TRUE \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python3 -X faulthandler -u examples/math_gsm8k/qwen3_grpo_demo.py \
  --mesh_fsdp "$MESH_FSDP" --mesh_tp "$MESH_TP" \
  --batch_size "$BATCH" --mini_batch_size "$MINI" \
  --train_micro_batch_size "$MICRO" \
  --compute_logps_micro_batch_size "$LOGPS" \
  --max_steps "$MAX_STEPS" \
  --rollout_vllm_hbm_utilization "$ROLLOUT_HBM" \
  --grad_accum "$GRAD_ACCUM" \
  "${pack_args[@]}" \
  "${prof_args[@]}" \
  "${perf_args[@]}" \
  2>&1 | tee "$log"

rc=${PIPESTATUS[0]}
echo "===== done (exit=$rc) ====="

# ---------------------------------------------------------------------------
# Best-effort summary: step timing / HBM / packing efficiency from the log.
# ---------------------------------------------------------------------------
echo
echo "################ SUMMARY ################"
if [ -f "$log" ]; then
  echo "--- step timing / HBM ---"
  grep -inE "steps?/sec|sec/step|step_time|train_step|HBM" "$log" | tail -12
  echo "--- packing efficiency (dummy_ratio; want << 1) ---"
  grep -inE "dummy_ratio|pack_sequences" "$log" | tail -6
  [ "${PROFILER_STEPS}" != "0" ] && echo "--- xprof (kernel) trace: $trace_dir ---"
  [ "${ENABLE_PERF_V2}" != "0" ] && \
    echo "--- Perfetto (full-run) trace: $PERF_TRACE_DIR  (open at ui.perfetto.dev) ---"
fi
exit "$rc"
