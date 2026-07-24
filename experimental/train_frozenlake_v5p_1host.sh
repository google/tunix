#!/bin/bash
# Real (full) agentic FrozenLake GRPO training on a single-host TPU VM
# (v5p, 4 chips), Qwen3-8B via examples/frozenlake/train_frozenlake_qwen3.py —
# the recipe whose convergence is established (swe-evaluation runs; solve_ratio
# climbs on held-out eval). The FrozenLake analogue of train_v5p_1host_pack.sh.
#
# ONE run yields BOTH:
#   * convergence  -- wandb loss/reward + eval rewards/solve_ratio every 10
#                     steps (the script wires the held-out test set).
#   * performance  -- full-run Perfetto (v1 spans + v2 trace) plus a short
#                     xprof kernel window. Same as the gsm8k run.
#
# Usage (on the TPU VM, tunix repo root, deps present -- e.g. inside the
# tunix_base_image container):
#   # packed (segment-aware CL3 + weighted stream)
#   RUN_TAG=cl3_frozenlake_pack bash experimental/train_frozenlake_v5p_1host.sh
#   # unpack parity (same stream+weighted accumulation, packing OFF)
#   MAX_TOKEN_PER_TPU=0 RUN_TAG=cl3_frozenlake_unpack \
#     bash experimental/train_frozenlake_v5p_1host.sh
#
# 4-chip sizing rationale (v5p ~95GB/chip; Qwen3-8B):
#   mesh (2,2) = 2 fsdp x 2 tp -> pack_size = fsdp = 2; the script derives the
#     vLLM rollout's data/tensor_parallel_size from the mesh (dp2 tp2).
#   batch 16 / mini 16 / micro 4 / num_gen 8: the converged swe-evaluation
#     recipe (batch 64 on 64 chips) scaled by chip count; 128 episodes/step.
#     All optimizer/algo hyperparams (LR 1e-6, rloo, gspo-token, eps .003/.005)
#     stay at the script's converged defaults.
#   memory: actor fp32 8GB + Adam 16GB + grads 8GB + ref bf16 4GB + vLLM(0.20)
#     19GB + logits/activations ~10GB ~= 65GB/chip < 95GB.
#   packing: single seq max = prompt 2048 + response 2048 = 4096; budget 16384
#     -> ~4 seqs/row (same density as gsm8k at 8192).
#   steps: MAX_STEPS -> --num_batches with --num_epochs 1 (the script computes
#     max_steps = num_batches * epochs); 200 x batch16 = 3200 prompts <= the
#     generated train set (10000).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

# ---- knobs (same ENV interface as train_v5p_1host_pack.sh) ----------------
ENGINE="${ROLLOUT_ENGINE:-vllm}"           # vllm | vanilla
MESH_FSDP="${MESH_FSDP:-2}"                # mesh (2,2) = 2 fsdp x 2 tp
MESH_TP="${MESH_TP:-2}"
BATCH="${BATCH:-16}"
MINI="${MINI:-16}"
MICRO="${MICRO:-4}"
LOGPS="${LOGPS:-4}"
NUM_GEN="${NUM_GEN:-8}"                    # RLOO baseline samples (keep 8)
MAX_TOKEN_PER_TPU="${MAX_TOKEN_PER_TPU:-8192}"   # packing budget; 0 = DISABLE
MAX_SEGMENTS_PER_ROW="${MAX_SEGMENTS_PER_ROW:-}"  # segment cap; empty = budget-derived
MAX_STEPS="${MAX_STEPS:-200}"              # training updates (= --num_batches,
NUM_EPOCHS="${NUM_EPOCHS:-1}"              #   with --num_epochs 1)
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

mkdir -p "$LOG_DIR"
case "$TRACE_DEST" in gs://*) ;; *) mkdir -p "$TRACE_DEST" ;; esac
cd "$TUNIX_DIR"

# ---------------------------------------------------------------------------
# vLLM/tpu_inference needs the qwen3 rope-scaling shim (same as the gsm8k
# wrapper -- Qwen3-8B goes through the same tpu_inference qwen3 model).
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
# The training script only READS /tmp/data/frozenlake/{train,test}.parquet;
# generate them once (idempotent: loads existing files when present).
# ---------------------------------------------------------------------------
PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" python3 - <<'EOF'
from examples.frozenlake import data
data.create_dataset(split="train", data_dir="/tmp/data/frozenlake")
data.create_dataset(split="test", data_dir="/tmp/data/frozenlake")
print("frozenlake datasets ready")
EOF

# ---------------------------------------------------------------------------
# Assemble args. Packing, profiler and perf flags are added conditionally.
# ---------------------------------------------------------------------------
pack_args=()
if [ "${MAX_TOKEN_PER_TPU}" != "0" ]; then
  pack_args+=(--max_seq_token_per_tpu "$MAX_TOKEN_PER_TPU")
  [ -n "${MAX_SEGMENTS_PER_ROW}" ] && \
    pack_args+=(--max_segments_per_packed_row "$MAX_SEGMENTS_PER_ROW")
fi

prof_args=()
trace_dir="$TRACE_DEST/${RUN_TAG}"
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

log="$LOG_DIR/${RUN_TAG}.log"
echo "===== FROZENLAKE[qwen3-8b] packing=${MAX_TOKEN_PER_TPU} mesh=${MESH_FSDP}x${MESH_TP} "\
"batch=${BATCH}/${MINI}/${MICRO}/${LOGPS} num_gen=${NUM_GEN} steps=${MAX_STEPS} "\
"(log: $log) ====="

ROLLOUT_ENGINE="$ENGINE" \
PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
WANDB_MODE="${WANDB_MODE:-online}" \
python3 -X faulthandler -u examples/frozenlake/train_frozenlake_qwen3.py \
  --mesh_fsdp "$MESH_FSDP" --mesh_tp "$MESH_TP" \
  --batch_size "$BATCH" --mini_batch_size "$MINI" \
  --train_micro_batch_size "$MICRO" \
  --compute_logps_micro_batch_size "$LOGPS" \
  --num_generations "$NUM_GEN" \
  --num_batches "$MAX_STEPS" --num_epochs "$NUM_EPOCHS" \
  --run_name "$RUN_TAG" \
  "${pack_args[@]}" \
  "${prof_args[@]}" \
  "${perf_args[@]}" \
  2>&1 | tee "$log"

rc=${PIPESTATUS[0]}
echo "===== done (exit=$rc) ====="

# ---------------------------------------------------------------------------
# Best-effort summary: solve rate / step timing / packing efficiency.
# ---------------------------------------------------------------------------
echo
echo "################ SUMMARY ################"
if [ -f "$log" ]; then
  echo "--- convergence (want solve_ratio climbing over steps) ---"
  grep -inE "solve_ratio" "$log" | tail -8
  echo "--- step timing / HBM ---"
  grep -inE "steps?/sec|sec/step|step_time|train_step|HBM" "$log" | tail -8
  echo "--- packing efficiency (dummy_ratio; want << 1) + row count/depth ---"
  grep -inE "dummy_ratio|pack_sequences|pack_size|max_seq_token_per_tpu" "$log" | tail -8
  [ "${PROFILER_STEPS}" != "0" ] && echo "--- xprof (kernel) trace: $trace_dir ---"
  [ "${ENABLE_PERF_V2}" != "0" ] && \
    echo "--- Perfetto (full-run) trace: $PERF_TRACE_DIR  (open at ui.perfetto.dev) ---"
fi
exit "$rc"
