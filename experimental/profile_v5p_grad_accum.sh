#!/bin/bash
# Profile optax vs stream gradient accumulation on a single-host TPU VM
# (e.g. v5p, 4 chips). Runs the SAME recipe as the GKE ablation yamls
# (experimental/gsm8k_refactor_{optax,stream}.yaml) minus the JobSet shell:
# both variants are one qwen3_grpo_demo.py invocation; the only differences
# between variants are --grad_accum and the profiler output dir.
#
# Usage (on the TPU VM, from the tunix repo root, deps present — e.g. inside
# the tunix_base_image container):
#   bash experimental/profile_v5p_grad_accum.sh
#
# Everything is overridable via env vars, e.g. quick profile-only run with
# vanilla rollout and local traces:
#   MAX_STEPS=12 ROLLOUT_ENGINE=vanilla TRACE_DEST=/tmp/xprof \
#     bash experimental/profile_v5p_grad_accum.sh
#
# Defaults (see tasks/grad_accum_profiling_v5p/plan.md for rationale):
#   full 200-step gsm8k run per variant (same as the GKE ablation; the xprof
#   trace still only covers steps SKIP..SKIP+PSTEPS); mesh (fsdp=4, tp=1);
#   batch 16 / mini 16 / micro 4 / logps 4 — keeps the accumulation depth at
#   4 microsteps (= the 64-chip run's 64/16) and per-fsdp-shard micro batch
#   at 1 sequence (= the 64-chip run's 16/16); traces written straight to
#   gs://yuxzhang-tunix-models/xprof/${RUN_TAG}_{optax,stream} (the same
#   bucket convention as the loss_accum_ablation profiling yamls).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

# Traces go straight to the bucket by default, following the convention of the
# loss_accum_ablation profiling yamls (gs://yuxzhang-tunix-models/xprof/<run>);
# jax.profiler.start_trace writes gs:// directly, one timestamped session
# subdir per run, so the folder accumulates without clobbering. Set
# TRACE_DEST to a local path to keep traces on the VM (then they get tar'd).
TRACE_DEST="${TRACE_DEST:-gs://yuxzhang-tunix-models/xprof}"
RUN_TAG="${RUN_TAG:-v5p_refactor}"
LOG_DIR="${LOG_DIR:-/tmp/grad_accum_logs}"
ENGINE="${ROLLOUT_ENGINE:-vllm}"        # vllm | vanilla
MESH_FSDP="${MESH_FSDP:-4}"
MESH_TP="${MESH_TP:-1}"
BATCH="${BATCH:-16}"
MINI="${MINI:-16}"
MICRO="${MICRO:-4}"
LOGPS="${LOGPS:-4}"
MAX_STEPS="${MAX_STEPS:-200}"           # full gsm8k run like the GKE ablation;
                                        # trace still only covers SKIP..SKIP+PSTEPS
# The profiler counts MICRO-steps (peft_trainer._iter_steps increments per
# micro-batch), not RL steps. With accumulation depth mini/micro = 4 (true for
# all presets here: 16/4, 64/16, 8/2), SKIP=4 starts exactly at an
# accumulation-cycle boundary and PSTEPS=8 covers TWO full cycles => 2
# flush/update boundaries (the optax-vs-stream comparison target) + the
# first post-flush add + 2 rollout segments. Use SKIP=5 PSTEPS=3 to match
# the window of the older GKE ablation traces.
SKIP="${PROFILER_SKIP:-4}"
PSTEPS="${PROFILER_STEPS:-8}"
VARIANTS="${VARIANTS:-optax stream}"

mkdir -p "$LOG_DIR"
case "$TRACE_DEST" in gs://*) ;; *) mkdir -p "$TRACE_DEST" ;; esac
cd "$TUNIX_DIR"

# ---------------------------------------------------------------------------
# ROPE hot-patch for vllm/tpu_inference qwen3 (PR #2838) — same replaces as
# the GKE yamls, but the file is auto-located and the patch is idempotent.
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
# Run both variants sequentially with identical settings.
# ---------------------------------------------------------------------------
for v in $VARIANTS; do
  trace_dir="$TRACE_DEST/${RUN_TAG}_${v}"
  log="$LOG_DIR/${RUN_TAG}_${v}.log"
  echo
  echo "===== [$v] grad-accum profile: trace -> $trace_dir  (log: $log) ====="
  ROLLOUT_ENGINE="$ENGINE" \
  PYTHONPATH="$TUNIX_DIR:${PYTHONPATH:-}" \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  SKIP_JAX_PRECOMPILE=True \
  XLA_PYTHON_CLIENT_PREALLOCATE=TRUE \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python3 -X faulthandler -u examples/math_gsm8k/qwen3_grpo_demo.py \
    --mesh_fsdp "$MESH_FSDP" --mesh_tp "$MESH_TP" \
    --batch_size "$BATCH" --mini_batch_size "$MINI" \
    --train_micro_batch_size "$MICRO" \
    --compute_logps_micro_batch_size "$LOGPS" \
    --max_steps "$MAX_STEPS" \
    --grad_accum "$v" \
    --profiler_log_dir "$trace_dir" \
    --profiler_skip_steps "$SKIP" --profiler_steps "$PSTEPS" \
    2>&1 | tee "$log"
  echo "===== [$v] done (exit=${PIPESTATUS[0]}) ====="
done

# ---------------------------------------------------------------------------
# Best-effort summary: step/HBM lines from the logs + trace locations.
# ---------------------------------------------------------------------------
echo
echo "################ SUMMARY ################"
for v in $VARIANTS; do
  trace_dir="$TRACE_DEST/${RUN_TAG}_${v}"
  log="$LOG_DIR/${RUN_TAG}_${v}.log"
  echo "--- [$v] ---"
  [ -f "$log" ] || { echo "  (no log)"; continue; }
  grep -inE "steps?/sec|sec/step|step_time|train_step|HBM" "$log" | tail -12
  echo "  trace: $trace_dir"
  case "$TRACE_DEST" in
    gs://*)
      gsutil ls "$trace_dir/plugins/profile/" 2>/dev/null | tail -3 || \
        echo "  (gsutil ls failed - check from an authed machine)"
      ;;
    *)
      # Local traces: pack per variant for the xprof visualization web UI.
      if [ -d "$trace_dir" ]; then
        tar -czf "$TRACE_DEST/${RUN_TAG}_${v}_trace.tar.gz" -C "$TRACE_DEST" "${RUN_TAG}_${v}"
        echo "  packed: $TRACE_DEST/${RUN_TAG}_${v}_trace.tar.gz"
      fi
      ;;
  esac
done
echo
echo "xprof web UI: point it at $TRACE_DEST/${RUN_TAG}_{optax,stream}"
echo "logs: $LOG_DIR/"
