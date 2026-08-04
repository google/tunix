#!/bin/bash
# gsm8k document-mask A/B, profiled -- cloned from profile_v5p_grad_accum.sh so
# the trace capture is the one that has already been shaken out, not a new one.
#
# Both arms run the identical gsm8k config; the only difference is whether
# splash is told the packed row's segment layout:
#
#   arm off (TUNIX_SPLASH_DOCMASK=0)  today: a causal mask over the whole packed
#           row, so splash schedules budget^2/2 blocks and `segment_ids` zeroes
#           the cross-segment ones afterwards.
#   arm on  (TUNIX_SPLASH_DOCMASK=1)  the chunk's layout, rounded OUT to block
#           boundaries, is built into a mask on the host and passed to the model
#           as an argument.  The mask is a SUPERSET of the true block-diagonal
#           one, so `segment_ids` still masks exactly and the output is bitwise
#           unchanged -- only the schedule shrinks.
#
# Why the kernel is an ARGUMENT and not a module-level global: a global read
# inside jit is a trace-time constant, so a later layout is silently ignored.
# Measured -- declaring layout A, running, then declaring B returned A's answer
# bit for bit, with a truncated segment and no error.
#
# Unlike profile_v5p_grad_accum.sh this does NOT `git fetch` a branch inside the
# container: the change is not committed, so the patched files are bind-mounted
# read-only ONE AT A TIME over the image's own tunix.  Mounting the whole tree
# would swap the packer too and silently change what is being measured.
#
# Usage on the TPU VM:
#   bash experimental/profile_v5p_docmask.sh              # both arms
#   bash experimental/profile_v5p_docmask.sh on           # one arm
#   MAX_STEPS=12 TRACE_DEST=/tmp/xprof bash experimental/profile_v5p_docmask.sh
#
# Read experimental/splash_docmask_design.md before interpreting the numbers: a
# chunk holding one near-L_max sequence gets no speed-up at all, and the
# per-mask-shape histogram tells you how often that happens.
set -uo pipefail

IMAGE="${IMAGE:-tunix_frozenlake_image:vllm-tpu0.25.0}"
PATCH_DIR="${PATCH_DIR:-/mnt/disks/tunix-data/p18_splash/p20c}"
RUN_TAG="${RUN_TAG:-docmask_$(date +%m%d_%H%M)}"
LOG_DIR="${LOG_DIR:-/mnt/disks/tunix-data/p18_splash/gsm8k_docmask}"
# jax.profiler writes gs:// directly, same as the grad-accum ablation; point it
# at a local path to keep traces on the VM instead.
TRACE_DEST="${TRACE_DEST:-$LOG_DIR/traces}"
PERF_DEST="${PERF_DEST:-$LOG_DIR/perfetto}"

# The xprof window skips warm-up, so the trace covers steady-state steps only.
SKIP="${PROFILER_SKIP:-5}"
PSTEPS="${PROFILER_STEPS:-3}"

MESH_FSDP="${MESH_FSDP:-4}"; MESH_TP="${MESH_TP:-1}"
BATCH="${BATCH:-8}"; MINI="${MINI:-8}"; MICRO="${MICRO:-1}"; LOGPS="${LOGPS:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
MAX_RESPONSE="${MAX_RESPONSE:-1024}"
# Production values: train_v5p_1host_pack.sh defaults MAX_TOKEN_PER_TPU to 8192,
# the gsm8k packing yamls pass 4096.  This used to default to 2048, which is not
# a production value and is the flattest end of the curve -- causal work grows
# quadratically with the budget while the segment mask grows linearly, so 2048
# measures ~0.45x on the kernel where 8192 measures ~0.18-0.31x.
BUDGET="${BUDGET:-8192}"          # max_seq_token_per_tpu
ROLLOUT_HBM="${ROLLOUT_HBM:-0.3}"
NUM_HEADS="${NUM_HEADS:-16}"      # qwen3_1p7b; splash_mask.attach refuses to guess

case "${1:-all}" in
  off|on) ARMS="$1" ;;
  all)    ARMS="off on" ;;
  *) echo "usage: $0 [off|on|all]"; exit 1 ;;
esac

mkdir -p "$LOG_DIR"
case "$TRACE_DEST" in gs://*) ;; *) mkdir -p "$TRACE_DEST" ;; esac
case "$PERF_DEST"  in gs://*) ;; *) mkdir -p "$PERF_DEST"  ;; esac

MOUNTS=(
  -v "$PATCH_DIR/model.py":/app/tunix/models/qwen3/model.py:ro
  -v "$PATCH_DIR/common.py":/app/tunix/rl/common.py:ro
  -v "$PATCH_DIR/utils.py":/app/tunix/rl/utils.py:ro
  -v "$PATCH_DIR/algo_core.py":/app/tunix/rl/algo_core.py:ro
  -v "$PATCH_DIR/splash_mask.py":/app/tunix/rl/splash_mask.py:ro
  -v "$PATCH_DIR/rl_learner.py":/app/tunix/rl/rl_learner.py:ro
)
for f in "${MOUNTS[@]}"; do
  case "$f" in -v) continue;; esac
  src="${f%%:*}"; [ -f "$src" ] || { echo "FATAL: missing $src"; exit 2; }
done

# ---------------------------------------------------------------------------
# Preflight.  Printing a flag is not enough: an earlier gate ran three arms
# against the image's own code because only a print guarded it, and the whole
# run was VOID.  This also dry-runs argparse, because the
# demo's flags differ between branches and a typo there wastes the whole run.
# ---------------------------------------------------------------------------
echo "=== PREFLIGHT ==="
sudo docker run --rm "${MOUNTS[@]}" -e JAX_PLATFORMS=cpu \
  --entrypoint python3 "$IMAGE" -c "
import argparse, inspect, sys
from tunix.rl import common, splash_mask
from tunix.models.qwen3 import model as m
ok = True
for name, cond in (
    ('splash_mask.attach',            hasattr(splash_mask, 'attach')),
    ('TrainExample.segment_layout',   'segment_layout' in common.TrainExample.__dataclass_fields__),
    ('TrainExample.splash_kernel',    'splash_kernel' in common.TrainExample.__dataclass_fields__),
    ('common forwards splash_kernel', 'splash_kernel' in inspect.signature(common.compute_per_token_logps).parameters),
    ('Qwen3.__call__ takes it',       'splash_kernel' in m.Qwen3.__call__.__code__.co_varnames),
    ('Attention.block takes it',      'splash_kernel' in m.Attention.block.__code__.co_varnames),
):
    print(f'  {name:<32}{cond}'); ok &= bool(cond)
import tunix.rl.rl_learner as L
wired = 'splash_mask.attach' in open(L.__file__).read()
print(f'  {\"learner attaches it\":<32}{wired}'); ok &= wired
# The falsified NumpyMask path must be unreachable: kernel_for has to route to
# the computable mask, whose MaskInfo fetches no tiles.  A silent fallback here
# would produce wrong numbers more slowly, with no other symptom.
import numpy as _np
comp = hasattr(splash_mask, 'SegmentCausalMask')
print(f'  {\"SegmentCausalMask present\":<32}{comp}'); ok &= comp
if comp:
    _sid = _np.concatenate([_np.full(n, i, _np.int32)
                            for i, n in enumerate([512] * 8)])
    _k = splash_mask.build_segment_kernel(4096, _sid, 256, 4)
    _fi = _k.fwd_mask_info
    _computable = _fi.q_sequence is not None and _fi.partial_mask_blocks is None
    print(f'  {\"computable path (no tiles)\":<32}{_computable}')
    ok &= _computable
    _routed = 'build_segment_kernel' in inspect.getsource(splash_mask.kernel_for)
    print(f'  {\"kernel_for routes to it\":<32}{_routed}'); ok &= _routed
# every flag this script passes must exist in THIS image's demo
src = open('/app/examples/math_gsm8k/qwen3_grpo_demo.py').read()
need = ['--mesh_fsdp','--mesh_tp','--batch_size','--mini_batch_size',
        '--train_micro_batch_size','--compute_logps_micro_batch_size',
        '--max_steps','--max_response_length','--max_seq_token_per_tpu',
        '--rollout_vllm_hbm_utilization','--profiler_log_dir',
        '--profiler_skip_steps','--profiler_steps','--enable_perf_v2',
        '--perf_trace_dir']
missing = [f for f in need if f not in src]
print(f'  {\"demo accepts every flag\":<32}{not missing}' + (f'  missing={missing}' if missing else ''))
ok &= not missing
sys.exit(0 if ok else 3)
" || { echo "PREFLIGHT FAIL -- not launching."; exit 3; }
echo "  OK"; echo

# ---------------------------------------------------------------------------
run_arm () {
  local arm="$1" docmask="$2"
  local trace_dir="$TRACE_DEST/${RUN_TAG}_${arm}"
  local perf_dir="$PERF_DEST/${RUN_TAG}_${arm}"
  local log="$LOG_DIR/${RUN_TAG}_${arm}.log"
  echo "===== [$arm] DOCMASK=$docmask  xprof -> $trace_dir  perfetto -> $perf_dir"
  echo "      (log: $log) ====="
  sudo docker run --rm --privileged --net=host \
    "${MOUNTS[@]}" \
    -v "$LOG_DIR":"$LOG_DIR" \
    -e TUNIX_SPLASH_DOCMASK="$docmask" \
    -e TUNIX_SPLASH_NUM_HEADS="$NUM_HEADS" \
    -e PYTHONUNBUFFERED=1 \
    -e WANDB_MODE="${WANDB_MODE:-offline}" \
    -e SKIP_JAX_PRECOMPILE=True \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=TRUE \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    -w /app --entrypoint python3 "$IMAGE" \
    -X faulthandler -u examples/math_gsm8k/qwen3_grpo_demo.py \
      --mesh_fsdp "$MESH_FSDP" --mesh_tp "$MESH_TP" \
      --batch_size "$BATCH" --mini_batch_size "$MINI" \
      --train_micro_batch_size "$MICRO" \
      --compute_logps_micro_batch_size "$LOGPS" \
      --max_steps "$MAX_STEPS" \
      --max_response_length "$MAX_RESPONSE" \
      --max_seq_token_per_tpu "$BUDGET" \
      --rollout_vllm_hbm_utilization "$ROLLOUT_HBM" \
      --profiler_log_dir "$trace_dir" \
      --profiler_skip_steps "$SKIP" --profiler_steps "$PSTEPS" \
      --enable_perf_v2 --perf_trace_dir "$perf_dir" \
      2>&1 | tee "$log"
  echo "===== [$arm] done (exit=${PIPESTATUS[0]}) ====="
}

for arm in $ARMS; do
  case "$arm" in off) run_arm off 0 ;; on) run_arm on 1 ;; esac
done

echo
echo "=== 对账要看什么 ==="
echo "  1. 两臂 loss / grad_norm 曲线应在噪声内一致 —— 文档 mask 是超集,模块级已证逐位不变"
echo "  2. xprof:同名 *_segmented_{fwd,dq,dkv} kernel,arm on 的 grid 迭代数应更少"
echo "  3. perfetto(v2):attention 段的墙钟应缩短,projections/MLP 段应不变"
echo "  4. arm on 日志里的 (grid_width, partial_blocks) 直方图:partial_blocks 应恒为 1、"
echo "     组合数 <=8;超了说明真实布局比合成分布更碎,编译预算要重估"
echo "  5. 前几步有几次额外编译(每个 mask 形状一次),之后应稳态零编译"
echo
echo "  logs:     $LOG_DIR/${RUN_TAG}_*.log"
echo "  xprof:    $TRACE_DEST/${RUN_TAG}_*"
echo "  perfetto: $PERF_DEST/${RUN_TAG}_*"
