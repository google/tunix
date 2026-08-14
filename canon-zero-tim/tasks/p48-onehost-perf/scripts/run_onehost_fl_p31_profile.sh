#!/usr/bin/env bash
# P49 Phase 0: one-host FrozenLake P31-convergence profiling run (Qwen3-8B,
# canonical TRAIN mode, CANON_P31_MAX_STEPS-truncated).  Derived from
# canon-zero-tim/tasks/p41-optimizer-residency/scripts/
# run_frozenlake_onehost_resident.sh (the 417.6s reference vehicle).
# Mode deltas vs that template (M1-M6, per the phase0 Finding):
#   M1  P31 TRAIN instead of the L3+P27 update canary:
#       CANON_P31_CONVERGENCE=1, CANON_ALIGNMENT_TRAIN=1,
#       CANON_ALIGNMENT_UPDATE_CANARY=0 (required_p31 keeps L3=1 and P27=1).
#   M2  (revised by r6) CANON_UPDATE_REPORT is REQUIRED by the G6 update even
#       in TRAIN mode (agentic_rl_learner.py:1217); only the canary
#       classifier stays dropped.  success = exit 0 + per-step gates.
#   M3  the full P31 geometry pin on the CLI (train_frozenlake_qwen3.py:405-430).
#   M4  CANON_PRE_ALIGN_GATE=1 + report -> per-step [CANON_ALIGN_PRE].
#   M5  timeout default 14400s (8B multi-turn step time is itself unmeasured).
#   M6  no compilation cache: the 8B cold-compile account is Phase-0 data.
#   M8  pinned-host-offload optimizer placement: with the resident placement
#       the r4 attempt passed the step-0 gate and then OOMed inside
#       segmented_grpo_value_and_grad (2.32G wanted, 504M free) -- 8B resident
#       adam state (~16G/chip) does not coexist with 32x6144 activations on
#       one host.  The P41 pair classifier proved resident/offload updates
#       bitwise_equal, so placement changes no numerics; the offload h2d/d2h
#       per-step transfers are reported as their own line in the profile.
set -euo pipefail

label="${1:?usage: run_onehost_fl_p31_profile.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P49.FLPROF] invalid label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
model_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
model_sha="$(cat "$model_cache/refs/main")"
model="$model_cache/snapshots/$model_sha"
data=/mnt/disks/tunix-data/frozenlake/data_p31_10k
deps=/mnt/disks/tunix-data/frozenlake/deps
image=tunix_frozenlake_image:vllm-tpu0.25.0
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
root=/mnt/disks/tunix-data/logp_probe_1host/p49_fl_p31_${label}
canon_out="$root/canon"
raw="$root/raw.log"
align="$root/alignment.jsonl"
pre_align="$root/pre_alignment.jsonl"
update="$root/update.json"
runtime="$root/runtime.json"
driver="$root/driver.log"
container="p49_fl_p31_${label}"
timeout_seconds="${P49_FL_TIMEOUT_SECONDS:-28800}"
p31_max_steps="${P49_MAX_STEPS:-3}"
# P49_COMPILE_CACHE=1 enables the P48-proven persistent compilation cache
# (key = contract tree hashes + FL mode tag; hit evidence via entry-count
# deltas and the stage-call collapse, per the P48 Phase-2 methodology).
compile_cache="${P49_COMPILE_CACHE:-0}"

cache_env=()
if [ "$compile_cache" = "1" ] && [ -n "${P49_CACHE_DIR:-}" ]; then
  # Explicit namespace override: observer-only source changes keep the traced
  # HLO identical, so JAX's internal keys still hit the prior executables --
  # and a hit under the old namespace IS the program-identity proof.
  cache_dir="$P49_CACHE_DIR"
  mkdir -p "$cache_dir"
  cache_env=( -e JAX_COMPILATION_CACHE_DIR="$cache_dir" \
              -e JAX_DEBUG_LOG_MODULES=jax._src.compilation_cache )
elif [ "$compile_cache" = "1" ]; then
  contract_tree="$(git -C "$repo" rev-parse HEAD:tunix HEAD:examples \
    HEAD:canon-zero-tim/src HEAD:canon-zero-tim/patches \
    HEAD:canon-zero-tim/install.sh | sha256sum | cut -c1-12)"
  cache_dir="/mnt/disks/tunix-data/jax_cache/p49_${contract_tree}_vllm-tpu0250_dp1tp4_fl8b_p31off"
  mkdir -p "$cache_dir"
  cache_env=( -e JAX_COMPILATION_CACHE_DIR="$cache_dir" \
              -e JAX_DEBUG_LOG_MODULES=jax._src.compilation_cache )
fi

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
[[ "$model_sha" =~ ^[0-9a-f]{40}$ ]]
test -s "$model/config.json"
test "$(find "$model" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)" -eq 5
# P31 pins the dataset at train/test=10000/100 (train_frozenlake_qwen3.py:
# 741-745); the on-disk data/ dir holds an 8/4-row remnant from bounded runs.
# Point at a dedicated dir and let the demo's own create_dataset self-generate
# the contract dataset deterministically on first touch (data.py:110-121,
# seed 42) -- the generated parquets persist for later runs.  The dir is
# created once on the host with sudo (frozenlake/ is root-owned; r3 died on
# a plain mkdir); the runner only asserts it exists.
test -d "$data"
test -s "$deps/gymnasium-1.3.0-py3-none-any.whl"
test -s "$deps/farama_notifications-0.0.6-py3-none-any.whl"
test ! -e "$root"
# Refuse to launch while any non-infrastructure container is running: the
# TPU is single-tenant, and r5 died colliding with our own zombie r4.
infra='instance_agent|tpu-runtime|vbarcontrolagent|healthagent|google-runtime-monitor|google-collectd|monitoringagent'
if [ "$(sudo docker ps --format '{{.Names}}' | grep -vcE "$infra")" -ne 0 ]; then
  echo "[P49.FLPROF] refusing: TPU busy (non-infra container running)" >&2
  sudo docker ps --format '{{.Names}} {{.Status}}' | grep -vE "$infra" >&2
  exit 3
fi
mkdir -p "$root/wandb" "$root/logs"

{
  echo "[P49.FLPROF] source=$source_sha diff_sha256=$diff_sha image=$image"
  echo "[P49.FLPROF] topology=DP1xTP4 model=Qwen3-8B mode=P31_TRAIN trajectories=32 micro=2 grad_acc=16"
  echo "[P49.FLPROF] p31_max_steps=$p31_max_steps timeout_seconds=$timeout_seconds compile_cache=$compile_cache"
  if [ "$compile_cache" = "1" ]; then
    echo "[P49.FLPROF] cache_dir=$cache_dir entries_before=$(find "$cache_dir" -type f | wc -l)"
  fi
  echo "[P49.FLPROF] MODE_DELTAS_VS_P41_TEMPLATE=M1_p31_train M2_no_update_report M3_p31_cli_pin M4_pre_align_gate M5_timeout14400 M6_no_compile_cache M7_selfgen_p31_dataset_dir M8_offload_optimizer_8b_resident_ooms"
} >"$driver"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen8b \
  >>"$driver" 2>&1

{
  echo "[P49.FLPROF] RUN_BEGIN"
  sha256sum \
    "$repo/tunix/rl/agentic/agentic_rl_learner.py" \
    "$repo/tunix/rl/canonical_qwen3_adapter.py" \
    "$repo/tunix/rl/dp_workloads.py" \
    "$repo/tunix/sft/peft_trainer.py" \
    "$repo/examples/frozenlake/train_frozenlake_qwen3.py" \
    "$model/config.json" "$model/model.safetensors.index.json" "$0"
} >"$raw"

set +e
started="$(date +%s)"
timeout --foreground --signal=TERM --kill-after=120s "${timeout_seconds}s" \
sudo docker run --rm --privileged --net=host --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$repo":"$repo":ro \
  -v "$canon_out":"$canon_out":ro \
  -v "$canon_out/attn_iface_patched.py":"$sp/layers/common/attention_interface.py":ro \
  -v "$canon_out/linear_p22xk.py":"$sp/layers/jax/linear.py":ro \
  -v "$canon_out/embed_patched.py":"$sp/layers/jax/embed.py":ro \
  -v "$canon_out/tpu_runner_p21_l30.py":"$sp/runner/tpu_runner.py":ro \
  -v "$canon_out/qwen3_p22xk.py":"$sp/models/jax/qwen3.py":ro \
  -v "$canon_out/qwen2_p22xk.py":"$sp/models/jax/qwen2.py":ro \
  "${cache_env[@]}" \
  -e PYTHONPATH="$canon_out:$repo" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e MODEL_DOWNLOAD_DIR="$model" -e FROZENLAKE_DATA_DIR="$data" \
  -e WANDB_MODE=disabled -e WANDB_DIR="$root/wandb" \
  -e CANON_P29_LOG_DIR="$root/logs" \
  -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e CANON_FROZENLAKE_L3=1 -e CANON_FROZENLAKE_P27=1 \
  -e CANON_P31_CONVERGENCE=1 \
  -e CANON_P31_MAX_STEPS="$p31_max_steps" \
  -e CANON_P31_ENABLE_EVAL=0 \
  -e CANON_P29_FULL_TRAIN=1 \
  -e CANON_P28_G3_ONLY=0 -e CANON_P28_G4_ONLY=0 \
  -e CANON_P28_G5_ONLY=0 -e CANON_P28_G5C_ONLY=0 \
  -e CANON_P28_G6_UPDATE=1 \
  -e CANON_P28_SEGMENTED_FORWARD=1 -e CANON_P28_SEGMENTED_VJP=0 \
  -e CANON_P28_SEGMENTED_PULLBACK=0 -e CANON_P28_SEGMENTED_TRAIN=1 \
  -e CANON_P28_G5C_SHARED_LOGSOFTMAX=1 \
  -e CANON_P27_TRAJECTORY_MICRO=2 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=0 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=0 -e CANON_ALIGNMENT_TRAIN=1 \
  -e CANON_ALIGNMENT_EXPECTED_RED=0 \
  -e CANON_ALIGN_REPORT="$align" -e CANON_UPDATE_REPORT="$update" \
  -e CANON_PRE_ALIGN_GATE=1 -e CANON_PRE_ALIGN_REPORT="$pre_align" \
  -e CANON_BATCHED_EVIDENCE="${P49_BATCHED_EVIDENCE:-0}" \
  -e CANON_P28_LAYER_SCAN="${P49_LAYER_SCAN:-}" \
  -e CANON_P28_BATCHED_REPORT="${P49_BATCHED_REPORT:-}" \
  -e CANON_OPT_STATE_RESIDENT=0 -e CANON_P30_OPT_STATE_OFFLOAD=1 \
  -e CANON_P30_SPARSE_GRAD_ASSEMBLY=1 \
  -e CANON_P30_FUSED_PAIR_ACCUMULATION=0 \
  -e CANON_P30_REUSE_SEGMENTED_ENGINE=1 \
  -e CANON_P30_RELEASE_CAPTURED_STATE=1 \
  -e CANON_P30_POST_COMMIT_GC=0 -e CANON_P30_DONATE_MODEL=0 \
  -e CANON_P30_SHARDING_PROFILE=1 -e CANON_P30_RESHARD_ACCUMULATOR=1 \
  -e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4 -e FL_SHARED_MESH=1,4 \
  -e XLA_FLAGS="$XTRA_XLA" \
  -e CANON_RPA_D=128,512,128,512 -e CANON_RPA_P=128,512,128,512 \
  -e CANON_RPA_M=128,512,128,512 -e MIN_TOKEN_BUCKET=256 \
  -e CANON_FIXED_AR=1 -e CANON_FIXED_AR_EMBED=1 \
  -e CANON_RPA_VJP=1 -e CANON_RPA_VJP2=1 -e CANON_VJP2_MAX_SEQS=1 \
  -e CANON_ENGINE_MODULE_C=1 -e CANON_PROMPT_PROCESSED_LOGPROBS=1 \
  -e CANON_LOGPROB_M=256 -e CANON_PALLAS_LOGSOFTMAX=1 \
  -e CANON_PALLAS_ALL_PROJ=1 -e CANON_PALLAS_ALL_RMSNORM=1 \
  -e CANON_PALLAS_SWIGLU=1 -e CANON_PALLAS_MPAD=1 \
  -e CANON_PALLAS_SWIGLU_MPAD=1 -e CANON_PALLAS_CANONICAL_VJP=1 \
  -e CANON_QWEN3_HIDDEN_SIZE=4096 -e CANON_QWEN3_INTERMEDIATE_SIZE=12288 \
  -e CANON_QWEN3_NUM_ATTENTION_HEADS=32 -e CANON_QWEN3_NUM_KV_HEADS=8 \
  -e CANON_QWEN3_HEAD_DIM=128 -e CANON_QWEN3_TP_SIZE=4 \
  -e CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3 \
  -e CANON_PALLAS_MATMUL= -e CANON_PALLAS_MATERIALIZE= \
  -e CANON_CUT= -e CANON_TAIL= -e CANON_POSTRPA_M= \
  -w "$repo" "$image" bash -lc "
    set -e
    python3 '$pkg/tasks/p41-optimizer-residency/scripts/admit_frozenlake_runtime.py' \
      --gymnasium-wheel '$deps/gymnasium-1.3.0-py3-none-any.whl' \
      --farama-wheel '$deps/farama_notifications-0.0.6-py3-none-any.whl' \
      --report '$runtime'
    python3 - <<'PY'
import jax
devices = jax.devices()
print(f'[P49.FLPROF] devices={len(devices)} ids={[d.id for d in devices]} platform={jax.default_backend()}')
if len(devices) != 4 or jax.default_backend() != 'tpu':
  raise SystemExit('P49 FL profile requires exactly four TPU devices')
PY
    exec python3 -u examples/frozenlake/train_frozenlake_qwen3.py \
      --batch_size=4 --mini_batch_size=4 --num_batches=150 \
      --num_generations=8 --max_prompt_length=4096 \
      --max_response_length=2048 --max_concurrency=8 \
      --vllm_max_num_seqs=16 --vllm_max_num_batched_tokens=256 \
      --env_max_steps=5 --num_test_batches=25 --eval_every_n_steps=25 \
      --temperature=0.7 --top_k=0 --top_p=1.0 \
      --advantage_estimator=rloo
  " >>"$raw" 2>&1 &
docker_wait_pid=$!
# Crash watchdog: r4/r6 showed that after an uncaught main-thread exception
# the container lingers (non-daemon engine threads) until the outer timeout,
# wasting hours of ceiling.  A [rank0] traceback in raw.log is terminal on
# this stack; give teardown 180s of grace, then stop the container.
(
  while kill -0 "$docker_wait_pid" 2>/dev/null; do
    sleep 60
    if grep -aq '^\[rank0\]: Traceback (most recent call last)' "$raw"; then
      sleep 180
      if kill -0 "$docker_wait_pid" 2>/dev/null; then
        echo "[P49.FLPROF] crash_watchdog stopping lingering container" >>"$raw"
        sudo docker stop "$container" >/dev/null 2>&1 || true
      fi
      break
    fi
  done
) &
crash_watchdog_pid=$!
wait "$docker_wait_pid"
docker_rc=$?
kill "$crash_watchdog_pid" 2>/dev/null || true
# The outer timeout TERMs the docker client, not the container; r6 lingered
# four hours holding the TPU.  Stop it explicitly if it survived.
if sudo docker ps --format '{{.Names}}' | grep -q "^${container}\$"; then
  echo "[P49.FLPROF] post_run_cleanup stopping lingering container" >>"$raw"
  sudo docker stop "$container" >/dev/null 2>&1 || true
fi
set -e
elapsed=$(( $(date +%s) - started ))
echo "[P49.FLPROF] RUN_END docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
canon_postflight "$raw" 2>&1 | sed 's/^/[P49.FLPROF.POSTFLIGHT] /' >>"$raw" || true

steps_done="$(grep -ac 'Global step .* completed in' "$raw" || true)"
pre_pass="$(grep -ac '\[CANON_ALIGN_PRE\] step=.* verdict=PASS' "$raw" || true)"
echo "[P49.FLPROF] SUMMARY steps_done=${steps_done:-0} pre_align_pass=${pre_pass:-0}" \
  | tee -a "$driver" >>"$raw"
if [ "$docker_rc" -ne 0 ]; then
  echo "[P49.FLPROF] run failed exit=$docker_rc" | tee -a "$driver"
  exit 1
fi
sha256sum "$raw" "$align" "$pre_align" "$runtime" >>"$driver" 2>/dev/null || \
  sha256sum "$raw" >>"$driver"
if [ "$compile_cache" = "1" ]; then
  echo "[P49.FLPROF] CACHE_SUMMARY entries_after=$(find "$cache_dir" -type f | wc -l) bytes=$(du -sb "$cache_dir" | awk '{print $1}')" | tee -a "$driver"
fi
echo "[P49.FLPROF] RUN_COMPLETE root=$root" | tee -a "$driver"
