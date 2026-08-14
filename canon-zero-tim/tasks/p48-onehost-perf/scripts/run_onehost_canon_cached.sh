#!/usr/bin/env bash
# P48 Phase 2: one canonical bounded L3 resident arm with the JAX persistent
# compilation cache enabled.  Run twice with the same label prefix:
#   run_onehost_canon_cached.sh <label>_cold   (fresh cache dir key -> fills)
#   run_onehost_canon_cached.sh <label>_warm   (same key -> must hit)
# Discipline (tasks/p48_onehost_perf_optimization/phase2_compile_cache.md):
# - cache dir scoped by the full executable contract: source sha + image tag
#   + topology + mode (l3-resident switch set is constant in this vehicle);
#   a dirty worktree is refused so the key stays honest;
# - hit/miss judged by JAX's own log lines (JAX_DEBUG_LOG_MODULES on
#   jax._src.compilation_cache), never by timing deltas;
# - warm and cold must pass the same numeric gates ([CANON_ALIGN] all-zero,
#   update.json canary, hashes vs the pre-cache baselines).
# Docker env is the P41 resident arm byte-for-byte plus exactly two vars:
# JAX_COMPILATION_CACHE_DIR and JAX_DEBUG_LOG_MODULES.
set -euo pipefail

label="${1:?usage: run_onehost_canon_cached.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'') echo "[P48.CACHED] invalid label: $label" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
assets=/mnt/disks/tunix-data/gsm8k_zero_tim
model="$assets/models"
hf_model_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-1.7B
hf_snapshot_sha="$(cat "$hf_model_cache/refs/main")"
hf_snapshot="$hf_model_cache/snapshots/$hf_snapshot_sha"
evidence=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
source_sha="$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" diff --quiet HEAD -- tunix examples canon-zero-tim/install.sh canon-zero-tim/src canon-zero-tim/patches || {
  echo "[P48.CACHED] refusing: dirty worktree would poison the cache contract key" >&2
  exit 2
}
# Key on the tree hashes of the paths that actually shape the compiled
# programs, not HEAD: a script-only commit must not rotate the namespace
# (learned from p48cache_warmro_20260813, which landed in an empty dir).
contract_tree="$(git -C "$repo" rev-parse HEAD:tunix HEAD:examples \
  HEAD:canon-zero-tim/src HEAD:canon-zero-tim/patches \
  HEAD:canon-zero-tim/install.sh | sha256sum | cut -c1-12)"
cache_key="p48_${contract_tree}_vllm-tpu0250_dp1tp4_l3res"
cache_dir="/mnt/disks/tunix-data/jax_cache/$cache_key"
root="$evidence/p48_cached_${label}"
canon_out="$root/canon"
driver="$root/driver.log"
timeout_seconds="${P48_ONEHOST_TIMEOUT_SECONDS:-2700}"

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -s "$model/config.json"
test -s "$assets/data/gsm8k/1.0.0/dataset_info.json"
[[ "$hf_snapshot_sha" =~ ^[0-9a-f]{40}$ ]] || exit 2
test ! -e "$root"
mkdir -p "$root" "$cache_dir"

# P48_CACHE_RO=1 remounts the cache dir read-only inside the container: a
# run that still shows the warm collapse with zero new entries proves the
# executables were READ from the cache (the "hit" judgment without relying
# on timing), because JAX_DEBUG_LOG_MODULES emits nothing in this image.
cache_ro_mount=()
if [ "${P48_CACHE_RO:-0}" = "1" ]; then
  cache_ro_mount=( -v "$cache_dir":"$cache_dir":ro )
fi

entries_before="$(find "$cache_dir" -type f | wc -l)"
{
  echo "[P48.CACHED] source=$source_sha image=$image cache_key=$cache_key"
  echo "[P48.CACHED] cache_entries_before=$entries_before cache_ro=${P48_CACHE_RO:-0} timeout_seconds=$timeout_seconds"
} >"$driver"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen1p7b \
  >>"$driver" 2>&1

state="$root/resident"
raw="$state/raw.log"
align="$state/alignment.jsonl"
update="$state/update.json"
container="p48_cached_${label}"
mkdir -p "$state/wandb" "$state/logs"
echo "[P48.CACHED] ARM_BEGIN arm=resident cache_dir=$cache_dir" >"$raw"

set +e
started="$(date +%s)"
timeout --foreground --signal=TERM --kill-after=120s "${timeout_seconds}s" \
sudo docker run --rm --privileged --net=host --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$model":"$hf_snapshot":ro \
  -v "$repo":"$repo":ro \
  -v "$canon_out":"$canon_out":ro \
  -v "$canon_out/attn_iface_patched.py":"$sp/layers/common/attention_interface.py":ro \
  -v "$canon_out/linear_p22xk.py":"$sp/layers/jax/linear.py":ro \
  -v "$canon_out/embed_patched.py":"$sp/layers/jax/embed.py":ro \
  -v "$canon_out/tpu_runner_p21_l30.py":"$sp/runner/tpu_runner.py":ro \
  -v "$canon_out/qwen3_p22xk.py":"$sp/models/jax/qwen3.py":ro \
  -v "$canon_out/qwen2_p22xk.py":"$sp/models/jax/qwen2.py":ro \
  "${cache_ro_mount[@]}" \
  -e JAX_COMPILATION_CACHE_DIR="$cache_dir" \
  -e JAX_DEBUG_LOG_MODULES=jax._src.compilation_cache \
  -e PYTHONPATH="$canon_out:$repo" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e GSM8K_ARTIFACT_ROOT="$assets" -e GSM8K_LOG_DIR="$state/logs" \
  -e WANDB_MODE=disabled -e WANDB_DIR="$state/wandb" \
  -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e CANON_GSM8K_L3=1 -e CANON_GSM8K_TRAIN=0 \
  -e CANON_GSM8K_UPDATE_CANARY=1 -e CANON_GSM8K_GRAD_PROBE=1 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=0 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=1 -e CANON_ALIGNMENT_TRAIN=0 \
  -e CANON_ALIGN_REPORT="$align" -e CANON_UPDATE_REPORT="$update" \
  -e CANON_OPT_STATE_RESIDENT=1 \
  -e CANON_P30_OPT_STATE_OFFLOAD=0 \
  -e CANON_P28_SEGMENTED_FORWARD=1 -e CANON_P28_SEGMENTED_VJP=0 \
  -e CANON_P28_SEGMENTED_TRAIN=1 -e CANON_P28_G5C_ONLY=0 \
  -e CANON_P28_G6_UPDATE=1 -e CANON_P30_DONATE_MODEL=0 \
  -e CANON_P41_OPTIMIZER_BENCH=1 \
  -e CANON_P30_SPARSE_GRAD_ASSEMBLY=1 \
  -e CANON_P30_FUSED_PAIR_ACCUMULATION=0 \
  -e CANON_P30_REUSE_SEGMENTED_ENGINE=1 \
  -e CANON_P30_RELEASE_CAPTURED_STATE=1 \
  -e CANON_P30_POST_COMMIT_GC=0 \
  -e CANON_P30_SHARDING_PROFILE=1 -e CANON_P30_RESHARD_ACCUMULATOR=1 \
  -e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4 -e FL_SHARED_MESH=1,4 \
  -e XLA_FLAGS="$XTRA_XLA" \
  -e CANON_RPA_D=128,512,128,512 -e CANON_RPA_P=128,512,128,512 \
  -e CANON_RPA_M=128,512,128,512 -e MIN_TOKEN_BUCKET=256 \
  -e CANON_FIXED_AR=1 -e CANON_FIXED_AR_EMBED=1 \
  -e CANON_RPA_VJP2=1 -e CANON_VJP2_MAX_SEQS=1 \
  -e CANON_ENGINE_MODULE_C=1 -e CANON_PROMPT_PROCESSED_LOGPROBS=1 \
  -e CANON_LOGPROB_M=256 -e CANON_PALLAS_LOGSOFTMAX=1 \
  -e CANON_PALLAS_ALL_PROJ=1 -e CANON_PALLAS_ALL_RMSNORM=1 \
  -e CANON_PALLAS_SWIGLU=1 -e CANON_PALLAS_MPAD=1 \
  -e CANON_PALLAS_SWIGLU_MPAD=1 -e CANON_PALLAS_CANONICAL_VJP=1 \
  -e CANON_QWEN3_HIDDEN_SIZE=2048 -e CANON_QWEN3_INTERMEDIATE_SIZE=6144 \
  -e CANON_QWEN3_NUM_ATTENTION_HEADS=16 -e CANON_QWEN3_NUM_KV_HEADS=8 \
  -e CANON_QWEN3_HEAD_DIM=128 -e CANON_QWEN3_TP_SIZE=4 \
  -e CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3 \
  -w "$repo" "$image" bash -lc "
    set -e
    echo '[P48.CACHED] container_start'
    python3 - <<'PY'
import jax
devices = jax.devices()
print(f'[P48.CACHED] devices={len(devices)} ids={[d.id for d in devices]} platform={jax.default_backend()}')
if len(devices) != 4 or jax.default_backend() != 'tpu':
  raise SystemExit('P48 cached arm requires exactly four TPU devices')
PY
    exec python3 -u examples/math_gsm8k/qwen3_grpo_demo.py \
      --mesh_fsdp=1 --mesh_tp=4 --batch_size=1 --mini_batch_size=1 \
      --train_micro_batch_size=1 --compute_logps_micro_batch_size=1 \
      --train_trajectory_micro_batch_size=2 \
      --max_steps=1 --num_generations=2 --max_prompt_length=256 \
      --max_response_length=64 --max_concurrency=1 \
      --rollout_vllm_hbm_utilization=0.20 \
      --rollout_vllm_max_num_seqs=2 \
      --rollout_vllm_max_num_batched_tokens=256
  " >>"$raw" 2>&1
docker_rc=$?
set -e
elapsed=$(( $(date +%s) - started ))
echo "[P48.CACHED] ARM_END arm=resident docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
canon_postflight "$raw" 2>&1 | sed 's/^/[P48.CACHED.POSTFLIGHT] /' >>"$raw" || true

entries_after="$(find "$cache_dir" -type f | wc -l)"
cache_bytes="$(du -sb "$cache_dir" | awk '{print $1}')"
hits="$(grep -ac 'ache hit' "$raw" || true)"
writes="$(grep -ac 'ompilation cache\|Writing .* to persistent' "$raw" || true)"
{
  echo "[P48.CACHED] CACHE_SUMMARY entries_before=$entries_before entries_after=$entries_after cache_bytes=$cache_bytes log_hit_lines=${hits:-0} log_cache_lines=${writes:-0}"
  echo "[P48.CACHED] RUN_END docker_exit=$docker_rc elapsed_seconds=$elapsed"
} | tee -a "$driver" >>"$raw"
if [ "$docker_rc" -ne 0 ] || [ ! -s "$update" ]; then
  echo "[P48.CACHED] arm failed exit=$docker_rc update=$update" | tee -a "$driver"
  exit 1
fi
sha256sum "$raw" "$align" "$update" >>"$driver"
echo "[P48.CACHED] RUN_COMPLETE root=$root" | tee -a "$driver"
