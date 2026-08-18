#!/usr/bin/env bash
# One-host Qwen3-1.7B GSM8K profiling run: 3 training steps in CANON_GSM8K_TRAIN
# mode with the shipping perf envs on, capturing one warm step (step 2) as an
# XProf xplane + perfetto_trace.json.gz via the CANON_XPROF_* learner hook.
# Derived from tasks/p41-optimizer-residency/scripts/run_onehost_pair.sh; one
# arm, resident optimizer placement. Profile-run numbers are for shape only,
# never for A/B deltas (the python tracer perturbs step time).
set -euo pipefail

label="${1:?usage: run_onehost_gsm8k_xprof.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P51.XPROF] invalid label: $label" >&2
    exit 2
    ;;
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
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
root="$evidence/p51_gsm8k_xprof_${label}"
canon_out="$root/canon"
driver="$root/driver.log"
timeout_seconds="${P51_ONEHOST_TIMEOUT_SECONDS:-2700}"
max_steps="${P51_MAX_STEPS:-3}"
xprof_skip="${P51_XPROF_SKIP_STEPS:-2}"
xprof_steps="${P51_XPROF_STEPS:-1}"

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -s "$model/config.json"
test -s "$assets/data/gsm8k/1.0.0/dataset_info.json"
[[ "$hf_snapshot_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[P51.XPROF] invalid Qwen3-1.7B cache ref: $hf_snapshot_sha" >&2
  exit 2
}
test ! -e "$root"
mkdir -p "$root"

{
  echo "[P51.XPROF] source=$source_sha diff_sha256=$diff_sha image=$image"
  echo "[P51.XPROF] topology=DP1xTP4 batch=4 generations=8 traj_micro=2"
  echo "[P51.XPROF] max_steps=$max_steps capture=step$xprof_skip x$xprof_steps"
  echo "[P51.XPROF] perf_envs=CANON_BATCHED_EVIDENCE=1,CANON_P28_BATCHED_REPORT=1"
} >"$driver"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen1p7b \
  >>"$driver" 2>&1

state="$root/train"
raw="$state/raw.log"
align="$state/alignment.jsonl"
update="$state/update.json"
xprof_dir="$state/xprof"
container="p51_gsm8k_xprof_${label}"
mkdir -p "$state/wandb" "$state/logs" "$xprof_dir"
{
  echo "[P51.XPROF] RUN_BEGIN"
  sha256sum \
    "$repo/tunix/rl/agentic/agentic_rl_learner.py" \
    "$repo/tunix/rl/dp_workloads.py" \
    "$repo/tunix/sft/peft_trainer.py" \
    "$repo/tunix/rl/canonical_qwen3_adapter.py" \
    "$repo/examples/math_gsm8k/qwen3_grpo_demo.py" \
    "$model/config.json" "$model/model.safetensors.index.json" "$0"
} >"$raw"

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
  -e PYTHONPATH="$canon_out:$repo" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e GSM8K_ARTIFACT_ROOT="$assets" -e GSM8K_LOG_DIR="$state/logs" \
  -e WANDB_MODE=disabled -e WANDB_DIR="$state/wandb" \
  -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e CANON_GSM8K_L3=0 -e CANON_GSM8K_TRAIN=1 \
  -e CANON_GSM8K_GRAD_PROBE=0 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=0 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=0 -e CANON_ALIGNMENT_TRAIN=1 \
  -e CANON_ALIGN_REPORT="$align" -e CANON_UPDATE_REPORT="$update" \
  -e CANON_OPT_STATE_RESIDENT=1 \
  -e CANON_P30_OPT_STATE_OFFLOAD=0 \
  -e CANON_P28_SEGMENTED_FORWARD=1 -e CANON_P28_SEGMENTED_VJP=0 \
  -e CANON_P28_SEGMENTED_TRAIN=1 -e CANON_P28_G5C_ONLY=0 \
  -e CANON_P28_G6_UPDATE=1 -e CANON_P30_DONATE_MODEL=0 \
  -e CANON_P30_SPARSE_GRAD_ASSEMBLY=1 \
  -e CANON_P30_FUSED_PAIR_ACCUMULATION=0 \
  -e CANON_P30_REUSE_SEGMENTED_ENGINE=1 \
  -e CANON_P30_RELEASE_CAPTURED_STATE=1 \
  -e CANON_P30_POST_COMMIT_GC=0 \
  -e CANON_P30_SHARDING_PROFILE=1 -e CANON_P30_RESHARD_ACCUMULATOR=1 \
  -e CANON_BATCHED_EVIDENCE=1 \
  -e CANON_P28_BATCHED_REPORT=1 \
  -e CANON_XPROF_DIR="$xprof_dir" \
  -e CANON_XPROF_SKIP_STEPS="$xprof_skip" \
  -e CANON_XPROF_STEPS="$xprof_steps" \
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
    echo '[P51.XPROF] container_start'
    python3 - <<'PY'
import jax
devices = jax.devices()
print(f'[P51.XPROF] devices={len(devices)} ids={[d.id for d in devices]} platform={jax.default_backend()}')
if len(devices) != 4 or jax.default_backend() != 'tpu':
  raise SystemExit('P51 one-host profiling requires exactly four TPU devices')
PY
    exec python3 -u examples/math_gsm8k/qwen3_grpo_demo.py \
      --mesh_fsdp=1 --mesh_tp=4 --batch_size=4 --mini_batch_size=4 \
      --train_micro_batch_size=4 --compute_logps_micro_batch_size=4 \
      --train_trajectory_micro_batch_size=2 \
      --max_steps=$max_steps --num_generations=8 --max_prompt_length=512 \
      --max_response_length=1024 --max_concurrency=8 \
      --rollout_vllm_hbm_utilization=0.30 \
      --rollout_vllm_max_num_seqs=16 \
      --rollout_vllm_max_num_batched_tokens=256
  " >>"$raw" 2>&1
docker_rc=$?
set -e
if sudo docker ps --format '{{.Names}}' | grep -q "^${container}\$"; then
  echo "[P51.XPROF] post_run_cleanup stopping lingering container" >>"$raw"
  sudo docker stop "$container" >/dev/null 2>&1 || true
fi
elapsed=$(( $(date +%s) - started ))
echo "[P51.XPROF] RUN_END docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
canon_postflight "$raw" 2>&1 | sed 's/^/[P51.XPROF.POSTFLIGHT] /' >>"$raw" || true

steps_done="$(grep -ac 'Global step .* completed in' "$raw" || true)"
align_pass="$(grep -ac 'verdict=PASS' "$raw" || true)"
xprof_started="$(grep -ac '\[P51.XPROF\] start dir=' "$raw" || true)"
xprof_stopped="$(grep -ac '\[P51.XPROF\] stop captured_steps=' "$raw" || true)"
xplane_count="$(find "$xprof_dir" -name '*.xplane.pb' 2>/dev/null | wc -l)"
perfetto_count="$(find "$xprof_dir" \( -name '*perfetto*' -o -name '*.trace.json.gz' \) 2>/dev/null | wc -l)"
{
  echo "[P51.XPROF] SUMMARY steps_done=${steps_done:-0} align_pass=${align_pass:-0}" \
    "xprof_started=${xprof_started:-0} xprof_stopped=${xprof_stopped:-0}" \
    "xplane_files=${xplane_count} perfetto_files=${perfetto_count}"
  find "$xprof_dir" -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -10
} | tee -a "$driver" >>"$raw"
if [ "$docker_rc" -ne 0 ] || [ "${xplane_count}" -eq 0 ] || [ "${perfetto_count}" -eq 0 ]; then
  echo "[P51.XPROF] RED docker_exit=$docker_rc xplane=$xplane_count perfetto=$perfetto_count" | tee -a "$driver"
  exit 1
fi
echo "[P51.XPROF] GREEN artifacts under $xprof_dir" | tee -a "$driver"
