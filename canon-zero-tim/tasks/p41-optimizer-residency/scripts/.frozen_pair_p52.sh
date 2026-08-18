#!/usr/bin/env bash
# Run one bounded Qwen3-1.7B optimizer-placement pair on the authorized v5p host.
set -euo pipefail

label="${1:?usage: run_onehost_pair.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P41.OPTIMIZER] invalid label: $label" >&2
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
root="$evidence/p41_optimizer_${label}"
canon_out="$root/canon"
result="$root/pair.classification.json"
driver="$root/driver.log"
timeout_seconds="${P41_ONEHOST_TIMEOUT_SECONDS:-2700}"

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -s "$model/config.json"
test -s "$assets/data/gsm8k/1.0.0/dataset_info.json"
[[ "$hf_snapshot_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[P41.OPTIMIZER] invalid Qwen3-1.7B cache ref: $hf_snapshot_sha" >&2
  exit 2
}
test ! -e "$root"
mkdir -p "$root"

{
  echo "[P41.OPTIMIZER] source=$source_sha diff_sha256=$diff_sha image=$image"
  echo "[P41.OPTIMIZER] topology=DP1xTP4 trajectories=2 microbatches=1 updates=1"
  echo "[P41.OPTIMIZER] arms=offload,resident timeout_seconds=$timeout_seconds"
} >"$driver"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen1p7b \
  >>"$driver" 2>&1

run_arm() {
  local arm="$1" resident="$2" offload="$3"
  local state="$root/$arm"
  local raw="$state/raw.log"
  local align="$state/alignment.jsonl"
  local update="$state/update.json"
  local container="p41_optimizer_${label}_${arm}"
  mkdir -p "$state/wandb" "$state/logs"
  {
    echo "[P41.OPTIMIZER] ARM_BEGIN arm=$arm resident=$resident offload=$offload"
    sha256sum \
      "$repo/tunix/rl/agentic/agentic_rl_learner.py" \
      "$repo/tunix/rl/dp_workloads.py" \
      "$repo/tunix/sft/peft_trainer.py" \
      "$repo/examples/math_gsm8k/qwen3_grpo_demo.py" \
      "$model/config.json" "$model/model.safetensors.index.json" "$0"
  } >"$raw"

  set +e
  local started
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
    -e CANON_GSM8K_L3=1 -e CANON_GSM8K_TRAIN=0 \
    -e CANON_GSM8K_UPDATE_CANARY=1 -e CANON_GSM8K_GRAD_PROBE=1 \
    -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=0 \
    -e CANON_ALIGNMENT_UPDATE_CANARY=1 -e CANON_ALIGNMENT_TRAIN=0 \
    -e CANON_ALIGN_REPORT="$align" -e CANON_UPDATE_REPORT="$update" \
    -e CANON_OPT_STATE_RESIDENT="$resident" \
    -e CANON_P30_OPT_STATE_OFFLOAD="$offload" \
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
      echo '[P41.OPTIMIZER] container_start arm=$arm'
      python3 - <<'PY'
import jax
devices = jax.devices()
print(f'[P41.OPTIMIZER] devices={len(devices)} ids={[d.id for d in devices]} platform={jax.default_backend()}')
if len(devices) != 4 or jax.default_backend() != 'tpu':
  raise SystemExit('P41 one-host canary requires exactly four TPU devices')
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
  local docker_rc=$?
  set -e
  local elapsed=$(( $(date +%s) - started ))
  echo "[P41.OPTIMIZER] ARM_END arm=$arm docker_exit=$docker_rc elapsed_seconds=$elapsed" \
    >>"$raw"
  canon_postflight "$raw" 2>&1 | sed "s/^/[P41.$arm.POSTFLIGHT] /" \
    >>"$raw" || true
  if [ "$docker_rc" -ne 0 ] || [ ! -s "$update" ]; then
    echo "[P41.OPTIMIZER] arm failed: $arm exit=$docker_rc update=$update" \
      | tee -a "$driver"
    return 1
  fi
  sha256sum "$raw" "$align" "$update" >>"$driver"
}

run_arm offload 0 1
run_arm resident 1 0

python3 "$script_dir/classify_onehost_pair.py" \
  --offload "$root/offload/update.json" \
  --resident "$root/resident/update.json" \
  --output "$result" | tee -a "$driver"
sha256sum "$result" "$0" "$script_dir/classify_onehost_pair.py" | tee -a "$driver"
echo "[P41.OPTIMIZER] PAIR_COMPLETE result=$result" | tee -a "$driver"
