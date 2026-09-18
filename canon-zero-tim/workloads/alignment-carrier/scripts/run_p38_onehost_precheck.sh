#!/usr/bin/env bash
# Run one bounded P38 pre-backward reproduction on the authorized v5p host.
set -euo pipefail

label="${1:?usage: run_p38_onehost_precheck.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P38.ONEHOST] REFUSING: invalid label: $label" >&2
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
state="$evidence/p38_onehost_${label}"
raw="$evidence/p38_onehost_${label}.raw.log"
wrapper="$evidence/p38_onehost_${label}.wrapper.log"
result="$evidence/p38_onehost_${label}.result.json"
pre_report="$state/pre_alignment.jsonl"
canon_out="$state/canon"
container="p38_onehost_${label}"
timeout_seconds="${P38_ONEHOST_TIMEOUT_SECONDS:-2700}"

source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -s "$model/config.json"
[[ "$hf_snapshot_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "invalid Qwen3-1.7B cache ref: $hf_snapshot_sha" >&2
  exit 2
}
test -s "$assets/data/gsm8k/1.0.0/dataset_info.json"
for path in "$state" "$raw" "$wrapper" "$result"; do
  if [ -e "$path" ]; then
    echo "[P38.ONEHOST] REFUSING: evidence path exists: $path" >&2
    exit 3
  fi
done
mkdir -p "$state/wandb" "$state/logs"

{
  echo "[P38.ONEHOST] source=$source_sha diff_sha256=$diff_sha image=$image"
  echo "[P38.ONEHOST] topology=DP1xTP4 prompts=2 generations=8 trajectories=16 local_m=256"
  echo "[P38.ONEHOST] mutation=none stop=before_backward wandb=disabled timeout=$timeout_seconds"
  sha256sum "$0" \
    "$pkg/install.sh" \
    "$repo/tunix/rl/alignment.py" \
    "$repo/tunix/rl/canonical_qwen3_adapter.py" \
    "$repo/tunix/rl/agentic/agentic_grpo_learner.py" \
    "$repo/tunix/rl/rollout/vllm_rollout.py" \
    "$repo/examples/math_gsm8k/qwen3_grpo_demo.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} >"$raw"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen1p7b \
  >>"$raw" 2>&1

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
  -e CANON_GSM8K_TRAIN=1 -e CANON_GSM8K_L3=0 \
  -e CANON_GSM8K_UPDATE_CANARY=0 -e CANON_GSM8K_GRAD_PROBE=0 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=0 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=0 -e CANON_ALIGNMENT_TRAIN=1 \
  -e CANON_PRE_ALIGN_GATE=1 -e CANON_PRE_ALIGN_REPORT="$pre_report" \
  -e CANON_P38_PRECHECK_ONLY=1 \
  -e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4 -e CANON_TARGET_M=256 \
  -e CANON_P32_TRAIN_ADMITTED=0 -e FL_SHARED_MESH=1,4 \
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
    for pair in \
      'attn_iface_patched.py layers/common/attention_interface.py' \
      'linear_p22xk.py layers/jax/linear.py' \
      'embed_patched.py layers/jax/embed.py' \
      'tpu_runner_p21_l30.py runner/tpu_runner.py' \
      'qwen3_p22xk.py models/jax/qwen3.py' \
      'qwen2_p22xk.py models/jax/qwen2.py'; do
      set -- \$pair
      cmp '$canon_out/'\"\$1\" '$sp/'\"\$2\"
    done
    echo '[P38.ONEHOST] OVERLAY_BYTE_IDENTITY PASS files=6'
    echo '[P38.ONEHOST] container_start'
    python3 - <<'PY'
import jax
devices = jax.devices()
print(f'[P38.ONEHOST] devices={len(devices)} ids={[d.id for d in devices]} platform={jax.default_backend()}')
if len(devices) != 4 or jax.default_backend() != 'tpu':
  raise SystemExit('P38 one-host reproduction requires exactly four TPU devices')
PY
    exec python3 -u examples/math_gsm8k/qwen3_grpo_demo.py \
      --mesh_fsdp=1 --mesh_tp=4 --batch_size=2 --mini_batch_size=2 \
      --train_micro_batch_size=2 --compute_logps_micro_batch_size=2 \
      --train_trajectory_micro_batch_size=16 \
      --max_steps=1 --num_generations=8 \
      --max_prompt_length=1024 --max_response_length=1024 \
      --max_concurrency=16 --rollout_vllm_hbm_utilization=0.20 \
      --rollout_vllm_max_num_seqs=16 \
      --rollout_vllm_max_num_batched_tokens=256 \
      --wandb_project=zero-tim-p38-onehost \
      --wandb_run_name=p38-onehost-$label
  " >>"$raw" 2>&1
docker_rc=$?
elapsed="$(( $(date +%s) - started ))"
set -e

echo "[P38.ONEHOST] docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
canon_postflight "$raw" 2>&1 | sed 's/^/[P38.ONEHOST.POSTFLIGHT] /' >>"$raw" || true

if [ ! -s "$pre_report" ]; then
  echo "[P38.ONEHOST] missing pre-alignment report" | tee -a "$wrapper"
  exit 1
fi

set +e
PYTHONPATH="$repo" python3 \
  "$script_dir/classify_p38_onehost.py" \
  --report "$pre_report" --raw "$raw" --output "$result" | tee -a "$wrapper"
classifier_rc="${PIPESTATUS[0]}"
set -e
sha256sum "$raw" "$pre_report" "$result" "$0" | tee -a "$wrapper"
echo "[P38.ONEHOST] classifier_exit=$classifier_rc result=$result" | tee -a "$wrapper"
exit "$classifier_rc"
