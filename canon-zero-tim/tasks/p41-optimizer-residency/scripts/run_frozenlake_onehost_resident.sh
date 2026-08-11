#!/usr/bin/env bash
# Run one strict Qwen3-8B device-resident FrozenLake update on the v5p host.
set -euo pipefail

label="${1:?usage: run_frozenlake_onehost_resident.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P41.FROZENLAKE] invalid label: $label" >&2
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
data=/mnt/disks/tunix-data/frozenlake/data
deps=/mnt/disks/tunix-data/frozenlake/deps
image=tunix_frozenlake_image:vllm-tpu0.25.0
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
root=/mnt/disks/tunix-data/logp_probe_1host/p41_frozenlake_${label}
canon_out="$root/canon"
raw="$root/raw.log"
align="$root/alignment.jsonl"
update="$root/update.json"
runtime="$root/runtime.json"
result="$root/resident.classification.json"
driver="$root/driver.log"
container="p41_frozenlake_${label}"
timeout_seconds="${P41_FROZENLAKE_TIMEOUT_SECONDS:-2700}"

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
[[ "$model_sha" =~ ^[0-9a-f]{40}$ ]]
test -s "$model/config.json"
test "$(find "$model" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)" -eq 5
test -s "$data/train.parquet"
test -s "$data/test.parquet"
test -s "$deps/gymnasium-1.3.0-py3-none-any.whl"
test -s "$deps/farama_notifications-0.0.6-py3-none-any.whl"
test ! -e "$root"
mkdir -p "$root/wandb" "$root/logs"

{
  echo "[P41.FROZENLAKE] source=$source_sha diff_sha256=$diff_sha image=$image"
  echo "[P41.FROZENLAKE] topology=DP1xTP4 model=Qwen3-8B trajectories=8 microbatches=4 updates=1"
  echo "[P41.FROZENLAKE] placement=device-resident strict_alignment=1 timeout_seconds=$timeout_seconds"
} >"$driver"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen8b \
  >>"$driver" 2>&1

{
  echo "[P41.FROZENLAKE] RUN_BEGIN"
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
  -e CANON_P29_FULL_TRAIN=1 \
  -e CANON_P28_G3_ONLY=0 -e CANON_P28_G4_ONLY=0 \
  -e CANON_P28_G5_ONLY=0 -e CANON_P28_G5C_ONLY=0 \
  -e CANON_P28_G6_UPDATE=1 \
  -e CANON_P28_SEGMENTED_FORWARD=1 -e CANON_P28_SEGMENTED_VJP=0 \
  -e CANON_P28_SEGMENTED_PULLBACK=0 -e CANON_P28_SEGMENTED_TRAIN=1 \
  -e CANON_P28_G5C_SHARED_LOGSOFTMAX=1 \
  -e CANON_P27_TRAJECTORY_MICRO=2 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=0 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=1 -e CANON_ALIGNMENT_TRAIN=0 \
  -e CANON_ALIGNMENT_EXPECTED_RED=0 \
  -e CANON_ALIGN_REPORT="$align" -e CANON_UPDATE_REPORT="$update" \
  -e CANON_OPT_STATE_RESIDENT=1 -e CANON_P30_OPT_STATE_OFFLOAD=0 \
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
    python3 '$script_dir/admit_frozenlake_runtime.py' \
      --gymnasium-wheel '$deps/gymnasium-1.3.0-py3-none-any.whl' \
      --farama-wheel '$deps/farama_notifications-0.0.6-py3-none-any.whl' \
      --report '$runtime'
    python3 - <<'PY'
import jax
devices = jax.devices()
print(f'[P41.FROZENLAKE] devices={len(devices)} ids={[d.id for d in devices]} platform={jax.default_backend()}')
if len(devices) != 4 or jax.default_backend() != 'tpu':
  raise SystemExit('P41 FrozenLake canary requires exactly four TPU devices')
PY
    exec python3 -u examples/frozenlake/train_frozenlake_qwen3.py \
      --batch_size=4 --mini_batch_size=4 --num_batches=1 \
      --num_generations=2 --max_prompt_length=2048 \
      --max_response_length=64 --max_concurrency=2 \
      --vllm_max_num_seqs=2 --vllm_max_num_batched_tokens=256 \
      --env_max_steps=2 --temperature=0.7 --top_k=0 --top_p=1.0
  " >>"$raw" 2>&1
docker_rc=$?
set -e
elapsed=$(( $(date +%s) - started ))
echo "[P41.FROZENLAKE] RUN_END docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
canon_postflight "$raw" 2>&1 | sed 's/^/[P41.FROZENLAKE.POSTFLIGHT] /' >>"$raw" || true

if [ "$docker_rc" -ne 0 ] || [ ! -s "$update" ]; then
  echo "[P41.FROZENLAKE] run failed exit=$docker_rc update=$update" | tee -a "$driver"
  exit 1
fi
set +e
python3 "$script_dir/classify_frozenlake_resident.py" \
  --update "$update" --output "$result" | tee -a "$driver"
classifier_rc=${PIPESTATUS[0]}
set -e
sha256sum "$raw" "$align" "$update" "$runtime" "$result" "$0" \
  "$script_dir/classify_frozenlake_resident.py" | tee -a "$driver"
echo "[P41.FROZENLAKE] CAPACITY_COMPLETE result=$result classifier_exit=$classifier_rc" \
  | tee -a "$driver"
exit "$classifier_rc"
