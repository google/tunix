#!/usr/bin/env bash
# Run one bounded P38 FrozenLake R0/R1 replay on the authorized v5p host.
set -euo pipefail

capsule="${1:?usage: run_p38_frozenlake_replay.sh <capsule.npz> <unique-label>}"
label="${2:?usage: run_p38_frozenlake_replay.sh <capsule.npz> <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P38.FL.REPLAY] REFUSING: invalid label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
capsule="$(realpath "$capsule")"
case "$capsule" in
  /mnt/disks/tunix-data/*|"$repo"/*) ;;
  *)
    echo "[P38.FL.REPLAY] REFUSING: capsule is outside mounted roots: $capsule" >&2
    exit 2
    ;;
esac
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
hf_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
hf_snapshot_sha="$(cat "$hf_cache/refs/main")"
model="$hf_cache/snapshots/$hf_snapshot_sha"
data=/mnt/disks/tunix-data/frozenlake/data_convergence_v1
evidence=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
state="$evidence/p38_fl_replay_${label}"
raw="$evidence/p38_fl_replay_${label}.raw.log"
wrapper="$evidence/p38_fl_replay_${label}.wrapper.log"
schedule_report="$state/schedule.json"
report="$state/replay.json"
classification="$state/replay.classification.json"
canon_out="$state/canon"
container="p38_fl_replay_${label}"
timeout_seconds="${P38_FL_REPLAY_TIMEOUT_SECONDS:-3600}"

source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -f "$capsule"
[[ "$hf_snapshot_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "invalid Qwen3-8B cache ref: $hf_snapshot_sha" >&2
  exit 2
}
test -s "$model/config.json"
test -s "$data/train.parquet"
for path in "$state" "$raw" "$wrapper"; do
  if [ -e "$path" ]; then
    echo "[P38.FL.REPLAY] REFUSING: evidence path exists: $path" >&2
    exit 3
  fi
done
mkdir -p "$state"

python3 "$script_dir/prepare_p38_frozenlake_replay.py" \
  --capsule "$capsule" --output "$schedule_report" --local-m 256 \
  >"$wrapper"

{
  echo "[P38.FL.REPLAY] source=$source_sha diff_sha256=$diff_sha image=$image"
  echo "[P38.FL.REPLAY] topology=DP1xTP4 model=Qwen3-8B local_m=256"
  echo "[P38.FL.REPLAY] schedule=mask-derived-v1 prefix_cache=disabled runtime_kv_cache=enabled"
  echo "[P38.FL.REPLAY] mutation=none no_backward=1 optimizer_commits=0 timeout=$timeout_seconds"
  sha256sum "$capsule" "$schedule_report" "$0" \
    "$script_dir/prepare_p38_frozenlake_replay.py" \
    "$script_dir/classify_p38_frozenlake_replay.py" \
    "$repo/tunix/rl/p38_frozenlake_replay.py" \
    "$repo/tunix/rl/canonical_qwen3_adapter.py" \
    "$repo/examples/frozenlake/train_frozenlake_qwen3.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} >"$raw"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen8b \
  >>"$raw" 2>&1

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
  -e WANDB_MODE=disabled \
  -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e CANON_FROZENLAKE_L3=1 -e CANON_P38_FROZENLAKE_REPLAY=1 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=1 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=0 -e CANON_ALIGNMENT_TRAIN=0 \
  -e CANON_P38_CAPSULE_INPUT="$capsule" \
  -e CANON_P38_REPLAY_REPORT="$report" \
  -e CANON_P38_EXPECTED_POLICY_VERSION=0 \
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
  -e CANON_QWEN3_HIDDEN_SIZE=4096 -e CANON_QWEN3_INTERMEDIATE_SIZE=12288 \
  -e CANON_QWEN3_NUM_ATTENTION_HEADS=32 -e CANON_QWEN3_NUM_KV_HEADS=8 \
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
      cmp '$canon_out/'"\$1" '$sp/'"\$2"
    done
    echo '[P38.FL.REPLAY] OVERLAY_BYTE_IDENTITY PASS files=6'
    python3 - <<'PY'
import jax
devices = jax.devices()
print(f'[P38.FL.REPLAY] devices={len(devices)} ids={[d.id for d in devices]} platform={jax.default_backend()}')
if len(devices) != 4 or jax.default_backend() != 'tpu':
  raise SystemExit('P38 FrozenLake replay requires exactly four TPU devices')
PY
    exec python3 -u examples/frozenlake/train_frozenlake_qwen3.py \
      --batch_size=1 --mini_batch_size=1 \
      --train_trajectory_micro_batch_size=1 \
      --max_steps=1 --num_generations=2 --num_batches=1 \
      --max_prompt_length=4096 --max_response_length=2048 \
      --max_concurrency=1 --vllm_max_num_seqs=1 \
      --vllm_max_num_batched_tokens=256 --env_max_steps=1 \
      --temperature=0.7 --top_k=0 --top_p=1.0
  " >>"$raw" 2>&1
docker_rc=$?
elapsed="$(( $(date +%s) - started ))"
set -e

echo "[P38.FL.REPLAY] docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
if [ "$docker_rc" -ne 0 ]; then
  tail -n 160 "$raw" | tee -a "$wrapper"
  exit "$docker_rc"
fi
test -s "$report"
python3 "$script_dir/classify_p38_frozenlake_replay.py" \
  --report "$report" --output "$classification" | tee -a "$wrapper"
sha256sum "$raw" "$schedule_report" "$report" "$classification" "$0" \
  | tee -a "$wrapper"
echo "[P38.FL.REPLAY] COMPLETE report=$report classification=$classification" \
  | tee -a "$wrapper"
