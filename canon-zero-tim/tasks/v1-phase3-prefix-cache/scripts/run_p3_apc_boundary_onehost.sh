#!/usr/bin/env bash
# Fixed-prefix, production-decode APC G-A discriminator on one DP1xTP4 host.
set -euo pipefail

mode="${1:?usage: run_p3_apc_boundary_onehost.sh control|repro|dirty <unique-label>}"
label="${2:?usage: run_p3_apc_boundary_onehost.sh control|repro|dirty <unique-label>}"
case "$mode" in
  control) apc=0; dirty_page=0 ;;
  repro) apc=1; dirty_page=0 ;;
  dirty) apc=1; dirty_page=1 ;;
  *) echo "invalid mode: $mode" >&2; exit 2 ;;
esac
case "$label" in
  *[!a-zA-Z0-9_-]*|'') echo "invalid label: $label" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
hf_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
hf_snapshot_sha="$(head -n 1 "$hf_cache/refs/main")"
model="$hf_cache/snapshots/$hf_snapshot_sha"
data=/mnt/disks/tunix-data/frozenlake/data_convergence_v1
evidence=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
state="$evidence/p3_apc_boundary_${label}_${mode}"
raw="$evidence/p3_apc_boundary_${label}_${mode}.raw.log"
report="$state/boundary.report.json"
classification="$state/boundary.classification.json"
canon_out="$state/canon"
container="p3_apc_boundary_${label}_${mode}"
timeout_seconds="${P3_APC_TIMEOUT_SECONDS:-3600}"

test "$(hostname)" = "${TIM_C2_EXPECT_HOSTNAME:-t1v-n-4a77ebd0-w-0}"
test -s "$model/config.json"
test -s "$model/model.safetensors.index.json"
test -s "$data/train.parquet"
for path in "$state" "$raw"; do
  if [ -e "$path" ]; then
    echo "[P3.APC.BOUNDARY] REFUSING evidence collision: $path" >&2
    exit 3
  fi
done
running_containers="$(sudo docker ps --format '{{.Names}}')"
case "$running_containers" in
  *p51_*) echo "[P3.APC.BOUNDARY] REFUSING while p51 is active" >&2; exit 4 ;;
  *p3_apc_*) echo "[P3.APC.BOUNDARY] REFUSING while another p3 APC run is active" >&2; exit 4 ;;
esac

mkdir -p "$state"
git -C "$repo" diff --binary HEAD > "$state/source.diff"
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sum="$(sha256sum "$state/source.diff")"
diff_sha="${diff_sum%% *}"
{
  echo "[P3.APC.BOUNDARY] source=$source_sha diff_sha256=$diff_sha image=$image mode=$mode apc=$apc dirty_page=$dirty_page"
  echo "[P3.APC.BOUNDARY] topology=DP1xTP4 model=Qwen3-8B M=256 fixed_prefixes=1"
  echo "[P3.APC.BOUNDARY] A=cache-hit-decode B=full-reset-prefill C=not-in-this-G-A-probe backward=0 optimizer_commits=0"
  echo "[P3.APC.BOUNDARY] prefixes=1535,1536,1537,1685,1686,1687,1788,1792,2047,2048,2049 a_decode_tokens=16"
  sha256sum "$0" "$script_dir/classify_p3_boundary.py" \
    "$repo/tunix/rl/rollout/vllm_rollout.py" \
    "$repo/examples/frozenlake/train_frozenlake_qwen3.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} > "$raw"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen8b \
  >> "$raw" 2>&1

xla_flags='--xla_cpu_max_isa=AVX2 --xla_allow_excess_precision=false'
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
  -e PYTHONPATH="$canon_out:$repo" -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e MODEL_DOWNLOAD_DIR="$model" -e FROZENLAKE_DATA_DIR="$data" \
  -e WANDB_MODE=disabled -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e CANON_FROZENLAKE_L3=1 \
  -e CANON_VLLM_ENABLE_PREFIX_CACHING="$apc" \
  -e CANON_P3_APC_DIRTY_PAGE="$dirty_page" \
  -e CANON_P3_APC_BOUNDARY_REPORT="$report" \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=1 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=0 -e CANON_ALIGNMENT_TRAIN=0 \
  -e CANON_P38_PRECHECK_ONLY=1 \
  -e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4 -e CANON_TARGET_M=256 \
  -e CANON_P32_TRAIN_ADMITTED=0 -e FL_SHARED_MESH=1,4 \
  -e XLA_FLAGS="$xla_flags" \
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
  -e FL_VLLM_HBM_UTIL=0.56 -e FL_STATELESS_OPTIMIZER=1 \
  -w "$repo" "$image" bash -lc "
    exec python3 -u examples/frozenlake/train_frozenlake_qwen3.py \\
      --batch_size=1 --mini_batch_size=1 \\
      --train_trajectory_micro_batch_size=1 \\
      --max_steps=1 --num_generations=2 --num_batches=1 \\
      --max_prompt_length=4096 --max_response_length=512 \\
      --max_concurrency=1 --vllm_max_num_seqs=1 \\
      --vllm_max_num_batched_tokens=256 --env_max_steps=1 \\
      --temperature=0.7 --top_k=0 --top_p=1.0
  " >> "$raw" 2>&1
docker_rc=$?
elapsed=$(( $(date +%s) - started ))
set -e

echo "[P3.APC.BOUNDARY] docker_exit=$docker_rc elapsed_seconds=$elapsed" >> "$raw"
if [ "$docker_rc" -ne 0 ]; then
  exit "$docker_rc"
fi
test -s "$report"
set +e
python3 "$script_dir/classify_p3_boundary.py" \
  --report "$report" --expect-apc "$apc" \
  --expect-dirty-page "$dirty_page" --output "$classification" \
  >> "$raw" 2>&1
classifier_rc=$?
set -e
sha256sum "$raw" "$report" "$classification" "$state/source.diff" "$0" \
  > "$state/SHA256SUMS"
if [ "$classifier_rc" -ne 0 ]; then
  exit "$classifier_rc"
fi
echo "[P3.APC.BOUNDARY] CLASSIFIED mode=$mode state=$state"
