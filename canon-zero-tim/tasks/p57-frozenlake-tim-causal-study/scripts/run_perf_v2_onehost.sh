#!/usr/bin/env bash
# Run the P57.1c Perf v2 step-boundary gate on one direct-attached v5p host.
set -euo pipefail

label="${1:?usage: run_perf_v2_onehost.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P57.PERF_V2.ONEHOST] invalid label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
model_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
model_sha="$(head -n 1 "$model_cache/refs/main")"
model="$model_cache/snapshots/$model_sha"
data=/mnt/disks/tunix-data/frozenlake/data
deps=/mnt/disks/tunix-data/frozenlake/deps
image=tunix_frozenlake_image:vllm-tpu0.25.0
image_id="$(sudo docker image inspect "$image" --format '{{.Id}}')"
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
root=/mnt/disks/tunix-data/logp_probe_1host/p57_perf_v2_${label}
canon_out="$root/canon"
raw="$root/raw.log"
align="$root/alignment.jsonl"
updates="$root/updates.jsonl"
runtime="$root/runtime.json"
perf_dir="$root/perf"
semantic="$root/semantic.json"
result="$root/classification.json"
driver="$root/driver.log"
container="p57_perf_v2_${label}"
timeout_seconds="${P57_PERF_V2_ONEHOST_TIMEOUT_SECONDS:-3600}"

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
[[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]
[[ "$model_sha" =~ ^[0-9a-f]{40}$ ]]
test -s "$model/config.json"
test "$(find "$model" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)" -eq 5
test -s "$data/train.parquet"
test -s "$data/test.parquet"
test -s "$deps/gymnasium-1.3.0-py3-none-any.whl"
test -s "$deps/farama_notifications-0.0.6-py3-none-any.whl"
active=""
while IFS= read -r running_container; do
  case "$running_container" in
    tpu-runtime|instance_agent|vbarcontrolagent|google-runtime-monitor|healthagent|google-collectd|monitoringagent)
      continue
      ;;
  esac
  if [ "$(sudo docker inspect --format '{{.HostConfig.Privileged}}' "$running_container")" = true ]; then
    active="${active}${active:+$'\n'}${running_container}"
  fi
done < <(sudo docker ps --format '{{.Names}}')
if [ -n "$active" ]; then
  echo "[P57.PERF_V2.ONEHOST] refusing a host with a non-system privileged container" >&2
  printf '%s\n' "$active" >&2
  exit 2
fi
unset running_container
if [ -e "$root" ]; then
  echo "[P57.PERF_V2.ONEHOST] refusing existing evidence root: $root" >&2
  exit 3
fi
mkdir -p "$root/wandb" "$root/logs" "$perf_dir"

{
  echo "[P57.PERF_V2.ONEHOST] source=$source_sha diff_sha256=$diff_sha"
  echo "[P57.PERF_V2.ONEHOST] image_id=$image_id"
  echo "[P57.PERF_V2.ONEHOST] topology=DP1xTP4 model=Qwen3-8B trajectories=8 microbatches=4 updates=3 concurrency=2"
  echo "[P57.PERF_V2.ONEHOST] perf_target_step=2 strict_alignment=1 placement=device-resident timeout_seconds=$timeout_seconds"
} >"$driver"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen8b \
  >>"$driver" 2>&1

{
  echo "[P57.PERF_V2.ONEHOST] RUN_BEGIN"
  sha256sum \
    "$repo/tunix/perf/experimental/timeline.py" \
    "$repo/tunix/perf/experimental/tracer.py" \
    "$repo/tunix/rl/agentic/agentic_rl_learner.py" \
    "$repo/examples/frozenlake/train_frozenlake_qwen3.py" \
    "$script_dir/classify_perf_v2_onehost.py" \
    "$script_dir/census_perf_v2_onehost.py" \
    "$model/config.json" "$model/model.safetensors.index.json" "$0"
} >"$raw"

set +e
started="$(date +%s)"
timeout --foreground --signal=TERM --kill-after=180s "${timeout_seconds}s" \
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
  -v "$canon_out/rpa_kernel_p66.py":"$sp/kernels/ragged_paged_attention/v3/kernel.py":ro \
  -e PYTHONPATH="$canon_out:$repo" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e MODEL_DOWNLOAD_DIR="$model" -e FROZENLAKE_DATA_DIR="$data" \
  -e WANDB_MODE=disabled -e WANDB_DIR="$root/wandb" \
  -e CANON_P29_LOG_DIR="$root/logs" \
  -e CANON_PERF_TRACE_DIR="$perf_dir" -e CANON_PERF_TRACE_EXPORT_STEP=2 \
  -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e CANON_FROZENLAKE_L3=1 -e CANON_FROZENLAKE_P27=1 \
  -e CANON_P29_FULL_TRAIN=1 \
  -e CANON_P28_G5_ONLY=0 -e CANON_P28_G5C_ONLY=0 \
  -e CANON_P28_G6_UPDATE=1 \
  -e CANON_P28_SEGMENTED_FORWARD=1 -e CANON_P28_SEGMENTED_VJP=0 \
  -e CANON_P28_SEGMENTED_PULLBACK=0 -e CANON_P28_SEGMENTED_TRAIN=1 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=0 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=1 -e CANON_ALIGNMENT_TRAIN=0 \
  -e CANON_ALIGNMENT_EXPECTED_RED=0 \
  -e CANON_ALIGN_REPORT="$align" -e CANON_UPDATE_REPORT="$updates" \
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
  -e CANON_PALLAS_MATMUL= -e CANON_PALLAS_MATERIALIZE= \
  -e CANON_CUT= -e CANON_TAIL= -e CANON_POSTRPA_M= \
  -w "$repo" "$image_id" bash -lc "
    set -e
    python3 '$pkg/tasks/p41-optimizer-residency/scripts/admit_frozenlake_runtime.py' \\
      --gymnasium-wheel '$deps/gymnasium-1.3.0-py3-none-any.whl' \\
      --farama-wheel '$deps/farama_notifications-0.0.6-py3-none-any.whl' \\
      --report '$runtime'
    python3 - <<'PY'
import jax
devices = jax.devices()
print(f'[P57.PERF_V2.ONEHOST] devices={len(devices)} ids={[d.id for d in devices]} platform={jax.default_backend()}')
if len(devices) != 4 or jax.default_backend() != 'tpu':
  raise SystemExit('P57 Perf v2 one-host gate requires exactly four TPU devices')
PY
    exec python3 -u examples/frozenlake/train_frozenlake_qwen3.py \\
      --mesh_dp=1 --mesh_tp=4 \\
      --batch_size=4 --mini_batch_size=4 --num_batches=3 \\
      --num_generations=2 --max_prompt_length=2048 \\
      --max_response_length=64 --max_concurrency=2 \\
      --vllm_max_num_seqs=2 --vllm_max_num_batched_tokens=256 \\
      --env_max_steps=2 --beta=0 --temperature=0.7 --top_k=0 --top_p=1.0
  " >>"$raw" 2>&1 &
docker_wait_pid=$!
wait "$docker_wait_pid"
docker_rc=$?
set -e
elapsed=$(( $(date +%s) - started ))
echo "[P57.PERF_V2.ONEHOST] RUN_END docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
sudo chmod -R a+rX "$root" || true
canon_postflight "$raw" 2>&1 | sed 's/^/[P57.PERF_V2.ONEHOST.POSTFLIGHT] /' >>"$raw" || true

semantic_rc=1
if find "$perf_dir" -maxdepth 1 -type f -name 'perfetto_trace_v2_*.pb' -print -quit | grep -q .; then
  set +e
  sudo docker run --rm --net=none \
    -v "$repo":"$repo":ro -v "$root":"$root" \
    -w "$repo" "$image_id" \
    python3 "$script_dir/census_perf_v2_onehost.py" \
      --perf-dir "$perf_dir" --output "$semantic" \
      --reference-inference disabled \
    >>"$driver" 2>&1
  semantic_rc=$?
  set -e
fi

set +e
python3 "$script_dir/classify_perf_v2_onehost.py" \
  --raw "$raw" --alignment "$align" --updates "$updates" \
  --semantic "$semantic" --docker-exit "$docker_rc" --output "$result" \
  >>"$driver" 2>&1
classifier_rc=$?
set -e

if [ "$semantic_rc" -ne 0 ] || [ "$classifier_rc" -ne 0 ]; then
  echo "[P57.PERF_V2.ONEHOST] RED docker=$docker_rc semantic=$semantic_rc classifier=$classifier_rc evidence=$root" >>"$driver"
  find "$root" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$root/SHA256SUMS"
  exit 1
fi
echo "[P57.PERF_V2.ONEHOST] GREEN optimizer_commits=3 strict_failures=0 perfetto=1 evidence=$root" >>"$driver"
find "$root" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$root/SHA256SUMS"
echo "P57_PERF_V2_ONEHOST_PASS optimizer_commits=3 strict_failures=0 perfetto=1 evidence=$root"
