#!/usr/bin/env bash
# Bounded Qwen3-8B DP1xTP4 APC reproduction/certification/performance runner.
set -euo pipefail

mode="${1:?usage: run_p3_apc_onehost.sh control|repro|cert|perf-control|perf-apc|xprof-control|xprof-apc|seam-layer|seam-full <unique-label> [layer]}"
label="${2:?usage: run_p3_apc_onehost.sh control|repro|cert|perf-control|perf-apc|xprof-control|xprof-apc|seam-layer|seam-full <unique-label> [layer]}"
layer="${3:-}"
case "$mode" in
  control) apc=0; seam_mode=""; purpose=reproduction; temperature=0.7; profile=0 ;;
  repro) apc=1; seam_mode=""; purpose=reproduction; temperature=0.7; profile=0 ;;
  cert) apc=1; seam_mode=""; purpose=certification; temperature=0.7; profile=0 ;;
  perf-control) apc=0; seam_mode=""; purpose=certification; temperature=0.0; profile=0 ;;
  perf-apc) apc=1; seam_mode=""; purpose=certification; temperature=0.0; profile=0 ;;
  xprof-control) apc=0; seam_mode=""; purpose=certification; temperature=0.0; profile=1 ;;
  xprof-apc) apc=1; seam_mode=""; purpose=certification; temperature=0.0; profile=1 ;;
  seam-layer) apc=1; seam_mode=layer; purpose=reproduction; temperature=0.7; profile=0 ;;
  seam-full) apc=1; seam_mode=full; purpose=reproduction; temperature=0.7; profile=0 ;;
  *) echo "invalid mode: $mode" >&2; exit 2 ;;
esac
case "$label" in
  *[!a-zA-Z0-9_-]*|'') echo "invalid label: $label" >&2; exit 2 ;;
esac
if [ "$seam_mode" = full ]; then
  case "$layer" in
    ''|*[!0-9]*) echo "seam-full requires a non-negative layer index" >&2; exit 2 ;;
  esac
elif [ -n "$layer" ]; then
  echo "layer is accepted only by seam-full" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
hf_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
hf_snapshot_sha="$(head -n 1 "$hf_cache/refs/main")"
model="$hf_cache/snapshots/$hf_snapshot_sha"
data=/mnt/disks/tunix-data/frozenlake/data_convergence_v1
evidence=/mnt/disks/tunix-data/logp_probe_1host
deps="$evidence/p38_incident_deps_nodeps"
image=tunix_frozenlake_image:vllm-tpu0.25.0
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
state="$evidence/p3_apc_${label}_${mode}"
raw="$evidence/p3_apc_${label}_${mode}.raw.log"
classification="$state/alignment.classification.json"
seam_classification="$state/seam.classification.json"
profile_classification="$state/profile.classification.json"
pre_report="$state/pre_alignment.jsonl"
capsule="$state/mismatch.npz"
round_file="$state/diagnostic_round"
capture="$state/capture"
canon_out="$state/canon"
xprof_dir="$state/xprof"
perf_dir="$state/perf"
container="p3_apc_${label}_${mode}"
timeout_seconds="${P3_APC_TIMEOUT_SECONDS:-3600}"

test "$(hostname)" = "${TIM_C2_EXPECT_HOSTNAME:-t1v-n-4a77ebd0-w-0}"
test -s "$model/config.json"
test -s "$data/train.parquet"
test -s "$deps/gymnasium/__init__.py"
for path in "$state" "$raw"; do
  if [ -e "$path" ]; then
    echo "[P3.APC] REFUSING evidence collision: $path" >&2
    exit 3
  fi
done
running_containers="$(sudo docker ps --format '{{.Names}}')"
case "$running_containers" in
  *p51_*) echo "[P3.APC] REFUSING while a p51 container is active" >&2; exit 4 ;;
  *"$container"*) echo "[P3.APC] REFUSING duplicate container" >&2; exit 4 ;;
esac

mkdir -p "$state"
printf '0\n' > "$round_file"
git -C "$repo" diff --binary HEAD > "$state/source.diff"
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sum="$(sha256sum "$state/source.diff")"
diff_sha="${diff_sum%% *}"
{
  echo "[P3.APC] source=$source_sha diff_sha256=$diff_sha image=$image mode=$mode"
  echo "[P3.APC] classifier_purpose=$purpose"
  echo "[P3.APC] topology=DP1xTP4 model=Qwen3-8B rounds=3 apc=$apc"
  echo "[P3.APC] performance_contract=greedy-matched-v1 temperature=$temperature max_concurrency=1"
  echo "[P3.APC] profile=$profile xprof_phase=$([ "$profile" = 1 ] && echo diagnostic || echo off)"
  echo "[P3.APC] A=rollout B=full-reset-prefill C=trainer-old backward=0 optimizer_commits=0"
  echo "[P3.APC] shape global=action-mask local_dp=1 canonical_m=256 scheduler_seqs=4 scheduler_tokens=256"
  sha256sum "$0" "$pkg/install.sh" "$repo/tunix/rl/alignment.py" \
    "$repo/tunix/rl/rollout/vllm_rollout.py" \
    "$repo/examples/frozenlake/train_frozenlake_qwen3.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} > "$raw"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen8b \
  >> "$raw" 2>&1

capture_env=()
if [ "$profile" = 1 ]; then
  mkdir -p "$xprof_dir" "$perf_dir"
  capture_env+=(
    -e CANON_XPROF_DIR="$xprof_dir"
    -e CANON_PERF_TRACE_DIR="$perf_dir"
    -e CANON_XPROF_SKIP_STEPS=1
    -e CANON_XPROF_STEPS=1
    -e CANON_XPROF_PYTHON_TRACER=0
    -e CANON_XPROF_HOST_TRACER=1
    -e CANON_XPROF_PHASE=diagnostic
  )
fi
if [ -n "$seam_mode" ]; then
  mkdir -p "$capture"
  capture_env+=(
    -e CANON_P38_SERVING_CAPTURE_DIR="$capture"
    -e CANON_P38_REQUEST_JOURNAL="$capture/p38_request_journal.jsonl"
    -e CANON_P38_INCIDENT_LEDGER="$capture/p38_incident_ledger.jsonl"
    -e CANON_P38_INCIDENT_MIN_PREFIX=0
    -e CANON_P38_INCIDENT_MAX_PREFIX=4096
    -e CANON_P38_INCIDENT_MAX_BYTES=134217728
    -e CANON_P38_DIAGNOSTIC_ROUND_FILE="$round_file"
    -e CANON_P38_SERVING_CAPTURE_MAX_CALLS=4
    -e CANON_P38_SERVING_CAPTURE_MIN_PREFIX=0
    -e CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=0,512,1024,1536,3072,4096
    -e CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER=5
    -e CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
    -e CANON_P38_SEAM_OBSERVER="$seam_mode"
    -e CANON_P38_SEAM_OBSERVER_DIR="$capture"
    -e CANON_P38_SEAM_MIN_POSITION=0
    -e CANON_P38_SEAM_MAX_POSITION=4096
    -e CANON_P38_SEAM_MAX_BYTES=4294967296
  )
  if [ "$seam_mode" = full ]; then
    capture_env+=(-e CANON_P38_SEAM_LAYER="$layer")
  fi
fi

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
  -e PYTHONPATH="$deps:$canon_out:$repo" -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e MODEL_DOWNLOAD_DIR="$model" -e FROZENLAKE_DATA_DIR="$data" \
  -e WANDB_MODE=disabled -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e CANON_FROZENLAKE_L3=1 \
  -e CANON_VLLM_ENABLE_PREFIX_CACHING="$apc" \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=1 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=0 -e CANON_ALIGNMENT_TRAIN=0 \
  -e CANON_PRE_ALIGN_GATE=1 -e CANON_PRE_ALIGN_REPORT="$pre_report" \
  -e CANON_P38_PRECHECK_ONLY=1 -e CANON_P38_CONTROLLED_EXIT=1 \
  -e CANON_P38_DIAGNOSTIC_ROUNDS=3 \
  -e CANON_P38_ONEHOST_REHEARSAL=1 \
  -e CANON_P38_DIAGNOSTIC_ROUND_FILE="$round_file" \
  -e CANON_P38_MISMATCH_CAPSULE="$capsule" \
  -e CANON_P38_MISMATCH_CAPSULE_MAX_ROWS=2 \
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
  "${capture_env[@]}" \
  -w "$repo" "$image" bash -lc "
    set +e
    python3 -u examples/frozenlake/train_frozenlake_qwen3.py \\
      --batch_size=2 --mini_batch_size=2 \\
      --train_trajectory_micro_batch_size=4 \\
      --max_steps=1 --num_generations=2 --num_batches=3 \\
      --max_prompt_length=4096 --max_response_length=512 \\
      --max_concurrency=1 --vllm_max_num_seqs=4 \\
      --vllm_max_num_batched_tokens=256 --env_max_steps=5 \\
      --temperature=$temperature --top_k=0 --top_p=1.0
    workload_rc=\$?
    chmod a+r '$pre_report' '$round_file' 2>/dev/null || true
    if [ -d '$xprof_dir' ]; then chmod -R a+rX '$xprof_dir'; fi
    if [ -d '$perf_dir' ]; then chmod -R a+rX '$perf_dir'; fi
    if [ -d '$capture' ]; then chmod -R a+rX '$capture'; fi
    for capsule_path in '$state'/mismatch*.npz; do
      [ ! -e \"\$capsule_path\" ] || chmod a+r \"\$capsule_path\"
    done
    exit \$workload_rc
  " >> "$raw" 2>&1
docker_rc=$?
elapsed=$(( $(date +%s) - started ))
set -e

echo "[P3.APC] docker_exit=$docker_rc elapsed_seconds=$elapsed" >> "$raw"
test -s "$pre_report"
python3 "$script_dir/classify_p3_alignment.py" \
  --raw "$raw" --report "$pre_report" --expect-apc "$apc" \
  --purpose "$purpose" \
  --output "$classification" >> "$raw" 2>&1

if [ "$profile" = 1 ]; then
  python3 "$script_dir/classify_p3_profile.py" \
    --raw "$raw" --state "$state" --expect-apc "$apc" \
    --output "$profile_classification" >> "$raw" 2>&1
fi

if [ -n "$seam_mode" ]; then
  capsules=()
  for capsule_path in "$state"/mismatch*.npz; do
    [ ! -e "$capsule_path" ] || capsules+=(--capsule "$capsule_path")
  done
  test "${#capsules[@]}" -gt 0
  python3 \
    "$pkg/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_seam.py" \
    --directory "$capture" "${capsules[@]}" --mode "$seam_mode" \
    --output "$seam_classification" >> "$raw" 2>&1
fi

sha256sum "$raw" "$pre_report" "$classification" "$round_file" "$0" \
  >> "$state/SHA256SUMS"
if [ -s "$seam_classification" ]; then
  sha256sum "$seam_classification" >> "$state/SHA256SUMS"
fi
if [ -s "$profile_classification" ]; then
  sha256sum "$profile_classification" >> "$state/SHA256SUMS"
fi
echo "[P3.APC] CLASSIFIED mode=$mode state=$state"
