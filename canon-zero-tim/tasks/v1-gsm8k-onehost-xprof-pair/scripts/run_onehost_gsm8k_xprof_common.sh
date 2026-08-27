#!/usr/bin/env bash
# Matched Qwen3-1.7B GSM8K DP4xTP1 Native/Zero-HP update XProf carrier.
set -euo pipefail
trap '' HUP

arm="${1:?usage: run_onehost_gsm8k_xprof_common.sh <native|zero-hp> <label>}"
label="${2:?usage: run_onehost_gsm8k_xprof_common.sh <native|zero-hp> <label>}"
case "$arm" in
  native|zero-hp) ;;
  *) echo "[V1.GSM8K.XPROF] invalid arm: $arm" >&2; exit 2 ;;
esac
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[V1.GSM8K.XPROF] invalid immutable label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$script_dir/finalize_gsm8k_xprof_evidence.sh"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
assets=/mnt/disks/tunix-data/gsm8k_zero_tim
model="$assets/models"
hf_model_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-1.7B
hf_snapshot_sha="$(head -n 1 "$hf_model_cache/refs/main")"
hf_snapshot="$hf_model_cache/snapshots/$hf_snapshot_sha"
evidence_root="${V1_GSM8K_XPROF_EVIDENCE_ROOT:-/mnt/disks/tunix-data/gsm8k-onehost-xprof}"
root="${V1_GSM8K_XPROF_ARTIFACT_DIR:-$evidence_root/v1_${arm}_${label}}"
image="${V1_GSM8K_XPROF_IMAGE:-tunix_frozenlake_image:vllm-tpu0.25.0}"
expected_hostname="${V1_GSM8K_XPROF_EXPECT_HOSTNAME:-t1v-n-4a77ebd0-w-0}"
timeout_seconds="${V1_GSM8K_XPROF_TIMEOUT_SECONDS:-7200}"
source_sha="$(git -C "$repo" rev-parse HEAD)"
source_diff_sha256="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
source_dirty="$(git -C "$repo" status --porcelain)"
source_untracked="$(git -C "$repo" ls-files --others --exclude-standard)"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
state="$root/train"
raw="$state/raw.log"
driver="$root/driver.log"
pre="$state/pre_alignment.jsonl"
align="$state/alignment.jsonl"
update="$state/updates.jsonl"
xprof_dir="$state/xprof"
perf_dir="$state/perf"
xprof_census="$state/xprof_census.txt"
semantic_census="$state/semantic_census.txt"
hierarchy_census="$state/hierarchy_census.txt"
trace_census="$state/trace_census.txt"
size_census="$state/xprof_size_census.txt"
size_receipt="$state/xprof_size_receipt.json"
classification="$state/classification.json"
canon_out="$root/canon"
container="v1_gsm8k_xprof_${arm//-/_}_${label}"
runtime_files=(
  "$repo/tunix/rl/agentic/agentic_rl_learner.py"
  "$repo/tunix/rl/canonical_qwen3_adapter.py"
  "$repo/tunix/rl/gsm8k_xprof.py"
  "$repo/examples/math_gsm8k/qwen3_grpo_demo.py"
  "$repo/canon-zero-tim/cluster/profiles/qwen3-1p7b-dp4-tp1-gsm8k-v1-hp.env"
  "$script_dir/run_onehost_gsm8k_xprof_common.sh"
  "$script_dir/run_onehost_gsm8k_xprof_inner.sh"
  "$script_dir/finalize_gsm8k_xprof_evidence.sh"
  "$script_dir/classify_gsm8k_xprof_arm.py"
  "$script_dir/census_gsm8k_xprof_hierarchy.py"
  "$script_dir/census_gsm8k_xprof_trace.py"
  "$script_dir/census_gsm8k_xprof_size.py"
  "$script_dir/census_gsm8k_xprof_modules.py"
  "$script_dir/census_gsm8k_semantic_trace.py"
)
runtime_manifest_sha256="$(sha256sum "${runtime_files[@]}" | sha256sum | awk '{print $1}')"

case "$timeout_seconds" in
  ''|*[!0-9]*) echo "[V1.GSM8K.XPROF] timeout must be an integer" >&2; exit 2 ;;
esac
if [ "$timeout_seconds" -lt 1 ]; then
  echo "[V1.GSM8K.XPROF] timeout must be positive" >&2
  exit 2
fi
if [ "$root" = "${root#/}" ] || [ -e "$root" ]; then
  echo "[V1.GSM8K.XPROF] artifact root must be a fresh absolute path: $root" >&2
  exit 2
fi
if [ "$(hostname)" != "$expected_hostname" ]; then
  echo "[V1.GSM8K.XPROF] hostname mismatch: actual=$(hostname) expected=$expected_hostname" >&2
  exit 2
fi
if [ -n "$source_dirty" ] && [ "${V1_GSM8K_XPROF_ALLOW_DIRTY:-0}" != "1" ]; then
  echo "[V1.GSM8K.XPROF] tracked tree is dirty; set V1_GSM8K_XPROF_ALLOW_DIRTY=1 only for development validation" >&2
  exit 2
fi
if [ -n "$source_untracked" ] && [ "${V1_GSM8K_XPROF_ALLOW_DIRTY:-0}" != "1" ]; then
  echo "[V1.GSM8K.XPROF] untracked files cannot enter acceptance evidence" >&2
  exit 2
fi
test -s "$model/config.json"
test -s "$model/model.safetensors.index.json"
test -s "$assets/data/gsm8k/1.0.0/dataset_info.json"
[[ "$hf_snapshot_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[V1.GSM8K.XPROF] invalid Qwen3-1.7B cache ref" >&2
  exit 2
}
image_id="$(sudo docker image inspect --format '{{.Id}}' "$image")"
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "[V1.GSM8K.XPROF] image has no immutable local id: $image" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "${NETRC:-$HOME/.netrc}" ]; then
  WANDB_API_KEY="$(awk '
    $1 == "machine" { machine = $2 }
    machine == "api.wandb.ai" {
      for (i = 1; i <= NF; i++) {
        if ($i == "password" && i < NF) { print $(i + 1); exit }
      }
    }
  ' "${NETRC:-$HOME/.netrc}")"
  export WANDB_API_KEY
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[V1.GSM8K.XPROF] W&B credential presence is required by the Zero-HP profile" >&2
  exit 2
fi
active="$(sudo docker ps --format '{{.Names}}' | grep -E '^(p51_|p59_|v1_gsm8k_xprof_)' || true)"
if [ -n "$active" ]; then
  echo "[V1.GSM8K.XPROF] REFUSING: the one-host TPU lane is busy" >&2
  printf '%s\n' "$active" >&2
  exit 2
fi

mkdir -p "$state/wandb" "$state/logs" "$xprof_dir" "$perf_dir"
{
  echo "[V1.GSM8K.XPROF] source=$source_sha diff_sha256=$source_diff_sha256 runtime_manifest_sha256=$runtime_manifest_sha256 image=$image image_id=$image_id model_snapshot=$hf_snapshot_sha"
  echo "[V1.GSM8K.XPROF] arm=$arm label=$label hostname=$expected_hostname topology=DP4xTP1"
  echo "[V1.GSM8K.XPROF] work=prompts8_generations8_response256_concurrency1 steps=3 capture=update:2->3"
  echo "[V1.GSM8K.XPROF] tracers=host:1,python:0,tpu:TRACE_ONLY_XLA labels=1 xprof_budget=soft:1200000000,hard:1500000000,basis:logical_regular_file_bytes"
  echo "[V1.GSM8K.XPROF] treatment=$([ "$arm" = native ] && echo stock-vanilla || echo strict-zero-hp-v1)"
} >"$driver"
{
  echo "[V1.GSM8K.XPROF] RUN_BEGIN arm=$arm"
  sha256sum \
    "$repo/tunix/rl/agentic/agentic_rl_learner.py" \
    "$repo/tunix/rl/canonical_qwen3_adapter.py" \
    "$repo/tunix/rl/gsm8k_xprof.py" \
    "$repo/examples/math_gsm8k/qwen3_grpo_demo.py" \
    "$script_dir/run_onehost_gsm8k_xprof_inner.sh" "$0" \
    "$model/config.json" "$model/model.safetensors.index.json"
} >"$raw"

docker_args=(
  sudo docker run --rm --privileged --net=host --ipc=host --name "$container"
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data
  -v "$model":"$hf_snapshot":ro
  -v "$repo":"$repo":ro
  -e PYTHONDONTWRITEBYTECODE=1
  -e HF_HOME=/mnt/disks/tunix-data/hf
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1
  -e GSM8K_ARTIFACT_ROOT="$assets" -e GSM8K_LOG_DIR="$state/logs"
  -e WANDB_API_KEY="$WANDB_API_KEY" -e WANDB_MODE=online -e WANDB_DIR="$state/wandb"
  -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0
  -e V1_GSM8K_XPROF_REPO="$repo"
  -e V1_GSM8K_XPROF_ARM="$arm"
  -e V1_GSM8K_XPROF_LABEL="$label"
  -e V1_GSM8K_XPROF_XLA_FLAGS="$XTRA_XLA"
  -e CANON_V1_GSM8K_XPROF_ARM="$arm"
  -e CANON_GSM8K_TRAIN=1 -e CANON_GSM8K_L3=0 -e CANON_GSM8K_GRAD_PROBE=0
  -e CANON_OPT_STATE_RESIDENT=1 -e CANON_P30_OPT_STATE_OFFLOAD=0
  -e CANON_VLLM_ENABLE_PREFIX_CACHING=0
  -e CANON_P60_DETERMINISTIC_AB=1
  -e CANON_BATCHED_EVIDENCE="${CANON_BATCHED_EVIDENCE:-}"
  -e CANON_DP_COLLECTIVE_REDUCE="${CANON_DP_COLLECTIVE_REDUCE:-}"
  -e CANON_DP_COMPARE_MODE="${CANON_DP_COMPARE_MODE:-}"
  -e CANON_DP_DISTINCT_SCHEDULE="${CANON_DP_DISTINCT_SCHEDULE:-}"
  -e CANON_DP_FINITE_FETCH="${CANON_DP_FINITE_FETCH:-}"
  -e CANON_P71_SCAN="${CANON_P71_SCAN:-}"
  -e CANON_EXPECT_TRAIN_MESH_IDS=0,2,1,3
  -e CANON_XPROF_DIR="$xprof_dir"
  -e CANON_XPROF_SKIP_STEPS=2 -e CANON_XPROF_STEPS=1
  -e CANON_XPROF_PHASE="${CANON_XPROF_PHASE:-update}" -e CANON_XPROF_HOST_TRACER=1
  -e CANON_XPROF_PYTHON_TRACER=0 -e CANON_XPROF_TPU_TRACE_MODE="${CANON_XPROF_TPU_TRACE_MODE-TRACE_ONLY_XLA}"
  -e CANON_XPROF_LABELS=1
  -e CANON_PERF_TRACE_DIR="$perf_dir" -e CANON_PERF_TRACE_EXPORT_STEP=2
  -w "$repo"
)

if [ "$arm" = zero-hp ]; then
  bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen1p7b_tp1 \
    >>"$driver" 2>&1
  docker_args+=(
    -v "$canon_out":"$canon_out":ro
    -v "$canon_out/attn_iface_patched.py":"$sp/layers/common/attention_interface.py":ro
    -v "$canon_out/linear_p22xk.py":"$sp/layers/jax/linear.py":ro
    -v "$canon_out/embed_patched.py":"$sp/layers/jax/embed.py":ro
    -v "$canon_out/tpu_runner_p21_l30.py":"$sp/runner/tpu_runner.py":ro
    -v "$canon_out/qwen3_p22xk.py":"$sp/models/jax/qwen3.py":ro
    -v "$canon_out/qwen2_p22xk.py":"$sp/models/jax/qwen2.py":ro
    -e PYTHONPATH="$canon_out:$repo"
    -e CANON_SHIM_ROOT="$canon_out"
    -e CANON_P32_TRAIN_ADMITTED=1
    -e CANON_P32_DP_REDUCTION_ADMITTED=1
    -e CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
    -e CANON_P33_RUN_STAGE=three-update -e CANON_P33_NO_COMMIT=0
    -e CANON_P59_KIND=v1 -e CANON_P59_DP4_SERIAL_MESH_BRIDGE=1
    -e CANON_P59_DP4_TAIL8=0 -e CANON_P59_RANK_PARALLEL_BACKWARD=1
    -e CANON_GSM8K_ALIGNMENT_WARN_ONLY=0
    -e CANON_PRE_ALIGN_REPORT="$pre" -e CANON_ALIGN_REPORT="$align"
    -e CANON_UPDATE_REPORT="$update"
  )
else
  docker_args+=(
    -e PYTHONPATH="$repo"
    -e CANON_GSM8K_VANILLA=1
    -e CANON_P59_RANK_PARALLEL_BACKWARD=0
    -e CANON_P28_G6_UPDATE=0
  )
fi

set +e
started="$(date +%s)"
timeout --signal=TERM --kill-after=180s "${timeout_seconds}s" \
  "${docker_args[@]}" "$image" \
  bash "$script_dir/run_onehost_gsm8k_xprof_inner.sh" >>"$raw" 2>&1 &
docker_wait_pid=$!
(
  while kill -0 "$docker_wait_pid" 2>/dev/null; do
    sleep 60
    if grep -aq '^\[rank0\]: Traceback (most recent call last)' "$raw"; then
      sleep 180
      if kill -0 "$docker_wait_pid" 2>/dev/null; then
        echo "[V1.GSM8K.XPROF] crash_watchdog stopping lingering container" >>"$raw"
        sudo docker stop "$container" >/dev/null 2>&1 || true
      fi
      break
    fi
  done
) &
watchdog_pid=$!
wait "$docker_wait_pid"
docker_rc=$?
kill "$watchdog_pid" 2>/dev/null || true
set -e
if sudo docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
  sudo docker stop "$container" >/dev/null 2>&1 || true
fi
elapsed=$(( $(date +%s) - started ))
echo "[V1.GSM8K.XPROF] RUN_END arm=$arm docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
# The container entrypoint normalizes its root-owned train artifacts before it
# exits.  This check is intentionally non-mutating: a failure means the
# container-side durability contract regressed.
if ! test -r "$raw" || find "$state" -type f ! -readable -print -quit | grep -q .; then
  echo "[V1.GSM8K.XPROF] artifact readability contract failed" >>"$driver"
  docker_rc=97
fi

xprof_census_rc=1
semantic_census_rc=1
hierarchy_census_rc=1
trace_census_rc=1
set +e
python3 "$script_dir/census_gsm8k_xprof_size.py" \
  --run-root "$root" --output "$size_receipt" \
  >"$size_census" 2>&1
size_census_rc=$?
set -e
if [ "$docker_rc" -eq 0 ]; then
  set +e
  python3 "$script_dir/census_gsm8k_xprof_modules.py" \
    --arm "$arm" --run-root "$root" \
    >"$xprof_census" 2>&1
  xprof_census_rc=$?
  sudo docker run --rm --ipc=host \
    -v "$repo":"$repo":ro -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
    -w "$repo" "$image" \
    python3 "$script_dir/census_gsm8k_semantic_trace.py" \
      --arm "$arm" --run-root "$root" \
    >"$semantic_census" 2>&1
  semantic_census_rc=$?
  if [ "$arm" = zero-hp ]; then
    python3 "$script_dir/census_gsm8k_xprof_hierarchy.py" \
      --run-root "$root" --expected-update-step 2 \
      >"$hierarchy_census" 2>&1
    hierarchy_census_rc=$?
    python3 "$script_dir/census_gsm8k_xprof_trace.py" \
      --run-root "$root" --expected-update-step 2 \
      >"$trace_census" 2>&1
    trace_census_rc=$?
  fi
  set -e
fi

set +e
classifier_args=(
  python3 "$script_dir/classify_gsm8k_xprof_arm.py"
  --arm "$arm" --run-root "$root" --source-sha "$source_sha" \
  --source-diff-sha256 "$source_diff_sha256" \
  --runtime-manifest-sha256 "$runtime_manifest_sha256" \
  --model-snapshot "$hf_snapshot_sha" --image-id "$image_id" \
  --xprof-census-rc "$xprof_census_rc" \
  --semantic-census-rc "$semantic_census_rc" \
  --size-census-rc "$size_census_rc" \
  --output "$classification"
)
if [ "$arm" = zero-hp ]; then
  classifier_args+=(
    --require-hierarchy --hierarchy-census-rc "$hierarchy_census_rc"
    --trace-census-rc "$trace_census_rc"
  )
fi
"${classifier_args[@]}" >>"$driver" 2>&1
classifier_rc=$?
set -e

sha_inputs=("$raw" "$driver")
for path in \
    "$xprof_census" "$semantic_census" "$classification" \
    "$hierarchy_census" "$trace_census" "$size_census" "$size_receipt" \
    "$pre" "$align" "$update"; do
  [ -e "$path" ] && sha_inputs+=("$path")
done
while IFS= read -r -d '' path; do
  sha_inputs+=("$path")
done < <(find "$xprof_dir" -type f -print0 | sort -z)
while IFS= read -r -d '' path; do
  sha_inputs+=("$path")
done < <(find "$perf_dir" -type f -name 'perfetto_trace_v2_*.pb' -print0 | sort -z)
if ! gsm8k_xprof_choose_terminal \
    "$arm" "$root" "$docker_rc" "$classifier_rc"; then
  echo "[V1.GSM8K.XPROF] SHA_LEDGER_RED stage=select root=$root" >&2
  exit 98
fi
if ! gsm8k_xprof_write_terminal_manifest \
    "$GSM8K_XPROF_TERMINAL_MARKER" "$driver" "$root/SHA256SUMS" \
    "${sha_inputs[@]}"; then
  echo "[V1.GSM8K.XPROF] SHA_LEDGER_RED stage=write root=$root" >&2
  exit 98
fi
# Verification is intentionally immediate. After this point the runner may
# write only to stdout/stderr, never to a file covered by SHA256SUMS.
if ! gsm8k_xprof_verify_manifest "$root/SHA256SUMS"; then
  echo "[V1.GSM8K.XPROF] SHA_LEDGER_RED stage=verify root=$root" >&2
  exit 98
fi
echo "[V1.GSM8K.XPROF] SHA_LEDGER_PASS entries=${#sha_inputs[@]} root=$root"
printf '%s\n' "$GSM8K_XPROF_TERMINAL_MARKER"
exit "$GSM8K_XPROF_TERMINAL_RC"
