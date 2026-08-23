#!/usr/bin/env bash
# Shared mutation-free Qwen3-4B DeepSWE XProf carrier. The training command is
# intentionally a direct command with file redirection and no output pipeline.
set -euo pipefail

arm="${1:?usage: run_onehost_deepswe_xprof_common.sh <native|zero-hp> <unique-label>}"
label="${2:?usage: run_onehost_deepswe_xprof_common.sh <native|zero-hp> <unique-label>}"
case "$arm" in
  native|zero-hp) ;;
  *) echo "[P58.ONEHOST.XPROF] invalid arm: $arm" >&2; exit 2 ;;
esac
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P58.ONEHOST.XPROF] invalid immutable label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
python_bin="${DEEPSWE_TRAIN_PYTHON:-/mnt/disks/tunix-data/venvs/train/bin/python}"
model_path="${DEEPSWE_QWEN4B_MODEL_PATH:-/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
dataset_cache="${DEEPSWE_DATASET_CACHE:-/mnt/disks/tunix-data/dataset_cache}"
r2egym_root="${DEEPSWE_R2EGYM_ROOT:-/home/yuxuan/tunix/submodules/R2E-Gym}"
gold_whitelist="${DEEPSWE_ONEHOST_WHITELIST:-/home/yuxuan/code_rl_repro/google_dev/tunix/experimental/smoke_test_whitelist.jsonl}"
task_image="${DEEPSWE_ONEHOST_TASK_IMAGE:-namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d}"
expected_hostname="${P58_ONEHOST_EXPECT_HOSTNAME:?set P58_ONEHOST_EXPECT_HOSTNAME to the exact direct-TPU host}"
evidence_root="${P58_ONEHOST_EVIDENCE_ROOT:-/mnt/disks/tunix-data/deepswe-onehost-xprof}"
artifact_dir="${P58_ONEHOST_ARTIFACT_DIR:-$evidence_root/p58_${arm}_${label}}"
timeout_seconds="${P58_ONEHOST_TIMEOUT_SECONDS:-7200}"

case "$timeout_seconds" in
  ''|*[!0-9]*) echo "[P58.ONEHOST.XPROF] timeout must be an integer" >&2; exit 2 ;;
esac
if [ "$timeout_seconds" -lt 1 ]; then
  echo "[P58.ONEHOST.XPROF] timeout must be positive" >&2
  exit 2
fi
if [ "$artifact_dir" = "${artifact_dir#/}" ] || [ -e "$artifact_dir" ]; then
  echo "[P58.ONEHOST.XPROF] artifact directory must be a fresh absolute path: $artifact_dir" >&2
  exit 2
fi
if [ "$(hostname)" != "$expected_hostname" ]; then
  echo "[P58.ONEHOST.XPROF] hostname mismatch: actual=$(hostname) expected=$expected_hostname" >&2
  exit 2
fi
if [ ! -x "$python_bin" ]; then
  echo "[P58.ONEHOST.XPROF] missing interpreter: $python_bin" >&2
  exit 2
fi
for path in "$model_path" "$dataset_cache" "$r2egym_root"; do
  if [ ! -d "$path" ]; then
    echo "[P58.ONEHOST.XPROF] missing prerequisite directory: $path" >&2
    exit 2
  fi
done
for path in "$model_path/model.safetensors.index.json" "$gold_whitelist"; do
  if [ ! -s "$path" ]; then
    echo "[P58.ONEHOST.XPROF] missing prerequisite file: $path" >&2
    exit 2
  fi
done

source_sha="$(git -C "$repo" rev-parse HEAD)"
source_diff_sha256="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
source_branch="$(git -C "$repo" branch --show-current)"
source_dirty="$(git -C "$repo" status --porcelain)"
source_untracked="$(git -C "$repo" ls-files --others --exclude-standard)"
if [ -z "$source_branch" ] || [[ "$source_branch" != local/* ]]; then
  echo "[P58.ONEHOST.XPROF] requires a named local/* branch" >&2
  exit 2
fi
if [ -n "$source_dirty" ] && [ "${P58_ONEHOST_ALLOW_DIRTY:-0}" != "1" ]; then
  echo "[P58.ONEHOST.XPROF] acceptance evidence requires a clean tracked tree" >&2
  echo "[P58.ONEHOST.XPROF] set P58_ONEHOST_ALLOW_DIRTY=1 only for development evidence" >&2
  exit 2
fi
if [ -n "$source_untracked" ]; then
  echo "[P58.ONEHOST.XPROF] untracked runtime files cannot be provenance-sealed" >&2
  exit 2
fi
r2egym_sha="$(git -C "$r2egym_root" rev-parse HEAD)"
if [ "$r2egym_sha" != "0d94c4eb9431cd195c55a7ea3abd54006c9a1735" ]; then
  echo "[P58.ONEHOST.XPROF] R2E-Gym source changed: $r2egym_sha" >&2
  exit 2
fi

tpu_inference_path="$($python_bin -c 'import pathlib,tpu_inference; print(pathlib.Path(tpu_inference.__file__).resolve().parent)')"
if [ ! -d "$tpu_inference_path" ]; then
  echo "[P58.ONEHOST.XPROF] missing tpu_inference package: $tpu_inference_path" >&2
  exit 2
fi
if [[ ! "$source_sha" =~ ^[0-9a-f]{40}$ ]] || \
   [[ ! "$source_diff_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[P58.ONEHOST.XPROF] invalid source identity" >&2
  exit 2
fi
if [ "${model_path##*/}" != "cdbee75f17c01a7cc42f958dc650907174af0554" ]; then
  echo "[P58.ONEHOST.XPROF] Qwen3-4B snapshot identity changed: $model_path" >&2
  exit 2
fi
task_image_id="$($python_bin -c 'import docker,sys; image=docker.from_env().images.get(sys.argv[1]); print(image.id)' "$task_image")"
if [[ ! "$task_image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "[P58.ONEHOST.XPROF] task image has no immutable local id" >&2
  exit 2
fi

mkdir -p "$artifact_dir/xprof-update" "$artifact_dir/perfetto" "$artifact_dir/install"
raw_log="$artifact_dir/raw.log"
install_log="$artifact_dir/install.log"
runner_sha256="$(sha256sum "$script_dir/run_onehost_deepswe_xprof_common.sh" | awk '{print $1}')"
run_id="p58-onehost-xprof-${arm}-${label}"

export CANON_DEEPSWE_ONEHOST_SMOKE=1
export CANON_DEEPSWE_ONEHOST_STAGE=backward-no-commit
export CANON_DEEPSWE_ONEHOST_ROLLOUT_ONLY=0
export CANON_DEEPSWE_ONEHOST_NO_COMMIT=1
export CANON_DEEPSWE_ONEHOST_DEBUG_DIR="$artifact_dir"
export CANON_DEEPSWE_ONEHOST_REPORT="$artifact_dir/backward_no_commit.json"
export CANON_DEEPSWE_ONEHOST_TASK_IMAGE="$task_image"
export CANON_P58_ONEHOST_XPROF_ARM="$arm"
export CANON_P58_EXPECT_HOSTNAME="$expected_hostname"
export CANON_P58_MODEL_SNAPSHOT="$model_path"
export CANON_P58_R2EGYM_COMMIT="$r2egym_sha"
export CANON_P58_TASK_IMAGE_ID="$task_image_id"
export CANON_P58_RUNNER_SHA256="$runner_sha256"
export CANON_EXPECT_COMMIT="$source_sha"
export CANON_SOURCE_BRANCH="$source_branch"
export CANON_P58_SOURCE_DIFF_SHA256="$source_diff_sha256"
export CANON_RUN_ID="$run_id"
export CANON_P34_DEEPSWE=0
export CANON_P34_DISABLE_SAMPLER_IS=1
export CANON_P34_DISABLE_TIS=1
export CANON_P39_64CHIP_PILOT=0
export CANON_P43_DEEPSWE_DEBUG=0
export CANON_P44_DEEPSWE_PARITY=0
export CANON_P46_DEEPSWE_TRAIN=0
export CANON_P58_DEEPSWE_TIM=0
export CANON_P58_TIM_ADMITTED=0
unset CANON_P58_TIM_ARM

export CANON_ALIGNMENT_GATE=1
export CANON_ALIGNMENT_GATE_ONLY=0
export CANON_ALIGNMENT_UPDATE_CANARY=0
export CANON_ALIGNMENT_TRAIN=1
export CANON_PRE_ALIGN_GATE=1
export CANON_PRE_ALIGN_REPORT="$artifact_dir/pre_alignment.jsonl"
export CANON_ALIGN_REPORT="$artifact_dir/alignment.jsonl"
export CANON_UPDATE_REPORT="$artifact_dir/updates.jsonl"

export CANON_XPROF_DIR="$artifact_dir/xprof-update"
export CANON_XPROF_SKIP_STEPS=0
export CANON_XPROF_STEPS=1
export CANON_XPROF_PHASE=update
export CANON_XPROF_HOST_TRACER=1
export CANON_XPROF_PYTHON_TRACER=0
export CANON_XPROF_TPU_TRACE_MODE=TRACE_COMPUTE
export CANON_XPROF_LABELS=1
export CANON_PERF_TRACE_DIR="$artifact_dir/perfetto"
export CANON_PERF_TRACE_EXPORT_STEP=0

export CANON_VLLM_ENABLE_PREFIX_CACHING=0
export CANON_P38_FIXED_LM_HEAD=0
export CANON_P59_RANK_PARALLEL_BACKWARD=0
export CANON_P28_SEGMENTED_FORWARD=0
export CANON_P28_SEGMENTED_VJP=0
export CANON_P28_SEGMENTED_TRAIN=0
export CANON_P28_G6_UPDATE=0
export CANON_P28_BATCHED_REPORT=0
export CANON_P28_BATCHED_REVERSE=0
export CANON_BATCHED_EVIDENCE=0
export CANON_OPT_STATE_RESIDENT=1
export CANON_P30_OPT_STATE_OFFLOAD=0

export DATASET_CACHE="$dataset_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
export JAX_PLATFORMS=tpu,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SKIP_JAX_PRECOMPILE=true
export VLLM_ENABLE_V1_MULTIPROCESSING=0
unset DOCKER_HOST JAX_BACKEND_TARGET PATHWAYS_HEAD

if [ "$arm" = "zero-hp" ]; then
  # shellcheck disable=SC1091
  source "$pkg/cluster/profiles/_canonical_engine.env"
  export FL_SHARED_MESH=1,4
  export CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3
  export CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=0
  export CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=0
  export CANON_CONTINUE_DECODE=8
  export CANON_FIXED_AR_GATHER=1
  export CANON_PALLAS_GATHERED_LOGPROBS=1
  export CANON_LOGPROB_STEP_FUSION=1
  export CANON_FUSED_TREE_OPS=0
  export CANON_PALLAS_NORM_MATMUL=0
  export CANON_PALLAS_INPUT_FUSION=0
  export CANON_SAMPLE_SPLIT_FUSION=0
  export CANON_ENGINE_LOGPROB_READBACK=0
  export CANON_ANCHOR_OVERLAP=0
  shim_root="$artifact_dir/install/canonical-shims"
  bash "$pkg/install.sh" "$shim_root" --from-path "$tpu_inference_path" --model qwen4b \
    > "$install_log" 2>&1
  export CANON_SHIM_ROOT="$shim_root"
  export PYTHONPATH="$shim_root:$repo:$r2egym_root${PYTHONPATH:+:$PYTHONPATH}"
else
  unset CANON_FIXED_AR CANON_FIXED_AR_EMBED
  unset CANON_RPA_D CANON_RPA_P CANON_RPA_M CANON_LOGPROB_M
  unset CANON_PALLAS_ALL_PROJ CANON_PALLAS_ALL_RMSNORM
  unset CANON_PALLAS_SWIGLU CANON_PALLAS_MPAD
  unset CANON_PALLAS_SWIGLU_MPAD CANON_PALLAS_CANONICAL_VJP
  unset CANON_SHIM_ROOT
  export XLA_FLAGS=--xla_cpu_max_isa=AVX2
  export CANON_RPA_VJP2=0
  export CANON_VJP2_MAX_SEQS=0
  export CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=1
  export CANON_PROMPT_PROCESSED_LOGPROBS=0
  export CANON_PALLAS_LOGSOFTMAX=0
  export CANON_ENGINE_MODULE_C=0
  export CANON_CONTINUE_DECODE=0
  export CANON_FIXED_AR_GATHER=0
  export CANON_PALLAS_GATHERED_LOGPROBS=0
  export CANON_LOGPROB_STEP_FUSION=0
  export CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=1
  native_overlay_root="$artifact_dir/install/native-overlay"
  native_package="$native_overlay_root/tpu_inference"
  mkdir -p "$native_package"
  cp -a "$tpu_inference_path/." "$native_package/"
  native_state="$artifact_dir/install/native-state"
  mkdir -p "$native_state"
  printf '%s\n' \
    'export CANON_PROFILE_FILE=cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env' \
    'export CANON_P34_DEEPSWE=0' \
    'export CANON_P58_DEEPSWE_TIM=0' \
    'export CANON_P58_TIM_ADMITTED=0' \
    'export CANON_P58_ONEHOST_XPROF_ARM=native' \
    'export CANON_DEEPSWE_ONEHOST_SMOKE=1' \
    'export CANON_DEEPSWE_ONEHOST_STAGE=backward-no-commit' \
    'export CANON_DEEPSWE_ONEHOST_NO_COMMIT=1' \
    'export CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=1' \
    'export CANON_PROMPT_PROCESSED_LOGPROBS=0' \
    'export CANON_ENGINE_MODULE_C=0' \
    > "$native_state/env.sh"
  printf '%s\n' "$native_package" > "$native_state/tpu_inference_path"
  CANON_STATE="$native_state" CANON_PKG="$pkg" \
    bash "$pkg/cluster/steps/p58_install_stock_prompt_observer.sh" \
      > "$install_log" 2>&1
  export PYTHONPATH="$native_overlay_root:$repo:$r2egym_root${PYTHONPATH:+:$PYTHONPATH}"
fi

{
  echo "[P58.ONEHOST.XPROF] source=$source_sha diff_sha256=$source_diff_sha256 branch=$source_branch dirty=$([ -n "$source_dirty" ] && echo 1 || echo 0)"
  echo "[P58.ONEHOST.XPROF] arm=$arm label=$label hostname=$expected_hostname topology=DP1xTP4"
  echo "[P58.ONEHOST.XPROF] model=$model_path r2egym=$r2egym_sha task_image=$task_image task_image_id=$task_image_id"
  echo "[P58.ONEHOST.XPROF] capture=update-repeat xplane=required trace_json=required perfetto=required commit=0"
} > "$raw_log"

"$python_bin" -c '
import jax
devices = jax.devices()
assert len(devices) == 4, devices
assert all(device.platform == "tpu" for device in devices), devices
print("[P58.ONEHOST.XPROF] DEVICE_PASS count=4 platform=tpu", flush=True)
' >> "$raw_log" 2>&1
"$python_bin" -c '
import docker
import r2egym
client = docker.from_env()
assert client.ping() is True
print("[P58.ONEHOST.XPROF] R2E_PASS docker=1 import=1", flush=True)
' >> "$raw_log" 2>&1

set +e
timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" \
  "$python_bin" examples/deepswe/train_deepswe_nb.py \
    --seed 42 \
    --model_version Qwen3-4B-Instruct-2507 \
    --model_absolute_path "$model_path" \
    --gold_whitelist "$gold_whitelist" \
    --batch_size 1 \
    --mini_batch_size 1 \
    --train_micro_batch_size 1 \
    --compute_logps_micro_batch_size 1 \
    --rollout_micro_batch_size 1 \
    --num_generations 2 \
    --num_iterations 1 \
    --max_prompt_length 3584 \
    --max_response_length 512 \
    --max_turns 2 \
    --max_steps 1 \
    --num_epochs 1 \
    --eval_every_n_steps 10 \
    --max_concurrency 1 \
    --temperature 0.7 \
    --rollout_engine vllm \
    --vllm_utilization 0.3 \
    --rollout_vllm_max_num_seqs 2 \
    --max_num_batched_tokens 512 \
    --rollout_mesh_dp 1 \
    --rollout_mesh_tp 4 \
    --train_mesh_dp 1 \
    --train_mesh_tp 4 \
    --ckpt_dir none \
    --dtype bfloat16 \
    --param_dtype bfloat16 \
    --use_rollout_logps \
    --logging_level INFO \
    >> "$raw_log" 2>&1
run_status=$?
set -e
if [ "$run_status" -ne 0 ]; then
  echo "[P58.ONEHOST.XPROF] run failed status=$run_status raw=$raw_log" >&2
  exit "$run_status"
fi

classification="$artifact_dir/classification.json"
"$python_bin" "$script_dir/classify_onehost_xprof.py" \
  --arm "$arm" \
  --artifact-dir "$artifact_dir" \
  --source-sha "$source_sha" \
  --expected-hostname "$expected_hostname" \
  --output "$classification"
