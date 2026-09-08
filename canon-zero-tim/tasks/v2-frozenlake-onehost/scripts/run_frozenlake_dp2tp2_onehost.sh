#!/usr/bin/env bash
# One-host-only Qwen3-8B FrozenLake no-commit carrier.
# Direct calls retain the historical DP2xTP2 default.  Other geometries enter
# only through run_frozenlake_matrix_onehost.sh, which sets V2_FL_GEOMETRY.
set -euo pipefail

workload="${1:?usage: run_frozenlake_dp2tp2_onehost.sh <p45|m15> <r0|r0b|r0c|r0d|r1|r2|r3> <fresh-label> <measure|certify|profile>}"
arm="${2:?usage: run_frozenlake_dp2tp2_onehost.sh <p45|m15> <r0|r0b|r0c|r0d|r1|r2|r3> <fresh-label> <measure|certify|profile>}"
label="${3:?usage: run_frozenlake_dp2tp2_onehost.sh <p45|m15> <r0|r0b|r0c|r0d|r1|r2|r3> <fresh-label> <measure|certify|profile>}"
mode="${4:?usage: run_frozenlake_dp2tp2_onehost.sh <p45|m15> <r0|r0b|r0c|r0d|r1|r2|r3> <fresh-label> <measure|certify|profile> [none|capture|replay] [capsule.npz]}"
capsule_mode="${5:-none}"
capsule_source="${6:-}"
geometry="${V2_FL_GEOMETRY:-dp2-tp2}"
case "$geometry" in
  dp4-tp1)
    dp_size=4; tp_size=1; model_dir=qwen8b_tp1
    profile_rel=cluster/profiles/qwen3-8b-dp4-tp1-frozenlake-onehost.env
    ;;
  dp2-tp2)
    dp_size=2; tp_size=2; model_dir=qwen8b_tp2
    profile_rel=cluster/profiles/qwen3-8b-dp2-tp2-frozenlake-onehost.env
    ;;
  dp1-tp4)
    dp_size=1; tp_size=4; model_dir=qwen8b
    profile_rel=cluster/profiles/qwen3-8b-dp1-tp4-frozenlake-onehost.env
    ;;
  *) echo "invalid geometry: $geometry" >&2; exit 2 ;;
esac
case "$workload" in p45|m15) ;; *) echo "invalid workload: $workload" >&2; exit 2;; esac
segmented_actor_logps_default=0
if [ "$workload" = p45 ] && [ "$geometry" = dp4-tp1 ]; then
  segmented_actor_logps_default=1
fi
segmented_actor_logps="${V2_FL_P78_SEGMENTED_ACTOR_LOGPS:-$segmented_actor_logps_default}"
case "$segmented_actor_logps" in 0|1) ;; *) echo "invalid P78 segmented actor-logps selector" >&2; exit 2;; esac
if [ "$segmented_actor_logps" = 1 ] && { [ "$workload" != p45 ] || [ "$geometry" != dp4-tp1 ]; }; then
  echo "P78 segmented actor logps admit only P45 DP4xTP1" >&2
  exit 2
fi
case "$arm" in r0|r0b|r0c|r0d|r1|r2|r3) ;; *) echo "invalid arm: $arm" >&2; exit 2;; esac
if { [ "$arm" = r0b ] || [ "$arm" = r0c ] || [ "$arm" = r0d ]; } && \
   { [ "$workload" != p45 ] || [ "$geometry" != dp2-tp2 ]; }; then
  echo "$arm P75/P76/P77 capacity arm admits only workload p45 DP2xTP2" >&2
  exit 2
fi
if [ "$geometry" = dp1-tp4 ] && [ "$arm" = r2 ]; then
  echo "DP1 has no reduce-once arm" >&2
  exit 2
fi
case "$mode" in measure|certify|profile) ;; *) echo "invalid mode: $mode" >&2; exit 2;; esac
case "$capsule_mode:$mode" in
  none:measure|none:certify) ;;
  capture:measure) ;;
  replay:certify|replay:profile) ;;
  *) echo "invalid capsule/classification mode: $capsule_mode/$mode" >&2; exit 2;;
esac
if [ "$mode" = profile ] && { [ "$geometry" != dp2-tp2 ] || [ "$workload" != p45 ] || { [ "$arm" != r1 ] && [ "$arm" != r2 ]; }; }; then
  echo "profile mode admits only P45 DP2xTP2 arm r1 or r2" >&2
  exit 2
fi
if [ "$capsule_mode" = replay ] && [ -z "$capsule_source" ]; then
  echo "replay requires an absolute captured capsule path" >&2
  exit 2
fi
if [ "$capsule_mode" != replay ] && [ -n "$capsule_source" ]; then
  echo "capsule source is valid only for replay" >&2
  exit 2
fi
case "$label" in *[!a-z0-9_-]*|'') echo "invalid fresh label: $label" >&2; exit 2;; esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
expected_repo=/mnt/disks/tunix-data/worktrees/gsm8k_rescore_fix_0828
if [ "$repo" != "$expected_repo" ] || [ "$(realpath "$repo")" != "$repo" ]; then
  echo "[V2.FL.ONEHOST] physical worktree mismatch: $repo" >&2
  exit 2
fi
if [ -n "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)" ]; then
  echo "[V2.FL.ONEHOST] acceptance launches require a clean tracked/untracked tree" >&2
  exit 2
fi

pkg="$repo/canon-zero-tim"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
image=tunix_frozenlake_image:vllm-tpu0.25.0
model_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
model_revision="$(head -n 1 "$model_cache/refs/main")"
model="$model_cache/snapshots/$model_revision"
data=/mnt/disks/tunix-data/frozenlake/data_convergence_v1
deps=/mnt/disks/tunix-data/frozenlake/deps
evidence_root=/mnt/disks/tunix-data/frozenlake-onehost-v2
run_identity="$workload"
if [ "$geometry" != dp2-tp2 ]; then
  run_identity="${workload}_${geometry}"
fi
root="$evidence_root/${run_identity}_${arm}_${label}"
canon_out="$root/canon"
raw="$root/raw.log"
driver="$root/driver.log"
container="v2_fl_${workload}_${geometry}_${arm}_${label}"
timeout_seconds="${V2_FL_TIMEOUT_SECONDS:-14400}"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
xprof_dir=
perf_trace_dir=
xprof_report=
xprof_census=
xprof_skip=
xprof_steps=
xprof_phase=
xprof_host_tracer=
xprof_python_tracer=
xprof_tpu_trace_mode=
xprof_labels=
if [ "$mode" = profile ]; then
  xprof_dir="$root/xprof-update"
  perf_trace_dir="$root/perfetto"
  xprof_report="$root/warm_xprof_repeat.json"
  xprof_census="$root/xplane_census.json"
  xprof_skip=0
  xprof_steps=1
  xprof_phase=update
  xprof_host_tracer=1
  xprof_python_tracer=0
  xprof_tpu_trace_mode=TRACE_ONLY_XLA
  xprof_labels=1
fi
seal_evidence() {
  find "$root" -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum >"$root/SHA256SUMS"
  sha256sum -c "$root/SHA256SUMS" >/dev/null
}
capsule_path=
capsule_sha256=
capsule_binding_sha256=
capsule_capture_run=
case "$capsule_mode" in
  none) ;;
  capture)
    capsule_path="$root/training_capsule.npz"
    ;;
  replay)
    if [ "${capsule_source#/}" = "$capsule_source" ]; then
      echo "replay capsule path must be absolute: $capsule_source" >&2
      exit 2
    fi
    capsule_path="$(realpath "$capsule_source")"
    case "$capsule_path" in
      "$evidence_root"/*/training_capsule.npz) ;;
      *) echo "replay capsule is outside the one-host evidence root: $capsule_path" >&2; exit 2;;
    esac
    test -s "$capsule_path"
    test -s "$capsule_path.model.json"
    capsule_sha256="$(sha256sum "$capsule_path" | awk '{print $1}')"
    capsule_binding_sha256="$(sha256sum "$capsule_path.model.json" | awk '{print $1}')"
    capsule_dir_name="$(basename "$(dirname "$capsule_path")")"
    capsule_dir_tail="${capsule_dir_name#${run_identity}_}"
    if [ "$capsule_dir_tail" = "$capsule_dir_name" ]; then
      echo "replay capsule geometry identity is invalid: $capsule_dir_name" >&2
      exit 2
    fi
    case "$capsule_dir_tail" in
      r0_*|r0b_*|r0c_*|r0d_*|r1_*|r2_*|r3_*) ;;
      *) echo "replay capsule run identity is invalid: $capsule_dir_name" >&2; exit 2;;
    esac
    capsule_capture_run="${capsule_dir_tail#*_}"
    case "$capsule_capture_run" in
      *[!a-z0-9_-]*|'') echo "replay capsule label is invalid: $capsule_capture_run" >&2; exit 2;;
    esac
    ;;
esac
case "$arm" in
  r0b) report_adjoint_buckets=1; chunk_dependency_ticket=0; chunk_backpressure=0 ;;
  r0c) report_adjoint_buckets=1; chunk_dependency_ticket=1; chunk_backpressure=0 ;;
  r0d) report_adjoint_buckets=1; chunk_dependency_ticket=0; chunk_backpressure=1 ;;
  *) report_adjoint_buckets=0; chunk_dependency_ticket=0; chunk_backpressure=0 ;;
esac

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
if [ "$mode" = profile ]; then
  test -x /home/yuxuan/miniconda3/bin/python3
  test "$(/home/yuxuan/miniconda3/bin/python3 -c 'import importlib.metadata; print(importlib.metadata.version("xprof"))')" = 2.23.1
fi
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test ! -e "$root"
test -s "$model/config.json"
test "$(find "$model" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)" -eq 5
test -s "$data/train.parquet"
test -s "$data/test.parquet"
test -s "$data/p57/m15/main/train.parquet"
test -s "$data/p57/m15/main/test.parquet"
test -s "$deps/gymnasium-1.3.0-py3-none-any.whl"
test -s "$deps/farama_notifications-0.0.6-py3-none-any.whl"

image_id="$(sudo docker image inspect "$image" --format '{{.Id}}')"
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
model_sha256="$(sha256sum "$model/config.json" "$model/model.safetensors.index.json" | sha256sum | awk '{print $1}')"
if [ "$workload" = m15 ]; then
  train_file="$data/p57/m15/main/train.parquet"
  test_file="$data/p57/m15/main/test.parquet"
else
  train_file="$data/train.parquet"
  test_file="$data/test.parquet"
fi
train_sha256="$(sha256sum "$train_file" | awk '{print $1}')"
test_sha256="$(sha256sum "$test_file" | awk '{print $1}')"
runner_sha256="$(sha256sum "$0" "$script_dir/run_frozenlake_matrix_onehost.sh" "$script_dir/run_frozenlake_dp2tp2_inner.sh" "$script_dir/classify_frozenlake_dp2tp2.py" "$script_dir/classify_frozenlake_warm_xprof.py" "$script_dir/census_frozenlake_warm_xprof.py" | sha256sum | awk '{print $1}')"
image_sha="${image_id#sha256:}"

system_container='^(tpu-runtime|instance_agent|vbarcontrolagent|google-runtime-monitor|runtime-monitor|healthagent|google-collectd|collectd|monitoringagent)$'
other_containers() {
  sudo docker ps --format '{{.Names}}' | grep -Ev "$system_container" || true
}
idle_since=
while true; do
  others="$(other_containers)"
  now="$(date +%s)"
  if [ -z "$others" ]; then
    if [ -z "$idle_since" ]; then idle_since="$now"; fi
    idle_seconds=$((now - idle_since))
    echo "[V2.FL.ONEHOST] idle_seconds=$idle_seconds/120"
    if [ "$idle_seconds" -ge 120 ]; then break; fi
  else
    idle_since=
    echo "[V2.FL.ONEHOST] busy; observe-only wait: $others"
  fi
  sleep 15
done
if [ -n "$(other_containers)" ]; then
  echo "[V2.FL.ONEHOST] contender appeared at final admission check" >&2
  exit 2
fi

mkdir -p "$root" "$root/wandb" "$root/logs"
if [ "$mode" = profile ]; then
  mkdir -p "$root/xprof-update" "$root/perfetto"
fi
{
  echo "[V2.FL.ONEHOST] source=$source_sha diff_sha256=$diff_sha"
  echo "[V2.FL.ONEHOST] image_id=$image_id model_revision=$model_revision"
  echo "[V2.FL.ONEHOST] workload=$workload arm=$arm mode=$mode capsule_mode=$capsule_mode topology=DP${dp_size}xTP${tp_size} stage=backward-no-commit"
  echo "[V2.FL.ONEHOST] timeout_seconds=$timeout_seconds idle_120s=PASS root=$root"
} >"$driver"
bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model "$model_dir" \
  >>"$driver" 2>&1

{
  echo "[V2.FL.ONEHOST] RUN_BEGIN"
  sha256sum "$0" "$script_dir/run_frozenlake_matrix_onehost.sh" \
    "$script_dir/run_frozenlake_dp2tp2_inner.sh" \
    "$script_dir/classify_frozenlake_dp2tp2.py" \
    "$script_dir/classify_frozenlake_warm_xprof.py" \
    "$script_dir/census_frozenlake_warm_xprof.py" \
    "$pkg/$profile_rel" \
    "$pkg/cluster/profiles/_qwen3-8b-frozenlake-four-chip-onehost.env" \
    "$repo/tunix/rl/dp_workloads.py" \
    "$repo/tunix/rl/canonical_qwen3_adapter.py" \
    "$repo/examples/frozenlake/train_frozenlake_qwen3.py"
} >"$raw"

set +e
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
  -e MODEL_DOWNLOAD_DIR="$model" \
  -e FROZENLAKE_DATA_DIR="$data" \
  -e CANON_PRE_ALIGN_REPORT="$root/pre_alignment.jsonl" \
  -e CANON_ALIGN_REPORT="$root/alignment.jsonl" \
  -e CANON_UPDATE_REPORT="$root/updates.json" \
  -e CANON_P29_LOG_DIR="$root/logs" \
  -e WANDB_DIR="$root/wandb" \
  -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e XLA_FLAGS="$XTRA_XLA" \
  -e V2_FL_REPO="$repo" -e V2_FL_ROOT="$root" \
  -e V2_FL_WORKLOAD="$workload" -e V2_FL_ARM="$arm" -e V2_FL_LABEL="$label" \
  -e V2_FL_GEOMETRY="$geometry" -e V2_FL_DP_SIZE="$dp_size" \
  -e V2_FL_TP_SIZE="$tp_size" -e V2_FL_MODEL_DIR="$model_dir" \
  -e V2_FL_PROFILE_REL="$profile_rel" \
  -e CANON_P75_REPORT_ADJOINT_BUCKETS="$report_adjoint_buckets" \
  -e CANON_P76_CHUNK_DEPENDENCY_TICKET="$chunk_dependency_ticket" \
  -e CANON_P77_CHUNK_BACKPRESSURE="$chunk_backpressure" \
  -e CANON_P78_SEGMENTED_ACTOR_LOGPS="$segmented_actor_logps" \
  -e V2_FL_MODE="$mode" \
  -e CANON_XPROF_DIR="$xprof_dir" \
  -e CANON_PERF_TRACE_DIR="$perf_trace_dir" \
  -e V2_FL_XPROF_REPORT="$xprof_report" \
  -e CANON_XPROF_SKIP_STEPS="$xprof_skip" \
  -e CANON_XPROF_STEPS="$xprof_steps" \
  -e CANON_XPROF_PHASE="$xprof_phase" \
  -e CANON_XPROF_HOST_TRACER="$xprof_host_tracer" \
  -e CANON_XPROF_PYTHON_TRACER="$xprof_python_tracer" \
  -e CANON_XPROF_TPU_TRACE_MODE="$xprof_tpu_trace_mode" \
  -e CANON_XPROF_LABELS="$xprof_labels" \
  -e V2_FL_CAPSULE_MODE="$capsule_mode" \
  -e V2_FL_CAPSULE_CAPTURE_RUN="$capsule_capture_run" \
  -e CANON_V2_TRAINING_CAPSULE_MODE="${capsule_mode#none}" \
  -e CANON_V2_TRAINING_CAPSULE="$capsule_path" \
  -e CANON_V2_TRAINING_CAPSULE_SHA256="$capsule_sha256" \
  -e CANON_V2_MODEL_BINDING_SHA256="$capsule_binding_sha256" \
  -e V2_FL_DEPS="$deps" -e V2_FL_SOURCE_SHA="$source_sha" \
  -e V2_FL_DIFF_SHA="$diff_sha" -e V2_FL_IMAGE_SHA="$image_sha" \
  -e V2_FL_MODEL_SHA256="$model_sha256" \
  -e V2_FL_TRAIN_SHA256="$train_sha256" -e V2_FL_TEST_SHA256="$test_sha256" \
  -e V2_FL_RUNNER_SHA256="$runner_sha256" \
  -w "$repo" "$image_id" bash "$script_dir/run_frozenlake_dp2tp2_inner.sh" \
  >>"$raw" 2>&1 &
docker_pid=$!
set -e
started="$(date +%s)"
contention=0
timed_out=0
while kill -0 "$docker_pid" 2>/dev/null; do
  sleep 15
  contenders="$(other_containers | grep -vFx "$container" || true)"
  if [ -n "$contenders" ]; then
    echo "[V2.FL.ONEHOST] contender=$contenders stopping_own=$container" >>"$driver"
    sudo docker stop "$container" >/dev/null 2>&1 || true
    contention=1
    break
  fi
  if [ $(( $(date +%s) - started )) -ge "$timeout_seconds" ]; then
    echo "[V2.FL.ONEHOST] timeout stopping_own=$container" >>"$driver"
    sudo docker stop "$container" >/dev/null 2>&1 || true
    timed_out=1
    break
  fi
done
set +e
wait "$docker_pid"
docker_rc=$?
set -e
elapsed=$(( $(date +%s) - started ))
echo "[V2.FL.ONEHOST] RUN_END docker_exit=$docker_rc elapsed_seconds=$elapsed contention=$contention timeout=$timed_out" >>"$raw"
sudo chmod -R a+rX "$root" || true

xplane_census_rc=0
if [ "$mode" = profile ]; then
  set +e
  /home/yuxuan/miniconda3/bin/python3 \
    "$script_dir/census_frozenlake_warm_xprof.py" \
    --root "$root" --arm "$arm" --output "$xprof_census" \
    >>"$driver" 2>&1
  xplane_census_rc=$?
  set -e
fi

if [ "$mode" = profile ]; then
  classifier_args=(
    "$script_dir/classify_frozenlake_warm_xprof.py"
    --root "$root" --arm "$arm"
    --docker-exit "$docker_rc"
    --anchor-registry "$script_dir/gradient_anchors.json"
    --output "$root/classification.json"
  )
else
  classifier_args=(
    "$script_dir/classify_frozenlake_dp2tp2.py"
    --root "$root" --workload "$workload" --geometry "$geometry" --arm "$arm"
    --docker-exit "$docker_rc"
    --anchor-registry "$script_dir/gradient_anchors.json"
    --output "$root/classification.json"
  )
  if [ "$mode" = certify ]; then
    classifier_args+=(--require-anchor)
  fi
fi
set +e
/mnt/disks/tunix-data/venvs/train/bin/python "${classifier_args[@]}" \
  >>"$driver" 2>&1
classifier_rc=$?
set -e
if [ "$classifier_rc" -ne 0 ] || [ "$xplane_census_rc" -ne 0 ] || [ "$contention" -ne 0 ] || [ "$timed_out" -ne 0 ]; then
  echo "[V2.FL.ONEHOST] RED docker=$docker_rc xplane_census=$xplane_census_rc classifier=$classifier_rc evidence=$root" >>"$driver"
  seal_evidence
  exit 1
fi
verdict="$(/mnt/disks/tunix-data/venvs/train/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$root/classification.json")"
echo "[V2.FL.ONEHOST] $verdict evidence=$root" >>"$driver"
seal_evidence
echo "V2_FROZENLAKE_ONEHOST_${verdict} workload=$workload arm=$arm evidence=$root"
