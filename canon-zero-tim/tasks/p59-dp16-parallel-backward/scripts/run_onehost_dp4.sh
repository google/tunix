#!/usr/bin/env bash
# Run one immutable Qwen3-1.7B DP4xTP1 P59 arm.
set -euo pipefail

kind="${1:?usage: run_onehost_dp4.sh <candidate|control|profile|tail|numerical-control|numerical-candidate> <unique-label>}"
label="${2:?usage: run_onehost_dp4.sh <candidate|control|profile|tail|numerical-control|numerical-candidate> <unique-label>}"
case "$kind" in
  candidate) rank_parallel=1; capture=0; numerical=0; tail8=0; run_stage=three-update; expected_steps=3; expected_align=51 ;;
  control) rank_parallel=0; capture=0; numerical=0; tail8=0; run_stage=three-update; expected_steps=3; expected_align=51 ;;
  profile) rank_parallel=1; capture=1; numerical=0; tail8=0; run_stage=three-update; expected_steps=3; expected_align=51 ;;
  tail) rank_parallel=1; capture=0; numerical=0; tail8=1; run_stage=p59-eight-update; expected_steps=8; expected_align=136 ;;
  numerical-control) rank_parallel=0; capture=0; numerical=1; tail8=0; run_stage=one-update; expected_steps=1; expected_align=17 ;;
  numerical-candidate) rank_parallel=1; capture=0; numerical=1; tail8=0; run_stage=one-update; expected_steps=1; expected_align=17 ;;
  *) echo "[P59.DP4] invalid kind: $kind" >&2; exit 2 ;;
esac
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P59.DP4] invalid label: $label" >&2
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
hf_snapshot_sha="$(head -n 1 "$hf_model_cache/refs/main")"
hf_snapshot="$hf_model_cache/snapshots/$hf_snapshot_sha"
evidence=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
root="$evidence/p59_dp4_${kind}_${label}"
canon_out="$root/canon"
state="$root/train"
raw="$state/raw.log"
pre="$state/pre_alignment.jsonl"
align="$state/alignment.jsonl"
update="$state/updates.jsonl"
classification="$state/classification.json"
xprof_inspection="$state/xprof_inspection.json"
xprof_dir="$state/xprof"
perf_dir="$state/perf"
driver="$root/driver.log"
container="p59_dp4_${kind}_${label}"
timeout_seconds="${P59_DP4_TIMEOUT_SECONDS:-7200}"
deterministic_ab="${CANON_P60_DETERMINISTIC_AB:-0}"
if [ "$numerical" = 1 ]; then
  deterministic_ab="${CANON_P60_DETERMINISTIC_AB:-1}"
  if [ "$deterministic_ab" != 1 ]; then
    echo "[P61.NUMERICAL] numerical carrier requires CANON_P60_DETERMINISTIC_AB=1" >&2
    exit 2
  fi
fi
case "$deterministic_ab" in
  0) ab_concurrency=64; ab_response_length=1024 ;;
  1) ab_concurrency=1; ab_response_length=256 ;;
  *) echo "[P59.DP4] CANON_P60_DETERMINISTIC_AB must be exactly 0 or 1" >&2; exit 2 ;;
esac

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -s "$model/config.json"
test -s "$assets/data/gsm8k/1.0.0/dataset_info.json"
[[ "$hf_snapshot_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[P59.DP4] invalid Qwen3-1.7B cache ref" >&2
  exit 2
}
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
  echo "[P59.DP4] WANDB_API_KEY or api.wandb.ai netrc entry is required for canonical telemetry" >&2
  exit 2
fi
active="$(sudo docker ps --format '{{.Names}}' | grep -E '^(p51_|p59_)' || true)"
if [ -n "$active" ]; then
  echo "[P59.DP4] REFUSING: another P51/P59 TPU container is active" >&2
  printf '%s\n' "$active" >&2
  exit 2
fi
if [ -e "$root" ]; then
  echo "[P59.DP4] REFUSING: run label already exists: $root" >&2
  exit 2
fi
mkdir -p "$state/wandb" "$state/logs" "$xprof_dir" "$perf_dir"

{
  echo "[P59.DP4] source=$source_sha diff_sha256=$diff_sha image=$image"
  echo "[P59.DP4] kind=$kind rank_parallel=$rank_parallel capture=$capture tail8=$tail8 stage=$run_stage capture_phase=$([ "$capture" = 1 ] && echo backward_group_1 || echo none)"
  echo "[P59.DP4] topology=DP4xTP1 prompts=8 generations=8 trajectories=64 groups=16"
  echo "[P59.DP4] expected_pullbacks=$([ "$rank_parallel" = 1 ] && echo 16 || echo 64)"
  echo "[P59.DP4] zero_tim_gate=$expected_align/$expected_align strict expected_fail=0"
  echo "[P59.DP4] deterministic_ab=$deterministic_ab engine_seed=42 max_concurrency=$ab_concurrency max_response_length=$ab_response_length"
  echo "[P61.NUMERICAL] enabled=$numerical capture=$([ "$numerical" = 1 ] && echo full_gradient_and_update || echo none) performance_eligible=$([ "$numerical" = 1 ] && echo 0 || echo 1)"
  echo "[P59.DP4] p56_recipe=batched_reverse,fused_tree_ops,dp_row_padded_logprobs,fixed_ar_gather,norm_matmul,logprob_step_fusion,continue_decode8"
} >"$driver"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen1p7b_tp1 \
  >>"$driver" 2>&1

{
  echo "[P59.DP4] RUN_BEGIN"
  sha256sum \
    "$repo/tunix/rl/agentic/agentic_rl_learner.py" \
    "$repo/tunix/rl/dp_training.py" \
    "$repo/tunix/rl/dp_workloads.py" \
    "$repo/tunix/rl/canonical_qwen3_adapter.py" \
    "$repo/examples/math_gsm8k/qwen3_grpo_demo.py" \
    "$pkg/cluster/profiles/qwen3-1p7b-dp4-tp1-gsm8k-p59.env" \
    "$pkg/tests/p33_workloads/classify_run.py" \
    "$pkg/tests/p59_backward/classify_and_analyze.py" \
    "$script_dir/run_dp4_inner.sh" "$0" \
    "$model/config.json" "$model/model.safetensors.index.json"
} >"$raw"

active="$(sudo docker ps --format '{{.Names}}' | grep -E '^(p51_|p59_)' || true)"
if [ -n "$active" ]; then
  echo "[P59.DP4] REFUSING: TPU lane became busy before launch" | tee -a "$driver" "$raw" >&2
  printf '%s\n' "$active" | tee -a "$driver" "$raw" >&2
  exit 2
fi

set +e
started="$(date +%s)"
timeout --signal=TERM --kill-after=180s "${timeout_seconds}s" \
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
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e WANDB_MODE=online -e WANDB_DIR="$state/wandb" \
  -e P59_REPO="$repo" -e P59_XLA_FLAGS="$XTRA_XLA" \
  -e JOBSET_RESTART_ATTEMPT=0 -e CANON_P59_KIND="$kind" \
  -e CANON_P32_TRAIN_ADMITTED=1 \
  -e CANON_P32_DP_REDUCTION_ADMITTED=1 \
  -e CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1 \
  -e CANON_P33_RUN_STAGE="$run_stage" -e CANON_P33_NO_COMMIT=0 \
  -e CANON_OPT_STATE_RESIDENT=1 -e CANON_P30_OPT_STATE_OFFLOAD=0 \
  -e CANON_P59_RANK_PARALLEL_BACKWARD="$rank_parallel" \
  -e CANON_P59_DP4_SERIAL_MESH_BRIDGE=1 \
  -e CANON_P59_DP4_TAIL8="$tail8" \
  -e CANON_P60_DETERMINISTIC_AB="$deterministic_ab" \
  -e CANON_P61_BACKWARD_NUMERICAL_DIR="$([ "$numerical" = 1 ] && echo "$state/p61_numerical")" \
  -e CANON_WANDB_RUN_NAME="p59-dp4-${kind}-${label}" \
  -e CANON_PRE_ALIGN_REPORT="$pre" -e CANON_ALIGN_REPORT="$align" \
  -e CANON_UPDATE_REPORT="$update" \
  -e CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3 \
  -e CANON_EXPECT_TRAIN_MESH_IDS=0,2,1,3 \
  -e CANON_XPROF_DIR= \
  -e CANON_P59_XPROF_BACKWARD_DIR="$([ "$capture" = 1 ] && echo "$xprof_dir")" \
  -e CANON_PERF_TRACE_DIR="$perf_dir" \
  -e CANON_XPROF_SKIP_STEPS=1 -e CANON_XPROF_STEPS=1 \
  -e CANON_XPROF_PHASE=update -e CANON_XPROF_HOST_TRACER=1 \
  -e CANON_XPROF_PYTHON_TRACER=0 -e CANON_XPROF_LABELS="$capture" \
  -w "$repo" "$image" bash "$script_dir/run_dp4_inner.sh" \
  >>"$raw" 2>&1 &
docker_wait_pid=$!
(
  while kill -0 "$docker_wait_pid" 2>/dev/null; do
    sleep 60
    if grep -aq '^\[rank0\]: Traceback (most recent call last)' "$raw"; then
      sleep 180
      if kill -0 "$docker_wait_pid" 2>/dev/null; then
        echo "[P59.DP4] crash_watchdog stopping lingering container" >>"$raw"
        sudo docker stop "$container" >/dev/null 2>&1 || true
      fi
      break
    fi
  done
) &
crash_watchdog_pid=$!
wait "$docker_wait_pid"
docker_rc=$?
kill "$crash_watchdog_pid" 2>/dev/null || true
set -e
if sudo docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
  echo "[P59.DP4] post_run_cleanup stopping lingering container" >>"$raw"
  sudo docker stop "$container" >/dev/null 2>&1 || true
fi
elapsed=$(( $(date +%s) - started ))
echo "[P59.DP4] RUN_END docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
chmod -R a+rX "$state" || true

classifier_kind="$kind"
if [ "$kind" = profile ]; then
  classifier_kind=profile
fi
classifier_rc=1
if [ "$docker_rc" -eq 0 ]; then
  set +e
  PYTHONPATH="$repo" python3 "$pkg/tests/p59_backward/classify_and_analyze.py" \
    --kind "$classifier_kind" --workload gsm8k-p59-dp4-tp1 \
    --dp-size 4 --tp-size 1 --run-log "$raw" \
    --pre-alignment-report "$pre" --update-report "$update" \
    --alignment-report "$align" --output "$classification" \
    >>"$driver" 2>&1
  classifier_rc=$?
  set -e
fi

xprof_inspection_rc=0
if [ "$capture" = 1 ]; then
  xprof_inspection_rc=1
  if [ "$docker_rc" -eq 0 ]; then
    set +e
    PYTHONPATH="$repo" python3 \
      "$pkg/tests/p59_backward/inspect_xprof_capture.py" \
      --run-root "$root" --output "$xprof_inspection" >>"$driver" 2>&1
    xprof_inspection_rc=$?
    set -e
  fi
fi

align_pass="$(grep -aEc '^\[CANON_ALIGN(_PRE)?\].*verdict=PASS( |$)' "$raw" || true)"
align_fail="$(grep -aEc '^\[CANON_ALIGN(_PRE)?\].*verdict=FAIL( |$)' "$raw" || true)"
steps_done="$(grep -ac 'Global step .* completed in' "$raw" || true)"
xplane_count="$(find "$xprof_dir" -name '*.xplane.pb' 2>/dev/null | wc -l)"
xplane_bytes="$(find "$xprof_dir" -name '*.xplane.pb' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
perfetto_count="$(find "$xprof_dir" \( -name '*perfetto*' -o -name '*.trace.json.gz' \) 2>/dev/null | wc -l)"
red=""
[ "$docker_rc" -eq 0 ] || red="$red docker_exit=$docker_rc"
[ "$classifier_rc" -eq 0 ] || red="$red classifier_exit=$classifier_rc"
[ "$steps_done" -eq "$expected_steps" ] || red="$red steps=$steps_done/$expected_steps"
[ "$align_pass" -eq "$expected_align" ] || red="$red align_pass=$align_pass/$expected_align"
[ "$align_fail" -eq 0 ] || red="$red align_fail=$align_fail"
if [ "$capture" = 1 ]; then
  [ "$xplane_count" -gt 0 ] || red="$red xplane=$xplane_count"
  [ "$xplane_bytes" -gt 0 ] || red="$red xplane_bytes=$xplane_bytes"
  [ "$perfetto_count" -gt 0 ] || red="$red perfetto=$perfetto_count"
  [ "$xprof_inspection_rc" -eq 0 ] || red="$red xprof_inspection_exit=$xprof_inspection_rc"
fi
{
  echo "[P59.DP4] SUMMARY kind=$kind steps=$steps_done/$expected_steps align=$align_pass/$expected_align fail=$align_fail classifier_rc=$classifier_rc"
  grep -a 'Global step .* completed in\|^\[PERF\] ' "$raw" || true
  if [ -n "$red" ]; then
    echo "[P59.DP4] RED$red"
  else
    echo "[P59.DP4] GREEN kind=$kind zero_tim=$expected_align/$expected_align fail=0 root=$root"
  fi
} | tee -a "$driver"

sha_inputs=("$raw" "$driver" "$pre" "$align" "$update" "$classification")
if [ -e "$xprof_inspection" ]; then
  sha_inputs+=("$xprof_inspection")
fi
if [ "$numerical" = 1 ]; then
  sha_inputs+=(
    "$state/p61_numerical/model_before/manifest.json"
    "$state/p61_numerical/gradient/manifest.json"
    "$state/p61_numerical/model_after/manifest.json"
  )
fi
sha256sum "${sha_inputs[@]}" >"$root/SHA256SUMS" 2>/dev/null || true
if [ -n "$red" ]; then
  exit 1
fi
