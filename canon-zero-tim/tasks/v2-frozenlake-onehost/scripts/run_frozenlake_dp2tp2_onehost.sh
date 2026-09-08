#!/usr/bin/env bash
# One-host-only Qwen3-8B FrozenLake DP2xTP2 no-commit capacity carrier.
set -euo pipefail

workload="${1:?usage: run_frozenlake_dp2tp2_onehost.sh <p45|m15> <r0|r1|r2|r3> <fresh-label> <measure|certify>}"
arm="${2:?usage: run_frozenlake_dp2tp2_onehost.sh <p45|m15> <r0|r1|r2|r3> <fresh-label> <measure|certify>}"
label="${3:?usage: run_frozenlake_dp2tp2_onehost.sh <p45|m15> <r0|r1|r2|r3> <fresh-label> <measure|certify>}"
mode="${4:?usage: run_frozenlake_dp2tp2_onehost.sh <p45|m15> <r0|r1|r2|r3> <fresh-label> <measure|certify>}"
case "$workload" in p45|m15) ;; *) echo "invalid workload: $workload" >&2; exit 2;; esac
case "$arm" in r0|r1|r2|r3) ;; *) echo "invalid arm: $arm" >&2; exit 2;; esac
case "$mode" in measure|certify) ;; *) echo "invalid mode: $mode" >&2; exit 2;; esac
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
root="$evidence_root/${workload}_${arm}_${label}"
canon_out="$root/canon"
raw="$root/raw.log"
driver="$root/driver.log"
container="v2_fl_${workload}_${arm}_${label}"
timeout_seconds="${V2_FL_TIMEOUT_SECONDS:-14400}"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
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
runner_sha256="$(sha256sum "$0" "$script_dir/run_frozenlake_dp2tp2_inner.sh" "$script_dir/classify_frozenlake_dp2tp2.py" | sha256sum | awk '{print $1}')"
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
{
  echo "[V2.FL.ONEHOST] source=$source_sha diff_sha256=$diff_sha"
  echo "[V2.FL.ONEHOST] image_id=$image_id model_revision=$model_revision"
  echo "[V2.FL.ONEHOST] workload=$workload arm=$arm mode=$mode topology=DP2xTP2 stage=backward-no-commit"
  echo "[V2.FL.ONEHOST] timeout_seconds=$timeout_seconds idle_120s=PASS root=$root"
} >"$driver"
bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen8b_tp2 \
  >>"$driver" 2>&1

{
  echo "[V2.FL.ONEHOST] RUN_BEGIN"
  sha256sum "$0" "$script_dir/run_frozenlake_dp2tp2_inner.sh" \
    "$script_dir/classify_frozenlake_dp2tp2.py" \
    "$pkg/cluster/profiles/qwen3-8b-dp2-tp2-frozenlake-onehost.env" \
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
  -e V2_FL_MODE="$mode" \
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

classifier_args=(
  "$script_dir/classify_frozenlake_dp2tp2.py"
  --root "$root" --workload "$workload" --arm "$arm"
  --docker-exit "$docker_rc"
  --anchor-registry "$script_dir/gradient_anchors.json"
  --output "$root/classification.json"
)
if [ "$mode" = certify ]; then
  classifier_args+=(--require-anchor)
fi
set +e
/mnt/disks/tunix-data/venvs/train/bin/python "${classifier_args[@]}" \
  >>"$driver" 2>&1
classifier_rc=$?
set -e
find "$root" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$root/SHA256SUMS"
if [ "$classifier_rc" -ne 0 ] || [ "$contention" -ne 0 ] || [ "$timed_out" -ne 0 ]; then
  echo "[V2.FL.ONEHOST] RED docker=$docker_rc classifier=$classifier_rc evidence=$root" >>"$driver"
  exit 1
fi
verdict="$(/mnt/disks/tunix-data/venvs/train/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$root/classification.json")"
echo "[V2.FL.ONEHOST] $verdict evidence=$root" >>"$driver"
echo "V2_FROZENLAKE_ONEHOST_${verdict} workload=$workload arm=$arm evidence=$root"
