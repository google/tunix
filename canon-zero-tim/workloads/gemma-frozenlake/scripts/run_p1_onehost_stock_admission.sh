#!/usr/bin/env bash
# Runs one explicitly approved Gemma E4B-IT P1 admission on a direct v5p host.
set -euo pipefail

workload="${1:?usage: run_p1_onehost_stock_admission.sh <p45|m15> <unique-label>}"
label="${2:?usage: run_p1_onehost_stock_admission.sh <p45|m15> <unique-label>}"
case "$workload" in p45|m15) ;; *) echo "invalid workload" >&2; exit 2 ;; esac
case "$label" in *[!a-zA-Z0-9_-]*|'') echo "invalid label" >&2; exit 2 ;; esac
: "${HF_TOKEN:?HF_TOKEN is required for the pinned checkpoint}"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
profile="$repo/canon-zero-tim/cluster/profiles/gemma4-e4b-dp1-tp4-frozenlake.env"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
image="${GEMMA4_P1_IMAGE:-tunix_frozenlake_image:vllm-tpu0.25.0}"
expected_image=sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
image_id="$(sudo docker image inspect "$image" --format '{{.Id}}')"
root="/mnt/disks/tunix-data/gemma4_e4b_p1/${label}"
raw="$root/raw.log"
manifest="$root/manifest.json"
post_manifest="$root/post_manifest.json"
classification="$root/classification.json"
launch_preflight="$root/launch_preflight.json"
container="gemma4_e4b_p1_${label}"
timeout_seconds="${GEMMA4_P1_TIMEOUT_SECONDS:-10800}"
deps=/mnt/disks/tunix-data/frozenlake/deps

test -f "$canon_env"
# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$image_id" = "$expected_image"
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -f "$deps/gymnasium-1.3.0-py3-none-any.whl"
test -f "$deps/farama_notifications-0.0.6-py3-none-any.whl"
if [ -n "$(git -C "$repo" status --porcelain)" ]; then
  echo "refusing to launch P1 from a dirty runtime tree" >&2
  exit 5
fi
if [ -e "$root" ]; then
  echo "refusing existing evidence root: $root" >&2
  exit 3
fi

active=""
while IFS= read -r running_container; do
  case "$running_container" in
    tpu-runtime|instance_agent|vbarcontrolagent|google-runtime-monitor|healthagent|google-collectd|monitoringagent) continue ;;
  esac
  if [ "$(sudo docker inspect --format '{{.HostConfig.Privileged}}' "$running_container")" = true ]; then
    active="${active}${active:+$'\n'}${running_container}"
  fi
done < <(sudo docker ps --format '{{.Names}}')
if [ -n "$active" ]; then
  echo "refusing host with non-system privileged container" >&2
  printf '%s\n' "$active" >&2
  exit 4
fi

mkdir -p "$root/logs" "$root/wandb" "/mnt/disks/tunix-data/jax-cache/gemma4-e4b-p1"
python3 "$script_dir/render_p1_onehost_stock_admission.py" \
  --repo "$repo" --workload "$workload" --image-digest "$image_id" \
  --output "$manifest"

export CANON_GEMMA4_E4B_P1_ADMISSION="$workload"
# shellcheck disable=SC1090
source "$profile"

{
  echo "[GEMMA4_E4B_P1_HOST] {\"hostname\":\"$(hostname)\",\"schema\":\"gemma4-e4b-p1-host-v1\"}"
  echo "[GEMMA4_E4B_P1_DRIVER] workload=$workload source=$(git -C "$repo" rev-parse HEAD)"
  echo "[GEMMA4_E4B_P1_DRIVER] image_digest=$image_id profile=$CANON_PROFILE"
  echo "[GEMMA4_E4B_P1_DRIVER] rollout_only=1 backward=0 optimizer_commits=0"
} >"$raw"

set +e
python3 "$script_dir/preflight_p1_launch.py" \
  --repo "$repo" --workload "$workload" --manifest "$manifest" \
  --output "$launch_preflight" >>"$raw" 2>&1
preflight_rc=$?
set -e
if [ "$preflight_rc" -ne 0 ]; then
  find "$root" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$root/SHA256SUMS"
  echo "GEMMA4_E4B_P1_PREFLIGHT_FAIL workload=$workload evidence=$root" >&2
  exit 1
fi

set +e
timeout --foreground --signal=TERM --kill-after=180s "${timeout_seconds}s" \
sudo --preserve-env=HF_TOKEN docker run --rm --privileged --net=host --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$repo":"$repo":ro \
  -w "$repo" \
  --env HF_TOKEN \
  -e PYTHONPATH="$repo" -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e WANDB_MODE=disabled -e WANDB_DIR="$root/wandb" \
  -e GEMMA4_P1_LOCAL_LOG_DIR="$root/logs" \
  -e JAX_COMPILATION_CACHE_DIR=/mnt/disks/tunix-data/jax-cache/gemma4-e4b-p1 \
  -e CANON_GEMMA4_E4B_P1_ADMISSION="$workload" \
  -e CANON_PROFILE="$CANON_PROFILE" \
  -e CANON_ENGINE_MODULE_C=0 -e CANON_VLLM_ENABLE_PREFIX_CACHING=0 \
  -e CANON_EXPECT_JAX_VERSION="$CANON_EXPECT_JAX_VERSION" \
  -e CANON_EXPECT_PATHWAYS_RELEASE="$CANON_EXPECT_PATHWAYS_RELEASE" \
  -e CANON_EXPECT_VISIBLE_DEVICES=4 -e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4 \
  -e FL_ROLLOUT_MESH=1,4 -e FL_TRAINER_MESH=1,4 \
  -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  "$image_id" bash -lc "
    set -e
    python3 -m pip install --no-index \
      '$deps/farama_notifications-0.0.6-py3-none-any.whl' \
      '$deps/gymnasium-1.3.0-py3-none-any.whl'
    exec python3 -u examples/frozenlake/train_frozenlake.py
  " >>"$raw" 2>&1
docker_rc=$?
set -e
sudo chmod -R a+rX "$root" || true
python3 "$script_dir/render_p1_onehost_stock_admission.py" \
  --repo "$repo" --workload "$workload" --image-digest "$image_id" \
  --output "$post_manifest"

set +e
python3 "$script_dir/classify_p1_admission.py" \
  --manifest "$manifest" --post-manifest "$post_manifest" \
  --raw "$raw" --docker-exit "$docker_rc" \
  --output "$classification"
classifier_rc=$?
set -e
find "$root" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$root/SHA256SUMS"
if [ "$classifier_rc" -ne 0 ]; then
  echo "GEMMA4_E4B_P1_FAIL workload=$workload evidence=$root" >&2
  exit 1
fi
echo "GEMMA4_E4B_P1_PASS workload=$workload evidence=$root"
