#!/usr/bin/env bash
# Real four-chip v5p mechanism gate for the Attempt-3 RPA-local repair.
set -euo pipefail

label="${1:?usage: run_onehost_rpa_v5p.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "P59 RPA one-host label must use letters, digits, underscore, or dash" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
image="tunix_frozenlake_image:vllm-tpu0.25.0"
expected_image_id="sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a"
image_id="$(sudo docker image inspect "$image" --format '{{.Id}}')"
evidence=/mnt/disks/tunix-data/logp_probe_1host
root="$evidence/p59_rpa_${label}"
overlay="$root/overlay"
raw="$root/raw.log"
driver="$root/driver.log"
container="p59_rpa_${label}"
source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
timeout_seconds="${P59_RPA_ONEHOST_TIMEOUT_SECONDS:-3600}"

test "$(hostname)" = t1v-n-4a77ebd0-w-0
test "$image_id" = "$expected_image_id"
if [[ -e "$root" ]]; then
  echo "P59 RPA one-host run label already exists: $root" >&2
  exit 2
fi
active="$(sudo docker ps --format '{{.Names}}' | grep -E '^(p51_|p59_)' || true)"
if [[ -n "$active" ]]; then
  echo "P59 RPA one-host refuses an active P51/P59 TPU container" >&2
  printf '%s\n' "$active" >&2
  exit 2
fi

mkdir -p "$root"
{
  echo "P59_RPA_ONEHOST source=$source_sha diff_sha256=$diff_sha"
  echo "P59_RPA_ONEHOST image=$image image_id=$image_id"
  echo "P59_RPA_ONEHOST p59_topology=DP2xTP2 ordinary_topology=DP1xTP4"
  echo "P59_RPA_ONEHOST scope=mechanism-only optimizer_commits=0"
} >"$driver"

bash "$pkg/install.sh" "$overlay" --from-image "$image" --model qwen1p7b \
  >>"$driver" 2>&1
sha256sum \
  "$pkg/patches/tpu_inference/25-attention-p59-local-kv.patch" \
  "$pkg/MANIFEST.sha256" \
  "$pkg/tests/p59_backward/probe_onehost_rpa_v5p.py" \
  "$pkg/tests/p59_backward/run_onehost_rpa_v5p.sh" >"$raw"

active="$(sudo docker ps --format '{{.Names}}' | grep -E '^(p51_|p59_)' || true)"
if [[ -n "$active" ]]; then
  echo "P59 RPA one-host TPU lane became busy before launch" >>"$raw"
  exit 2
fi

set +e
started="$(date +%s)"
timeout --signal=TERM --kill-after=180s "${timeout_seconds}s" \
sudo docker run --rm --privileged --network=host --ipc=host \
  --name "$container" \
  -v "$repo:/workspace:ro" \
  -v "$overlay:/opt/canon-overlay:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=tpu \
  -e PYTHONPATH=/opt/canon-overlay:/workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT=/opt/canon-overlay \
  -e CANON_P59_RANK_PARALLEL_BACKWARD=1 \
  -e CANON_RPA_VJP2=1 \
  -e CANON_VJP2_MAX_SEQS=1 \
  "$image_id" \
  python3 canon-zero-tim/tests/p59_backward/probe_onehost_rpa_v5p.py \
  >>"$raw" 2>&1 &
wait_pid=$!
wait "$wait_pid"
docker_rc=$?
set -e
elapsed=$(( $(date +%s) - started ))
echo "P59_RPA_ONEHOST_END docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"

if sudo docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
  sudo docker stop "$container" >/dev/null 2>&1 || true
fi
terminal_count="$(grep -ac '^P59_RPA_ONEHOST_V5P_PASS ' "$raw" || true)"
local_marker_count="$(grep -ac '^\[PATHTRACE\] P59_RPA_LOCAL_KV_READY tp=2 local_q_heads=4 local_kv_heads=1 cache_heads=1 packing=2$' "$raw" || true)"
traceback_count="$(grep -ac 'Traceback (most recent call last)' "$raw" || true)"
red=""
[[ "$docker_rc" -eq 0 ]] || red="$red docker_exit=$docker_rc"
[[ "$terminal_count" -eq 1 ]] || red="$red terminal=$terminal_count/1"
[[ "$local_marker_count" -eq 1 ]] || red="$red local_marker=$local_marker_count/1"
[[ "$traceback_count" -eq 0 ]] || red="$red traceback=$traceback_count"
{
  echo "P59_RPA_ONEHOST_SUMMARY docker_exit=$docker_rc elapsed_seconds=$elapsed terminal=$terminal_count local_marker=$local_marker_count traceback=$traceback_count"
  if [[ -n "$red" ]]; then
    echo "P59_RPA_ONEHOST_RED$red"
  else
    echo "P59_RPA_ONEHOST_GREEN root=$root"
  fi
} >>"$driver"
sha256sum "$raw" "$driver" >"$root/SHA256SUMS"
chmod -R a+rX "$root" || true
if [[ -n "$red" ]]; then
  exit 1
fi
