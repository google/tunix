#!/usr/bin/env bash
# Host-side pinned-image launcher for the P58.17 direct-v5p seam probe.
set -euo pipefail

label="${1:?usage: run_onehost_deepswe_seam_probe_docker.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P58.ONEHOST.DOCKER] invalid immutable label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
workspace_root="$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)"
workspace_root="$(dirname "$workspace_root")"
r2egym_root="${DEEPSWE_R2EGYM_ROOT:-/home/yuxuan/tunix/submodules/R2E-Gym}"
docker_sdk_path="${P58_ONEHOST_DOCKER_SDK_PATH:-/mnt/disks/tunix-data/venvs/train/lib/python3.11/site-packages/docker}"
image="${P58_ONEHOST_IMAGE:-tunix_frozenlake_image:vllm-tpu0.25.0}"
expected_hostname="${P58_ONEHOST_EXPECT_HOSTNAME:-t1v-n-4a77ebd0-w-0}"
evidence_root="${P58_ONEHOST_EVIDENCE_ROOT:-/mnt/disks/tunix-data/deepswe-onehost-xprof}"
artifact_dir="${P58_ONEHOST_ARTIFACT_DIR:-$evidence_root/p58_zero-hp_${label}}"
container="p58_seam_${label//-/_}"

if [ "$(hostname)" != "$expected_hostname" ]; then
  echo "[P58.ONEHOST.DOCKER] hostname mismatch: actual=$(hostname) expected=$expected_hostname" >&2
  exit 2
fi
if [ "$artifact_dir" = "${artifact_dir#/}" ] || [ -e "$artifact_dir" ]; then
  echo "[P58.ONEHOST.DOCKER] artifact directory must be a fresh absolute path: $artifact_dir" >&2
  exit 2
fi
for path in "$workspace_root" "$repo" "$r2egym_root" "$docker_sdk_path" /mnt/disks/tunix-data /var/run/docker.sock; do
  if [ ! -e "$path" ]; then
    echo "[P58.ONEHOST.DOCKER] missing host prerequisite: $path" >&2
    exit 2
  fi
done

image_id="$(sudo docker image inspect --format '{{.Id}}' "$image")"
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "[P58.ONEHOST.DOCKER] image has no immutable local id: $image" >&2
  exit 2
fi
active="$(sudo docker ps --format '{{.Names}}' | grep -E '^(p51_|p59_|p66|v1_gsm8k_xprof_|p58_seam_)' || true)"
if [ -n "$active" ]; then
  echo "[P58.ONEHOST.DOCKER] one-host TPU lane is busy" >&2
  printf '%s\n' "$active" >&2
  exit 2
fi

echo "[P58.ONEHOST.DOCKER] RUN_BEGIN label=$label image_id=$image_id artifact_dir=$artifact_dir"
sudo docker run --rm --privileged --net=host --ipc=host --uts=host \
  --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$workspace_root:$workspace_root:ro" \
  -v "$repo:$repo:ro" \
  -v "$r2egym_root:$r2egym_root:ro" \
  -v "$docker_sdk_path:/opt/p58-deps/docker:ro" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w "$repo" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/p58-deps \
  -e DEEPSWE_TRAIN_PYTHON=/usr/local/bin/python3 \
  -e DEEPSWE_R2EGYM_ROOT="$r2egym_root" \
  -e P58_ONEHOST_EXPECT_HOSTNAME="$expected_hostname" \
  -e P58_ONEHOST_ARTIFACT_DIR="$artifact_dir" \
  -e P58_ONEHOST_ALLOW_DIRTY="${P58_ONEHOST_ALLOW_DIRTY:-0}" \
  "$image" \
  bash -euo pipefail -c '
    git config --global --add safe.directory "$1"
    git config --global --add safe.directory "$2"
    git config --global --add safe.directory "$2/.git"
    exec bash "$1/canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_seam_probe.sh" "$3"
  ' bash "$repo" "$r2egym_root" "$label"
echo "[P58.ONEHOST.DOCKER] RUN_END label=$label artifact_dir=$artifact_dir"
