#!/usr/bin/env bash
# Render-only preparation for the exact 128-chip P58.19 coarse seam run.
# This script never talks to Kubernetes and never publishes source or images.
set -euo pipefail

usage() {
  echo "usage: P58_EXPECT_SOURCE_SHA=<40hex> $0 RUN_ID CLIENT_IMAGE@sha256:... WORKER_NODEPOOL OUTPUT_YAML [MODEL_PVC]" >&2
}

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  usage
  exit 2
fi

run_id="$1"
client_image="$2"
worker_nodepool="$3"
output_yaml="$4"
model_pvc="${5:-haoyugao-cpu-np-pvc}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pkg="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$pkg/.." && pwd)"
expected_source="${P58_EXPECT_SOURCE_SHA:-}"

if ! [[ "$expected_source" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[prepare] P58_EXPECT_SOURCE_SHA must be the published 40-character source SHA" >&2
  exit 2
fi
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=normal)" ]; then
  echo "[prepare] refusing a dirty source tree" >&2
  exit 1
fi
actual_source="$(git -C "$repo" rev-parse HEAD)"
published_source="$(git -C "$repo" rev-parse origin/yuxzhang/canon-zero-tim)"
if [ "$actual_source" != "$expected_source" ] || \
   [ "$published_source" != "$expected_source" ]; then
  echo "[prepare] local HEAD, expected SHA, and operator branch must be identical" >&2
  exit 1
fi
if [ -e "$output_yaml" ]; then
  echo "[prepare] refusing to overwrite $output_yaml" >&2
  exit 1
fi

python3 "$pkg/cluster/render_p58_deepswe_tim.py" \
  --base "$pkg/cluster/jobset-64chip.yaml" \
  --output "$output_yaml" \
  --source-commit "$actual_source" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$client_image" \
  --run-id "$run_id" \
  --stage full \
  --arm zero \
  --cpu-nodepool canon-cpu-pool \
  --sandbox-nodepool deepswe-cpu-pool-2 \
  --worker-nodepool "$worker_nodepool" \
  --model-pvc "$model_pvc" \
  --seam-localization coarse

echo "P58_COARSE_SEAM_PREPARE_PASS source=$actual_source output=$output_yaml rounds=3 backward=0 optimizer_commits=0"
echo "[prepare] render only; Kubernetes dry-run/apply requires separate operator approval"
