#!/usr/bin/env bash
# Render-only preparation for three parallel-capable P58 checked-VMA JobSets.
set -euo pipefail

usage() {
  echo "usage: P58_EXPECT_SOURCE_SHA=<40hex> $0 WAVE_ID CLIENT_IMAGE@sha256:... WORKER_NODEPOOL OUTPUT_DIR [MODEL_PVC]" >&2
}

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  usage
  exit 2
fi

wave_id="$1"
client_image="$2"
worker_nodepool="$3"
output_dir="$4"
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
  echo "[prepare] local HEAD, expected SHA, and origin/yuxzhang/canon-zero-tim must be identical" >&2
  exit 1
fi
if [ -e "$output_dir" ]; then
  echo "[prepare] refusing to overwrite $output_dir" >&2
  exit 1
fi

python3 "$script_dir/render_p58_checked_vma_aba_wave.py" \
  --output-dir "$output_dir" \
  --source-commit "$actual_source" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$client_image" \
  --wave-id "$wave_id" \
  --cpu-nodepool cpu-np \
  --worker-nodepool "$worker_nodepool" \
  --model-pvc "$model_pvc"

python3 "$script_dir/verify_p58_checked_vma_aba_wave.py" \
  --output-dir "$output_dir"

sha256sum "$output_dir"/jobsets/*.yaml \
  "$output_dir/wave-render-receipt.json" \
  "$output_dir/wave-verify.json"
echo "P58_CHECKED_VMA_ABA_WAVE_READY source=$actual_source output=$output_dir jobs=3 tpu_request=384 sandbox_concurrency=384"
echo "[prepare] render only; server dry-run, image publication, and parallel apply require separate approval"
