#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "usage: $0 <native|is|zero> <source-sha> <output-dir> <p45-run-id> <m15-run-id> <campaign-root>" >&2
  exit 2
fi

wave="$1"
source_sha="$2"
output_root="$3"
p45_run_id="$4"
m15_run_id="$5"
campaign_root="$6"

case "$wave" in
  native) arm=mismatch ;;
  is) arm=is ;;
  zero) arm=zero ;;
  *) echo "wave must be native, is, or zero" >&2; exit 2 ;;
esac
renderer_mode=()
if [ "$wave" = "zero" ]; then
  renderer_mode+=(--high-performance)
fi
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "source SHA must be exactly 40 lowercase hex characters" >&2
  exit 2
}
[ ! -e "$output_root" ] || {
  echo "refusing to overwrite output root: $output_root" >&2
  exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../../.." && pwd)"
renderer="$repo/canon-zero-tim/cluster/render_p57_frozenlake_tim.py"
git -C "$repo" cat-file -e "$source_sha^{commit}"
[ "$(git -C "$repo" rev-parse HEAD)" = "$source_sha" ] || {
  echo "refusing to render from a checkout that is not the requested source SHA" >&2
  exit 2
}
[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ] || {
  echo "refusing to render from a dirty worktree" >&2
  exit 2
}
expected_updates=300
mkdir -p "$output_root"

python3 "$renderer" \
  --source-commit "$source_sha" \
  --run-id "$p45_run_id" \
  --output-dir "$output_root/p45" \
  --campaign-tag "${campaign_root}-p45" \
  --checkpoint-mode new \
  --expected-updates "$expected_updates" \
  --run-kind train \
  --arm "$arm" \
  "${renderer_mode[@]}"

python3 "$renderer" \
  --source-commit "$source_sha" \
  --run-id "$m15_run_id" \
  --output-dir "$output_root/m15" \
  --campaign-tag "${campaign_root}-m15" \
  --checkpoint-mode new \
  --expected-updates "$expected_updates" \
  --run-kind train \
  --workload-candidate m15 \
  --data-split main \
  --arm "$arm" \
  "${renderer_mode[@]}"

p45_manifest="$(find "$output_root/p45" -maxdepth 1 -name 'jobset-*.yaml' -print)"
m15_manifest="$(find "$output_root/m15" -maxdepth 1 -name 'jobset-*.yaml' -print)"
[ -n "$p45_manifest" ] && [ -n "$m15_manifest" ] || {
  echo "render did not produce both manifests" >&2
  exit 1
}

python3 "$script_dir/verify_three_arm_manifests.py" \
  --wave "$wave" \
  --source "$source_sha" \
  --p45 "$p45_manifest" \
  --m15 "$m15_manifest"
sha256sum "$p45_manifest" "$m15_manifest"
echo "P57_THREE_ARM_RENDER_PASS wave=$wave output=$output_root"
