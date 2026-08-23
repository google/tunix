#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 <native|is|zero> <source-sha> <output-dir> <run-id-root> <campaign-root>" >&2
  exit 2
fi

wave="$1"
source_sha="$2"
output_root="$3"
run_id_root="$4"
campaign_root="$5"

case "$wave" in
  native) arm=mismatch ;;
  is) arm=is ;;
  zero) arm=zero ;;
  *) echo "wave must be native, is, or zero" >&2; exit 2 ;;
esac
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "source SHA must be exactly 40 lowercase hex characters" >&2
  exit 2
}
[[ "$run_id_root" =~ ^[a-z0-9][a-z0-9-]{0,5}$ ]] || {
  echo "run-id root must be 1-6 lowercase alphanumeric/hyphen characters" >&2
  exit 2
}
[ ! -e "$output_root" ] || {
  echo "refusing to overwrite output root: $output_root" >&2
  exit 2
}
git cat-file -e "$source_sha^{commit}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../../.." && pwd)"
renderer="$repo/canon-zero-tim/cluster/render_p57_frozenlake_tim.py"
base_yaml="$repo/canon-zero-tim/cluster/jobset-64chip.yaml"
[ -f "$base_yaml" ] || { echo "base YAML is absent: $base_yaml" >&2; exit 2; }

# The primary 0..300 curve is emitted by each training JobSet.  This renderer
# remains only as a recovery audit for the clean initial policy and final
# rolling checkpoint; intermediate checkpoints are intentionally not retained.
for workload in p45 m15; do
  workload_letter=p
  candidate_args=()
  if [ "$workload" = m15 ]; then
    workload_letter=m
    candidate_args=(--workload-candidate m15 --data-split main)
  fi
  for step in 0 300; do
    mode=resume
    if [ "$step" -eq 0 ]; then mode=new; fi
    run_id="${run_id_root}${workload_letter}${step}"
    python3 "$renderer" \
      --base "$base_yaml" \
      --source-commit "$source_sha" \
      --run-id "$run_id" \
      --output-dir "$output_root/$workload/step-$step" \
      --campaign-tag "${campaign_root}-${workload}" \
      --checkpoint-mode "$mode" \
      --expected-updates 300 \
      --run-kind eval \
      --checkpoint-step "$step" \
      --arm "$arm" \
      "${candidate_args[@]}"
  done
done

python3 "$script_dir/verify_eval_schedule.py" \
  --root "$output_root" \
  --wave "$wave" \
  --source "$source_sha" \
  --campaign-root "$campaign_root"
find "$output_root" -type f -name 'jobset-*.yaml' -print0 \
  | sort -z \
  | xargs -0 sha256sum
echo "P57_RECOVERY_EVAL_RENDER_PASS wave=$wave output=$output_root manifests=4"
