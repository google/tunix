#!/usr/bin/env bash
# Run the frozen P66 ordinary/segmented no-commit backward pair in order.
set -euo pipefail

ordinary_label="${1:?usage: run_backward_ab.sh <ordinary-label> <segmented-label> <ab-label> <tier1-baseline.json>}"
segmented_label="${2:?usage: run_backward_ab.sh <ordinary-label> <segmented-label> <ab-label> <tier1-baseline.json>}"
ab_label="${3:?usage: run_backward_ab.sh <ordinary-label> <segmented-label> <ab-label> <tier1-baseline.json>}"
tier1_baseline="${4:?usage: run_backward_ab.sh <ordinary-label> <segmented-label> <ab-label> <tier1-baseline.json>}"
for label in "$ordinary_label" "$segmented_label" "$ab_label"; do
  case "$label" in
    *[!a-zA-Z0-9_-]*|'')
      echo "[P66.BACKWARD] invalid immutable label: $label" >&2
      exit 2
      ;;
  esac
done
if [ ! -s "$tier1_baseline" ]; then
  echo "[P66.BACKWARD] missing Tier-1 baseline: $tier1_baseline" >&2
  exit 2
fi
tier1_baseline="$(realpath "$tier1_baseline")"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
runner="$pkg/tasks/p59-dp16-parallel-backward/scripts/run_onehost_dp4.sh"
comparator="$pkg/tests/p66_backward/compare_arms.py"
evidence=/mnt/disks/tunix-data/logp_probe_1host
ordinary_root="$evidence/p59_dp4_p66-ordinary_${ordinary_label}"
segmented_root="$evidence/p59_dp4_p66-segmented_${segmented_label}"
ab_root="$evidence/p66_backward_ab_${ab_label}"
result="$ab_root/backward_ab.json"
driver="$ab_root/driver.log"

if [ -e "$ordinary_root" ] || [ -e "$segmented_root" ] || [ -e "$ab_root" ]; then
  echo "[P66.BACKWARD] refusing reused run label" >&2
  printf '%s\n' "$ordinary_root" "$segmented_root" "$ab_root" >&2
  exit 2
fi
mkdir -p "$ab_root"
{
  echo "[P66.BACKWARD] BEGIN ordinary_label=$ordinary_label segmented_label=$segmented_label ab_label=$ab_label"
  echo "[P66.BACKWARD] topology=DP4xTP1 optimizer_commits=0 align=17/17_per_arm fail=0"
  echo "[P66.BACKWARD] baseline=$tier1_baseline baseline_sha256=$(sha256sum "$tier1_baseline" | awk '{print $1}')"
  echo "[P66.BACKWARD] performance_eligible=0 reason=full_tree_d2h_and_io"
} >"$driver"

CANON_P60_DETERMINISTIC_AB=1 bash "$runner" \
  p66-ordinary "$ordinary_label" >>"$driver" 2>&1
CANON_P60_DETERMINISTIC_AB=1 bash "$runner" \
  p66-segmented "$segmented_label" >>"$driver" 2>&1

set +e
python3 "$comparator" \
  --ordinary-root "$ordinary_root/train/p66_backward" \
  --segmented-root "$segmented_root/train/p66_backward" \
  --ordinary-update "$ordinary_root/train/updates.jsonl" \
  --segmented-update "$segmented_root/train/updates.jsonl" \
  --ordinary-classification "$ordinary_root/train/classification.json" \
  --segmented-classification "$segmented_root/train/classification.json" \
  --tier1-baseline "$tier1_baseline" --output "$result" \
  >>"$driver" 2>&1
comparator_status=$?
set -e

verdict=MISSING_RESULT
if [ -s "$result" ]; then
  verdict="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("verdict", "MISSING_VERDICT"))' \
    "$result")"
fi
case "$verdict:$comparator_status" in
  P66_GRADIENT_KEEP:0) classification=KEEP ;;
  REJECT_ZERO_TIM:1|INCONCLUSIVE_CARRIER:1|P66_GRADIENT_REJECT:1)
    classification=CLASSIFIED_NON_KEEP
    ;;
  *)
    classification=INVALID_COMPARATOR_OUTCOME
    if [ "$comparator_status" -eq 0 ]; then
      comparator_status=2
    fi
    ;;
esac
{
  echo "[P66.BACKWARD] TERMINAL verdict=$verdict classification=$classification comparator_status=$comparator_status"
  echo "[P66.BACKWARD] result=$result"
} >>"$driver"

manifest_inputs=(
  "$driver" "$tier1_baseline"
  "$ordinary_root/SHA256SUMS" "$segmented_root/SHA256SUMS"
  "$runner" "$comparator" "$0"
)
if [ -s "$result" ]; then
  manifest_inputs=("$result" "${manifest_inputs[@]}")
fi
sha256sum "${manifest_inputs[@]}" >"$ab_root/SHA256SUMS"
exit "$comparator_status"
