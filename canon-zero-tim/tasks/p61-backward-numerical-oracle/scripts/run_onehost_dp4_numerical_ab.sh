#!/usr/bin/env bash
# Launch the frozen P61 serial/parallel one-update numerical pair in order.
set -euo pipefail

serial_label="${1:?usage: run_onehost_dp4_numerical_ab.sh <serial-label> <parallel-label> <ab-label> <tier1-baseline.json>}"
parallel_label="${2:?usage: run_onehost_dp4_numerical_ab.sh <serial-label> <parallel-label> <ab-label> <tier1-baseline.json>}"
ab_label="${3:?usage: run_onehost_dp4_numerical_ab.sh <serial-label> <parallel-label> <ab-label> <tier1-baseline.json>}"
tier1_baseline="${4:?usage: run_onehost_dp4_numerical_ab.sh <serial-label> <parallel-label> <ab-label> <tier1-baseline.json>}"
for label in "$serial_label" "$parallel_label" "$ab_label"; do
  case "$label" in
    *[!a-zA-Z0-9_-]*|'')
      echo "[P61.NUMERICAL] invalid immutable label: $label" >&2
      exit 2
      ;;
  esac
done
if [ ! -s "$tier1_baseline" ]; then
  echo "[P61.NUMERICAL] missing Tier-1 baseline: $tier1_baseline" >&2
  exit 2
fi
tier1_baseline="$(realpath "$tier1_baseline")"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
test_mode="${P61_NUMERICAL_TEST_MODE:-0}"
case "$test_mode" in
  0)
    for variable in \
      P61_NUMERICAL_TEST_RUNNER \
      P61_NUMERICAL_TEST_COMPARATOR \
      P61_NUMERICAL_EVIDENCE_ROOT; do
      if [ -n "${!variable:-}" ]; then
        echo "[P61.NUMERICAL] $variable requires test mode" >&2
        exit 2
      fi
    done
    runner="$pkg/tasks/p59-dp16-parallel-backward/scripts/run_onehost_dp4.sh"
    comparator="$pkg/tests/p61_backward/compare_full_trees.py"
    evidence=/mnt/disks/tunix-data/logp_probe_1host
    ;;
  1)
    : "${P61_NUMERICAL_TEST_RUNNER:?test runner unset}"
    : "${P61_NUMERICAL_TEST_COMPARATOR:?test comparator unset}"
    : "${P61_NUMERICAL_EVIDENCE_ROOT:?test evidence root unset}"
    case "$P61_NUMERICAL_EVIDENCE_ROOT" in
      /tmp/*) ;;
      *) echo "[P61.NUMERICAL] test evidence root must be under /tmp" >&2; exit 2 ;;
    esac
    runner="$P61_NUMERICAL_TEST_RUNNER"
    comparator="$P61_NUMERICAL_TEST_COMPARATOR"
    evidence="$P61_NUMERICAL_EVIDENCE_ROOT"
    ;;
  *)
    echo "[P61.NUMERICAL] P61_NUMERICAL_TEST_MODE must be 0 or 1" >&2
    exit 2
    ;;
esac
serial_root="$evidence/p59_dp4_numerical-control_${serial_label}"
parallel_root="$evidence/p59_dp4_numerical-candidate_${parallel_label}"
ab_root="$evidence/p61_dp4_numerical_ab_${ab_label}"
result="$ab_root/numerical_ab.json"
driver="$ab_root/driver.log"

if [ -e "$serial_root" ] || [ -e "$parallel_root" ] || [ -e "$ab_root" ]; then
  echo "[P61.NUMERICAL] refusing reused run label" >&2
  printf '%s\n' "$serial_root" "$parallel_root" "$ab_root" >&2
  exit 2
fi
mkdir -p "$ab_root"
{
  echo "[P61.NUMERICAL] BEGIN serial_label=$serial_label parallel_label=$parallel_label ab_label=$ab_label"
  echo "[P61.NUMERICAL] test_mode=$test_mode"
  echo "[P61.NUMERICAL] topology=DP4xTP1 steps=1/1 align=17/17_per_arm fail=0"
  echo "[P61.NUMERICAL] baseline=$tier1_baseline baseline_sha256=$(sha256sum "$tier1_baseline" | awk '{print $1}')"
  echo "[P61.NUMERICAL] performance_eligible=0 reason=full_tree_d2h_and_io"
} >"$driver"

CANON_P60_DETERMINISTIC_AB=1 bash "$runner" \
  numerical-control "$serial_label" >>"$driver" 2>&1
CANON_P60_DETERMINISTIC_AB=1 bash "$runner" \
  numerical-candidate "$parallel_label" >>"$driver" 2>&1

set +e
python3 "$comparator" \
  --control-root "$serial_root/train/p61_numerical" \
  --candidate-root "$parallel_root/train/p61_numerical" \
  --control-update "$serial_root/train/updates.jsonl" \
  --candidate-update "$parallel_root/train/updates.jsonl" \
  --control-classification "$serial_root/train/classification.json" \
  --candidate-classification "$parallel_root/train/classification.json" \
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
  NUMERICAL_KEEP_DP4_PROXY:0)
    classification=KEEP
    ;;
  REJECT_ZERO_TIM:1|INCONCLUSIVE_CARRIER:1|NUMERICAL_REJECT:1)
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
  echo "[P61.NUMERICAL] TERMINAL verdict=$verdict classification=$classification comparator_status=$comparator_status"
  echo "[P61.NUMERICAL] result=$result"
} >>"$driver"

manifest_inputs=(
  "$driver" "$tier1_baseline"
  "$serial_root/SHA256SUMS" "$parallel_root/SHA256SUMS"
  "$runner" "$comparator" "$0"
)
if [ -s "$result" ]; then
  manifest_inputs=("$result" "${manifest_inputs[@]}")
fi
sha256sum "${manifest_inputs[@]}" >"$ab_root/SHA256SUMS"
exit "$comparator_status"
