#!/usr/bin/env bash
# Recover and audit only bounded e0w5 evidence. Never launch or mutate GCS.
set -euo pipefail

analysis_source="${1:?usage: run_m15_e0w5_gcs_return.sh <full-analysis-sha> <original-e0w5-render-dir> <new-output-dir> [scratch-parent]}"
render_dir="${2:?usage: run_m15_e0w5_gcs_return.sh <full-analysis-sha> <original-e0w5-render-dir> <new-output-dir> [scratch-parent]}"
output="${3:?usage: run_m15_e0w5_gcs_return.sh <full-analysis-sha> <original-e0w5-render-dir> <new-output-dir> [scratch-parent]}"
scratch_parent="${4:-/tmp}"
target_source="2f61f8fc7cf073964a9adbd30e78de872426a4d2"
incident_manifest_sha="9c81c858e3e9b1e9d68ebbdf332de1e550bc9536de27dd121d8d1bf9a9a3bc60"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
incident="$canon/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_e0w5_paired_20260831"

refuse() {
  echo "[M15.E0W5.RECOVERY] REFUSING status=$1 classification=NONE" >&2
  exit 2
}

[[ "$analysis_source" =~ ^[0-9a-f]{40}$ ]] || refuse INVALID_ANALYSIS_SOURCE
[ -d "$render_dir" ] || refuse ORIGINAL_RENDER_UNAVAILABLE
[ -d "$scratch_parent" ] || refuse SCRATCH_PARENT_UNAVAILABLE
[ ! -e "$output" ] || refuse OUTPUT_ALREADY_EXISTS
[ ! -e "$output.partial" ] || refuse STALE_PARTIAL_OUTPUT

branch="$(git -C "$repo" branch --show-current)"
case "$branch" in
  local/*) ;;
  *) refuse NONLOCAL_BRANCH ;;
esac
head="$(git -C "$repo" rev-parse HEAD)"
[ "$head" = "$analysis_source" ] || refuse HEAD_ANALYSIS_SOURCE_MISMATCH
git -C "$repo" merge-base --is-ancestor "$target_source" "$analysis_source" || \
  refuse TARGET_SOURCE_NOT_ANCESTOR
python3 "$canon/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py" \
  --repo "$repo" --require-clean

(cd "$incident" && sha256sum -c SHA256SUMS --quiet)
read -r actual_incident_manifest_sha _ < <(sha256sum "$incident/SHA256SUMS")
[ "$actual_incident_manifest_sha" = "$incident_manifest_sha" ] || \
  refuse INCIDENT_MANIFEST_DRIFT
python3 "$script_dir/validate_m15_e0w5_recovery_render.py" \
  --render-dir "$render_dir" --target-source "$target_source"

raw_log="$(mktemp -p "$scratch_parent" m15-e0w5-gcs-return.XXXXXX.log)"
set +e
bash "$script_dir/run_m15_multiround_gcs_return.sh" \
  "$render_dir" "$output" "$scratch_parent" 1 >"$raw_log" 2>&1
return_exit=$?
set -e
raw_log_sha="$(sha256sum "$raw_log" | awk '{print $1}')"
if [ "$return_exit" -ne 0 ]; then
  preserved_scratch="$(sed -n 's/^\[M15.MULTIROUND\] FAILURE_PRESERVED scratch=//p' "$raw_log" | tail -n 1)"
  [ -n "$preserved_scratch" ] || preserved_scratch="NONE"
  echo "[M15.E0W5.RECOVERY] INCONCLUSIVE status=OFFICIAL_RETURN_FAILED return_exit=$return_exit target_source=$target_source analysis_source=$analysis_source raw_log=$raw_log raw_log_sha256=$raw_log_sha preserved_scratch=$preserved_scratch"
  echo "[M15.E0W5.RECOVERY] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0 target_pass=0 first_red_localized=0 numerical_repair_authorized=0"
  exit 3
fi

(cd "$output" && sha256sum -c SHA256SUMS --quiet)
read -r status off_sealed on_sealed localized candidate three_round < <(
  python3 - "$output/MULTIROUND_SUMMARY.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
off = value["arms"]["off"]
on = value["arms"]["on"]
on_names = [
    row.get("classification", "")
    for row in on["rounds"] if row.get("status") == "SEALED"
]
localized_names = {
    "M15_LAYER_FIRST_RED_LOCALIZED",
    "M15_HIDDEN_EXACT_TAIL_FIRST_RED_LOCALIZED",
    "M15_INTERNAL_FIRST_RED_LOCALIZED",
}
candidate_names = {
    "M15_LAYER_FIRST_RED_CANDIDATE_SET",
    "M15_INTERNAL_FIRST_RED_CANDIDATE_SET",
}
off_names = [
    row.get("classification", "")
    for row in off["rounds"] if row.get("status") == "SEALED"
]
three_round = (
    off["sealed_rounds"] == 3
    and on["sealed_rounds"] == 3
    and off_names == ["M15_OBSERVER_CONTROL_EXACT"] * 3
    and len(set(on_names)) == 1
)
print(
    value["status"], off["sealed_rounds"], on["sealed_rounds"],
    sum(name in localized_names for name in on_names),
    sum(name in candidate_names for name in on_names),
    int(three_round),
)
PY
)
manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
echo "[M15.E0W5.RECOVERY] COMPLETE status=$status target_source=$target_source analysis_source=$analysis_source off_sealed_rounds=$off_sealed on_sealed_rounds=$on_sealed localized_classifier_rounds=$localized candidate_set_rounds=$candidate three_round_numerical=$three_round target_pass=0 first_red_localized=0 numerical_repair_authorized=0 manifest_sha256=$manifest_sha raw_log=$raw_log raw_log_sha256=$raw_log_sha output=$output"
echo "[M15.E0W5.RECOVERY] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0"
if [ "$status" != "COMPLETE" ] || [ "$three_round" -ne 1 ]; then
  exit 3
fi
