#!/usr/bin/env bash
# Recover and admit the official Attempt-18 E0 compact return. GCS reads only.
set -euo pipefail

analysis_source="${1:?usage: run_m15_attempt18_e0_return_recovery.sh <full-analysis-sha> <verified-render-dir> <new-output-dir> [scratch-parent]}"
render_dir="${2:?usage: run_m15_attempt18_e0_return_recovery.sh <full-analysis-sha> <verified-render-dir> <new-output-dir> [scratch-parent]}"
output="${3:?usage: run_m15_attempt18_e0_return_recovery.sh <full-analysis-sha> <verified-render-dir> <new-output-dir> [scratch-parent]}"
scratch_parent="${4:-/tmp}"
runtime_source="12207e3281db13461350fe7ef68dbaadfe713a58"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"

if [[ ! "$analysis_source" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[M15.E0.RECOVERY] REFUSING analysis source must be one full lowercase SHA" >&2
  exit 2
fi
test -d "$render_dir"
test -d "$scratch_parent"
test ! -e "$output"

branch="$(git -C "$repo" branch --show-current)"
case "$branch" in
  local/*) ;;
  *) echo "[M15.E0.RECOVERY] REFUSING branch must be local/*" >&2; exit 2 ;;
esac
head="$(git -C "$repo" rev-parse HEAD)"
[ "$head" = "$analysis_source" ] || {
  echo "[M15.E0.RECOVERY] REFUSING HEAD does not equal the supplied analysis source" >&2
  exit 2
}
[ -z "$(git -C "$repo" status --porcelain)" ] || {
  echo "[M15.E0.RECOVERY] REFUSING worktree is dirty" >&2
  exit 2
}

python3 "$canon/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py" \
  --repo "$repo" --require-clean
(cd "$render_dir" && sha256sum -c SHA256SUMS --quiet)
python3 - "$render_dir" "$runtime_source" <<'PY'
from pathlib import Path
import json
import re
import sys

root = Path(sys.argv[1])
expected_source = sys.argv[2]
contract = json.loads((root / "RUN_CONTRACT.json").read_text(encoding="utf-8"))
if not (
    contract.get("schema") == "m15-attempt18-e0-kv-render-v1"
    and contract.get("source_commit") == expected_source
    and re.fullmatch(r"[a-z0-9]([a-z0-9-]{0,14}[a-z0-9])?",
                     str(contract.get("run_id", "")))
    and contract.get("rounds") == 1
    and contract.get("observer", {}).get("layer") == 0
    and contract.get("observer", {}).get("target_prefix_tokens") == 1226
    and contract.get("observer", {}).get("target_aliases") == 8
    and contract.get("b_full_reset_immutable") is True
    and contract.get("control_and_treatment_differ_only_at_apc") is True
    and contract.get("launch_authorized") is False
    and contract.get("target_executed") is False
    and contract.get("remote_mutation") is False
):
  raise SystemExit("preserved render contract does not match Attempt-18 E0")
arms = contract.get("arms")
if not isinstance(arms, list) or {row.get("arm") for row in arms} != {"off", "on"}:
  raise SystemExit("preserved render contract does not contain the matched pair")
PY
python3 "$script_dir/test_review_m15_attempt18_e0_return.py"

raw_log="$(mktemp -p "$scratch_parent" m15-e0-return-recovery.XXXXXX.log)"
set +e
bash "$script_dir/run_m15_attempt18_e0_kv_gcs_return.sh" \
  "$render_dir" "$output" "$scratch_parent" >"$raw_log" 2>&1
return_rc=$?
set -e
if [ "$return_rc" -ne 0 ]; then
  read -r raw_sha _ < <(sha256sum "$raw_log")
  echo "[M15.E0.RECOVERY] INCONCLUSIVE official_return_exit=$return_rc raw_log=$raw_log raw_log_sha256=$raw_sha" >&2
  tail -n 20 "$raw_log" >&2
  exit "$return_rc"
fi

set +e
python3 "$script_dir/review_m15_attempt18_e0_return.py" \
  --return-dir "$output" --expected-source "$runtime_source" \
  --raw-log "$raw_log" >>"$raw_log" 2>&1
review_rc=$?
set -e
if [ "$review_rc" -ne 0 ]; then
  read -r raw_sha _ < <(sha256sum "$raw_log")
  echo "[M15.E0.RECOVERY] INCONCLUSIVE intake_exit=$review_rc output_preserved=$output raw_log=$raw_log raw_log_sha256=$raw_sha" >&2
  tail -n 20 "$raw_log" >&2
  exit "$review_rc"
fi

tail -n 1 "$raw_log"
read -r raw_sha _ < <(sha256sum "$raw_log")
read -r manifest_sha _ < <(sha256sum "$output/SHA256SUMS")
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/E0_KV_RETURN.json")"
echo "[M15.E0.RECOVERY] COMPLETE status=$status runtime_source=$runtime_source manifest_sha256=$manifest_sha raw_log=$raw_log raw_log_sha256=$raw_sha"
echo "[M15.E0.RECOVERY] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0"
