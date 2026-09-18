#!/usr/bin/env bash
# Read-only wrapper for the salvage-first Attempt-19 E0 KV3 return.
set -euo pipefail

analysis_source="${1:?usage: run_m15_attempt19_e0_kv3_return_recovery.sh <full-analysis-sha> <verified-render-dir> <new-output-dir> [scratch-parent]}"
render_dir="${2:?usage: run_m15_attempt19_e0_kv3_return_recovery.sh <full-analysis-sha> <verified-render-dir> <new-output-dir> [scratch-parent]}"
output="${3:?usage: run_m15_attempt19_e0_kv3_return_recovery.sh <full-analysis-sha> <verified-render-dir> <new-output-dir> [scratch-parent]}"
scratch_parent="${4:-/tmp}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"

if [[ ! "$analysis_source" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[M15.E0.KV3.RECOVERY] REFUSING analysis source must be one full lowercase SHA" >&2
  exit 2
fi
test -d "$render_dir"
test -d "$scratch_parent"
test ! -e "$output"

branch="$(git -C "$repo" branch --show-current)"
case "$branch" in
  local/*) ;;
  *) echo "[M15.E0.KV3.RECOVERY] REFUSING branch must be local/*" >&2; exit 2 ;;
esac
head="$(git -C "$repo" rev-parse HEAD)"
[ "$head" = "$analysis_source" ] || {
  echo "[M15.E0.KV3.RECOVERY] REFUSING HEAD does not equal the supplied analysis source" >&2
  exit 2
}
[ -z "$(git -C "$repo" status --porcelain)" ] || {
  echo "[M15.E0.KV3.RECOVERY] REFUSING worktree is dirty" >&2
  exit 2
}

python3 "$canon/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py" \
  --repo "$repo" --require-clean
(cd "$render_dir" && sha256sum -c SHA256SUMS --quiet)

raw_log="$(mktemp -p "$scratch_parent" m15-e0-kv3-return-recovery.XXXXXX.log)"
set +e
bash "$script_dir/run_m15_attempt19_e0_kv3_gcs_return.sh" \
  "$render_dir" "$output" "$scratch_parent" >"$raw_log" 2>&1
return_rc=$?
set -e
read -r raw_sha _ < <(sha256sum "$raw_log")
if [ "$return_rc" -eq 0 ]; then
  read -r manifest_sha _ < <(sha256sum "$output/SHA256SUMS")
  status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/E0_KV3_RETURN.json")"
  tail -n 2 "$raw_log"
  echo "[M15.E0.KV3.RECOVERY] COMPLETE status=$status analysis_source=$analysis_source manifest_sha256=$manifest_sha raw_log=$raw_log raw_log_sha256=$raw_sha"
  echo "[M15.E0.KV3.RECOVERY] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0"
  exit 0
fi
if [ "$return_rc" -eq 3 ] && [ -s "$output/E0_KV3_RETURN.json" ]; then
  read -r manifest_sha _ < <(sha256sum "$output/SHA256SUMS")
  status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/E0_KV3_RETURN.json")"
  echo "[M15.E0.KV3.RECOVERY] INCONCLUSIVE status=$status partial_rounds_preserved=1 output=$output manifest_sha256=$manifest_sha raw_log=$raw_log raw_log_sha256=$raw_sha" >&2
  tail -n 10 "$raw_log" >&2
  exit 3
fi
echo "[M15.E0.KV3.RECOVERY] INCONCLUSIVE official_return_exit=$return_rc raw_log=$raw_log raw_log_sha256=$raw_sha" >&2
tail -n 20 "$raw_log" >&2
exit "$return_rc"
