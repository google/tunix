#!/usr/bin/env bash
# Re-audit an already downloaded Attempt-20 round-0 checkpoint without GCS.
set -euo pipefail

preserved="${1:?usage: run_m15_attempt20_on_round0_preserved_scratch_audit.sh <preserved-scratch> <new-output-dir> [scratch-parent]}"
output="${2:?usage: run_m15_attempt20_on_round0_preserved_scratch_audit.sh <preserved-scratch> <new-output-dir> [scratch-parent]}"
scratch_parent="${3:-$(dirname "$preserved")}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
target_source="97e813de84f6c8b3e2ba911fc96ff8397b199603"

test -d "$preserved"
test -d "$preserved/remote"
test -s "$preserved/remote/CLASSIFIER_INPUT_ARCHIVE.tar"
test -s "$preserved/remote/CLASSIFIER_INPUT_RECEIPT.json"
test -s "$preserved/remote/CLASSIFIER_INPUT_SHA256SUMS"
test -s "$preserved/classifier.log"
test -d "$scratch_parent"
test ! -e "$output"

branch="$(git -C "$repo" branch --show-current)"
case "$branch" in
  local/*) ;;
  *) echo "[M15.E0V.SCRATCH] REFUSING branch must be local/*" >&2; exit 2 ;;
esac
analysis_source="$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" merge-base --is-ancestor "$target_source" "$analysis_source"
python3 "$canon/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py" \
  --repo "$repo" --require-clean

work="$(mktemp -d -p "$scratch_parent" m15-attempt20-r0-audit.XXXXXX)"
raw_log="$(mktemp -p "$scratch_parent" m15-attempt20-r0-audit.XXXXXX.log)"
audit_ok=0
preserve_on_failure() {
  if [ "$audit_ok" -ne 1 ]; then
    echo "[M15.E0V.SCRATCH] INCONCLUSIVE classification=NONE work_preserved=$work raw_log=$raw_log" >&2
  fi
}
trap preserve_on_failure EXIT

python3 "$script_dir/review_m15_attempt20_on_round0.py" \
  --archive "$preserved/remote/CLASSIFIER_INPUT_ARCHIVE.tar" \
  --manifest "$preserved/remote/CLASSIFIER_INPUT_SHA256SUMS" \
  --receipt "$preserved/remote/CLASSIFIER_INPUT_RECEIPT.json" \
  --expected-source "$target_source" \
  --analysis-source "$analysis_source" \
  --scratch "$work" \
  --output "$output" >"$raw_log" 2>&1

python3 - "$output" "$preserved/classifier.log" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
original_log = Path(sys.argv[2])
report_path = root / "ATTEMPT20_ON_R0_RECOVERY.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("classification_available") is not False:
  raise SystemExit("preserved failure unexpectedly produced a classification")
if report.get("status") not in (
    "TOKEN_HISTORY_JOIN_MISMATCH", "INVALID_OR_CLASSIFIER_FAILED"
):
  raise SystemExit("preserved failure audit returned an unknown status")
original_copy = root / "original_classifier_error.log"
shutil.copyfile(original_log, original_copy)
report["original_classifier_log_sha256"] = hashlib.sha256(
    original_copy.read_bytes()
).hexdigest()
report["preserved_scratch_reaudit"] = True
report_path.write_text(
    json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
names = sorted(
    path.name for path in root.iterdir()
    if path.is_file() and path.name != "SHA256SUMS"
)
(root / "SHA256SUMS").write_text(
    "".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names
    ),
    encoding="ascii",
)
PY

(cd "$output" && sha256sum -c SHA256SUMS --quiet)
read -r result_status < <(python3 - "$output/ATTEMPT20_ON_R0_RECOVERY.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
read -r manifest_sha _ < <(sha256sum "$output/SHA256SUMS")
read -r raw_log_sha _ < <(sha256sum "$raw_log")
audit_ok=1
rm -rf -- "$work"
trap - EXIT
echo "[M15.E0V.SCRATCH] AUDIT_COMPLETE status=$result_status classification=NONE round_input=1 three_round_verdict=0 target_pass=0 numerical_repair_authorized=0 manifest_sha256=$manifest_sha raw_log=$raw_log raw_log_sha256=$raw_log_sha"
echo "[M15.E0V.SCRATCH] LOCAL_ONLY gcs_read=0 gcs_write=0 kubernetes=0 tpu=0"
