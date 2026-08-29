#!/usr/bin/env bash
# Read-only recovery and offline request binding for Attempt 17 (d36).
set -euo pipefail

output="${1:?usage: run_m15_attempt17_d36_offline_binding.sh <output-dir> [scratch-parent]}"
scratch_parent="${2:-/tmp}"
test -d "$scratch_parent"
test ! -e "$output"

script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
source_commit="16c224aa80eb6b3a544be19f693c0542ab4b0dcb"
analysis_commit="$(git -C "$repo" rev-parse HEAD)"
evidence="$canon/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt17_d36_operator_return_20260829"
expected_classification="$evidence/on.round-000000.classification.json"

case "$(git -C "$repo" branch --show-current)" in
  local/*) ;;
  *) echo "[M15.D36.OFFLINE] REFUSING: use a clean local/* worktree" >&2; exit 2 ;;
esac
if [ -n "$(git -C "$repo" status --porcelain)" ]; then
  echo "[M15.D36.OFFLINE] REFUSING: analysis worktree is dirty" >&2
  exit 2
fi
python3 "$canon/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py" \
  --repo "$repo" --require-clean
test "$(git -C "$repo" rev-parse "$source_commit")" = "$source_commit"
(cd "$evidence" && sha256sum -c SHA256SUMS --quiet)

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2" >/dev/null; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
else
  echo "[M15.D36.OFFLINE] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" m15-d36-offline.XXXXXX)"
cleanup() {
  rc=$?
  if [ "$rc" -eq 0 ]; then
    rm -rf -- "$scratch"
  else
    echo "[M15.D36.OFFLINE] FAILED scratch_preserved=$scratch" >&2
  fi
}
trap cleanup EXIT

bash "$script_dir/prepare_m15_multiround_pair.sh" \
  "$source_commit" d36 "$scratch/render" full 0

python3 - "$scratch/render" "$evidence/JOBSET_STATUS.json" \
  "$source_commit" <<'PY'
import json
import pathlib
import sys
import yaml

render = pathlib.Path(sys.argv[1])
expected = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
source = sys.argv[3]
actual = {}
for path in sorted(render.glob("*.yaml")):
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  container = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"][0]
  env = {row["name"]: str(row["value"]) for row in container["env"] if "value" in row}
  arm = env.get("CANON_APC_M15_TARGET_DEBUG")
  if arm not in ("off", "on") or arm in actual:
    raise SystemExit("d36 reconstructed render has invalid arm membership")
  if (
      document["metadata"]["name"] != expected[arm]["jobset"]
      or env.get("CANON_EXPECT_COMMIT") != source
      or env.get("CANON_P38_DIAGNOSTIC_ROUNDS") != "3"
      or env.get("CANON_P38_SEAM_OBSERVER") != "full"
      or env.get("CANON_P38_SEAM_LAYER") != "0"
  ):
    raise SystemExit("d36 reconstructed render differs from committed receipts")
  actual[arm] = env["CANON_P38_GCS_PREFIX"]
if set(actual) != {"off", "on"}:
  raise SystemExit("d36 reconstructed render is incomplete")
print("M15_D36_RENDER_IDENTITY_PASS source=16c224aa rounds=3 observer=full seam_layer=0")
PY

bash "$script_dir/run_m15_multiround_gcs_return.sh" \
  "$scratch/render" "$scratch/core" "$scratch_parent"

on_root="$(python3 - "$scratch/render" <<'PY'
import pathlib
import sys
import yaml

for path in pathlib.Path(sys.argv[1]).glob("*.yaml"):
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  container = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"][0]
  env = {row["name"]: str(row["value"]) for row in container["env"] if "value" in row}
  if env.get("CANON_APC_M15_TARGET_DEBUG") == "on":
    print(env["CANON_P38_GCS_PREFIX"])
    break
else:
  raise SystemExit("d36 treatment root is absent from reconstructed render")
PY
)"

remote_round="$on_root/wide/rounds/000000"
gcs_cp "$remote_round/WIDE_SHA256SUMS" "$scratch/WIDE_SHA256SUMS"
gcs_cp "$remote_round/m15_wide_seam_bundle.tar" \
  "$scratch/m15_wide_seam_bundle.tar"

python3 - "$scratch/WIDE_SHA256SUMS" \
  "$scratch/m15_wide_seam_bundle.tar" \
  "$scratch/core/MULTIROUND_SUMMARY.json" "$source_commit" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

manifest_path, bundle_path, summary_path = map(pathlib.Path, sys.argv[1:4])
source = sys.argv[4]
matches = []
for line in manifest_path.read_text(encoding="ascii").splitlines():
  digest, separator, name = line.partition("  ")
  if separator == "  " and name == "m15_wide_seam_bundle.tar" and re.fullmatch(r"[0-9a-f]{64}", digest):
    matches.append(digest)
if len(matches) != 1:
  raise SystemExit("d36 round manifest has no unique compact bundle identity")
actual = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
if actual != matches[0]:
  raise SystemExit("d36 compact bundle differs from WIDE_SHA256SUMS")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
round0 = summary.get("arms", {}).get("on", {}).get("rounds", [{}])[0]
if (
    summary.get("source_commit") != source
    or round0.get("status") != "SEALED"
    or round0.get("classification") != "M15_INTERNAL_FIRST_RED_CANDIDATE_SET"
    or round0.get("bundle_sha256") != actual
):
  raise SystemExit("d36 refreshed remote summary differs from the sealed treatment receipt")
print("M15_D36_BUNDLE_IDENTITY_PASS treatment_round=0 sealed=1")
PY

python3 "$script_dir/review_m15_attempt17_d36_candidate.py" \
  --bundle "$scratch/m15_wide_seam_bundle.tar" \
  --expected-classification "$expected_classification" \
  --source-commit "$source_commit" \
  --analysis-commit "$analysis_commit" \
  --core-summary "$scratch/core/MULTIROUND_SUMMARY.json" \
  --scratch-parent "$scratch_parent" \
  --output "$output"

(cd "$output" && sha256sum -c SHA256SUMS --quiet)
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/D36_OFFLINE_REVIEW.json")"
manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
echo "[M15.D36.OFFLINE] COMPLETE status=$status manifest_sha256=$manifest_sha output=$output"
echo "[M15.D36.OFFLINE] TARGET_NOT_RUN gcs_read=1 gcs_write=0 kubernetes=0 tpu=0"
