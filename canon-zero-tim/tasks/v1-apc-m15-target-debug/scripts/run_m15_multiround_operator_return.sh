#!/usr/bin/env bash
# Build one self-hashed small return containing numerical and operator receipts.
set -euo pipefail

render_dir="${1:?usage: run_m15_multiround_operator_return.sh <render-dir> <output-dir> [scratch-parent] [namespace]}"
output="${2:?usage: run_m15_multiround_operator_return.sh <render-dir> <output-dir> [scratch-parent] [namespace]}"
scratch_parent="${3:-/tmp}"
namespace="${4:-default}"
test -d "$render_dir"
test -d "$scratch_parent"
test ! -e "$output"
command -v kubectl >/dev/null 2>&1 || {
  echo "[M15.OPERATOR.RETURN] REFUSING: kubectl is required" >&2
  exit 2
}

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2" >/dev/null; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
  gcs_size() {
    gcloud storage objects describe "$1" --format='value(size)' 2>/dev/null
  }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
  gcs_size() {
    gsutil stat "$1" 2>/dev/null | awk -F': ' '/Content-Length:/{print $2}'
  }
else
  echo "[M15.OPERATOR.RETURN] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" m15-operator-return.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/jobsets" "$scratch/raw-logs"

script_dir="$(cd "$(dirname "$0")" && pwd)"
bash "$script_dir/run_m15_multiround_gcs_return.sh" \
  "$render_dir" "$scratch/core" "$scratch_parent"

mapfile -t arm_rows < <(python3 - "$render_dir" <<'PY'
import pathlib
import re
import sys
import yaml

root = pathlib.Path(sys.argv[1])
paths = sorted(root.glob("jobset-v1-apc-m15-*-*.yaml"))
if len(paths) != 2:
  raise SystemExit("render directory must contain exactly two M15 YAMLs")
for path in paths:
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  container = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"][0]
  env = {row["name"]: str(row["value"]) for row in container["env"] if "value" in row}
  arm = env.get("CANON_APC_M15_TARGET_DEBUG", "")
  source = env.get("CANON_EXPECT_COMMIT", "")
  jobset = document["metadata"]["name"]
  gcs_root = env.get("CANON_P38_GCS_PREFIX", "")
  if arm not in ("off", "on") or not re.fullmatch(r"[0-9a-f]{40}", source):
    raise SystemExit("rendered arm/source contract drifted")
  if not re.fullmatch(r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/[a-z0-9-]+/attempt-0", gcs_root):
    raise SystemExit("rendered GCS root drifted")
  print(f"{arm}\t{source}\t{jobset}\t{gcs_root}")
PY
)
[ "${#arm_rows[@]}" -eq 2 ] || {
  echo "[M15.OPERATOR.RETURN] REFUSING: rendered pair is incomplete" >&2
  exit 2
}

for row in "${arm_rows[@]}"; do
  IFS=$'\t' read -r arm source jobset gcs_root <<< "$row"
  raw_jobset="$scratch/$arm.jobset.raw.json"
  set +e
  kubectl get jobset --namespace "$namespace" "$jobset" -o json \
    > "$raw_jobset" 2>/dev/null
  query_rc=$?
  set -e
  python3 - "$raw_jobset" "$scratch/jobsets/$arm.json" \
    "$arm" "$source" "$jobset" "$query_rc" <<'PY'
import json
import pathlib
import sys

raw, output, arm, source, jobset, query_rc = sys.argv[1:]
query_rc = int(query_rc)
record = {
    "schema": "m15-apc-jobset-status-v1",
    "arm": arm,
    "source_commit": source,
    "jobset": jobset,
    "query_status": "QUERY_FAILED",
    "query_exit_code": query_rc,
    "terminal_condition": None,
    "conditions": [],
}
if query_rc == 0:
  value = json.loads(pathlib.Path(raw).read_text(encoding="utf-8"))
  metadata = value.get("metadata", {})
  labels = metadata.get("labels", {})
  if (metadata.get("name") != jobset
      or labels.get("canon.zero-tim/apc-m15-arm") != arm
      or labels.get("canon.zero-tim/source") != source[:8]):
    raise SystemExit("JobSet identity drifted")
  conditions = []
  terminal = None
  for item in value.get("status", {}).get("conditions", []):
    condition = {
        key: item.get(key)
        for key in ("type", "status", "reason", "message", "lastTransitionTime")
        if item.get(key) is not None
    }
    conditions.append(condition)
    if item.get("status") == "True" and item.get("type") in ("Completed", "Failed"):
      terminal = item["type"]
  record.update({
      "query_status": "PASS",
      "terminal_condition": terminal,
      "conditions": conditions,
      "uid": metadata.get("uid"),
      "generation": metadata.get("generation"),
  })
pathlib.Path(output).write_text(
    json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
PY

  root_manifest="$scratch/$arm.root-SHA256SUMS"
  log_status="ABSENT"
  log_sha=""
  log_bytes=""
  if gcs_exists "$gcs_root/SHA256SUMS"; then
    gcs_cp "$gcs_root/SHA256SUMS" "$root_manifest"
    log_sha="$(python3 - "$root_manifest" <<'PY'
import pathlib
import re
import sys

matches = []
for line in pathlib.Path(sys.argv[1]).read_text(encoding="ascii").splitlines():
  digest, separator, name = line.partition("  ")
  if separator == "  " and name == "run.log" and re.fullmatch(r"[0-9a-f]{64}", digest):
    matches.append(digest)
if len(matches) == 1:
  print(matches[0])
PY
)"
    log_bytes="$(gcs_size "$gcs_root/run.log" || true)"
    if [[ "$log_sha" =~ ^[0-9a-f]{64}$ ]] && \
       [[ "$log_bytes" =~ ^[1-9][0-9]*$ ]]; then
      log_status="PRESENT"
    else
      log_status="INCOMPLETE"
    fi
  fi
  python3 - "$scratch/raw-logs/$arm.json" "$arm" "$source" "$jobset" \
    "$log_status" "$log_sha" "$log_bytes" <<'PY'
import json
import pathlib
import sys

output, arm, source, jobset, status, digest, size = sys.argv[1:]
record = {
    "schema": "m15-apc-raw-log-receipt-v1",
    "arm": arm,
    "source_commit": source,
    "jobset": jobset,
    "status": status,
    "object_identity": f"{jobset}/attempt-0/run.log",
    "sha256": digest or None,
    "bytes": int(size) if size.isdigit() else None,
    "payload_returned": False,
}
pathlib.Path(output).write_text(
    json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
PY
done

python3 "$script_dir/package_m15_multiround_operator_return.py" \
  --render-dir "$render_dir" \
  --core-return "$scratch/core" \
  --jobset-receipts "$scratch/jobsets" \
  --raw-log-receipts "$scratch/raw-logs" \
  --output "$output"
(cd "$output" && sha256sum -c SHA256SUMS --quiet)
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/OPERATOR_RETURN_SUMMARY.json")"
summary_sha="$(sha256sum "$output/OPERATOR_RETURN_SUMMARY.json" | awk '{print $1}')"
manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
echo "[M15.OPERATOR.RETURN] COMPLETE status=$status summary_sha256=$summary_sha manifest_sha256=$manifest_sha output=$output"
