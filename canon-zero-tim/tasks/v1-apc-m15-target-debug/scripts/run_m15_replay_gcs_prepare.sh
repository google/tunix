#!/usr/bin/env bash
# Verify one immutable APC-on Attempt-0 root and publish a replay input plan.
set -euo pipefail

source_uri="${1:?usage: run_m15_replay_gcs_prepare.sh <apc-on-attempt-0-gs-uri> [scratch-parent]}"
scratch_parent="${2:-/tmp}"
bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
case "$source_uri" in
  "$bucket_root"*/attempt-0) ;;
  *) echo "[M15.APC.REPLAY.PREPARE] REFUSING invalid attempt URI" >&2; exit 2 ;;
esac
derived_uri="$source_uri/derived/m15-replay-input-plan-v1"
test -d "$scratch_parent"

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2"; }
  gcs_sync_up() { gcloud storage rsync --recursive "$1" "$2"; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_sync_up() { gsutil -m rsync -r "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
else
  echo "[M15.APC.REPLAY.PREPARE] REFUSING gcloud or gsutil is required" >&2
  exit 2
fi
if gcs_exists "$derived_uri/files/SHA256SUMS"; then
  echo "[M15.APC.REPLAY.PREPARE] REFUSING immutable analysis already exists: $derived_uri" >&2
  exit 3
fi

scratch="$(mktemp -d -p "$scratch_parent" m15-apc-replay-prepare.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
root="$scratch/root"
capture="$scratch/capture"
audit="$scratch/audit"
analysis="$scratch/analysis"
mkdir -p "$root" "$capture"

gcs_cp "$source_uri/SHA256SUMS" "$root/SHA256SUMS"
while IFS= read -r line; do
  digest="${line%%  *}"
  name="${line#*  }"
  case "$name" in
    ''|*/*|../*|*'..'*)
      echo "[M15.APC.REPLAY.PREPARE] REFUSING unsafe root member: $name" >&2
      exit 2
      ;;
  esac
  case "$digest" in
    *[!0-9a-f]*|'')
      echo "[M15.APC.REPLAY.PREPARE] REFUSING invalid SHA for $name" >&2
      exit 2
      ;;
  esac
  [ "${#digest}" -eq 64 ] || {
    echo "[M15.APC.REPLAY.PREPARE] REFUSING invalid SHA length for $name" >&2
    exit 2
  }
  gcs_cp "$source_uri/$name" "$root/$name"
  actual="$(sha256sum "$root/$name" | awk '{print $1}')"
  [ "$actual" = "$digest" ] || {
    echo "[M15.APC.REPLAY.PREPARE] REFUSING root SHA drifted: $name" >&2
    exit 2
  }
done < "$root/SHA256SUMS"
for marker in PREFLIGHT.json COLLECTED.json COMPLETE.json; do
  gcs_cp "$source_uri/$marker" "$root/$marker"
done

while IFS= read -r member; do
  case "$member" in
    /*|../*|*/../*|*'/..')
      echo "[M15.APC.REPLAY.PREPARE] REFUSING unsafe tar member: $member" >&2
      exit 2
      ;;
  esac
done < <(tar -tf "$root/serving-capture.tar")
tar -xf "$root/serving-capture.tar" -C "$capture"

script_dir="$(cd "$(dirname "$0")" && pwd)"
python3 "$script_dir/audit_m15_replay_capture.py" \
  --root-dir "$root" \
  --capture-dir "$capture" \
  --source-gcs-uri "$source_uri" \
  --output-dir "$audit"
(cd "$audit" && sha256sum -c SHA256SUMS --quiet)
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$audit/RETURN_RECEIPT.json")"
[ "$status" = "FRESH_TARGET_RED_FROZEN" ] || {
  echo "[M15.APC.REPLAY.PREPARE] REFUSING source is not a frozen APC-on red: $status" >&2
  exit 2
}

python3 "$script_dir/analyze_m15_replay_carrier.py" \
  --producer-unit "$capture/m15_producer_unit.npz" \
  --serving-envelope "$capture/m15_replay_envelope.jsonl" \
  --first-red-contract "$capture/m15_first_red_replay/first_red_contract.json" \
  --replay-contract "$capture/m15_full_replay_carrier/replay_contract.json" \
  --m15-classification "$capture/m15_apc_target.classification.json" \
  --upstream-audit-receipt "$audit/RETURN_RECEIPT.json" \
  --source-gcs-uri "$source_uri" \
  --output-dir "$analysis"
(cd "$analysis" && sha256sum -c SHA256SUMS --quiet)

upload="$scratch/upload"
mkdir "$upload"
cp -a "$analysis/." "$upload/"
rm -- "$upload/SHA256SUMS"
gcs_sync_up "$upload" "$derived_uri/files"
gcs_cp "$analysis/SHA256SUMS" "$derived_uri/files/SHA256SUMS"
analysis_sha="$(sha256sum "$analysis/REPLAY_ANALYSIS.json" | awk '{print $1}')"
manifest_sha="$(sha256sum "$analysis/SHA256SUMS" | awk '{print $1}')"
prefix_sha="$(sha256sum "$analysis/replay-prefix-plan.jsonl" | awk '{print $1}')"
prefix_bytes="$(wc -c < "$analysis/replay-prefix-plan.jsonl" | tr -d ' ')"
summary="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("red_rows="+",".join(str(x["source_row"]) for x in d["numerical"]["red_rows"])+" replay_prefix_end_call="+str(d["carrier"]["replay_prefix_end_call"]))' "$analysis/REPLAY_ANALYSIS.json")"
echo "[M15.APC.REPLAY.PREPARE] COMPLETE status=M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED analysis_sha256=$analysis_sha manifest_sha256=$manifest_sha prefix_sha256=$prefix_sha prefix_bytes=$prefix_bytes $summary destination=$derived_uri"
