#!/usr/bin/env bash
# Seal a complete P38 operator bundle without hashing SHA256SUMS into itself.
set -euo pipefail

evidence="${1:?usage: seal_p38_evidence.sh <evidence-dir> <run-id>}"
run_id="${2:?usage: seal_p38_evidence.sh <evidence-dir> <run-id>}"
case "$run_id" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P38.EVIDENCE] REFUSING: invalid run id: $run_id" >&2
    exit 2
    ;;
esac
test -d "$evidence"

required_nonempty=(
  source_commit.txt render-intent256.txt render.txt
  rendered-intent256.yaml rendered-stock.yaml intent-diff.json
  dry-run-stock.txt apply.txt
  jobset-name.txt head-pod-name.txt log-follow-rc.txt jobset.final.yaml
  head-pod.final.yaml head-pod.describe.txt head.full.log
  pre-alignment.jsonl serving-classification.json
  "${run_id}-mismatch-capsule.npz" "${run_id}-serving-capture.tar"
)
required_present=(
  head.follow.log pathways-proxy.log pathways-rm.log head.previous.log
  head-pod.events.txt
)
for name in "${required_nonempty[@]}"; do
  if [ ! -s "$evidence/$name" ]; then
    echo "[P38.EVIDENCE] REFUSING: required nonempty artifact missing: $name" >&2
    exit 1
  fi
done
for name in "${required_present[@]}"; do
  if [ ! -e "$evidence/$name" ]; then
    echo "[P38.EVIDENCE] REFUSING: required artifact missing: $name" >&2
    exit 1
  fi
done
archive_listing="$(mktemp)"
trap 'rm -f "$archive_listing"' EXIT
tar -tf "$evidence/${run_id}-serving-capture.tar" > "$archive_listing"
grep -qx './p38_request_journal.jsonl' "$archive_listing"
grep -qx './p38_incident_ledger.jsonl' "$archive_listing"

manifest="$evidence/SHA256SUMS"
if [ -e "$manifest" ]; then
  echo "[P38.EVIDENCE] REFUSING: manifest already exists: $manifest" >&2
  exit 1
fi
(
  cd "$evidence"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 sha256sum > SHA256SUMS
  if grep -q 'SHA256SUMS$' SHA256SUMS; then
    echo "[P38.EVIDENCE] REFUSING: checksum manifest included itself" >&2
    exit 1
  fi
  sha256sum -c SHA256SUMS --quiet
)
echo "[P38.EVIDENCE] BUNDLE_COMPLETE run_id=$run_id manifest=$manifest"
