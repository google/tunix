#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEAL="$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/seal_p38_evidence.sh"

make_fixture() {
  local directory="$1" run_id="$2" archive_dir
  mkdir -p "$directory"
  for name in \
    source_commit.txt render-intent256.txt render.txt \
    rendered-intent256.yaml rendered-stock.yaml intent-diff.json \
    dry-run-stock.txt apply.txt \
    jobset-name.txt head-pod-name.txt log-follow-rc.txt jobset.final.yaml \
    head-pod.final.yaml head-pod.describe.txt head.full.log \
    pre-alignment.jsonl serving-classification.json \
    "${run_id}-mismatch-capsule.npz"; do
    printf 'fixture:%s\n' "$name" > "$directory/$name"
  done
  for name in head.follow.log pathways-proxy.log pathways-rm.log \
              head.previous.log head-pod.events.txt; do
    : > "$directory/$name"
  done
  archive_dir="$(mktemp -d)"
  printf '{}\n' > "$archive_dir/p38_request_journal.jsonl"
  printf '{}\n' > "$archive_dir/p38_incident_ledger.jsonl"
  tar -C "$archive_dir" -cf "$directory/${run_id}-serving-capture.tar" .
  rm -r "$archive_dir"
}

tmp="$(mktemp -d)"
trap 'rm -r "$tmp"' EXIT
make_fixture "$tmp/pass" p38s12b
bash "$SEAL" "$tmp/pass" p38s12b > "$tmp/pass.log"
grep -q 'BUNDLE_COMPLETE' "$tmp/pass.log"
! grep -q 'SHA256SUMS$' "$tmp/pass/SHA256SUMS"
(cd "$tmp/pass" && sha256sum -c SHA256SUMS --quiet)

make_fixture "$tmp/missing" p38s12b
rm "$tmp/missing/jobset.final.yaml"
if bash "$SEAL" "$tmp/missing" p38s12b > "$tmp/missing.log" 2>&1; then
  echo "[P38.EVIDENCE] seal accepted an incomplete bundle" >&2
  exit 1
fi
grep -q 'required nonempty artifact missing: jobset.final.yaml' "$tmp/missing.log"

make_fixture "$tmp/missing-intent" p38s12b
rm "$tmp/missing-intent/intent-diff.json"
if bash "$SEAL" "$tmp/missing-intent" p38s12b \
    > "$tmp/missing-intent.log" 2>&1; then
  echo "[P38.EVIDENCE] seal accepted a bundle without intent-diff" >&2
  exit 1
fi
grep -q 'required nonempty artifact missing: intent-diff.json' \
  "$tmp/missing-intent.log"
echo "[P38.EVIDENCE] SEAL_TEST_PASS complete=accepted missing=rejected self_hash=absent"
