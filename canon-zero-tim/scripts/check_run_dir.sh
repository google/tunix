#!/usr/bin/env bash
# check_run_dir.sh — structural postflight for one packaged run directory.
# Usage: check_run_dir.sh <run_dir>
# Pass criteria (exit 0, prints RUN_DIR_STRUCTURE_OK):
#   - SHA256SUMS exists, verifies, and does not list itself (P38s13a incident);
#   - PACKAGING.txt exists (provenance of dedup/compress decisions);
#   - a verdict is present: verdict.json or *classif*.json;
#   - no zero-byte files (a truncated artifact is worse than a missing one).
# Any failure prints RUN_DIR_STRUCTURE_FAIL reason=... and exits nonzero.
set -euo pipefail
D=${1:?usage: check_run_dir.sh <run_dir>}
fail() { echo "RUN_DIR_STRUCTURE_FAIL dir=$D reason=$1"; exit 1; }
[ -d "$D" ] || fail "not_a_directory"
[ -s "$D/SHA256SUMS" ] || fail "missing_SHA256SUMS"
grep -q 'SHA256SUMS' "$D/SHA256SUMS" && fail "sums_list_themselves"
( cd "$D" && sha256sum -c SHA256SUMS >/dev/null 2>&1 ) || fail "sums_do_not_verify"
[ -s "$D/PACKAGING.txt" ] || fail "missing_PACKAGING"
ls "$D"/verdict.json >/dev/null 2>&1 || ls "$D"/*classif*.json >/dev/null 2>&1 || fail "missing_verdict"
Z=$(find "$D" -maxdepth 1 -type f -size 0 | head -1)
[ -z "$Z" ] || fail "zero_byte_file:$(basename "$Z")"
echo "RUN_DIR_STRUCTURE_OK dir=$D"
