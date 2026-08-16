#!/usr/bin/env bash
# Negative-control suite for scripts/package_run.sh.
# Encodes three real incidents: duplicate pod logs (P38s12e), self-including
# SHA256SUMS (P38s13a), and missing-piece admission (must be INCONCLUSIVE, not
# silently green and not a refusal).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PKG="$HERE/../../scripts/package_run.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $1" >&2; exit 1; }

mk_src() { # complete five-piece source with one duplicate
  local d="$1"; mkdir -p "$d"
  printf 'line1\nline2\n%s\n' "$(head -c 2048 /dev/zero | tr '\0' 'x')" > "$d/run.log"
  echo '{"step":0}' > "$d/pre_alignment.jsonl"
  printf 'NPZBYTES' > "$d/mismatch_capsule.npz"
  tar -cf "$d/serving_capture.tar" -C "$d" pre_alignment.jsonl
  echo '{"verdict":"PASS"}' > "$d/serving-classification.json"
  cp "$d/run.log" "$d/run_copy.log"          # exact duplicate -> must be dropped
  echo selfsha > "$d/SHA256SUMS"             # stale source sums -> must be skipped
}

# T1 positive + dedup + self-exclusion + byte identity through compression
mk_src "$TMP/src1"
"$PKG" "$TMP/src1" "$TMP/out1" >/dev/null
grep -q 'dedup: dropped' "$TMP/out1/PACKAGING.txt" || fail "duplicate not dropped"
grep -q 'SHA256SUMS' "$TMP/out1/SHA256SUMS" && fail "SHA256SUMS includes itself"
( cd "$TMP/out1" && sha256sum -c SHA256SUMS >/dev/null ) || fail "sums do not verify"
if command -v zstd >/dev/null 2>&1; then zstd -dq "$TMP/out1/run.log.zst" -o "$TMP/rt.log"; else gunzip -c "$TMP/out1/run.log.gz" > "$TMP/rt.log"; fi
cmp -s "$TMP/src1/run.log" "$TMP/rt.log" || fail "compression is not byte-identical on round trip"
grep -q 'completeness: all core pieces present' "$TMP/out1/PACKAGING.txt" || fail "complete set not recognized"
test ! -f "$TMP/out1/verdict.json" || fail "verdict.json must not be synthesized when classification exists"

# T2 missing capsule -> packaged anyway, synthesized INCONCLUSIVE verdict
mk_src "$TMP/src2"; rm "$TMP/src2/mismatch_capsule.npz"
"$PKG" "$TMP/src2" "$TMP/out2" >/dev/null
grep -q 'INCONCLUSIVE' "$TMP/out2/verdict.json" || fail "missing piece did not yield INCONCLUSIVE"
grep -q 'missing=capsule' "$TMP/out2/PACKAGING.txt" || fail "missing piece not named"

# T3 immutability: refuse to write into a non-empty destination
"$PKG" "$TMP/src1" "$TMP/out1" >/dev/null 2>&1 && fail "non-empty dest must be refused"

echo "PACKAGE_RUN_TESTS PASS (3/3)"
