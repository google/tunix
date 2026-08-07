#!/usr/bin/env bash
# Check that every artifact EVIDENCE.md cites still exists and still hashes to what was
# recorded.  A claim whose artifact has vanished or changed is not a weaker claim -- it is an
# unverifiable one, and this exits nonzero so that shows up in CI rather than in a review.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CANON_ARTIFACT_ROOT:-/mnt/disks/tunix-data/logp_probe_1host}"
MAN="$HERE/evidence/artifacts.sha256"
PACKAGE_MAN="$HERE/evidence/package_artifacts.sha256"
[ -f "$MAN" ] || { echo "missing $MAN" >&2; exit 1; }
[ -f "$PACKAGE_MAN" ] || { echo "missing $PACKAGE_MAN" >&2; exit 1; }
ok=0; bad=0; gone=0

verify_manifest() {
  local manifest="$1" root="$2" label="$3"
  while read -r want rel; do
    [ -n "$want" ] || continue
    f="$root/$rel"
    if [ ! -f "$f" ]; then
      printf '  GONE   %s:%s\n' "$label" "$rel"
      gone=$((gone+1))
      continue
    fi
    got="$(sha256sum "$f" | cut -d' ' -f1)"
    if [ "$got" = "$want" ]; then
      printf '  OK     %s:%s\n' "$label" "$rel"
      ok=$((ok+1))
    else
      printf '  CHANGED %s:%s  (recorded %s.. now %s..)\n' "$label" "$rel" \
        "$(echo "$want" | cut -c1-12)" "$(echo "$got" | cut -c1-12)"
      bad=$((bad+1))
    fi
  done < "$manifest"
}

verify_manifest "$MAN" "$ROOT" external
verify_manifest "$PACKAGE_MAN" "$HERE" package
echo "  --- ok=$ok changed=$bad gone=$gone (external_root=$ROOT)"
[ "$bad" = 0 ] && [ "$gone" = 0 ] || exit 1
echo "  EVIDENCE VERIFIED"
