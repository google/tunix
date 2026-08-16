#!/usr/bin/env bash
# capture_golden.sh — golden regression net for the cluster renderers (phase 2a).
#
# Modes:
#   capture_golden.sh capture   render each manifest entry TWICE with pinned inputs,
#                               require byte-identical self-determinism, then record
#                               sha256 of every output file into goldens/<name>.sha256
#   capture_golden.sh check     re-render once per entry and compare against goldens
#                               (this is the gate any future renderer change must pass)
#
# Pinned inputs are constants below: goldens assert "same inputs -> same bytes", which
# is exactly the property renderer consolidation (phase 2b) must preserve. Active-thread
# renderers (p38-serving, p45, p46) are deliberately absent until their campaigns pause.
set -euo pipefail
MODE=${1:-check}
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$HERE/../.." && pwd)                       # canon-zero-tim/
CL="$PKG/cluster"
GOLD="$HERE/goldens"
SHA_PIN="a94d6c0cd0e08b9bed418331974b8694eb49507e"   # any fixed 40-hex; a real branch commit
RUN_PIN="golden0"
IMG_PIN="golden.example/canon-image@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
# Known-pending entries (kept in the manifest so the gap stays visible):
#   chip128 — its intended base jobset yaml is not in cluster/ (both in-tree bases lack the
#             CANON_EXPECT_COMMIT env entry / initContainers layout it patches);
#   p44     — requires the reviewed 1851-image clean whitelist artifact (operator-owned).
BASE64C="$CL/jobset-64chip.yaml"
WL="$HERE/fixtures/whitelist_golden.txt"
WLSHA="4d7d720d3a39281cbddbf0177ea60caf610041df9ba9a17440605c91b48aacfd"

# name|command with {OUT} placeholder (workdir = repo root so relative defaults resolve)
MANIFEST=(
  "p33|python3 $CL/render_p33_jobsets.py --source-commit $SHA_PIN --run-id $RUN_PIN --output-dir {OUT}"
  "p35|python3 $CL/render_p35_jobset.py --source-commit $SHA_PIN --run-id $RUN_PIN --output {OUT}/p35.yaml"
  "p36|python3 $CL/render_p36_proxy_xla_jobset.py --source-commit $SHA_PIN --run-id $RUN_PIN --output {OUT}/p36.yaml"
  "p38aval|python3 $CL/render_p38_aval_jobset.py --source-commit $SHA_PIN --run-id $RUN_PIN --output {OUT}/p38aval.yaml"
  "chip128|CANON_EXPECT_COMMIT=$SHA_PIN python3 $CL/render_128chip_jobset.py --source-commit $SHA_PIN --run-id $RUN_PIN --output-dir {OUT}"
  "p34|python3 $CL/render_p34_jobset.py --base $BASE64C --output {OUT}/p34.yaml --source-commit $SHA_PIN --client-image $IMG_PIN --run-id $RUN_PIN --stage one-update --whitelist $WL --whitelist-sha256 $WLSHA"
  "p39|python3 $CL/render_p39_deepswe_pilot.py --base $BASE64C --output {OUT}/p39.yaml --source-commit $SHA_PIN --client-image $IMG_PIN --run-id $RUN_PIN --stage one-update --cpu-nodepool golden-pool --worker-nodepool golden-workers --model-pvc golden-pvc --whitelist $WL --whitelist-sha256 $WLSHA"
  "p43|python3 $CL/render_p43_deepswe_debug.py --base $BASE64C --output {OUT}/p43.yaml --source-commit $SHA_PIN --client-image $IMG_PIN --run-id $RUN_PIN --stage one-update --cpu-nodepool golden-pool --worker-nodepool golden-workers --model-pvc golden-pvc --whitelist $WL --whitelist-sha256 $WLSHA"
  "p44|python3 $CL/render_p44_deepswe_parity.py --base $BASE64C --output {OUT}/p44.yaml --source-commit $SHA_PIN --client-image $IMG_PIN --run-id $RUN_PIN --stage one-update --topology 64 --cpu-nodepool golden-pool --worker-nodepool golden-workers --model-pvc golden-pvc --whitelist $WL --whitelist-sha256 $WLSHA"
)

render() { # $1 cmd-template $2 outdir -> 0 ok
  local out="$2"; mkdir -p "$out"
  ( cd "$PKG/.." && eval "${1//\{OUT\}/$out}" ) >/dev/null 2>"$out/.stderr"
}
digest() { ( cd "$1" && find . -type f ! -name '.stderr' -print0 | sort -z | xargs -0 sha256sum ); }

mkdir -p "$GOLD"
pass=0; fail=0; nondet=0; mismatch=0
for entry in "${MANIFEST[@]}"; do
  name=${entry%%|*}; cmd=${entry#*|}
  T=$(mktemp -d)
  if ! render "$cmd" "$T/a"; then
    echo "GOLDEN $name PENDING render_error: $(tail -1 "$T/a/.stderr" 2>/dev/null | cut -c1-120)"
    fail=$((fail+1)); rm -rf "$T"; continue
  fi
  if [ "$MODE" = capture ]; then
    render "$cmd" "$T/b" || { echo "GOLDEN $name PENDING second_render_error"; fail=$((fail+1)); rm -rf "$T"; continue; }
    if [ "$(digest "$T/a")" != "$(digest "$T/b")" ]; then
      echo "GOLDEN $name DETERMINISM_FAIL"; nondet=$((nondet+1)); rm -rf "$T"; continue
    fi
    digest "$T/a" > "$GOLD/$name.sha256"
    echo "GOLDEN $name CAPTURED $(wc -l < "$GOLD/$name.sha256") files"
    pass=$((pass+1))
  else
    [ -f "$GOLD/$name.sha256" ] || { echo "GOLDEN $name NO_GOLDEN"; fail=$((fail+1)); rm -rf "$T"; continue; }
    if [ "$(digest "$T/a")" = "$(cat "$GOLD/$name.sha256")" ]; then
      echo "GOLDEN $name MATCH"; pass=$((pass+1))
    else
      echo "GOLDEN $name MISMATCH (renderer output changed vs golden)"; mismatch=$((mismatch+1))
    fi
  fi
  rm -rf "$T"
done
echo "GOLDEN_SUMMARY mode=$MODE ok=$pass pending=$fail nondet=$nondet mismatch=$mismatch"
[ $((nondet+mismatch)) -eq 0 ]
