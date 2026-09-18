#!/usr/bin/env bash
# Zero-TIM arm, FrozenLake short-horizon: the canonical overlay makes decode, prefill re-score and the training forward bit-identical, so no correction is needed.
# This is a thin wrapper; the scripts underneath keep their campaign codenames: p67 (v1 phase-4) = the full-recipe wave, p45 = short-horizon,
# m15 = the long-horizon manifest, which the same wrapper also produces and Figure 4 does not use.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 <tip-sha40> <out-dir> <run-id>" >&2
  exit 2
fi
SHA="$1"
OUT="$2"
RUN_ID="$3"
# The renderers accept run ids of 1-16 lowercase letters, digits and hyphens (start/end alphanumeric);
# this wrapper derives "$RUN_ID-m15", so the id itself must be at most 12 characters.
if ! [[ "$RUN_ID" =~ ^[a-z0-9]([a-z0-9-]{0,10}[a-z0-9])?$ ]]; then
  echo "invalid run-id '$RUN_ID': use 1-12 lowercase letters, digits or hyphens, starting and ending with a letter or digit" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bash "$REPO/canon-zero-tim/workloads/full-recipes/scripts/prepare_p67_frozenlake_two_full_wave.sh" \
  "$SHA" "$OUT" "$RUN_ID-campaign" "$RUN_ID" "$RUN_ID-m15" \
  --token-continuity both-exact --token-continuity-debug-mode record-full \
  --train-geometry dp8-tp8-b256
echo "MANIFEST=$(cd "$OUT/frozenlake-short-horizon" && pwd)/jobset-frozenlake-short-horizon-zero-300.yaml"
