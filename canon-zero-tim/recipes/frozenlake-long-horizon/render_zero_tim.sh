#!/usr/bin/env bash
# Zero-TIM arm, FrozenLake long-horizon: the canonical overlay makes decode, prefill re-score and the training forward bit-identical, so no correction is needed.
# This is a thin wrapper; the scripts underneath keep their campaign codenames: p67 (v1 phase-4) = the full-recipe wave, m15 = long-horizon,
# p45 = the short-horizon manifest, which the same wrapper also produces and this recipe does not use.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 <tip-sha40> <out-dir> <run-id>" >&2
  exit 2
fi
SHA="$1"
OUT="$2"
RUN_ID="$3"
# The renderers accept run ids of 1-16 lowercase letters, digits and hyphens (start/end alphanumeric);
# this wrapper derives "$RUN_ID-short", so the id itself must be at most 10 characters.
if ! [[ "$RUN_ID" =~ ^[a-z0-9]([a-z0-9-]{0,8}[a-z0-9])?$ ]]; then
  echo "invalid run-id '$RUN_ID': use 1-10 lowercase letters, digits or hyphens, starting and ending with a letter or digit" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bash "$REPO/canon-zero-tim/workloads/full-recipes/scripts/prepare_p67_frozenlake_two_full_wave.sh" \
  "$SHA" "$OUT" "$RUN_ID-campaign" "$RUN_ID-short" "$RUN_ID" \
  --token-continuity both-exact --token-continuity-debug-mode record-full \
  --train-geometry dp8-tp8-b256
echo "MANIFEST=$(cd "$OUT/frozenlake-long-horizon" && pwd)/jobset-frozenlake-long-horizon-zero-300.yaml"
