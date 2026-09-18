#!/usr/bin/env bash
# Token-level TIS arm, FrozenLake long-horizon: the trainer is the old policy and every token is reweighted by the engine's decode logprob.
# This is a thin wrapper; the scripts underneath keep their campaign codenames: p57 = the three-arm FrozenLake study, m15 = long-horizon,
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
bash "$REPO/canon-zero-tim/workloads/frozenlake-three-arm/scripts/render_three_arm_wave.sh" \
  is "$SHA" "$OUT" "$RUN_ID-short" "$RUN_ID" "$RUN_ID-campaign"
echo "MANIFEST=$(cd "$OUT/long-horizon" && pwd)/jobset-frozenlake-long-horizon-is-300.yaml"
