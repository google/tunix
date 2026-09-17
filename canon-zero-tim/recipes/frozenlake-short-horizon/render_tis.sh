#!/usr/bin/env bash
# Token-level TIS arm, FrozenLake short-horizon: the trainer is the old policy and every token is reweighted by the engine's decode logprob.
# This is a thin wrapper; the scripts underneath keep their campaign codenames: p57 = the three-arm FrozenLake study, p45 = short-horizon,
# m15 = the long-horizon manifest, which the same wrapper also produces and Figure 4 does not use.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 <tip-sha40> <out-dir> <run-id>" >&2
  exit 2
fi
SHA="$1"
OUT="$2"
RUN_ID="$3"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bash "$REPO/canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh" \
  is "$SHA" "$OUT" "$RUN_ID" "$RUN_ID-m15" "$RUN_ID-campaign"
echo "MANIFEST=$(cd "$OUT/p45" && pwd)/jobset-p57-frozenlake-is-300.yaml"
