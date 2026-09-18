#!/usr/bin/env bash
# Zero-TIM arm, GSM8K DP16xTP4: the canonical overlay makes decode, prefill re-score and the
# training forward bit-identical, so no sampler correction is applied and alignment is strict.
# This is a thin wrapper; the script underneath keeps its campaign codenames: v1-hp = the
# high-performance Zero-TIM numerical profile, p74 = the full-recipe wave that introduced this
# manifest, which is still the name of the file it writes.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 <tip-sha40> <out-dir> <run-id>" >&2
  exit 2
fi
SHA="$1"
OUT="$2"
RUN_ID="$3"
# canon-zero-tim/cluster/render_jobsets.py accepts run ids of 1-16 lowercase letters, digits and
# hyphens (start and end alphanumeric); this wrapper passes the id through unchanged.
if ! [[ "$RUN_ID" =~ ^[a-z0-9]([a-z0-9-]{0,14}[a-z0-9])?$ ]]; then
  echo "invalid run-id '$RUN_ID': use 1-16 lowercase letters, digits or hyphens, starting and ending with a letter or digit" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bash "$REPO/canon-zero-tim/workloads/full-recipes/scripts/prepare_gsm8k_full_dp16tp4.sh" \
  "$SHA" "$OUT" "$RUN_ID"
echo "MANIFEST=$(cd "$OUT" && pwd)/jobset-v1-hp-gsm8k-dp16tp4-p74.yaml"
