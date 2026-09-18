#!/usr/bin/env bash
# Native control arm, GSM8K DP16xTP4: the same trainer and the same training command on the
# image's stock vllm-tpu engine, with every canonical selector removed and alignment off.
# This is a thin wrapper; the script underneath keeps its campaign codenames: v1ctl = the
# control arm of the v1 wave, mismatch = rollout and trainer logprobs are allowed to differ.
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
bash "$REPO/canon-zero-tim/workloads/gsm8k-native-full/prepare_gsm8k_native_full.sh" \
  "$SHA" "$OUT" "$RUN_ID"
echo "MANIFEST=$(cd "$OUT" && pwd)/jobset-v1-gsm8k-native-mismatch-full.yaml"
