#!/usr/bin/env bash
# Zero-TIM arm, DeepSWE-4B on 128 v5p chips: the canonical overlay makes decode, prefill re-score and the
# training forward bit-identical, so the rollout logprobs are the old policy and no correction is needed.
# This is a thin wrapper; the renderer underneath keeps its campaign codename: p58 = the DeepSWE
# native-vs-Zero-TIM comparison.  It renders the strict lane -- system-optimization arm "treatment",
# which sets CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=0 -- and not the warn-only --high-performance lane.
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "usage: $0 <tip-sha40> <out-dir> <run-id> <client-image@sha256:...> <worker-nodepool>" >&2
  exit 2
fi
SHA="$1"
OUT="$2"
RUN_ID="$3"
CLIENT_IMAGE="$4"
WORKER_NODEPOOL="$5"
if ! [[ "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid tip-sha40 '$SHA': the renderer wants the full 40-character commit the pods will run" >&2
  exit 2
fi
# The renderer refuses a client image that is not pinned by digest.
if ! [[ "$CLIENT_IMAGE" =~ ^[^[:space:]]+@sha256:[0-9a-f]{64}$ ]]; then
  echo "invalid client-image '$CLIENT_IMAGE': pin it by digest, <repo>@sha256:<64 hex>" >&2
  exit 2
fi
# The JobSet is named canon-p58-128s-zsoptt-full-<run-id> and the renderer caps a JobSet name at 36
# characters, so the 27-character prefix leaves 9 for the run id; the renderer itself would accept 16
# and then fail on the name, which is a much later and less obvious error.
if ! [[ "$RUN_ID" =~ ^[a-z0-9]([a-z0-9-]{0,7}[a-z0-9])?$ ]]; then
  echo "invalid run-id '$RUN_ID': use 1-9 lowercase letters, digits or hyphens, starting and ending with a letter or digit" >&2
  exit 2
fi
if [ -z "$WORKER_NODEPOOL" ]; then
  echo "invalid worker-nodepool: give the TPU node pool that holds the 32 v5p hosts" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
MANIFEST="$OUT/jobset-deepswe-4b-zero-1000.yaml"
# The renderer raises FileExistsError rather than overwrite; say so here instead.
if [ -e "$MANIFEST" ]; then
  echo "refusing to overwrite an existing manifest: $MANIFEST" >&2
  exit 2
fi
python3 "$REPO/canon-zero-tim/cluster/render_deepswe_comparison.py" \
  --base "$REPO/canon-zero-tim/cluster/jobset-64chip.yaml" \
  --output "$MANIFEST" \
  --source-commit "$SHA" \
  --client-image "$CLIENT_IMAGE" \
  --run-id "$RUN_ID" \
  --stage full \
  --arm zero \
  --topology 128 \
  --system-optimization-arm treatment \
  --jobset-namespace "${DEEPSWE_JOBSET_NAMESPACE:-trellis}" \
  --cpu-nodepool "${DEEPSWE_CPU_NODEPOOL:-cpu-np}" \
  --sandbox-nodepool "${DEEPSWE_SANDBOX_NODEPOOL:-sandbox-cpu-pool}" \
  --worker-nodepool "$WORKER_NODEPOOL"
echo "MANIFEST=$MANIFEST"
