#!/usr/bin/env bash
# Native arm, DeepSWE-4B on 128 v5p chips: the stock vllm-tpu engine and the ordinary training forward.
# The rollout logprobs are still used as the old policy, but nothing makes them match the trainer's, so
# the training-inference mismatch is present and uncorrected.  There is no sampler correction here: the
# renderer's --sampler-is modifier is a separate recipe (native-is) and this wrapper does not pass it.
# This is a thin wrapper; the renderer underneath keeps its campaign codename: p58 = the DeepSWE
# native-vs-Zero-TIM comparison.
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
# The JobSet is named canon-p58-ds4b-native-full-<run-id> and the renderer caps a JobSet name at 36
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
MANIFEST="$OUT/jobset-deepswe-4b-native-1000.yaml"
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
  --arm native \
  --topology 128 \
  --jobset-namespace "${DEEPSWE_JOBSET_NAMESPACE:-trellis}" \
  --cpu-nodepool "${DEEPSWE_CPU_NODEPOOL:-cpu-np}" \
  --sandbox-nodepool "${DEEPSWE_SANDBOX_NODEPOOL:-sandbox-cpu-pool}" \
  --worker-nodepool "$WORKER_NODEPOOL"
echo "MANIFEST=$MANIFEST"
