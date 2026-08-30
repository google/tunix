#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 <approved-40-sha> <digest-image> <output-yaml> <run-id> <worker-nodepool> <model-pvc>" >&2
  exit 2
fi

SOURCE_SHA="$1"
CLIENT_IMAGE="$2"
OUTPUT_YAML="$3"
RUN_ID="$4"
WORKER_NODEPOOL="$5"
MODEL_PVC="$6"

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$TASK_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"

if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source SHA must be exactly 40 lowercase hexadecimal characters" >&2
  exit 2
fi
if [[ ! "$CLIENT_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "client image must be digest-pinned" >&2
  exit 2
fi
if [[ -e "$OUTPUT_YAML" ]]; then
  echo "refusing to reuse output path: $OUTPUT_YAML" >&2
  exit 2
fi

git -C "$REPO_ROOT" cat-file -e "${SOURCE_SHA}^{commit}"
HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ "$HEAD_SHA" != "$SOURCE_SHA" ]]; then
  echo "checked-out HEAD does not match approved source SHA" >&2
  exit 1
fi
DIRTY="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
if [[ -n "$DIRTY" ]]; then
  echo "refusing to render from a dirty worktree" >&2
  exit 1
fi

python3 "$PKG_ROOT/cluster/render_p58_deepswe_tim.py" \
  --base "$PKG_ROOT/cluster/jobset-64chip.yaml" \
  --output "$OUTPUT_YAML" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE" \
  --run-id "$RUN_ID" \
  --stage full \
  --arm zero \
  --worker-nodepool "$WORKER_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --high-performance

sha256sum "$OUTPUT_YAML"
printf '%s\n' \
  "V1_DEEPSWE_ZERO_HP_RFULL_READY source=$SOURCE_SHA manifest=$OUTPUT_YAML transport=token-in-token-out launch=not-executed" \
  "Review the manifest and verify the published source and image by remote read-back before launch." \
  "kubectl apply -f $OUTPUT_YAML"
