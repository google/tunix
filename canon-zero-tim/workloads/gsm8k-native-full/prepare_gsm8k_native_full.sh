#!/usr/bin/env bash
# Render exactly one GSM8K Native/mismatch DP16xTP4 full control. Never launch.
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <approved-40-sha> <fresh-output-dir> <fresh-run-id>" >&2
  exit 2
fi

SOURCE_SHA="$1"
OUTPUT_DIR="$2"
RUN_ID="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RENDERER="$SCRIPT_DIR/render_gsm8k_native_full.py"

if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source SHA must be exactly 40 lowercase hexadecimal characters" >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" || -z "$RUN_ID" ]]; then
  echo "output directory and run ID must be non-empty" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to reuse output directory: $OUTPUT_DIR" >&2
  exit 2
fi

git -C "$REPO_ROOT" cat-file -e "${SOURCE_SHA}^{commit}"
HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ "$HEAD_SHA" != "$SOURCE_SHA" ]]; then
  echo "checked-out HEAD does not match approved source SHA: head=$HEAD_SHA approved=$SOURCE_SHA" >&2
  exit 1
fi
DIRTY="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
if [[ -n "$DIRTY" ]]; then
  echo "refusing to render from a dirty worktree" >&2
  exit 1
fi

python3 "$RENDERER" \
  --source-commit "$SOURCE_SHA" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID"

INDEX="$OUTPUT_DIR/manifest-index.json"
MANIFEST="$OUTPUT_DIR/jobset-v1-gsm8k-native-mismatch-full.yaml"
if [[ ! -s "$INDEX" || ! -s "$MANIFEST" ]]; then
  echo "renderer did not produce the Native manifest and index" >&2
  exit 1
fi

sha256sum "$INDEX" "$MANIFEST"
printf '%s\n' \
  "V1_GSM8K_NATIVE_FULL_READY manifests=1 source=$SOURCE_SHA output=$OUTPUT_DIR treatment=native-mismatch launch=not-executed" \
  "Review manifest-index.json and the resolved environment, then obtain launch approval." \
  "kubectl apply -f $MANIFEST"
