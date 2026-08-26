#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 <approved-40-sha> <output-dir> <campaign-root> <gsm8k-run-id> <p45-run-id> <m15-run-id>" >&2
  exit 2
fi

SOURCE_SHA="$1"
OUTPUT_DIR="$2"
CAMPAIGN_ROOT="$3"
GSM8K_RUN_ID="$4"
P45_RUN_ID="$5"
M15_RUN_ID="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RENDERER="$SCRIPT_DIR/render_three_full_recipes.py"

if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source SHA must be exactly 40 lowercase hexadecimal characters" >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" || -z "$CAMPAIGN_ROOT" || -z "$GSM8K_RUN_ID" || -z "$P45_RUN_ID" || -z "$M15_RUN_ID" ]]; then
  echo "output directory, campaign root, and run IDs must be non-empty" >&2
  exit 2
fi
if [[ "$GSM8K_RUN_ID" == "$P45_RUN_ID" || "$GSM8K_RUN_ID" == "$M15_RUN_ID" || "$P45_RUN_ID" == "$M15_RUN_ID" ]]; then
  echo "all three run IDs must be distinct" >&2
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
  --campaign-root "$CAMPAIGN_ROOT" \
  --gsm8k-run-id "$GSM8K_RUN_ID" \
  --p45-run-id "$P45_RUN_ID" \
  --m15-run-id "$M15_RUN_ID"

INDEX="$OUTPUT_DIR/manifest-index.json"
if [[ ! -s "$INDEX" ]]; then
  echo "renderer did not produce a non-empty manifest index" >&2
  exit 1
fi

sha256sum "$INDEX"
printf '%s\n' \
  "V1_HP_CHECKED_VMA_WAVE_READY manifests=3 source=$SOURCE_SHA output=$OUTPUT_DIR launch=not-executed" \
  "Review manifest-index.json, verify the pushed SHA by read-back, and obtain launch approval." \
  "kubectl apply -f $OUTPUT_DIR/gsm8k/jobset-v1-hp-gsm8k-full.yaml" \
  "kubectl apply -f $OUTPUT_DIR/frozenlake-p45/jobset-p57-frozenlake-zero-300.yaml" \
  "kubectl apply -f $OUTPUT_DIR/frozenlake-m15/jobset-p57-frozenlake-zero-m15-main-300.yaml"
