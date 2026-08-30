#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 6 ]]; then
  echo "usage: $0 <approved-40-sha> <output-dir> <campaign-root> <p45-run-id> <m15-run-id> [--m15-tito-exact]" >&2
  exit 2
fi

SOURCE_SHA="$1"
OUTPUT_DIR="$2"
CAMPAIGN_ROOT="$3"
P45_RUN_ID="$4"
M15_RUN_ID="$5"
M15_TITO_ARGS=()
M15_TITO_MODE=off
if [[ "$#" -eq 6 ]]; then
  if [[ "$6" != "--m15-tito-exact" ]]; then
    echo "optional sixth argument must be --m15-tito-exact" >&2
    exit 2
  fi
  M15_TITO_ARGS=(--m15-tito-exact)
  M15_TITO_MODE=exact
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RENDERER="$SCRIPT_DIR/render_p67_frozenlake_two_full_recipes.py"

if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source SHA must be exactly 40 lowercase hexadecimal characters" >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" || -z "$CAMPAIGN_ROOT" || -z "$P45_RUN_ID" || -z "$M15_RUN_ID" ]]; then
  echo "output directory, campaign root, and run IDs must be non-empty" >&2
  exit 2
fi
if [[ "$P45_RUN_ID" == "$M15_RUN_ID" ]]; then
  echo "P45 and M15 run IDs must be distinct" >&2
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
  --p45-run-id "$P45_RUN_ID" \
  --m15-run-id "$M15_RUN_ID" \
  "${M15_TITO_ARGS[@]}"

INDEX="$OUTPUT_DIR/manifest-index.json"
if [[ ! -s "$INDEX" ]]; then
  echo "renderer did not produce a non-empty manifest index" >&2
  exit 1
fi

sha256sum "$INDEX"
printf '%s\n' \
  "V1_P67_FROZENLAKE_WAVE_READY manifests=2 source=$SOURCE_SHA output=$OUTPUT_DIR m15_tito=$M15_TITO_MODE launch=not-executed" \
  "Review manifest-index.json and verify the pushed SHA by remote read-back before launch." \
  "kubectl apply -f $OUTPUT_DIR/frozenlake-p45/jobset-p57-frozenlake-zero-300.yaml" \
  "kubectl apply -f $OUTPUT_DIR/frozenlake-m15/jobset-p57-frozenlake-zero-m15-main-300.yaml"
