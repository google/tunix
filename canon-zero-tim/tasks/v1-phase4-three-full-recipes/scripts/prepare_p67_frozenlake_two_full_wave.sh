#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 9 ]]; then
  echo "usage: $0 <approved-40-sha> <output-dir> <campaign-root> <p45-run-id> <m15-run-id> [--token-continuity legacy|p45-exact|m15-exact|both-exact] [--token-continuity-debug|--token-continuity-debug-mode first-diff|record-full]" >&2
  exit 2
fi

SOURCE_SHA="$1"
OUTPUT_DIR="$2"
CAMPAIGN_ROOT="$3"
P45_RUN_ID="$4"
M15_RUN_ID="$5"
shift 5
TOKEN_CONTINUITY_ARGS=()
TOKEN_CONTINUITY_MODE=legacy
TOKEN_CONTINUITY_SEEN=0
TOKEN_CONTINUITY_DEBUG=off
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --m15-tito-exact)
      if [[ "$TOKEN_CONTINUITY_SEEN" = 1 ]]; then
        echo "token-continuity selector may be supplied only once" >&2
        exit 2
      fi
      TOKEN_CONTINUITY_ARGS+=(--m15-tito-exact)
      TOKEN_CONTINUITY_MODE=m15-exact
      TOKEN_CONTINUITY_SEEN=1
      shift
      ;;
    --token-continuity)
      if [[ "$TOKEN_CONTINUITY_SEEN" = 1 || "$#" -lt 2 ]]; then
        echo "--token-continuity requires one non-duplicate mode" >&2
        exit 2
      fi
      case "$2" in
      legacy|p45-exact|m15-exact|both-exact) ;;
      *)
        echo "token continuity must be legacy, p45-exact, m15-exact, or both-exact" >&2
        exit 2
        ;;
      esac
      TOKEN_CONTINUITY_ARGS+=(--token-continuity "$2")
      TOKEN_CONTINUITY_MODE="$2"
      TOKEN_CONTINUITY_SEEN=1
      shift 2
      ;;
    --token-continuity-debug)
      if [[ "$TOKEN_CONTINUITY_DEBUG" = on ]]; then
        echo "--token-continuity-debug may be supplied only once" >&2
        exit 2
      fi
      TOKEN_CONTINUITY_ARGS+=(--token-continuity-debug)
      TOKEN_CONTINUITY_DEBUG=on
      shift
      ;;
    --token-continuity-debug-mode)
      if [[ "$TOKEN_CONTINUITY_DEBUG" != off || "$#" -lt 2 ]]; then
        echo "--token-continuity-debug-mode requires one non-duplicate value" >&2
        exit 2
      fi
      case "$2" in
        first-diff|record-full) ;;
        *) echo "debug mode must be first-diff or record-full" >&2; exit 2 ;;
      esac
      TOKEN_CONTINUITY_ARGS+=(--token-continuity-debug-mode "$2")
      TOKEN_CONTINUITY_DEBUG="$2"
      shift 2
      ;;
    *)
      echo "unknown optional argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ "$TOKEN_CONTINUITY_DEBUG" != off && \
      "$TOKEN_CONTINUITY_MODE" = legacy ]]; then
  echo "token-continuity diagnostics require an exact treatment" >&2
  exit 2
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
  "${TOKEN_CONTINUITY_ARGS[@]}"

INDEX="$OUTPUT_DIR/manifest-index.json"
if [[ ! -s "$INDEX" ]]; then
  echo "renderer did not produce a non-empty manifest index" >&2
  exit 1
fi

sha256sum "$INDEX"
printf '%s\n' \
  "V1_P67_FROZENLAKE_WAVE_READY manifests=2 source=$SOURCE_SHA output=$OUTPUT_DIR token_continuity=$TOKEN_CONTINUITY_MODE token_continuity_debug=$TOKEN_CONTINUITY_DEBUG launch=not-executed" \
  "Review manifest-index.json and verify the pushed SHA by remote read-back before launch." \
  "kubectl apply -f $OUTPUT_DIR/frozenlake-p45/jobset-p57-frozenlake-zero-300.yaml" \
  "kubectl apply -f $OUTPUT_DIR/frozenlake-m15/jobset-p57-frozenlake-zero-m15-main-300.yaml"
