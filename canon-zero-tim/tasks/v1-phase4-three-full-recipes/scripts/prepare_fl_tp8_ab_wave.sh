#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <approved-40-sha> <fresh-output-dir> <fresh-p66-off-run-id> <fresh-serving-scope-run-id>" >&2
  exit 2
fi

SOURCE_SHA="$1"
OUTPUT_DIR="$2"
P66_OFF_RUN_ID="$3"
SERVING_SCOPE_RUN_ID="$4"
ROOT="$(git rev-parse --show-toplevel)"
RENDERER="$ROOT/canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/render_fl_tp8_ab_diagnostic.py"

[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "source SHA must be 40 lowercase hex" >&2
  exit 2
}
[ "$(git rev-parse HEAD)" = "$SOURCE_SHA" ] || {
  echo "refusing: worktree HEAD is not the approved source SHA" >&2
  exit 2
}
[ -z "$(git status --porcelain)" ] || {
  echo "refusing: render worktree is dirty" >&2
  exit 2
}
[ ! -e "$OUTPUT_DIR" ] || {
  echo "refusing: output directory already exists" >&2
  exit 2
}
[ "$P66_OFF_RUN_ID" != "$SERVING_SCOPE_RUN_ID" ] || {
  echo "refusing: run IDs must be distinct" >&2
  exit 2
}

python3 "$RENDERER" \
  --source-commit "$SOURCE_SHA" \
  --run-id "$P66_OFF_RUN_ID" \
  --output-dir "$OUTPUT_DIR/p66-off" \
  --workload p45 \
  --arm p66-off
python3 "$RENDERER" \
  --source-commit "$SOURCE_SHA" \
  --run-id "$SERVING_SCOPE_RUN_ID" \
  --output-dir "$OUTPUT_DIR/serving-scope" \
  --workload p45 \
  --arm serving-scope

sha256sum \
  "$OUTPUT_DIR/p66-off/jobset-v1-fl-tp8-ab-p45-p66-off.yaml" \
  "$OUTPUT_DIR/serving-scope/jobset-v1-fl-tp8-ab-p45-serving-scope.yaml"

echo "V1_FL_TP8_AB_WAVE_READY source=$SOURCE_SHA output=$OUTPUT_DIR jobs=2 backward=0 optimizer_commits=0"
echo "kubectl apply -f $OUTPUT_DIR/p66-off/jobset-v1-fl-tp8-ab-p45-p66-off.yaml"
echo "kubectl apply -f $OUTPUT_DIR/serving-scope/jobset-v1-fl-tp8-ab-p45-serving-scope.yaml"
