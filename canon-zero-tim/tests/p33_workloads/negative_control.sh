#!/usr/bin/env bash
# Prove that an unadmitted workload cannot cross the production launch gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="$(cd "$ROOT/.." && pwd)"
TMPDIR_PATH="$(mktemp -d)"
trap 'rm -r "$TMPDIR_PATH"' EXIT

# shellcheck disable=SC1091
set -a
source "$ROOT/cluster/profiles/_canonical_engine.env"
source "$ROOT/cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env"
set +a
# Simulate a launch request while leaving the measured DP reduction unadmitted.
export CANON_P32_TRAIN_ADMITTED=1
export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
export FL_SHARED_MESH=16,4

set +e
output="$(
  JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE:${PYTHONPATH:-}" \
    python3 "$ROOT/tests/p33_workloads/validate_workload.py" \
      --name gsm8k --output "$TMPDIR_PATH/should_not_exist.json" --launch \
      2>&1
)"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
  echo "$output"
  echo "NEGATIVE CONTROL FAIL: unadmitted workload launch was accepted" >&2
  exit 1
fi
grep -q "rank-local DP16 reduction gate is admitted" <<<"$output" || {
  echo "$output"
  echo "NEGATIVE CONTROL FAIL: launch failed without the admission verdict" >&2
  exit 1
}
test ! -e "$TMPDIR_PATH/should_not_exist.json"
echo "[P33.WORKLOAD] NEGATIVE_CONTROL PASS launch=refused reduction_admitted=0"
