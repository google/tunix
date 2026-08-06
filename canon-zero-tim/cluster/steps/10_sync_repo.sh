#!/usr/bin/env bash
# Verify -- not perform -- the repo sync.
#
# The manifest fetches the branch before this package exists on disk, so the sync itself
# cannot live here.  What CAN live here is the check that the sync landed where it was
# supposed to: a stale checkout is otherwise indistinguishable from a fresh one, and every
# number below would belong to code nobody chose.
set -euo pipefail
source "$CANON_STATE/env.sh"

cd "$CANON_PKG/.."
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "[sync] not a git checkout -- cannot verify provenance" >&2
  [ "${CANON_ALLOW_UNVERSIONED:-0}" = "1" ] || {
    echo "[sync] REFUSING (set CANON_ALLOW_UNVERSIONED=1 to override, and say so in the report)" >&2
    exit 1; }
  echo "[sync] CANON_ALLOW_UNVERSIONED=1 -- continuing without provenance"
  exit 0
fi
HEAD_SHA="$(git rev-parse HEAD)"
echo "[sync] HEAD=$HEAD_SHA"
echo "[sync] describe=$(git log -1 --format='%h %s' 2>/dev/null)"
echo "[sync] dirty_files=$(git status --porcelain 2>/dev/null | wc -l)"
if [ -n "${CANON_EXPECT_COMMIT:-}" ] && [ "$CANON_EXPECT_COMMIT" != "$HEAD_SHA" ]; then
  echo "[sync] REFUSING: expected commit $CANON_EXPECT_COMMIT, got $HEAD_SHA" >&2
  exit 1
fi
echo "[sync] provenance ok"
