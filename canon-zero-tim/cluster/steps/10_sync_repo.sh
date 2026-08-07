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
if [ -n "${CANON_EXPECT_COMMIT:-}" ] && [ "$CANON_EXPECT_COMMIT" != "$HEAD_SHA" ]; then
  echo "[sync] REFUSING: expected commit $CANON_EXPECT_COMMIT, got $HEAD_SHA" >&2
  exit 1
fi

tracked_dirty="$(git status --porcelain --untracked-files=no 2>/dev/null)"
package_untracked="$(git ls-files --others --exclude-standard -- canon-zero-tim 2>/dev/null)"
all_untracked="$(git ls-files --others --exclude-standard 2>/dev/null)"
count_lines() {
  if [ -z "$1" ]; then
    printf '0\n'
  else
    printf '%s\n' "$1" | wc -l | tr -d '[:space:]'
  fi
}
tracked_dirty_count="$(count_lines "$tracked_dirty")"
package_untracked_count="$(count_lines "$package_untracked")"
all_untracked_count="$(count_lines "$all_untracked")"
external_untracked_count="$((all_untracked_count - package_untracked_count))"

echo "[sync] tracked_dirty=$tracked_dirty_count"
echo "[sync] package_untracked=$package_untracked_count"
echo "[sync] external_untracked=$external_untracked_count"
if [ "$tracked_dirty_count" -ne 0 ]; then
  echo "[sync] REFUSING: tracked files differ from HEAD" >&2
  printf '%s\n' "$tracked_dirty" | sed 's/^/[sync] tracked: /' >&2
  exit 1
fi
if [ "$package_untracked_count" -ne 0 ]; then
  echo "[sync] REFUSING: untracked files can shadow canon-zero-tim package code" >&2
  printf '%s\n' "$package_untracked" | sed 's/^/[sync] package-untracked: /' >&2
  exit 1
fi
echo "[sync] provenance ok"
