#!/usr/bin/env bash
# Install the pinned R2E-Gym checkout that interactive DeepSWE stages require.
#
# The reference MLPerf launch cloned the floating upstream HEAD and patched it at
# runtime inside the pod.  This step keeps that supply chain but pins every input:
# the commit is CANON_R2EGYM_COMMIT, the behavioural patch is vendored in this
# package, and any drift fails closed instead of silently changing the agent
# environment.  Profiles that do not set CANON_R2EGYM_INSTALL=1 skip this step
# entirely, so GSM8K/FrozenLake launches are untouched.
set -euo pipefail
source "$CANON_STATE/env.sh"

if [ "${CANON_R2EGYM_INSTALL:-0}" != "1" ]; then
  echo "[r2egym] CANON_R2EGYM_INSTALL!=1 -- skipped (workload does not use R2E-Gym)"
  exit 0
fi

COMMIT="${CANON_R2EGYM_COMMIT:-}"
if ! printf '%s' "$COMMIT" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "[r2egym] FATAL: CANON_R2EGYM_COMMIT must be a 40-hex pinned commit, got '$COMMIT'" >&2
  exit 1
fi
PATCH="$CANON_PKG/patches/r2egym/r2egym.patch"
if [ ! -f "$PATCH" ]; then
  echo "[r2egym] FATAL: vendored patch missing: $PATCH" >&2
  exit 1
fi

DEST="$CANON_STATE/r2egym-src"
rm -rf "$DEST"
echo "[r2egym] cloning R2E-Gym at pinned $COMMIT"
git clone --quiet https://github.com/R2E-Gym/R2E-Gym.git "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
HEAD_NOW="$(git -C "$DEST" rev-parse HEAD)"
if [ "$HEAD_NOW" != "$COMMIT" ]; then
  echo "[r2egym] FATAL: checkout mismatch: $HEAD_NOW != $COMMIT" >&2
  exit 1
fi

echo "[r2egym] applying vendored patch sha256=$(sha256sum "$PATCH" | cut -d' ' -f1)"
git -C "$DEST" apply "$PATCH"

# The reference launch's source-level edits, applied to the pinned checkout.  HfFolder
# and the pod deadline are also handled in-process by r2egym_runtime_patch.py; these
# edits keep import time safe on huggingface_hub builds that removed HfFolder.
sed -i 's/, HfFolder//g' "$DEST/src/r2egym/agenthub/utils/utils.py"
sed -i 's/"restartPolicy": "Never",/"restartPolicy": "Never", "activeDeadlineSeconds": 10800,/g' "$DEST/src/r2egym/agenthub/runtime/docker.py"
sed -i 's/datasets==2.19/datasets/g' "$DEST/pyproject.toml"
sed -i 's/anthropic\[vertex\]==0.43.0/anthropic[vertex]/g' "$DEST/pyproject.toml"
if grep -q ", HfFolder" "$DEST/src/r2egym/agenthub/utils/utils.py"; then
  echo "[r2egym] FATAL: HfFolder import survived the source edit" >&2
  exit 1
fi
if grep -q "datasets==2.19" "$DEST/pyproject.toml"; then
  echo "[r2egym] FATAL: pyproject datasets pin survived the source edit" >&2
  exit 1
fi

echo "[r2egym] pip install -e checkout plus kubernetes for pod control"
pip3 install --quiet -e "$DEST"
pip3 install --quiet kubernetes
python3 - <<'PY'
import kubernetes
import r2egym
from r2egym.agenthub.action import Action

print("[r2egym] VERIFY import ok:", r2egym.__file__)
PY
echo "[r2egym] versions: $(pip3 show r2egym kubernetes 2>/dev/null | grep -E '^(Name|Version)' | tr '\n' ' ')"
echo "[r2egym] installed at pinned $COMMIT"
