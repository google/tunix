#!/usr/bin/env bash
# Install the exact Agent Sandbox client/Fleet source used by DeepSWE Fleet mode.
#
# Direct R2E is the default and pays no dependency or import cost. Fleet mode
# pins both Python packages to one reviewed upstream commit; a floating branch
# or a mismatched editable checkout is refused before any TPU program starts.
set -euo pipefail
source "$CANON_STATE/env.sh"

if [ "${CANON_DEEPSWE_SANDBOX_RUNTIME:-direct}" != "fleet" ]; then
  echo "[agent-sandbox] runtime=direct -- skipped"
  exit 0
fi

EXPECTED=7935857fee859bb18752ee04d8948b975e47ff20
COMMIT="${CANON_AGENT_SANDBOX_COMMIT:-}"
if [ "$COMMIT" != "$EXPECTED" ]; then
  echo "[agent-sandbox] FATAL: exact source commit required: expected=$EXPECTED actual=${COMMIT:-absent}" >&2
  exit 1
fi

DEST="$CANON_STATE/agent-sandbox-src"
if [ -e "$DEST" ]; then
  [ -d "$DEST/.git" ] || {
    echo "[agent-sandbox] FATAL: existing source path is not a git checkout: $DEST" >&2
    exit 1
  }
  echo "[agent-sandbox] reusing existing run-owned source checkout"
else
  echo "[agent-sandbox] cloning kubernetes-sigs/agent-sandbox at pinned $COMMIT"
  git clone --quiet https://github.com/kubernetes-sigs/agent-sandbox.git "$DEST"
  git -C "$DEST" checkout --quiet "$COMMIT"
fi
HEAD_NOW="$(git -C "$DEST" rev-parse HEAD)"
if [ "$HEAD_NOW" != "$COMMIT" ]; then
  echo "[agent-sandbox] FATAL: checkout mismatch: $HEAD_NOW != $COMMIT" >&2
  exit 1
fi

CLIENT="$DEST/clients/python/agentic-sandbox-client"
FLEET="$DEST/examples/agent-sandbox-rl"
for path in "$CLIENT/pyproject.toml" "$FLEET/pyproject.toml"; do
  [ -f "$path" ] || {
    echo "[agent-sandbox] FATAL: pinned source layout drifted: $path" >&2
    exit 1
  }
done

pip3 install --quiet -e "$CLIENT"
pip3 install --quiet -e "$FLEET"

AGENT_SANDBOX_EXPECTED_SOURCE="$DEST" python3 - <<'PY'
import inspect
import os
from pathlib import Path

import agent_sandbox_rl
from agent_sandbox_rl import FleetConfig, SandboxFleet
from agent_sandbox_rl.adapters.r2egym import make_fleet_repo_env
from k8s_agent_sandbox import SandboxClient

root = Path(os.environ["AGENT_SANDBOX_EXPECTED_SOURCE"]).resolve()
loaded = Path(inspect.getfile(agent_sandbox_rl)).resolve()
if root not in loaded.parents:
  raise RuntimeError(
      f"agent-sandbox-rl imported from wrong source: {loaded} not under {root}"
  )
parameters = inspect.signature(SandboxClient.create_sandbox).parameters
if "shutdown_after_seconds" not in parameters:
  raise RuntimeError("SandboxClient.create_sandbox lacks atomic shutdown TTL")
if not callable(make_fleet_repo_env):
  raise RuntimeError("R2E Fleet adapter is unavailable")
if not hasattr(SandboxFleet, "warm_images") or not hasattr(
    SandboxFleet, "unwarm_image"
):
  raise RuntimeError("SandboxFleet warm-window API drifted")
FleetConfig(max_concurrent=1, max_warmpool_size=1)
print(
    "[agent-sandbox] VERIFY imports=pass adapter=r2egym "
    "shutdown_ttl=present warm_window=present source=" + str(loaded)
)
PY

# Upstream Agent Sandbox preflight deliberately downgrades some RBAC failures
# to warnings. DeepSWE must reject them before worker wait/model initialization:
# a late Fleet create/watch/exec 403 otherwise burns the TPU allocation.
PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" python3 - <<'PY'
import os

from examples.deepswe.sandbox_fleet import verify_fleet_permissions

verify_fleet_permissions(
    namespace=os.environ.get("R2E_K8S_NAMESPACE", "default"),
    image_pull_secret=os.environ.get("IMAGE_PULL_SECRET", "dockerhub-pro") or None,
)
PY

echo "[agent-sandbox] installed at pinned $COMMIT"
