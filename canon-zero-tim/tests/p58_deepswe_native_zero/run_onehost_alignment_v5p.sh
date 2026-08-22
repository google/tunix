#!/usr/bin/env bash
# Direct-attached v5p gate for the P58 full-stage alignment admission repair.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${DEEPSWE_TRAIN_PYTHON:-/mnt/disks/tunix-data/venvs/train/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "missing P58 one-host interpreter: $PYTHON" >&2
  exit 2
fi

SOURCE_SHA="$(git rev-parse HEAD)"
SOURCE_DIRTY="$(git status --porcelain --untracked-files=no)"
if [[ -n "$SOURCE_DIRTY" && "${P58_ONEHOST_ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "P58 one-host evidence requires a clean tracked worktree" >&2
  echo "set P58_ONEHOST_ALLOW_DIRTY=1 only for development evidence" >&2
  exit 2
fi

unset JAX_BACKEND_TARGET PATHWAYS_HEAD
export JAX_PLATFORMS=tpu,cpu
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DEVICE_TIMEOUT_SECS="${P58_ONEHOST_DEVICE_TIMEOUT_SECS:-60}"
if [[ ! "$DEVICE_TIMEOUT_SECS" =~ ^[1-9][0-9]*$ ]]; then
  echo "P58 one-host device timeout must be a positive integer" >&2
  exit 2
fi

set +e
timeout "${DEVICE_TIMEOUT_SECS}s" "$PYTHON" -c '
import jax
import jax.numpy as jnp

devices = jax.devices("tpu")
assert len(devices) == 4, devices
assert all(device.platform == "tpu" for device in devices), devices
kinds = tuple(str(device.device_kind) for device in devices)
assert all("v5p" in kind.lower() for kind in kinds), kinds
value = jnp.matmul(
    jnp.arange(16, dtype=jnp.float32).reshape(4, 4),
    jnp.eye(4, dtype=jnp.float32),
).block_until_ready()
assert value.shape == (4, 4), value.shape
print(
    "[P58.ONEHOST] DEVICE_PASS count=4 kinds=" + ",".join(kinds),
    flush=True,
)
'
DEVICE_RC=$?
set -e
if [[ "$DEVICE_RC" -eq 124 ]]; then
  echo "P58_ONEHOST_ALIGNMENT_BLOCKED reason=device_inventory_timeout" \
    "timeout_secs=$DEVICE_TIMEOUT_SECS" >&2
  exit 3
fi
if [[ "$DEVICE_RC" -ne 0 ]]; then
  echo "P58 one-host TPU inventory failed: rc=$DEVICE_RC" >&2
  exit "$DEVICE_RC"
fi

"$PYTHON" canon-zero-tim/tests/p58_deepswe_native_zero/test_alignment_policy.py
"$PYTHON" \
  canon-zero-tim/tests/p58_deepswe_native_zero/test_environment_contract.py

echo "P58_ONEHOST_ALIGNMENT_ADMISSION_PASS source=$SOURCE_SHA devices=4" \
  "scope=renderer-profile-policy"
