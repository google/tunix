#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE_REF="${1:-tunix_frozenlake_image:vllm-tpu0.25.0}"
DOCKER="${DOCKER:-sudo docker}"
IMAGE_ID="$($DOCKER image inspect "$IMAGE_REF" --format '{{.Id}}')"
if [[ ! "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P45 one-host gate could not resolve immutable image ID: $IMAGE_ID" >&2
  exit 2
fi

infra='instance_agent|tpu-runtime|vbarcontrolagent|healthagent|google-runtime-monitor|google-collectd|monitoringagent'
if [ "$($DOCKER ps --format '{{.Names}}' | grep -vcE "$infra")" -ne 0 ]; then
  echo "P45 one-host checkpoint gate refuses a busy TPU" >&2
  $DOCKER ps --format '{{.Names}} {{.Status}}' | grep -vE "$infra" >&2
  exit 3
fi

container="p45_checkpoint_v5p_$$"
echo "P45_ONEHOST_CHECKPOINT_BEGIN image_id=$IMAGE_ID"
$DOCKER run --rm --privileged --net=host --name "$container" \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=tpu \
  -e PYTHONPATH=/workspace \
  -e XLA_FLAGS=--xla_allow_excess_precision=false \
  "$IMAGE_ID" \
  python3 canon-zero-tim/tests/p45_frozenlake_dp8_tp8/probe_checkpoint_v5p.py
