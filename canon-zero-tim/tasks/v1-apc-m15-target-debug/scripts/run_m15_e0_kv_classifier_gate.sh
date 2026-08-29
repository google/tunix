#!/usr/bin/env bash
# Run the E0 KV classifier host gate without permitting an implicit image pull.
set -euo pipefail

receipt="${1:?usage: run_m15_e0_kv_classifier_gate.sh <new-receipt-path> [auto|host|docker]}"
mode="${2:-auto}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
classifier="$canon/tests/p38_serving/test_kv_observer_classifier.py"
expected_image_id="sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a"

test ! -e "$receipt" || {
  echo "[M15.E0.KV] REFUSING classifier runtime receipt already exists" >&2
  exit 2
}
case "$mode" in
  auto|host|docker) ;;
  *)
    echo "[M15.E0.KV] REFUSING classifier runtime mode must be auto, host, or docker" >&2
    exit 2
    ;;
esac

route="$mode"
if [ "$route" = auto ]; then
  if python3 -c "import numpy" >/dev/null 2>&1; then
    route=host
  else
    route=docker
  fi
fi

image_id=""
pull_policy="not-applicable"
network_mode="not-applicable"
if [ "$route" = host ]; then
  python3 -c "import numpy" >/dev/null 2>&1 || {
    echo "[M15.E0.KV] REFUSING host classifier runtime lacks numpy" >&2
    exit 2
  }
  python3 "$classifier"
else
  read -r -a docker_cmd <<< "${DOCKER:-docker}"
  if [ "${#docker_cmd[@]}" -eq 0 ]; then
    echo "[M15.E0.KV] REFUSING DOCKER command is empty" >&2
    exit 2
  fi
  if ! image_id="$("${docker_cmd[@]}" image inspect "$expected_image_id" --format '{{.Id}}' 2>/dev/null)"; then
    echo "[M15.E0.KV] REFUSING pinned classifier image is not already local" >&2
    exit 2
  fi
  if [ "$image_id" != "$expected_image_id" ]; then
    echo "[M15.E0.KV] REFUSING pinned classifier image identity mismatch" >&2
    exit 2
  fi
  pull_policy=never
  network_mode=none
  "${docker_cmd[@]}" run --rm --pull=never --network=none \
    -v "$repo:/workspace:ro" \
    -w /workspace \
    -e PYTHONPATH=/workspace \
    "$image_id" \
    python3 "/workspace/canon-zero-tim/tests/p38_serving/test_kv_observer_classifier.py"
fi

python3 - "$receipt" "$route" "$image_id" "$pull_policy" "$network_mode" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
route = sys.argv[2]
image_id = sys.argv[3] or None
value = {
    "schema": "m15-e0-kv-classifier-runtime-v1",
    "status": "PASS",
    "route": route,
    "image_id": image_id,
    "pull_policy": sys.argv[4],
    "network_mode": sys.argv[5],
    "external_access": False,
}
path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

receipt_sha="$(sha256sum "$receipt" | awk '{print $1}')"
echo "[M15.E0.KV] KV_CLASSIFIER_RUNTIME_PASS route=$route image_id=${image_id:-none} pull=$pull_policy network=$network_mode external_access=0 receipt_sha256=$receipt_sha"
