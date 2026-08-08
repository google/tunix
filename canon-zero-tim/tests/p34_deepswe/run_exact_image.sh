#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
IMAGE="${1:-tunix_frozenlake_image:vllm-tpu0.25.0}"
DOCKER="${DOCKER:-sudo docker}"

$DOCKER image inspect "$IMAGE" --format 'P34_EXACT_IMAGE image_id={{.Id}}' >/dev/null
$DOCKER run --rm \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  "$IMAGE" \
  bash -euo pipefail -c '
    rm -rf /tmp/p34-overlay
    bash canon-zero-tim/install.sh /tmp/p34-overlay \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen32b
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_contract.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_script_contract.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_qwen32b_tp8.py
    PYTHONPATH=/workspace/canon-zero-tim/src/engine_shims/models/qwen32b \
      python3 canon-zero-tim/src/engine_shims/models/qwen32b/p22xf_contract.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_trajectory.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_update.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_render_p34_jobset.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_classify_run.py
    CANON_FIXED_AR=1 CANON_PALLAS_MATMUL=1 PYTHONPATH=/workspace \
      python3 canon-zero-tim/tests/p34_deepswe/probe_pallas_128.py
    echo "P34_EXACT_IMAGE_CPU_PASS unit_cases=32 pallas_cases=1 contract_cases=5 overlay=qwen32b"
  '
