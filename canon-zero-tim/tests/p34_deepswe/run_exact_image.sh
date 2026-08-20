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
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_env_contract.py
    JAX_PLATFORMS=cpu PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_sampler_contract.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_qwen32b_tp8.py
    PYTHONPATH=/workspace/canon-zero-tim/src/engine_shims/models/qwen32b \
      python3 canon-zero-tim/src/engine_shims/models/qwen32b/p22xf_contract.py
    PYTHONPATH=/tmp/p34-overlay python3 \
      canon-zero-tim/tests/p38_serving/probe_fixed_lm_head_overlay.py \
      --hidden-size 5120
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_trajectory.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_update.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_render_p34_jobset.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_classify_run.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_alignment_warning.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/test_scheduler_contract.py
    PYTHONPATH=/workspace python3 canon-zero-tim/tests/p34_deepswe/probe_scheduler_contract.py
    CANON_FIXED_AR=1 CANON_PALLAS_MATMUL=1 PYTHONPATH=/workspace \
      python3 canon-zero-tim/tests/p34_deepswe/probe_pallas_128.py
    PYTHONPATH=/tmp/p34-overlay python3 \
      canon-zero-tim/tests/p44_deepswe_qwen4b_parity/probe_swiglu_feature_padding.py \
      --feature 3200 --padded-feature 3328 --model qwen3-32b-tp8
    echo "P34_EXACT_IMAGE_CPU_PASS unit_cases=55 alignment_cases=3 pallas_cases=2 contract_cases=5 scheduler_cases=1 fixed_lm_head=1 overlay=qwen32b"
  '
