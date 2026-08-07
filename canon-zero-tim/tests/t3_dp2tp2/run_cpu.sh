#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=4"

python canon-zero-tim/tests/t3_dp2tp2/probe_contract.py
P32_SHIMS="$ROOT/canon-zero-tim/src/engine_shims/models/qwen8b"
MODEL_ENV=(
  CANON_P32_DP2TP2=1
  CANON_LOGPROB_M=512
  MIN_TOKEN_BUCKET=512
  CANON_QWEN3_HIDDEN_SIZE=4096
  CANON_QWEN3_INTERMEDIATE_SIZE=12288
  CANON_QWEN3_NUM_ATTENTION_HEADS=32
  CANON_QWEN3_NUM_KV_HEADS=8
  CANON_QWEN3_HEAD_DIM=128
  CANON_PALLAS_ALL_PROJ=1
  CANON_FIXED_AR=1
)
env "${MODEL_ENV[@]}" CANON_QWEN3_TP_SIZE=2 \
  PYTHONPATH="$P32_SHIMS:$ROOT" \
  python canon-zero-tim/tests/t3_dp2tp2/import_gate.py
env "${MODEL_ENV[@]}" CANON_QWEN3_TP_SIZE=4 \
  PYTHONPATH="$P32_SHIMS:$ROOT" \
  python "$P32_SHIMS/p22xf_contract.py"
if env "${MODEL_ENV[@]}" CANON_QWEN3_TP_SIZE=3 \
    PYTHONPATH="$P32_SHIMS:$ROOT" \
    python -c 'import p22xf_contract'; then
  echo "P32.D0.TP_FAMILY_NEGATIVE FAIL"
  exit 1
fi
echo "P32.D0.TP_FAMILY_NEGATIVE PASS"
python -m pytest -q \
  tests/rl/dp_training_test.py \
  tests/models/qwen3/qwen_dp_sharding_test.py
