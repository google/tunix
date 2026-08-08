#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/p34_deepswe/test_qwen32b_tp8.py
PYTHONPATH="$ROOT/canon-zero-tim/src/engine_shims/models/qwen32b:${PYTHONPATH:-}" \
  python3 canon-zero-tim/src/engine_shims/models/qwen32b/p22xf_contract.py
echo "P34_QWEN32B_TP8_STATIC_PASS tests=9"
