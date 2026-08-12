#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
python3 -m unittest discover -s canon-zero-tim/tests/p44_deepswe_qwen4b_parity -p 'test_*.py' -v
echo "P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS"
