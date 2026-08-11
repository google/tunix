#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s canon-zero-tim/tests/p43_deepswe_debug -p 'test_*.py' -v
echo "P43_DEEPSWE_DEBUG_CPU_PASS"
