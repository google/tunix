#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

cd "$ROOT"
python3 -m unittest discover -s canon-zero-tim/tests/deepswe_pilot -p 'test_*.py'
python3 -m py_compile \
  canon-zero-tim/cluster/render_deepswe_pilot.py \
  canon-zero-tim/tests/deepswe_pilot/classify_run.py
bash -n canon-zero-tim/cluster/profiles/qwen3-32b-dp4-tp8-deepswe-pilot.env
echo "P39_DEEPSWE_PILOT_CPU_PASS"
