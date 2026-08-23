#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo"

python3 -m unittest discover \
  -s canon-zero-tim/tests/v1_phase4 \
  -p 'test_*.py' \
  -v

echo "V1_HP_THREE_FULL_CPU_PASS manifests=3 strict=3 xprof=3"
