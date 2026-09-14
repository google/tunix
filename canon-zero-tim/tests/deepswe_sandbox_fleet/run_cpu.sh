#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/deepswe_sandbox_fleet/test_sandbox_fleet.py
echo "DEEPSWE_SANDBOX_FLEET_CPU_PASS"
