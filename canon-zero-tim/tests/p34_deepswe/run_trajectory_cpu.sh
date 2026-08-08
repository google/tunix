#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
JAX_PLATFORMS=cpu PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/p34_deepswe/test_trajectory.py
echo "P34_TRAJECTORY_CPU_PASS tests=5"
