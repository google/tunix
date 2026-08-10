#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/p34_deepswe/test_contract.py
PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/p34_deepswe/test_script_contract.py
PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/p34_deepswe/test_env_contract.py
JAX_PLATFORMS=cpu PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/p34_deepswe/test_sampler_contract.py
PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/p34_deepswe/test_render_p34_jobset.py
PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/p34_deepswe/test_classify_run.py
PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 \
  canon-zero-tim/tests/p34_deepswe/test_scheduler_contract.py
echo "P34_STATIC_PASS suites=7"
