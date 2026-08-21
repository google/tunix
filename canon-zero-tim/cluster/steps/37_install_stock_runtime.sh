#!/usr/bin/env bash
# Install non-numerical Python packages needed by the stock FrozenLake workload.
#
# Canonical runs historically received these packages as a side effect of Step
# 30. P57 stock-fast deliberately skips that step, so it must install only the
# runtime dependencies without building or exposing the canonical overlay.
set -euo pipefail
source "$CANON_STATE/env.sh"
# shellcheck disable=SC1091
source "$CANON_PKG/cluster/steps/p57_runtime_contract.sh"

if ! p57_is_stock_fast_calibration; then
  echo "[P57.STOCK_FAST] FATAL: stock runtime install used outside calibration" >&2
  exit 2
fi

if ! python3 -c "import gymnasium, sentencepiece, tiktoken" 2>/dev/null; then
  python3 -m pip install --break-system-packages --no-deps -q \
    'gymnasium==1.3.0' 'sentencepiece==0.2.2' 'tiktoken==0.13.0'
fi
python3 -c "import gymnasium, grain, numba, numpy, sentencepiece, tiktoken; print('[P57.STOCK_FAST] RUNTIME_DEPS_PASS packages=6')"
