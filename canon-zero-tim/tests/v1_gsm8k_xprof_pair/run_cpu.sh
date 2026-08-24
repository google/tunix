#!/usr/bin/env bash
set -euo pipefail
repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
python3 -m unittest discover \
  -s "$repo/canon-zero-tim/tests/v1_gsm8k_xprof_pair" \
  -p 'test_*.py'
bash -n "$repo/canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_common.sh"
bash -n "$repo/canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_inner.sh"
bash -n "$repo/canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_native.sh"
bash -n "$repo/canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh"
echo "V1_GSM8K_XPROF_CPU_PASS"
