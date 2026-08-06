#!/usr/bin/env bash
# Prove the P32 runner rejects a rank-dependent post-reduction gradient.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAX_PLATFORMS=cpu
export XLA_FLAGS="${XLA_FLAGS:-} --xla_force_host_platform_device_count=4"

set +e
output="$(python3 "$HERE/probe_dp_update.py" \
  --dp 4 --tp 1 --local-samples 16 --inject-rank-fault 2>&1)"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "$output"
  echo "NEGATIVE CONTROL FAIL: rank-dependent gradient was accepted" >&2
  exit 1
fi
grep -aq '^\[P32.DP\] VERDICT FAIL$' <<<"$output" || {
  echo "$output"
  echo "NEGATIVE CONTROL FAIL: probe failed without its verdict" >&2
  exit 1
}
echo "REJECTED (exit $rc) rank-dependent post-reduction gradient"
echo "===== T2-DP NEGATIVE CONTROL PASS ====="
