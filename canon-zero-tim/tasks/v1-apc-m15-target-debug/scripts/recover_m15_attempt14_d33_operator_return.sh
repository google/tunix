#!/usr/bin/env bash
# Recover the complete small d33 return without rerunning either TPU JobSet.
set -euo pipefail

output="${1:?usage: recover_m15_attempt14_d33_operator_return.sh <output-dir> [scratch-parent] [namespace]}"
scratch_parent="${2:-/tmp}"
namespace="${3:-default}"
test -d "$scratch_parent"
test ! -e "$output"

script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
evidence="$canon/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt14_paired_d33_20260828"

test -z "$(git -C "$repo" status --porcelain)" || {
  echo "[M15.D33.RECOVERY] REFUSING: checkout must be clean" >&2
  exit 2
}

scratch="$(mktemp -d -p "$scratch_parent" m15-d33-recovery.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
python3 "$script_dir/prepare_m15_attempt14_d33_recovery_contract.py" \
  --evidence "$evidence" \
  --output "$scratch/contract"

source_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' \
  "$scratch/contract/RECOVERY_INPUT_RECEIPT.json")"
test "$(git -C "$repo" rev-parse "$source_commit^{commit}")" = "$source_commit" || {
  echo "[M15.D33.RECOVERY] REFUSING: submitted source commit is unavailable" >&2
  exit 2
}

bash "$script_dir/run_m15_multiround_operator_return.sh" \
  "$scratch/contract" "$output" "$scratch_parent" "$namespace"
(cd "$output" && sha256sum -c SHA256SUMS --quiet)
test -f "$output/RECOVERY_INPUT_RECEIPT.json"
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$output/OPERATOR_RETURN_SUMMARY.json")"
manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
echo "[M15.D33.RECOVERY] COMPLETE status=$status manifest_sha256=$manifest_sha output=$output"
