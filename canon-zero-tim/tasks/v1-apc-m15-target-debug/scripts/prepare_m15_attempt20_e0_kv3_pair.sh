#!/usr/bin/env bash
# Prepare, but never launch, the repaired three-round M15 E0 KV3 pair.
set -euo pipefail

source_commit="${1:?usage: prepare_m15_attempt20_e0_kv3_pair.sh <full-source-sha> <fresh-run-id> <new-output-dir>}"
run_id="${2:?usage: prepare_m15_attempt20_e0_kv3_pair.sh <full-source-sha> <fresh-run-id> <new-output-dir>}"
output="${3:?usage: prepare_m15_attempt20_e0_kv3_pair.sh <full-source-sha> <fresh-run-id> <new-output-dir>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
incident="$canon/evidence/m15_e0_kv3_attempt19_incident/SHA256SUMS"
expected_incident_sha="bc824561d39ed4e0bb5df65f56baff68e86ac64b8694a073f13a40bf31ba1636"

test ! -e "$output"
actual_incident_sha="$(sha256sum "$incident" | awk '{print $1}')"
[ "$actual_incident_sha" = "$expected_incident_sha" ] || {
  echo "[M15.E0.KV3R] REFUSING Attempt 19 incident manifest drifted" >&2
  exit 2
}

bash "$script_dir/prepare_m15_attempt19_e0_kv3_pair.sh" \
  "$source_commit" "$run_id" "$output"

python3 - "$output" "$source_commit" "$expected_incident_sha" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
source = sys.argv[2]
incident_sha = sys.argv[3]
contract_path = root / "RUN_CONTRACT.json"
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if not (
    contract.get("schema") == "m15-attempt19-e0-kv3-render-v1"
    and contract.get("source_commit") == source
    and contract.get("rounds") == 3
    and contract.get("launch_authorized") is False
    and contract.get("target_executed") is False
):
  raise SystemExit("base E0 KV3 render contract is not admissible")
contract["execution_generation"] = "attempt20-carrier-repair-v1"
contract["carrier_repair"] = {
    "schema": "m15-e0-kv3-carrier-repair-v1",
    "status": "ADMITTED",
    "attempt19_incident_manifest_sha256": incident_sha,
    "red_join_boundary": "snapshot-prefix-or-next-token-boundary",
    "prompt_inventory": "round0-frozen-requeued-for-rounds1-and2",
    "required_runtime_markers": {
        "E0_KV3_PROMPT_BATCH_FROZEN": 1,
        "E0_KV3_PROMPT_BATCH_REQUEUED": 2,
        "unique_prompt_batch_sha256": 1,
    },
    "dataset_advance": False,
    "numerical_path_changed": False,
}
contract_path.write_text(
    json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
names = sorted(
    path.name for path in root.iterdir()
    if path.is_file() and path.name != "SHA256SUMS"
)
(root / "SHA256SUMS").write_text(
    "".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names
    ),
    encoding="ascii",
)
PY

(cd "$output" && sha256sum -c SHA256SUMS --quiet)
echo "[M15.E0.KV3R] RENDER_PASS source=$source_commit generation=attempt20-carrier-repair-v1 rounds=3 red_join=next-token-boundary prompt_inventory=frozen output=$output"
echo "[M15.E0.KV3R] TARGET_NOT_RUN pinned_exact_image=required launch_approval=required gcs=0 kubernetes=0 tpu=0"
