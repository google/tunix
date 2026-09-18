#!/usr/bin/env bash
# Validate the Attempt-20 carrier contract, then run the salvage-first read-only return.
set -euo pipefail

analysis_source="${1:?usage: run_m15_attempt20_e0_kv3_return_recovery.sh <full-analysis-sha> <verified-render-dir> <new-output-dir> [scratch-parent]}"
render_dir="${2:?usage: run_m15_attempt20_e0_kv3_return_recovery.sh <full-analysis-sha> <verified-render-dir> <new-output-dir> [scratch-parent]}"
output="${3:?usage: run_m15_attempt20_e0_kv3_return_recovery.sh <full-analysis-sha> <verified-render-dir> <new-output-dir> [scratch-parent]}"
scratch_parent="${4:-/mnt/disks/tunix-data}"
script_dir="$(cd "$(dirname "$0")" && pwd)"

(cd "$render_dir" && sha256sum -c SHA256SUMS --quiet)
python3 - "$render_dir/RUN_CONTRACT.json" "$analysis_source" <<'PY'
import json
from pathlib import Path
import sys

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
repair = contract.get("carrier_repair", {})
if not (
    contract.get("source_commit") == sys.argv[2]
    and contract.get("execution_generation") == "attempt20-carrier-repair-v1"
    and repair.get("schema") == "m15-e0-kv3-carrier-repair-v1"
    and repair.get("status") == "ADMITTED"
    and repair.get("red_join_boundary")
    == "snapshot-prefix-or-next-token-boundary"
    and repair.get("prompt_inventory")
    == "round0-frozen-requeued-for-rounds1-and2"
    and repair.get("dataset_advance") is False
    and repair.get("numerical_path_changed") is False
):
  raise SystemExit("Attempt-20 repaired carrier contract is absent or drifted")
PY

bash "$script_dir/run_m15_attempt19_e0_kv3_return_recovery.sh" \
  "$analysis_source" "$render_dir" "$output" "$scratch_parent"
echo "[M15.E0.KV3R.RECOVERY] COMPLETE generation=attempt20-carrier-repair-v1 gcs_read=1 gcs_write=0 kubernetes=0 tpu=0 output=$output"
