#!/usr/bin/env bash
# Render (but never launch) the frozen-weight three-round APC off/on pair.
set -euo pipefail

source_commit="${1:?usage: prepare_m15_multiround_pair.sh <full-source-sha> <run-id> <output-dir> [layer|full] [seam-layer]}"
run_id="${2:?usage: prepare_m15_multiround_pair.sh <full-source-sha> <run-id> <output-dir> [layer|full] [seam-layer]}"
output="${3:?usage: prepare_m15_multiround_pair.sh <full-source-sha> <run-id> <output-dir> [layer|full] [seam-layer]}"
observer="${4:-full}"
seam_layer="${5:-0}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
review_tmp="$(mktemp -d)"
trap 'rm -rf "$review_tmp"' EXIT

[ "${#source_commit}" -eq 40 ] || {
  echo "[M15.MULTIROUND] REFUSING: source must be a full 40-character SHA" >&2
  exit 2
}
case "$source_commit" in
  *[!0-9a-f]*)
    echo "[M15.MULTIROUND] REFUSING: source must be a full lowercase SHA" >&2
    exit 2
    ;;
esac
test ! -e "$output"
test "$(git -C "$repo" rev-parse "$source_commit")" = "$source_commit"
case "$observer" in
  layer) render_observer_args=(--observer layer) ;;
  full)
    case "$seam_layer" in
      ''|*[!0-9]*) echo "[M15.MULTIROUND] REFUSING: seam layer must be an integer" >&2; exit 2 ;;
    esac
    [ "$seam_layer" -ge 0 ] && [ "$seam_layer" -lt 36 ] || {
      echo "[M15.MULTIROUND] REFUSING: seam layer must be in [0,36)" >&2
      exit 2
    }
    render_observer_args=(--observer full --seam-layer "$seam_layer")
    ;;
  *) echo "[M15.MULTIROUND] REFUSING: observer must be layer or full" >&2; exit 2 ;;
esac

python3 "$script_dir/review_m15_attempt13_d32_inventory.py" \
  --inventory "$canon/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt13_d32_inventory_20260828" \
  --output "$review_tmp/return"
(cd "$review_tmp/return" && sha256sum -c SHA256SUMS --quiet)

python3 "$canon/cluster/render_v1_apc_m15_target_debug.py" \
  --source-commit "$source_commit" \
  --run-id "$run_id" \
  "${render_observer_args[@]}" \
  --output-dir "$output"
python3 "$script_dir/test_target_carrier.py"
python3 "$script_dir/test_resolved_env.py"

python3 - "$output" "$source_commit" "$run_id" "$observer" "$seam_layer" \
  "$review_tmp/return/D32_REVIEW.json" <<'PY'
import hashlib
import json
import pathlib
import sys
import yaml

root = pathlib.Path(sys.argv[1])
source = sys.argv[2]
run_id = sys.argv[3]
observer = sys.argv[4]
seam_layer = int(sys.argv[5]) if observer == "full" else None
review_path = pathlib.Path(sys.argv[6])
review = json.loads(review_path.read_text(encoding="utf-8"))
if not (
    review.get("status") == "PASS"
    and review.get("decision") == "D32_LIVE_ABSENT_WITH_COUNT_DRIFT"
    and review.get("inventory_transport_status") == "PASS"
    and review.get("live_absence_status") == "CONFIRMED"
    and review.get("count_contract_status") == "DRIFT"
    and review.get("d33_preparation_eligible") is True
    and review.get("d33_launch_authorized") is False
    and review.get("numerical_repair_authorized") is False
):
  raise SystemExit("D32 offline review does not admit d33 preparation")
rows = []
for path in sorted(root.glob("*.yaml")):
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  container = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"][0]
  env = {row["name"]: str(row["value"]) for row in container["env"] if "value" in row}
  rows.append({
      "arm": env["CANON_APC_M15_TARGET_DEBUG"],
      "diagnostic_rounds": int(env["CANON_P38_DIAGNOSTIC_ROUNDS"]),
      "jobset": document["metadata"]["name"],
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "yaml": path.name,
  })
if len(rows) != 2 or {row["arm"] for row in rows} != {"off", "on"}:
  raise SystemExit("rendered pair membership drifted")
if any(row["diagnostic_rounds"] != 3 for row in rows):
  raise SystemExit("rendered pair is not three-round")
(root / "D32_REVIEW.json").write_bytes(review_path.read_bytes())
(root / "RUN_CONTRACT.json").write_text(json.dumps({
    "schema": "m15-apc-three-round-render-v1",
    "source_commit": source,
    "run_id": run_id,
    "zero_backward": True,
    "zero_optimizer_commit": True,
    "observer": observer,
    "seam_layer": seam_layer,
    "d32_review": {
        "decision": review["decision"],
        "count_contract_status": review["count_contract_status"],
        "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "d33_preparation_eligible": True,
        "d33_launch_authorized": False,
        "numerical_repair_authorized": False,
    },
    "arms": rows,
}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
names = sorted([path.name for path in root.glob("*.yaml")]) + [
    "D32_REVIEW.json", "RUN_CONTRACT.json"
]
(root / "SHA256SUMS").write_text("".join(
    f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
    for name in names
), encoding="ascii")
PY
(cd "$output" && sha256sum -c SHA256SUMS --quiet)
echo "[M15.MULTIROUND] RENDER_PASS source=$source_commit rounds=3 observer=$observer seam_layer=$seam_layer output=$output"
echo "[M15.MULTIROUND] NOT_LAUNCHED use standalone kubectl apply only after approval"
