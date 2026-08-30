#!/usr/bin/env bash
# Read-only recovery of Attempt-20 treatment round 0 classifier input.
set -euo pipefail

render_dir="${1:?usage: run_m15_attempt20_on_round0_offline_recovery.sh <attempt20-render-dir> <new-output-dir> [scratch-parent]}"
output="${2:?usage: run_m15_attempt20_on_round0_offline_recovery.sh <attempt20-render-dir> <new-output-dir> [scratch-parent]}"
scratch_parent="${3:-/mnt/disks/tunix-data}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
evidence="$canon/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt20_e0_kv3_salvage_return_20260830"

if [ ! -d "$render_dir" ]; then
  echo "[M15.E0U.ON-R0] REFUSING status=ORIGINAL_RENDER_UNAVAILABLE classification=NONE three_round_verdict=0 numerical_repair_authorized=0" >&2
  exit 2
fi
if [ ! -d "$scratch_parent" ]; then
  echo "[M15.E0U.ON-R0] REFUSING status=SCRATCH_PARENT_UNAVAILABLE classification=NONE" >&2
  exit 2
fi
if [ -e "$output" ]; then
  echo "[M15.E0U.ON-R0] REFUSING status=OUTPUT_ALREADY_EXISTS classification=NONE" >&2
  exit 2
fi

branch="$(git -C "$repo" branch --show-current)"
case "$branch" in
  local/*) ;;
  *) echo "[M15.E0U.ON-R0] REFUSING branch must be local/*" >&2; exit 2 ;;
esac
analysis_source="$(git -C "$repo" rev-parse HEAD)"
if [ -n "$(git -C "$repo" status --porcelain)" ]; then
  echo "[M15.E0U.ON-R0] REFUSING worktree is dirty" >&2
  exit 2
fi
python3 "$canon/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py" \
  --repo "$repo" --require-clean
(cd "$evidence" && sha256sum -c SHA256SUMS --quiet)
target_source="$(python3 - "$evidence/E0_KV3_RETURN.json" <<'PY'
import json
from pathlib import Path
import re
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = str(value.get("source_commit", ""))
if not (
    value.get("status") == "ROUND_EVIDENCE_PARTIAL"
    and value.get("target_executed") is True
    and value.get("terminal_pair_complete") is False
    and value.get("numerical_repair_authorized") is False
    and value.get("arms", {}).get("off", {}).get("completed_rounds")
        == [0, 1, 2]
    and value.get("arms", {}).get("on", {}).get("completed_rounds") == []
    and re.fullmatch(r"[0-9a-f]{40}", source)
):
  raise SystemExit("committed Attempt-20 partial-return contract drifted")
print(source)
PY
)"
git -C "$repo" merge-base --is-ancestor "$target_source" "$analysis_source"
(cd "$render_dir" && sha256sum -c SHA256SUMS --quiet)

on_root="$(python3 - "$render_dir" "$target_source" <<'PY'
import json
from pathlib import Path
import re
import sys

import yaml

root = Path(sys.argv[1])
source = sys.argv[2]
contract = json.loads((root / "RUN_CONTRACT.json").read_text(encoding="utf-8"))
repair = contract.get("carrier_repair", {})
if not (
    contract.get("source_commit") == source
    and contract.get("execution_generation") == "attempt20-carrier-repair-v1"
    and contract.get("rounds") == 3
    and contract.get("launch_authorized") is False
    and contract.get("observer", {}).get("layer") == 0
    and contract.get("observer", {}).get("target_aliases_per_round") == 8
    and contract.get("durability", {}).get("profile") == "m15-e0-kv-v1"
    and repair.get("schema") == "m15-e0-kv3-carrier-repair-v1"
    and repair.get("status") == "ADMITTED"
    and repair.get("red_join_boundary")
        == "snapshot-prefix-or-next-token-boundary"
    and repair.get("prompt_inventory")
        == "round0-frozen-requeued-for-rounds1-and2"
    and repair.get("dataset_advance") is False
    and repair.get("numerical_path_changed") is False
):
  raise SystemExit("Attempt-20 render contract is absent or drifted")

matches = []
for path in sorted(root.glob("jobset-v1-apc-m15-*-kv3.yaml")):
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  containers = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"]
  container = next(item for item in containers if item["name"] == "jax-tpu")
  env = {item["name"]: str(item["value"])
         for item in container["env"] if "value" in item}
  if env.get("CANON_APC_M15_TARGET_DEBUG") != "on":
    continue
  remote = env.get("CANON_P38_GCS_PREFIX", "")
  if not (
      env.get("CANON_EXPECT_COMMIT") == source
      and env.get("CANON_P38_DURABILITY_PROFILE") == "m15-e0-kv-v1"
      and env.get("CANON_P38_DIAGNOSTIC_ROUNDS") == "3"
      and env.get("CANON_P38_KV_OBSERVER_LAYER") == "0"
      and re.fullmatch(r"gs://[^/]+/.+/attempt-0", remote)
  ):
    raise SystemExit("Attempt-20 treatment render identity drifted")
  matches.append(remote)
if len(matches) != 1:
  raise SystemExit("Attempt-20 treatment render is not unique")
print(matches[0])
PY
)"

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2" >/dev/null 2>&1; }
else
  echo "[M15.E0U.ON-R0] REFUSING gcloud or gsutil is required" >&2
  exit 2
fi

raw_log="$(mktemp -p "$scratch_parent" m15-attempt20-on-r0-recovery.XXXXXX.log)"
scratch="$(mktemp -d -p "$scratch_parent" m15-attempt20-on-r0-recovery.XXXXXX)"
mkdir -m 700 "$scratch/remote"
status_file="$scratch/status"
inventory="$scratch/retrieval.tsv"

recover_core() {
  local unavailable=0
  local remote_prefix="$on_root/rounds/000000/classifier-input"
  for name in CLASSIFIER_INPUT_RECEIPT.json CLASSIFIER_INPUT_SHA256SUMS; do
    if gcs_cp "$remote_prefix/$name" "$scratch/remote/$name"; then
      printf '%s\tretrieved\n' "$name" >> "$inventory"
    else
      printf '%s\tunavailable\n' "$name" >> "$inventory"
      unavailable=1
    fi
  done
  if [ "$unavailable" -ne 0 ]; then
    printf '%s\n' CLASSIFIER_INPUT_UNAVAILABLE > "$status_file"
    echo "[M15.E0U.ON-R0] small classifier-input receipt retrieval incomplete"
    return 3
  fi
  if gcs_cp "$remote_prefix/CLASSIFIER_INPUT_ARCHIVE.tar" \
      "$scratch/remote/CLASSIFIER_INPUT_ARCHIVE.tar"; then
    printf '%s\tretrieved\n' CLASSIFIER_INPUT_ARCHIVE.tar >> "$inventory"
  else
    printf '%s\tunavailable\n' CLASSIFIER_INPUT_ARCHIVE.tar >> "$inventory"
    printf '%s\n' CLASSIFIER_INPUT_UNAVAILABLE > "$status_file"
    echo "[M15.E0U.ON-R0] classifier-input archive retrieval incomplete"
    return 3
  fi
  if ! python3 "$script_dir/review_m15_attempt20_on_round0.py" \
      --archive "$scratch/remote/CLASSIFIER_INPUT_ARCHIVE.tar" \
      --manifest "$scratch/remote/CLASSIFIER_INPUT_SHA256SUMS" \
      --receipt "$scratch/remote/CLASSIFIER_INPUT_RECEIPT.json" \
      --expected-source "$target_source" \
      --analysis-source "$analysis_source" \
      --scratch "$scratch" \
      --output "$output"; then
    printf '%s\n' INVALID_OR_CLASSIFIER_FAILED > "$status_file"
    return 3
  fi
  printf '%s\n' RECOVERY_COMPLETE > "$status_file"
  return 0
}

set +e
recover_core >"$raw_log" 2>&1
recovery_rc=$?
set -e
read -r raw_sha _ < <(sha256sum "$raw_log")
if [ -s "$status_file" ]; then
  status="$(cat "$status_file")"
else
  status=UNEXPECTED_RECOVERY_FAILURE
fi

if [ "$recovery_rc" -ne 0 ]; then
  if [ ! -e "$output" ]; then
    python3 - "$output" "$analysis_source" "$target_source" \
        "$status" "$inventory" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
inventory = Path(sys.argv[5])
retrieval = {}
if inventory.is_file():
  for line in inventory.read_text(encoding="utf-8").splitlines():
    name, status = line.split("\t")
    retrieval[name] = status
report = {
    "schema": "m15-attempt20-on-round0-offline-recovery-v1",
    "status": sys.argv[4],
    "analysis_source_commit": sys.argv[2],
    "target_source_commit": sys.argv[3],
    "arm": "on",
    "diagnostic_round": 0,
    "classification": None,
    "classification_available": False,
    "b_full_reset_runtime_receipt_available": False,
    "all_num_cached_tokens_zero_runtime_receipt_available": False,
    "retrieval": retrieval,
    "rounds_recovered": [],
    "three_round_verdict": False,
    "terminal_pair_complete": False,
    "target_rerun": False,
    "numerical_repair_authorized": False,
    "remote_mutation": False,
    "claim_ceiling": (
        "NO_CLASSIFICATION / INCONCLUSIVE / NO_TARGET_PASS / "
        "B_RESET_RUNTIME_RECEIPT_UNAVAILABLE / "
        "NO_NUMERICAL_REPAIR_AUTHORIZATION"
    ),
}
output.mkdir(mode=0o700)
path = output / "ATTEMPT20_ON_R0_RECOVERY.json"
path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(output / "SHA256SUMS").write_text(
    f"{digest}  {path.name}\n", encoding="ascii"
)
PY
  fi
  (cd "$output" && sha256sum -c SHA256SUMS --quiet)
  read -r manifest_sha _ < <(sha256sum "$output/SHA256SUMS")
  echo "[M15.E0U.ON-R0] INCONCLUSIVE status=$status classification=NONE three_round_verdict=0 numerical_repair_authorized=0 manifest_sha256=$manifest_sha raw_log=$raw_log raw_log_sha256=$raw_sha scratch_preserved=$scratch" >&2
  echo "[M15.E0U.ON-R0] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0" >&2
  exit 3
fi

(cd "$output" && sha256sum -c SHA256SUMS --quiet)
read -r manifest_sha _ < <(sha256sum "$output/SHA256SUMS")
read -r result_status classification < <(python3 - "$output/ATTEMPT20_ON_R0_RECOVERY.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["status"], value["classification"])
PY
)
rm -rf -- "$scratch"
echo "[M15.E0U.ON-R0] RECOVERY_COMPLETE status=$result_status classification=$classification rounds=1 three_round_verdict=0 target_pass=0 numerical_repair_authorized=0 manifest_sha256=$manifest_sha raw_log=$raw_log raw_log_sha256=$raw_sha"
echo "[M15.E0U.ON-R0] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0"
