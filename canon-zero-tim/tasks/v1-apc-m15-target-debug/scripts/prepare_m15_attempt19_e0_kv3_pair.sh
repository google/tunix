#!/usr/bin/env bash
# Verify E0 admission and render, but never launch, the three-round layer-0 KV pair.
set -euo pipefail

source_commit="${1:?usage: prepare_m15_attempt19_e0_kv3_pair.sh <full-source-sha> <fresh-run-id> <new-output-dir>}"
run_id="${2:?usage: prepare_m15_attempt19_e0_kv3_pair.sh <full-source-sha> <fresh-run-id> <new-output-dir>}"
output="${3:?usage: prepare_m15_attempt19_e0_kv3_pair.sh <full-source-sha> <fresh-run-id> <new-output-dir>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
evidence="$canon/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt17_d3e_canonical_action_20260829"
review_tmp="$(mktemp -d -p /tmp m15-e0-kv3-admission.XXXXXX)"
preparation_complete=0
preserve_or_clean_scratch() {
  rc=$?
  trap - EXIT
  if [ "$preparation_complete" -eq 1 ]; then
    rm -rf -- "$review_tmp"
  else
    echo "[M15.E0.KV3] scratch_preserved=$review_tmp output=$output" >&2
  fi
  exit "$rc"
}
trap preserve_or_clean_scratch EXIT

if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[M15.E0.KV3] REFUSING source must be one full lowercase SHA" >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^[a-z0-9]([a-z0-9-]{0,14}[a-z0-9])?$ ]]; then
  echo "[M15.E0.KV3] REFUSING run id must be a fresh 1-16 character lowercase DNS label component" >&2
  exit 2
fi
test ! -e "$output"
branch="$(git -C "$repo" branch --show-current)"
case "$branch" in
  local/*) ;;
  *) echo "[M15.E0.KV3] REFUSING branch must be local/*" >&2; exit 2 ;;
esac
head="$(git -C "$repo" rev-parse HEAD)"
[ "$head" = "$source_commit" ] || {
  echo "[M15.E0.KV3] REFUSING HEAD does not equal the supplied source" >&2
  exit 2
}
[ -z "$(git -C "$repo" status --porcelain)" ] || {
  echo "[M15.E0.KV3] REFUSING worktree is dirty" >&2
  exit 2
}

python3 "$canon/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py" \
  --repo "$repo" --require-clean
python3 "$script_dir/review_m15_attempt18_e0_admission.py" \
  --evidence "$evidence" \
  --output "$review_tmp/D3E_ADMISSION.json"

python3 "$canon/cluster/render_v1_apc_m15_target_debug.py" \
  --source-commit "$source_commit" \
  --run-id "$run_id" \
  --observer kv3 \
  --output-dir "$output"

python3 "$script_dir/test_review_m15_attempt18_e0_admission.py"
python3 "$script_dir/test_target_carrier.py"
python3 "$script_dir/test_resolved_env.py"
python3 "$script_dir/test_m15_e0_kv_three_round.py"
bash "$script_dir/run_m15_e0_kv_classifier_gate.sh" \
  "$review_tmp/KV_CLASSIFIER_RUNTIME.json"
bash "$canon/tests/p38_serving/test_gcs_persistence.sh"

cp "$review_tmp/D3E_ADMISSION.json" "$output/D3E_ADMISSION.json"
cp "$review_tmp/KV_CLASSIFIER_RUNTIME.json" "$output/KV_CLASSIFIER_RUNTIME.json"
python3 - "$output" "$source_commit" "$run_id" <<'PY'
import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml

root = Path(sys.argv[1])
source = sys.argv[2]
run_id = sys.argv[3]
admission_path = root / "D3E_ADMISSION.json"
admission = json.loads(admission_path.read_text(encoding="utf-8"))
runtime_path = root / "KV_CLASSIFIER_RUNTIME.json"
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
if not (
    admission.get("status") == "E0_PREPARATION_ADMITTED"
    and admission.get("d3e_gate") == "FIRST_RED_LOCALIZED"
    and admission.get("launch_authorized") is False
    and admission.get("numerical_repair_authorized") is False
    and admission.get("target_prefix", {}).get("tokens") == 1226
    and admission.get("target_prefix", {}).get("aliases") == 8
    and admission.get("target_prefix", {}).get("logical_pages") == 77
):
  raise SystemExit("D3e admission report does not admit E0 KV3 rendering")
if not (
    runtime.get("schema") == "m15-e0-kv-classifier-runtime-v1"
    and runtime.get("status") == "PASS"
    and runtime.get("route") in {"host", "docker"}
    and runtime.get("external_access") is False
):
  raise SystemExit("KV classifier runtime receipt is not admissible")
if runtime["route"] == "docker" and not (
    runtime.get("image_id")
    == "sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a"
    and runtime.get("pull_policy") == "never"
    and runtime.get("network_mode") == "none"
):
  raise SystemExit("KV classifier Docker runtime is not pinned and offline")

rows = []
documents = []
for path in sorted(root.glob("*.yaml")):
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  containers = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"]
  container = next(item for item in containers if item["name"] == "jax-tpu")
  env = {item["name"]: str(item["value"])
         for item in container["env"] if "value" in item}
  arm = env["CANON_APC_M15_TARGET_DEBUG"]
  expected = {
      "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
      "CANON_P38_DURABILITY_PROFILE": "m15-e0-kv-v1",
      "CANON_P38_KV_OBSERVER_MAX_CANDIDATES": "8",
      "CANON_P38_KV_OBSERVER_MAX_PAGES": "96",
      "CANON_P38_KV_OBSERVER_MAX_BYTES": "134217728",
      "CANON_P38_KV_OBSERVER_MAX_READ_BYTES": "671088640",
      "CANON_P38_KV_OBSERVER_LAYER": "0",
      "CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS": "1226",
      "CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256": admission["target_prefix"]["sha256"],
  }
  wrong = {name: env.get(name) for name, value in expected.items()
           if env.get(name) != value}
  if wrong:
    raise SystemExit(f"rendered KV3 contract drifted: {wrong}")
  if any(name.startswith(("CANON_P38_SEAM", "CANON_P38_TAIL"))
         for name in env):
    raise SystemExit("rendered KV3 pair attached a seam observer")
  if env.get("CANON_VLLM_ENABLE_PREFIX_CACHING") != (
      "1" if arm == "on" else "0"
  ):
    raise SystemExit("rendered APC arm drifted")
  rows.append({
      "arm": arm,
      "jobset": document["metadata"]["name"],
      "yaml": path.name,
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
  })
  documents.append((arm, document))
if len(rows) != 2 or {row["arm"] for row in rows} != {"off", "on"}:
  raise SystemExit("rendered pair membership drifted")

def normalize(value, arm):
  if isinstance(value, str):
    return value.replace(f"-m15-{arm}-", "-m15-<ARM>-")
  if isinstance(value, list):
    return [normalize(item, arm) for item in value]
  if isinstance(value, dict):
    return {key: normalize(item, arm) for key, item in value.items()}
  return value

normalized = []
for arm, document in documents:
  candidate = copy.deepcopy(document)
  containers = candidate["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"]
  container = next(item for item in containers if item["name"] == "jax-tpu")
  for item in container["env"]:
    if item["name"] == "CANON_APC_M15_TARGET_DEBUG":
      item["value"] = "<ARM>"
    elif item["name"] == "CANON_VLLM_ENABLE_PREFIX_CACHING":
      item["value"] = "<APC>"
  candidate["metadata"]["labels"]["canon.zero-tim/apc-m15-arm"] = "<ARM>"
  normalized.append(normalize(candidate, arm))
if normalized[0] != normalized[1]:
  raise SystemExit("rendered pair differs beyond the signed APC treatment")

contract = {
    "schema": "m15-attempt19-e0-kv3-render-v1",
    "source_commit": source,
    "run_id": run_id,
    "d3e_admission_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
    "kv_classifier_runtime_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
    "observer": {
        "kind": "live-kv-prefix-fingerprint",
        "layer": 0,
        "target_prefix_tokens": 1226,
        "target_aliases_per_round": 8,
        "logical_pages": 77,
        "page_bound": 96,
        "claim_level": "bit-level-diagnostic-fingerprint-not-full-kv-bytes",
    },
    "durability": {
        "profile": "m15-e0-kv-v1",
        "classifier_input_checkpoint": "upload-readback-before-classification",
        "round_completion": "classifier-pass-upload-readback-before-learner-ack",
        "root_collection_required_for_round_salvage": False,
    },
    "rounds": 3,
    "zero_backward": True,
    "zero_optimizer_commit": True,
    "b_full_reset_immutable": True,
    "control_and_treatment_differ_only_at_apc": True,
    "pinned_exact_image_required": True,
    "launch_authorized": False,
    "target_executed": False,
    "remote_mutation": False,
    "arms": sorted(rows, key=lambda row: row["arm"]),
}
(root / "RUN_CONTRACT.json").write_text(
    json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
names = sorted(path.name for path in root.glob("*.yaml")) + [
    "D3E_ADMISSION.json", "KV_CLASSIFIER_RUNTIME.json", "RUN_CONTRACT.json"
]
(root / "SHA256SUMS").write_text("".join(
    f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
    for name in names
), encoding="ascii")
PY

(cd "$output" && sha256sum -c SHA256SUMS --quiet)
preparation_complete=1
echo "[M15.E0.KV3] RENDER_PASS source=$source_commit rounds=3 layer=0 aliases_per_round=8 pages=96 durability=m15-e0-kv-v1 output=$output"
echo "[M15.E0.KV3] TARGET_NOT_RUN pinned_exact_image=required launch_approval=required gcs=0 kubernetes=0 tpu=0"
