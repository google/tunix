#!/usr/bin/env bash
# Prepare, but never launch, a TiTO-aware three-round M15 APC layer pair.
set -euo pipefail

source_commit="${1:?usage: prepare_m15_e0v_tito_layer_pair.sh <full-source-sha> <fresh-run-id> <new-output-dir>}"
run_id="${2:?usage: prepare_m15_e0v_tito_layer_pair.sh <full-source-sha> <fresh-run-id> <new-output-dir>}"
output="${3:?usage: prepare_m15_e0v_tito_layer_pair.sh <full-source-sha> <fresh-run-id> <new-output-dir>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
incident="$canon/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt20_e0u_r0_recovery_20260830"
expected_incident_sha="827b4038d269870d5b72e4f432b9680c89d79923d8bb2952163daca0e60ea093"

if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[M15.E0V.PREPARE] REFUSING source must be one full lowercase SHA" >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^[a-z0-9]([a-z0-9-]{0,14}[a-z0-9])?$ ]]; then
  echo "[M15.E0V.PREPARE] REFUSING run id must be a fresh 1-16 character lowercase DNS label component" >&2
  exit 2
fi
test ! -e "$output"
branch="$(git -C "$repo" branch --show-current)"
case "$branch" in
  local/*) ;;
  *) echo "[M15.E0V.PREPARE] REFUSING branch must be local/*" >&2; exit 2 ;;
esac
head="$(git -C "$repo" rev-parse HEAD)"
[ "$head" = "$source_commit" ] || {
  echo "[M15.E0V.PREPARE] REFUSING HEAD does not equal supplied source" >&2
  exit 2
}
python3 "$canon/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py" \
  --repo "$repo" --require-clean
(cd "$incident" && sha256sum -c SHA256SUMS --quiet)
read -r incident_sha _ < <(sha256sum "$incident/SHA256SUMS")
[ "$incident_sha" = "$expected_incident_sha" ] || {
  echo "[M15.E0V.PREPARE] REFUSING Attempt-20 E0u incident manifest drifted" >&2
  exit 2
}

python3 "$canon/cluster/render_v1_apc_m15_target_debug.py" \
  --source-commit "$source_commit" \
  --run-id "$run_id" \
  --observer layer \
  --output-dir "$output"

python3 "$canon/tests/p57_frozenlake_tim/test_m15_token_continuity.py"
python3 "$script_dir/test_classify_m15_apc_debug_tito.py"
python3 "$script_dir/test_target_carrier.py"
python3 "$script_dir/test_resolved_env.py"

python3 - "$output" "$source_commit" "$run_id" "$incident_sha" <<'PY'
import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml

root = Path(sys.argv[1])
source = sys.argv[2]
run_id = sys.argv[3]
incident_sha = sys.argv[4]
rows = []
documents = []
for path in sorted(root.glob("*.yaml")):
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  containers = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"]
  container = next(item for item in containers if item["name"] == "jax-tpu")
  env = {
      item["name"]: str(item["value"])
      for item in container["env"] if "value" in item
  }
  arm = env.get("CANON_APC_M15_TARGET_DEBUG")
  expected = {
      "CANON_M15_TOKEN_CONTINUITY": "exact",
      "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
      "CANON_P38_DURABILITY_PROFILE": "m15-wide-v1",
      "CANON_P38_SEAM_OBSERVER": "layer",
      "CANON_P38_TAIL_OBSERVER": "1",
      "CANON_P38_PRECHECK_ONLY": "1",
      "CANON_P38_CONTROLLED_EXIT": "1",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
  }
  wrong = {
      name: env.get(name) for name, value in expected.items()
      if env.get(name) != value
  }
  if wrong:
    raise SystemExit(f"TiTO layer render contract drifted: {wrong}")
  historical_kv_names = (
      "CANON_P38_KV_OBSERVER_CLASSIFICATION",
      "CANON_P38_KV_OBSERVER_DIR",
      "CANON_P38_KV_OBSERVER_LAYER",
      "CANON_P38_KV_OBSERVER_MAX_BYTES",
      "CANON_P38_KV_OBSERVER_MAX_CANDIDATES",
      "CANON_P38_KV_OBSERVER_MAX_PAGES",
      "CANON_P38_KV_OBSERVER_MAX_READ_BYTES",
      "CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256",
      "CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS",
  )
  if any(name in env for name in historical_kv_names):
    raise SystemExit("TiTO layer re-baseline reused the historical KV prefix")
  if env.get("CANON_VLLM_ENABLE_PREFIX_CACHING") != (
      "1" if arm == "on" else "0"
  ):
    raise SystemExit("rendered APC arm drifted")
  if document["metadata"]["labels"].get(
      "canon.zero-tim/m15-token-continuity"
  ) != "exact":
    raise SystemExit("TiTO render label drifted")
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
    "schema": "m15-e0v-tito-layer-render-v1",
    "source_commit": source,
    "run_id": run_id,
    "attempt20_e0u_incident_manifest_sha256": incident_sha,
    "program_identity": "m15-apc-debug-exact-tito-layer-v1",
    "program_identity_changed": True,
    "historical_1226_prefix_reused": False,
    "historical_first_red_inherited": False,
    "observer": "layer",
    "durability_profile": "m15-wide-v1",
    "rounds": 3,
    "zero_backward": True,
    "zero_optimizer_commit": True,
    "b_full_reset_immutable": True,
    "control_and_treatment_differ_only_at_apc": True,
    "tito_exact_both_arms": True,
    "pinned_exact_image_required": True,
    "launch_authorized": False,
    "target_executed": False,
    "remote_mutation": False,
    "arms": sorted(rows, key=lambda row: row["arm"]),
}
(root / "RUN_CONTRACT.json").write_text(
    json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
names = sorted(path.name for path in root.iterdir() if path.is_file())
(root / "SHA256SUMS").write_text(
    "".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names if name != "SHA256SUMS"
    ),
    encoding="ascii",
)
PY

(cd "$output" && sha256sum -c SHA256SUMS --quiet)
echo "[M15.E0V.PREPARE] RENDER_PASS source=$source_commit identity=exact-tito-layer rounds=3 matched_pair=1 historical_prefix_reused=0 output=$output"
echo "[M15.E0V.PREPARE] TARGET_NOT_RUN pinned_exact_image=required launch_approval=required gcs=0 kubernetes=0 tpu=0"
