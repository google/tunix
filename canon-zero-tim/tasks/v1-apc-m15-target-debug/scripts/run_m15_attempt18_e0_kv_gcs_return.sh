#!/usr/bin/env bash
# Read only the compact E0 verdict artifacts from the registered evidence roots.
set -euo pipefail

render_dir="${1:?usage: run_m15_attempt18_e0_kv_gcs_return.sh <verified-render-dir> <new-output-dir> [scratch-parent]}"
output="${2:?usage: run_m15_attempt18_e0_kv_gcs_return.sh <verified-render-dir> <new-output-dir> [scratch-parent]}"
scratch_parent="${3:-/tmp}"
test -d "$render_dir"
test -d "$scratch_parent"
test ! -e "$output"
(cd "$render_dir" && sha256sum -c SHA256SUMS --quiet)

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2" >/dev/null 2>&1; }
else
  echo "[M15.E0.KV.RETURN] REFUSING gcloud or gsutil is required" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" m15-e0-kv-return.XXXXXX)"
success=0
cleanup() {
  if [ "$success" -eq 1 ]; then
    rm -rf -- "$scratch"
  else
    echo "[M15.E0.KV.RETURN] scratch_preserved=$scratch" >&2
  fi
}
trap cleanup EXIT

mapfile -t arm_rows < <(python3 - "$render_dir" <<'PY'
from pathlib import Path
import json
import re
import sys

import yaml

root = Path(sys.argv[1])
contract = json.loads((root / "RUN_CONTRACT.json").read_text(encoding="utf-8"))
if not (
    contract.get("schema") == "m15-attempt18-e0-kv-render-v1"
    and contract.get("rounds") == 1
    and contract.get("launch_authorized") is False
    and contract.get("observer", {}).get("layer") == 0
    and contract.get("observer", {}).get("target_aliases") == 8
):
  raise SystemExit("render contract does not describe the E0 KV pair")
seen = set()
for path in sorted(root.glob("jobset-v1-apc-m15-*-kv.yaml")):
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  containers = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"]
  container = next(item for item in containers if item["name"] == "jax-tpu")
  env = {item["name"]: str(item["value"])
         for item in container["env"] if "value" in item}
  arm = env.get("CANON_APC_M15_TARGET_DEBUG", "")
  source = env.get("CANON_EXPECT_COMMIT", "")
  remote = env.get("CANON_P38_GCS_PREFIX", "")
  if arm not in ("off", "on") or arm in seen:
    raise SystemExit("rendered E0 pair has invalid arms")
  if not re.fullmatch(r"[0-9a-f]{40}", source):
    raise SystemExit("rendered E0 source is invalid")
  if not remote.startswith("gs://") or not remote.endswith("/attempt-0"):
    raise SystemExit("rendered E0 evidence root is invalid")
  if env.get("CANON_P38_KV_OBSERVER_LAYER") != "0":
    raise SystemExit("rendered E0 layer selector drifted")
  seen.add(arm)
  print(f"{arm}\t{source}\t{remote}")
if seen != {"off", "on"}:
  raise SystemExit("rendered E0 pair is incomplete")
PY
)
[ "${#arm_rows[@]}" -eq 2 ] || {
  echo "[M15.E0.KV.RETURN] REFUSING rendered pair did not resolve" >&2
  exit 2
}

source_commit=""
for row in "${arm_rows[@]}"; do
  IFS=$'\t' read -r arm source remote <<< "$row"
  if [ -z "$source_commit" ]; then
    source_commit="$source"
  elif [ "$source_commit" != "$source" ]; then
    echo "[M15.E0.KV.RETURN] REFUSING paired source commits differ" >&2
    exit 2
  fi
  arm_dir="$scratch/$arm"
  mkdir -m 700 "$arm_dir"
  for name in PREFLIGHT.json COLLECTED.json COMPLETE.json SHA256SUMS \
      pre-alignment.jsonl serving-classification.json \
      kv-observer-classification.json; do
    gcs_cp "$remote/$name" "$arm_dir/$name" || {
      echo "[M15.E0.KV.RETURN] INCONCLUSIVE arm=$arm missing=$name" >&2
      exit 3
    }
    test -s "$arm_dir/$name"
  done
done

python3 - "$scratch" "$output" "$source_commit" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

scratch = Path(sys.argv[1])
output = Path(sys.argv[2])
source = sys.argv[3]

def sha256(path):
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()

def manifest(path):
  entries = {}
  for line in path.read_text(encoding="ascii").splitlines():
    digest, name = line.split()
    entries[name] = digest
  return entries

arms = {}
for arm in ("off", "on"):
  root = scratch / arm
  entries = manifest(root / "SHA256SUMS")
  for name in (
      "pre-alignment.jsonl", "serving-classification.json",
      "kv-observer-classification.json",
  ):
    if entries.get(name) != sha256(root / name):
      raise SystemExit(f"{arm} compact artifact failed remote manifest: {name}")
  preflight = json.loads((root / "PREFLIGHT.json").read_text())
  collected = json.loads((root / "COLLECTED.json").read_text())
  complete = json.loads((root / "COMPLETE.json").read_text())
  if not (
      preflight.get("schema") == "canon-p38-gcs-preflight-v1"
      and preflight.get("status") == "writable-and-source-verified"
      and preflight.get("source_verified") is True
      and collected.get("schema") == "canon-p38-gcs-collection-v1"
      and collected.get("status") == "collected"
      and complete.get("schema") == "canon-p38-gcs-completion-v1"
      and preflight.get("source_commit") == source
      and collected.get("source_commit") == source
      and complete.get("source_commit") == source
      and preflight.get("runtime_source_commit") == source
      and collected.get("runtime_source_commit") == source
      and complete.get("runtime_source_commit") == source
      and preflight.get("prefix") == collected.get("prefix")
      and collected.get("prefix") == complete.get("prefix")
      and complete.get("status") == "postflight-accepted"
      and complete.get("manifest_sha256") == sha256(root / "SHA256SUMS")
  ):
    raise SystemExit(f"{arm} terminal marker identity drifted")
  serving = json.loads(
      (root / "serving-classification.json").read_text()
  )
  if not (
      serving.get("schema_version") == 1
      and serving.get("verdict") == "PASS"
      and serving.get("scope") == "p38-serving-capture"
      and serving.get("source_commit") == source
  ):
    raise SystemExit(f"{arm} serving classifier is not source-bound PASS")
  records = [json.loads(line) for line in
             (root / "pre-alignment.jsonl").read_text().splitlines()
             if line.strip()]
  if len(records) != 1:
    raise SystemExit(f"{arm} does not have one alignment round")
  record = records[0]
  a_b = record.get("boundaries", {}).get("S_decode_vs_S_prefill", {})
  b_c = record.get("boundaries", {}).get("S_prefill_vs_T_old", {})
  if not (
      isinstance(record.get("N_action"), int)
      and record.get("N_action") > 0
      and isinstance(a_b.get("differing_bytes"), int)
      and a_b.get("differing_bytes") >= 0
      and isinstance(a_b.get("differing_elements"), int)
      and a_b.get("differing_elements") >= 0
      and b_c.get("differing_bytes") == 0
      and b_c.get("differing_elements") == 0
  ):
    raise SystemExit(f"{arm} alignment contract is invalid or B-C is red")
  kv = json.loads((root / "kv-observer-classification.json").read_text())
  if not (kv.get("status") == "PASS" and kv.get("pairs") == 8):
    raise SystemExit(f"{arm} KV classifier is incomplete")
  kv_all_pairs_equal = all(
      comparison.get("fingerprint_equal") is True
      for comparison in kv.get("comparisons", ())
  ) and len(kv.get("comparisons", ())) == 8
  arms[arm] = {
      "a_b_differing_bytes": a_b.get("differing_bytes"),
      "a_b_differing_elements": a_b.get("differing_elements"),
      "b_c_differing_bytes": 0,
      "n_action": record.get("N_action"),
      "kv_classification": kv.get("classification"),
      "kv_all_pairs_equal": kv_all_pairs_equal,
      "source_request_binding": kv.get("source_request_binding"),
      "root_manifest_sha256": sha256(root / "SHA256SUMS"),
      "kv_classification_sha256": sha256(
          root / "kv-observer-classification.json"
      ),
  }

if (arms["off"]["a_b_differing_bytes"] != 0
    or not arms["off"]["kv_all_pairs_equal"]):
  status = "CONTROL_RED_STOP"
elif arms["on"]["a_b_differing_bytes"] == 0:
  status = "TARGET_NON_REPRODUCTION"
else:
  binding = arms["on"]["source_request_binding"] or {}
  if binding.get("status") != "UNIQUE_FUTURE_PREFIX_BINDING":
    raise SystemExit("treatment red lacks a unique source-request binding")
  classification = arms["on"]["kv_classification"]
  if classification == "live_kv_fingerprint_differs_on_red_row":
    status = "LIVE_KV_FINGERPRINT_DIFFERS"
  elif classification == "live_kv_fingerprint_equal_on_red_row":
    status = "LIVE_KV_FINGERPRINT_EQUAL"
  else:
    raise SystemExit("treatment red lacks a mechanism classification")

output.mkdir(mode=0o700)
for arm in ("off", "on"):
  source_path = scratch / arm / "kv-observer-classification.json"
  (output / f"{arm}.kv-observer-classification.json").write_bytes(
      source_path.read_bytes()
  )
report = {
    "schema": "m15-attempt18-e0-kv-return-v1",
    "status": status,
    "source_commit": source,
    "arms": arms,
    "target_executed": True,
    "remote_mutation": False,
    "numerical_repair_authorized": False,
    "claim_ceiling": (
        "The KV result is a diagnostic fingerprint over the uniquely bound "
        "red request, not a collision-free proof of all KV bytes."
    ),
}
(output / "E0_KV_RETURN.json").write_text(
    json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
names = sorted(path.name for path in output.iterdir())
(output / "SHA256SUMS").write_text("".join(
    f"{sha256(output / name)}  {name}\n" for name in names
), encoding="ascii")
print(
    f"M15_E0_KV_RETURN_PASS status={status} "
    f"control_a_b={arms['off']['a_b_differing_bytes']} "
    f"treatment_a_b={arms['on']['a_b_differing_bytes']} b_c=0"
)
PY

(cd "$output" && sha256sum -c SHA256SUMS --quiet)
manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/E0_KV_RETURN.json")"
success=1
echo "[M15.E0.KV.RETURN] COMPLETE status=$status manifest_sha256=$manifest_sha output=$output"
echo "[M15.E0.KV.RETURN] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0"
