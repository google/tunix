#!/usr/bin/env bash
# Exercise the salvage-first return with no root or round objects available.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
render="$scratch/render"
output="$scratch/output"
source="$(git -C "$repo" rev-parse HEAD)"

python3 "$canon/cluster/render_v1_apc_m15_target_debug.py" \
  --source-commit "$source" --run-id kv3return \
  --observer kv3 --output-dir "$render" >/dev/null
python3 - "$render" "$source" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import yaml

root = Path(sys.argv[1])
source = sys.argv[2]
arms = []
for path in sorted(root.glob("*.yaml")):
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  container = next(
      value for value in
      document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"]
      if value["name"] == "jax-tpu"
  )
  env = {value["name"]: str(value["value"])
         for value in container["env"] if "value" in value}
  arms.append({
      "arm": env["CANON_APC_M15_TARGET_DEBUG"],
      "jobset": document["metadata"]["name"],
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "yaml": path.name,
  })
contract = {
    "arms": arms,
    "durability": {
        "profile": "m15-e0-kv-v1",
        "root_collection_required_for_round_salvage": False,
    },
    "launch_authorized": False,
    "observer": {"layer": 0, "target_aliases_per_round": 8},
    "rounds": 3,
    "schema": "m15-attempt19-e0-kv3-render-v1",
    "source_commit": source,
}
(root / "RUN_CONTRACT.json").write_text(
    json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8"
)
names = sorted(path.name for path in root.glob("*.yaml")) + [
    "RUN_CONTRACT.json"
]
(root / "SHA256SUMS").write_text("".join(
    f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
    for name in names
), encoding="ascii")
PY

mkdir -p "$scratch/bin" "$scratch/gcs"
cp "$canon/tests/p38_serving/fake_gcloud.sh" "$scratch/bin/gcloud"
chmod +x "$scratch/bin/gcloud"
export PATH="$scratch/bin:$PATH"
export FAKE_GCS_ROOT="$scratch/gcs"
return_rc=0
bash "$script_dir/run_m15_attempt19_e0_kv3_gcs_return.sh" \
  "$render" "$output" "$scratch" >"$scratch/return.log" 2>&1 || return_rc=$?
test "$return_rc" -eq 3
(cd "$output" && sha256sum -c SHA256SUMS --quiet)
python3 - "$output/E0_KV3_RETURN.json" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["schema"] == "m15-attempt19-e0-kv3-return-v1", value
assert value["status"] == "ROUND_EVIDENCE_PARTIAL", value
assert value["round_salvage_complete"] is False, value
assert value["terminal_pair_complete"] is False, value
assert value["target_executed"] is False, value
assert value["remote_mutation"] is False, value
PY
test -z "$(find "$FAKE_GCS_ROOT" -type f -print -quit)"
grep -q 'INCONCLUSIVE status=ROUND_EVIDENCE_PARTIAL' "$scratch/return.log"

# Populate only the compact registered objects and exercise the terminal path.
python3 - "$render" "$FAKE_GCS_ROOT" "$source" "$canon" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

render = Path(sys.argv[1])
gcs = Path(sys.argv[2])
source = sys.argv[3]
canon = Path(sys.argv[4])


def sha256(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


for yaml_path in sorted(render.glob("*.yaml")):
  document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
  container = next(
      value for value in
      document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"]
      if value["name"] == "jax-tpu"
  )
  env = {value["name"]: str(value["value"])
         for value in container["env"] if "value" in value}
  arm = env["CANON_APC_M15_TARGET_DEBUG"]
  remote = gcs / env["CANON_P38_GCS_PREFIX"][5:]
  remote.mkdir(parents=True)
  alignments = []
  for round_index in range(3):
    directory = remote / "rounds" / f"{round_index:06d}"
    (directory / "classifier-input").mkdir(parents=True)
    red = arm == "on"
    outcome = (
        "live_kv_fingerprint_differs_on_red_row"
        if red else "observer_pairs_valid_red_join_pending"
    )
    round_input = {
        "a_b_differing_bytes": 7 if red else 0,
        "a_b_differing_elements": 3 if red else 0,
        "arm": arm,
        "b_c_differing_bytes": 0,
        "b_c_differing_elements": 0,
        "diagnostic_round": round_index,
        "expected_source_commit": source,
        "kv_pairs": 8,
        "kv_records": 16,
        "runtime_source_commit": source,
        "schema": "m15-e0-kv-round-input-v1",
    }
    classification = {
        "classification": outcome,
        "comparisons": [
            {"diagnostic_round": round_index} for _ in range(8)
        ],
        "pairs": 8,
        "records": 16,
        "schema": "p38-live-kv-classification-v2",
        "status": "PASS",
    }
    if red:
      classification["source_request_binding"] = {
          "status": "UNIQUE_FUTURE_PREFIX_BINDING"
      }
    input_path = directory / "ROUND_INPUT.json"
    classification_path = directory / "kv-observer-classification.json"
    input_path.write_text(json.dumps(round_input), encoding="utf-8")
    classification_path.write_text(json.dumps(classification), encoding="utf-8")
    checkpoint = {
        "a_b_differing_bytes": round_input["a_b_differing_bytes"],
        "arm": arm,
        "diagnostic_round": round_index,
        "kv_pairs": 8,
        "kv_records": 16,
        "runtime_source_commit": source,
        "schema": "m15-e0-kv-classifier-input-receipt-v1",
        "source_commit": source,
        "status": "uploaded-readback-verified-before-classification",
    }
    checkpoint_path = directory / "classifier-input/CLASSIFIER_INPUT_RECEIPT.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )
    # The runtime aggregator reads the local sealed-round layout.  Keep this
    # temporary sibling only until aggregation; the registered remote layout
    # exposes the receipt below classifier-input/.
    (directory / "CLASSIFIER_INPUT_RECEIPT.json").write_bytes(
        checkpoint_path.read_bytes()
    )
    completion = {
        "arm": arm,
        "classification_sha256": sha256(classification_path),
        "classifier_input_receipt_sha256": sha256(checkpoint_path),
        "diagnostic_round": round_index,
        "round_input_sha256": sha256(input_path),
        "runtime_source_commit": source,
        "schema": "m15-e0-kv-round-completion-v1",
        "source_commit": source,
        "status": "sealed-uploaded-readback-verified",
    }
    (directory / "ROUND_COMPLETE.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    alignments.append({
        "boundaries": {
            "S_decode_vs_S_prefill": {
                "differing_bytes": 7 if red else 0,
                "differing_elements": 3 if red else 0,
            },
            "S_prefill_vs_T_old": {
                "differing_bytes": 0,
                "differing_elements": 0,
            },
        },
        "diagnostic_round": round_index,
    })
    for source_name, target_name in (
        ("ROUND_INPUT.json", f"ROUND_INPUT.{round_index:06d}.json"),
        ("kv-observer-classification.json",
         f"kv-observer-classification.{round_index:06d}.json"),
        ("ROUND_COMPLETE.json", f"ROUND_COMPLETE.{round_index:06d}.json"),
    ):
      (remote / target_name).write_bytes((directory / source_name).read_bytes())
    (remote / f"CLASSIFIER_INPUT_RECEIPT.{round_index:06d}.json").write_bytes(
        (directory / "classifier-input/CLASSIFIER_INPUT_RECEIPT.json").read_bytes()
    )
  subprocess.run([
      sys.executable,
      str(canon / "tasks/v1-apc-m15-target-debug/scripts/aggregate_m15_e0_kv_rounds.py"),
      "--root", str(remote / "rounds"), "--arm", arm, "--rounds", "3",
      "--expected-source", source,
      "--output", str(remote / "kv-observer-classification.json"),
  ], check=True, stdout=subprocess.DEVNULL)
  for round_index in range(3):
    (remote / "rounds" / f"{round_index:06d}" /
     "CLASSIFIER_INPUT_RECEIPT.json").unlink()
  (remote / "pre-alignment.jsonl").write_text(
      "".join(json.dumps(value) + "\n" for value in alignments),
      encoding="utf-8",
  )
  b_marker = (
      "[CANON_APC_M15_B_CONTRACT] reset_prefix_cache=True "
      "all_num_cached_tokens_zero=True\n"
  )
  (remote / "run.log").write_text(
      f"[sync] HEAD={source}\n"
      + b_marker * 3
      + f"[CANON_APC_M15_TARGET_CONTRACT] arm={arm} topology=DP8xTP8 workload=m15/main backward=0 optimizer_commits=0\n"
      + "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0\n",
      encoding="utf-8",
  )
  (remote / "PREFLIGHT.json").write_text(json.dumps({
      "prefix": env["CANON_P38_GCS_PREFIX"],
      "runtime_source_commit": source,
      "schema": "canon-p38-gcs-preflight-v1",
      "source_commit": source,
      "source_verified": True,
      "status": "writable-and-source-verified",
  }), encoding="utf-8")
  (remote / "COLLECTED.json").write_text(json.dumps({
      "arm": arm,
      "diagnostic_rounds": 3,
      "prefix": env["CANON_P38_GCS_PREFIX"],
      "runtime_source_commit": source,
      "schema": "m15-e0-kv-gcs-collection-v1",
      "source_commit": source,
      "status": "collected-from-three-sealed-kv-rounds",
  }), encoding="utf-8")
  names = [
      "run.log", "pre-alignment.jsonl", "kv-observer-classification.json",
  ]
  for round_index in range(3):
    names.extend([
        f"ROUND_INPUT.{round_index:06d}.json",
        f"kv-observer-classification.{round_index:06d}.json",
        f"CLASSIFIER_INPUT_RECEIPT.{round_index:06d}.json",
        f"ROUND_COMPLETE.{round_index:06d}.json",
    ])
  (remote / "SHA256SUMS").write_text("".join(
      f"{sha256(remote / name)}  {name}\n" for name in names
  ), encoding="ascii")
  (remote / "COMPLETE.json").write_text(json.dumps({
      "manifest_sha256": sha256(remote / "SHA256SUMS"),
      "prefix": env["CANON_P38_GCS_PREFIX"],
      "runtime_source_commit": source,
      "schema": "canon-p38-gcs-completion-v1",
      "source_commit": source,
      "status": "postflight-accepted",
  }), encoding="utf-8")
PY

full_output="$scratch/full-output"
bash "$script_dir/run_m15_attempt19_e0_kv3_gcs_return.sh" \
  "$render" "$full_output" "$scratch" >"$scratch/full-return.log" 2>&1
(cd "$full_output" && sha256sum -c SHA256SUMS --quiet)
python3 - "$full_output/E0_KV3_RETURN.json" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["status"] == "LIVE_KV_FINGERPRINT_DIFFERS_3_OF_3", value
assert value["round_salvage_complete"] is True, value
assert value["terminal_pair_complete"] is True, value
assert value["target_executed"] is True, value
assert value["remote_mutation"] is False, value
PY
grep -q 'COMPLETE status=LIVE_KV_FINGERPRINT_DIFFERS_3_OF_3' \
  "$scratch/full-return.log"
echo "[M15.E0.KV3.RETURN] SALVAGE_FIRST_TEST_PASS missing_root=preserved partial_exit=3 terminal_pair=PASS gcs_write=0"
