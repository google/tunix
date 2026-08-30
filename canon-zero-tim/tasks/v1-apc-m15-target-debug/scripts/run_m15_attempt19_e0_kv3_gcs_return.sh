#!/usr/bin/env bash
# Salvage three-round E0 KV receipts first; root collection is optional evidence.
set -euo pipefail

render_dir="${1:?usage: run_m15_attempt19_e0_kv3_gcs_return.sh <verified-render-dir> <new-output-dir> [scratch-parent]}"
output="${2:?usage: run_m15_attempt19_e0_kv3_gcs_return.sh <verified-render-dir> <new-output-dir> [scratch-parent]}"
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
  echo "[M15.E0.KV3.RETURN] REFUSING gcloud or gsutil is required" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" m15-e0-kv3-return.XXXXXX)"
terminal_complete=0
cleanup() {
  if [ "$terminal_complete" -eq 1 ]; then
    rm -rf -- "$scratch"
  else
    echo "[M15.E0.KV3.RETURN] scratch_preserved=$scratch" >&2
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
    contract.get("schema") == "m15-attempt19-e0-kv3-render-v1"
    and contract.get("rounds") == 3
    and contract.get("launch_authorized") is False
    and contract.get("observer", {}).get("layer") == 0
    and contract.get("observer", {}).get("target_aliases_per_round") == 8
    and contract.get("durability", {}).get("profile") == "m15-e0-kv-v1"
    and contract.get("durability", {}).get(
        "root_collection_required_for_round_salvage") is False
):
  raise SystemExit("render contract does not describe the E0 KV3 pair")
seen = set()
for path in sorted(root.glob("jobset-v1-apc-m15-*-kv3.yaml")):
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  containers = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"]
  container = next(item for item in containers if item["name"] == "jax-tpu")
  env = {item["name"]: str(item["value"])
         for item in container["env"] if "value" in item}
  arm = env.get("CANON_APC_M15_TARGET_DEBUG", "")
  source = env.get("CANON_EXPECT_COMMIT", "")
  remote = env.get("CANON_P38_GCS_PREFIX", "")
  if arm not in ("off", "on") or arm in seen:
    raise SystemExit("rendered E0 KV3 pair has invalid arms")
  if not re.fullmatch(r"[0-9a-f]{40}", source):
    raise SystemExit("rendered E0 KV3 source is invalid")
  if not remote.startswith("gs://") or not remote.endswith("/attempt-0"):
    raise SystemExit("rendered E0 KV3 evidence root is invalid")
  if not (
      env.get("CANON_P38_DURABILITY_PROFILE") == "m15-e0-kv-v1"
      and env.get("CANON_P38_DIAGNOSTIC_ROUNDS") == "3"
      and env.get("CANON_P38_KV_OBSERVER_LAYER") == "0"
  ):
    raise SystemExit("rendered E0 KV3 durability selector drifted")
  seen.add(arm)
  print(f"{arm}\t{source}\t{remote}")
if seen != {"off", "on"}:
  raise SystemExit("rendered E0 KV3 pair is incomplete")
PY
)
[ "${#arm_rows[@]}" -eq 2 ] || {
  echo "[M15.E0.KV3.RETURN] REFUSING rendered pair did not resolve" >&2
  exit 2
}

source_commit=""
for row in "${arm_rows[@]}"; do
  IFS=$'\t' read -r arm source remote <<< "$row"
  if [ -z "$source_commit" ]; then
    source_commit="$source"
  elif [ "$source_commit" != "$source" ]; then
    echo "[M15.E0.KV3.RETURN] REFUSING paired source commits differ" >&2
    exit 2
  fi
  arm_dir="$scratch/$arm"
  mkdir -m 700 "$arm_dir" "$arm_dir/rounds" "$arm_dir/root"

  # Salvage-first: do not make COLLECTED.json or COMPLETE.json a prerequisite
  # for downloading an already ACKed round.
  for round_index in 0 1 2; do
    printf -v round_text '%06d' "$round_index"
    round_dir="$arm_dir/rounds/$round_text"
    mkdir -m 700 "$round_dir"
    for spec in \
        "ROUND_INPUT.json:ROUND_INPUT.json" \
        "kv-observer-classification.json:kv-observer-classification.json" \
        "ROUND_COMPLETE.json:ROUND_COMPLETE.json" \
        "classifier-input/CLASSIFIER_INPUT_RECEIPT.json:CLASSIFIER_INPUT_RECEIPT.json"; do
      remote_name="${spec%%:*}"
      local_name="${spec#*:}"
      gcs_cp "$remote/rounds/$round_text/$remote_name" \
        "$round_dir/$local_name" || true
    done
  done

  root_names=(
    PREFLIGHT.json COLLECTED.json COMPLETE.json SHA256SUMS
    run.log pre-alignment.jsonl kv-observer-classification.json
  )
  for round_index in 0 1 2; do
    printf -v round_text '%06d' "$round_index"
    root_names+=(
      "ROUND_INPUT.$round_text.json"
      "kv-observer-classification.$round_text.json"
      "CLASSIFIER_INPUT_RECEIPT.$round_text.json"
      "ROUND_COMPLETE.$round_text.json"
    )
  done
  for name in "${root_names[@]}"; do
    gcs_cp "$remote/$name" "$arm_dir/root/$name" || true
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


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def expected_arm_status(arm: str, rows: list[dict]) -> str:
  outcomes = {row["classification"] for row in rows}
  if arm == "off":
    if any(row["a_b_differing_bytes"] != 0 for row in rows):
      return "CONTROL_RED_STOP"
    if outcomes != {"observer_pairs_valid_red_join_pending"}:
      return "CONTROL_CLASSIFIER_DRIFT"
    return "CONTROL_EXACT_3_OF_3"
  if outcomes == {"observer_pairs_valid_red_join_pending"}:
    return "TARGET_NON_REPRODUCTION_3_OF_3"
  if outcomes == {"live_kv_fingerprint_equal_on_red_row"}:
    return "LIVE_KV_FINGERPRINT_EQUAL_3_OF_3"
  if outcomes == {"live_kv_fingerprint_differs_on_red_row"}:
    return "LIVE_KV_FINGERPRINT_DIFFERS_3_OF_3"
  return "UNSTABLE_OR_INCOMPLETE_3_ROUND_TREATMENT"


arms = {}
terminal_pair = True
for arm in ("off", "on"):
  arm_root = scratch / arm
  completed = []
  missing = []
  round_rows = []
  for round_index in range(3):
    directory = arm_root / "rounds" / f"{round_index:06d}"
    required = {
        "input": directory / "ROUND_INPUT.json",
        "classification": directory / "kv-observer-classification.json",
        "completion": directory / "ROUND_COMPLETE.json",
        "classifier_input_receipt": directory / "CLASSIFIER_INPUT_RECEIPT.json",
    }
    absent = sorted(name for name, path in required.items()
                    if not path.is_file() or path.stat().st_size == 0)
    if absent:
      missing.append({"diagnostic_round": round_index, "missing": absent})
      continue
    round_input = load(required["input"])
    classification = load(required["classification"])
    completion = load(required["completion"])
    checkpoint = load(required["classifier_input_receipt"])
    if not (
        round_input.get("schema") == "m15-e0-kv-round-input-v1"
        and round_input.get("arm") == arm
        and round_input.get("diagnostic_round") == round_index
        and round_input.get("expected_source_commit") == source
        and round_input.get("runtime_source_commit") == source
        and round_input.get("kv_records") == 16
        and round_input.get("kv_pairs") == 8
        and round_input.get("b_c_differing_bytes") == 0
        and round_input.get("b_c_differing_elements") == 0
    ):
      raise SystemExit(f"{arm} round {round_index} input receipt drifted")
    if not (
        classification.get("schema") == "p38-live-kv-classification-v2"
        and classification.get("status") == "PASS"
        and classification.get("records") == 16
        and classification.get("pairs") == 8
        and {row.get("diagnostic_round")
             for row in classification.get("comparisons", ())}
            == {round_index}
    ):
      raise SystemExit(f"{arm} round {round_index} classifier drifted")
    if not (
        checkpoint.get("schema") == "m15-e0-kv-classifier-input-receipt-v1"
        and checkpoint.get("status")
            == "uploaded-readback-verified-before-classification"
        and checkpoint.get("arm") == arm
        and checkpoint.get("diagnostic_round") == round_index
        and checkpoint.get("source_commit") == source
        and checkpoint.get("runtime_source_commit") == source
        and checkpoint.get("kv_records") == 16
        and checkpoint.get("kv_pairs") == 8
        and checkpoint.get("a_b_differing_bytes")
            == round_input.get("a_b_differing_bytes")
    ):
      raise SystemExit(f"{arm} round {round_index} classifier checkpoint drifted")
    if not (
        completion.get("schema") == "m15-e0-kv-round-completion-v1"
        and completion.get("status") == "sealed-uploaded-readback-verified"
        and completion.get("arm") == arm
        and completion.get("diagnostic_round") == round_index
        and completion.get("source_commit") == source
        and completion.get("runtime_source_commit") == source
        and completion.get("round_input_sha256") == sha256(required["input"])
        and completion.get("classification_sha256")
            == sha256(required["classification"])
        and completion.get("classifier_input_receipt_sha256")
            == sha256(required["classifier_input_receipt"])
    ):
      raise SystemExit(f"{arm} round {round_index} completion drifted")
    outcome = classification.get("classification")
    red = int(round_input["a_b_differing_bytes"]) > 0
    if arm == "off" and red:
      raise SystemExit(f"APC-off control is red at round {round_index}")
    if red and not (
        outcome in {
            "live_kv_fingerprint_equal_on_red_row",
            "live_kv_fingerprint_differs_on_red_row",
        }
        and classification.get("source_request_binding", {}).get("status")
            == "UNIQUE_FUTURE_PREFIX_BINDING"
    ):
      raise SystemExit(f"{arm} round {round_index} red row is not bound")
    if not red and outcome != "observer_pairs_valid_red_join_pending":
      raise SystemExit(f"{arm} round {round_index} exact row classifier drifted")
    completed.append(round_index)
    round_rows.append({
        "a_b_differing_bytes": round_input["a_b_differing_bytes"],
        "a_b_differing_elements": round_input["a_b_differing_elements"],
        "b_c_differing_bytes": 0,
        "classification": outcome,
        "classification_sha256": sha256(required["classification"]),
        "diagnostic_round": round_index,
        "round_completion_sha256": sha256(required["completion"]),
    })

  arm_status = (
      expected_arm_status(arm, round_rows)
      if completed == [0, 1, 2]
      else "ROUND_EVIDENCE_PARTIAL"
  )
  root = arm_root / "root"
  root_required = [
      root / "PREFLIGHT.json", root / "COLLECTED.json",
      root / "COMPLETE.json", root / "SHA256SUMS",
      root / "run.log", root / "pre-alignment.jsonl",
      root / "kv-observer-classification.json",
  ]
  root_complete = all(path.is_file() and path.stat().st_size > 0
                      for path in root_required)
  if root_complete:
    manifest = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
      digest, name = line.split(maxsplit=1)
      manifest[name] = digest
    for name, digest in manifest.items():
      path = root / name
      if not path.is_file() or sha256(path) != digest:
        raise SystemExit(f"{arm} root manifest failed: {name}")
    expected_manifest_names = {
        "run.log", "pre-alignment.jsonl", "kv-observer-classification.json",
    }
    for round_index in range(3):
      round_text = f"{round_index:06d}"
      expected_manifest_names.update({
          f"ROUND_INPUT.{round_text}.json",
          f"kv-observer-classification.{round_text}.json",
          f"CLASSIFIER_INPUT_RECEIPT.{round_text}.json",
          f"ROUND_COMPLETE.{round_text}.json",
      })
    if set(manifest) != expected_manifest_names:
      raise SystemExit(f"{arm} root manifest membership drifted")
    preflight = load(root / "PREFLIGHT.json")
    collected = load(root / "COLLECTED.json")
    complete = load(root / "COMPLETE.json")
    aggregate = load(root / "kv-observer-classification.json")
    if not (
        preflight.get("schema") == "canon-p38-gcs-preflight-v1"
        and preflight.get("status") == "writable-and-source-verified"
        and preflight.get("source_verified") is True
        and preflight.get("source_commit") == source
        and preflight.get("runtime_source_commit") == source
        and collected.get("schema") == "m15-e0-kv-gcs-collection-v1"
        and collected.get("status") == "collected-from-three-sealed-kv-rounds"
        and collected.get("arm") == arm
        and collected.get("diagnostic_rounds") == 3
        and collected.get("source_commit") == source
        and collected.get("runtime_source_commit") == source
        and complete.get("schema") == "canon-p38-gcs-completion-v1"
        and complete.get("status") == "postflight-accepted"
        and complete.get("source_commit") == source
        and complete.get("runtime_source_commit") == source
        and complete.get("manifest_sha256") == sha256(root / "SHA256SUMS")
        and preflight.get("prefix") == collected.get("prefix")
        and collected.get("prefix") == complete.get("prefix")
        and aggregate.get("schema") == "m15-e0-kv-three-round-arm-v1"
        and aggregate.get("arm") == arm
        and aggregate.get("diagnostic_rounds") == 3
        and aggregate.get("runtime_source_commit") == source
        and aggregate.get("status") == arm_status
    ):
      raise SystemExit(f"{arm} root terminal contract drifted")
    for round_index in range(3):
      round_text = f"{round_index:06d}"
      direct = arm_root / "rounds" / round_text
      for root_name, direct_name in (
          (f"ROUND_INPUT.{round_text}.json", "ROUND_INPUT.json"),
          (f"kv-observer-classification.{round_text}.json",
           "kv-observer-classification.json"),
          (f"CLASSIFIER_INPUT_RECEIPT.{round_text}.json",
           "CLASSIFIER_INPUT_RECEIPT.json"),
          (f"ROUND_COMPLETE.{round_text}.json", "ROUND_COMPLETE.json"),
      ):
        if sha256(root / root_name) != sha256(direct / direct_name):
          raise SystemExit(
              f"{arm} root/direct round copy drifted: {root_name}")
    raw = (root / "run.log").read_text(encoding="utf-8", errors="replace")
    runtime_marker = f"[sync] HEAD={source}"
    b_marker = (
        "[CAN" "ON_APC_M15_B_CONTRACT] reset_prefix_cache=True "
        "all_num_cached_tokens_zero=True"
    )
    target_marker = (
        f"[CAN" f"ON_APC_M15_TARGET_CONTRACT] arm={arm} topology=DP8xTP8 "
        "workload=m15/main backward=0 optimizer_commits=0"
    )
    controlled_exit = (
        "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 "
        "optimizer_commits=0"
    )
    if not (
        raw.count(runtime_marker) == 1
        and raw.count(b_marker) >= 3
        and "all_num_cached_tokens_zero=False" not in raw
        and raw.count(target_marker) == 1
        and raw.count(controlled_exit) == 1
        and "OPTIMIZER_COMMIT" not in raw
    ):
      raise SystemExit(f"{arm} runtime reset/zero-commit receipt is incomplete")
    alignments = [
        json.loads(line)
        for line in (root / "pre-alignment.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(alignments) != 3 or {
        row.get("diagnostic_round") for row in alignments
    } != {0, 1, 2}:
      raise SystemExit(f"{arm} root alignment rounds are incomplete")
    direct_by_round = {row["diagnostic_round"]: row for row in round_rows}
    for alignment in alignments:
      round_index = alignment["diagnostic_round"]
      a_b = alignment.get("boundaries", {}).get(
          "S_decode_vs_S_prefill", {})
      b_c = alignment.get("boundaries", {}).get(
          "S_prefill_vs_T_old", {})
      if not (
          b_c.get("differing_bytes") == 0
          and b_c.get("differing_elements") == 0
          and a_b.get("differing_bytes")
              == direct_by_round[round_index]["a_b_differing_bytes"]
          and a_b.get("differing_elements")
              == direct_by_round[round_index]["a_b_differing_elements"]
      ):
        raise SystemExit(f"{arm} root alignment boundary drifted")
  else:
    terminal_pair = False
  arms[arm] = {
      "completed_rounds": completed,
      "missing": missing,
      "rounds": round_rows,
      "round_status": arm_status,
      "root_terminal_complete": root_complete,
  }

if arms["off"]["round_status"] not in {
    "CONTROL_EXACT_3_OF_3", "ROUND_EVIDENCE_PARTIAL"
}:
  pair_status = "CONTROL_RED_STOP"
elif (arms["off"]["round_status"] == "ROUND_EVIDENCE_PARTIAL"
      or arms["on"]["round_status"] == "ROUND_EVIDENCE_PARTIAL"):
  pair_status = "ROUND_EVIDENCE_PARTIAL"
elif not terminal_pair:
  pair_status = "ROUNDS_RECOVERED_ROOT_INCOMPLETE"
else:
  pair_status = arms["on"]["round_status"]

output.mkdir(mode=0o700)
report = {
    "arms": arms,
    "claim_ceiling": (
        "Each KV result is a diagnostic fingerprint over a uniquely bound "
        "request, not a collision-free proof of all KV bytes."
    ),
    "numerical_repair_authorized": False,
    "remote_mutation": False,
    "round_salvage_complete": all(
        value["completed_rounds"] == [0, 1, 2] for value in arms.values()
    ),
    "schema": "m15-attempt19-e0-kv3-return-v1",
    "source_commit": source,
    "status": pair_status,
    "target_executed": any(value["completed_rounds"] for value in arms.values()),
    "terminal_pair_complete": terminal_pair,
}
(output / "E0_KV3_RETURN.json").write_text(
    json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
(output / "SHA256SUMS").write_text(
    f"{sha256(output / 'E0_KV3_RETURN.json')}  E0_KV3_RETURN.json\n",
    encoding="ascii",
)
print(
    f"M15_E0_KV3_RETURN status={pair_status} "
    f"off_rounds={len(arms['off']['completed_rounds'])} "
    f"on_rounds={len(arms['on']['completed_rounds'])} "
    f"root_terminal={int(terminal_pair)}"
)
PY

(cd "$output" && sha256sum -c SHA256SUMS --quiet)
manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
read -r status is_terminal < <(python3 - "$output/E0_KV3_RETURN.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["status"], int(value["terminal_pair_complete"]))
PY
)
if [ "$is_terminal" -eq 1 ]; then
  terminal_complete=1
  echo "[M15.E0.KV3.RETURN] COMPLETE status=$status manifest_sha256=$manifest_sha output=$output"
  echo "[M15.E0.KV3.RETURN] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0"
  exit 0
fi
echo "[M15.E0.KV3.RETURN] INCONCLUSIVE status=$status output_preserved=$output manifest_sha256=$manifest_sha" >&2
echo "[M15.E0.KV3.RETURN] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0" >&2
exit 3
