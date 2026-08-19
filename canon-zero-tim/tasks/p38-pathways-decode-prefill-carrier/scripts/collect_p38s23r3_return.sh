#!/usr/bin/env bash
# Download, verify, and compact the P38s23r3 result for branch-side review.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: collect_p38s23r3_return.sh \
  --source-commit <40-hex-sha> \
  --head-log <complete-attempt-0-log> \
  --launch-dir <launch_p38s23r3-return-dir> \
  --output-dir <new-absolute-dir>
EOF
}

source_commit=""
head_log=""
launch_dir=""
output_dir=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-commit) source_commit="${2:-}"; shift 2 ;;
    --head-log) head_log="${2:-}"; shift 2 ;;
    --launch-dir) launch_dir="${2:-}"; shift 2 ;;
    --output-dir) output_dir="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[P38S23R3.RETURN] REFUSING: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[P38S23R3.RETURN] REFUSING: source commit must be 40 lowercase hex characters" >&2
  exit 2
fi
test -s "$head_log" || {
  echo "[P38S23R3.RETURN] REFUSING: complete head log is absent" >&2
  exit 2
}
test -d "$launch_dir" || {
  echo "[P38S23R3.RETURN] REFUSING: launch receipt directory is absent" >&2
  exit 2
}
case "$output_dir" in
  /*) ;;
  *) echo "[P38S23R3.RETURN] REFUSING: output dir must be absolute" >&2; exit 2 ;;
esac
case "$output_dir" in
  /|/home|/tmp) echo "[P38S23R3.RETURN] REFUSING: output target is too broad" >&2; exit 2 ;;
esac
test ! -e "$output_dir" || {
  echo "[P38S23R3.RETURN] REFUSING: output already exists: $output_dir" >&2
  exit 2
}

repo="$(git rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
archive_tool="$pkg/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_evidence_archive.py"
jobset="canon-p38-fl-stock-p38s23r3-${source_commit:0:8}"
gcs_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/$jobset/attempt-0"
if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2"; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
else
  echo "[P38S23R3.RETURN] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

mkdir -m 700 "$output_dir"
mkdir -m 700 "$output_dir/launch" "$output_dir/root" "$output_dir/rounds"
cp -- "$head_log" "$output_dir/head.full.log"
for name in source_commit.txt rendered-stock.yaml render.txt \
    semantic-preflight.txt dry-run.txt apply.txt LAUNCH_SHA256SUMS; do
  test -s "$launch_dir/$name" || {
    echo "[P38S23R3.RETURN] REFUSING: launch receipt is absent: $name" >&2
    exit 2
  }
  cp -- "$launch_dir/$name" "$output_dir/launch/$name"
done
(
  cd "$output_dir/launch"
  sha256sum -c LAUNCH_SHA256SUMS --quiet
)

gcs_cp "$gcs_root/PREFLIGHT.json" "$output_dir/root/PREFLIGHT.json"
test -s "$output_dir/root/PREFLIGHT.json"
for name in COLLECTED.json COMPLETE.json SHA256SUMS; do
  if gcs_exists "$gcs_root/$name"; then
    gcs_cp "$gcs_root/$name" "$output_dir/root/$name"
  fi
done
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
for round_index in 0 1 2; do
  printf -v sequence '%06d' "$round_index"
  destination="$output_dir/rounds/$sequence"
  mkdir -m 700 "$destination"
  for name in ROUND_ARCHIVE.tar SHA256SUMS ROUND_COMPLETE.json; do
    gcs_cp "$gcs_root/rounds/$sequence/$name" "$destination/$name"
    test -s "$destination/$name"
  done
  round_scratch="$scratch/$sequence"
  python3 "$archive_tool" extract \
    --archive "$destination/ROUND_ARCHIVE.tar" \
    --output "$round_scratch" > "$destination/archive-verify.txt"
  cmp -- "$destination/SHA256SUMS" "$round_scratch/SHA256SUMS"
  cp -- "$round_scratch/pre-alignment.jsonl" \
    "$destination/pre-alignment.jsonl"
  cp -- "$round_scratch/ROUND_INVENTORY.json" \
    "$destination/ROUND_INVENTORY.json"
  if [ -s "$round_scratch/mismatch-capsule.npz" ]; then
    cp -- "$round_scratch/mismatch-capsule.npz" \
      "$destination/mismatch-capsule.npz"
  fi
done

python3 - "$source_commit" "$output_dir" <<'PY'
import json
import hashlib
import pathlib
import re
import sys

source = sys.argv[1]
root = pathlib.Path(sys.argv[2])
log = (root / "head.full.log").read_text(encoding="utf-8", errors="replace")
if (root / "launch" / "source_commit.txt").read_text().strip() != source:
  raise SystemExit("launch source receipt drifted")
if f"[sync] HEAD={source}" not in log:
  raise SystemExit("head log does not attest the requested source commit")
if "timed out waiting for P38" in log:
  raise SystemExit("head log contains a P38 durability timeout")
if log.count("[CANON_P38] PRECHECK_ROUND_COMPLETE ") != 3:
  raise SystemExit("head log does not contain exactly three numerical rounds")
if log.count("[CANON_P38] ROUND_SEAL_ACKNOWLEDGED ") != 3:
  raise SystemExit("head log does not contain exactly three round acknowledgements")
if "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0" not in log:
  raise SystemExit("controlled diagnostic exit is absent")
for semantic_m, chunks in ((16, 1), (32, 1), (64, 1), (128, 1), (256, 1), (4096, 16)):
  pattern = rf"CANON_P38_FIXED_LM_HEAD=1 semantic_M={semantic_m}\b.*\bchunks={chunks}\b"
  if re.search(pattern, log) is None:
    raise SystemExit(f"fixed-lm-head receipt is absent: M={semantic_m}")

rounds = []
for round_index in range(3):
  sequence = f"{round_index:06d}"
  round_dir = root / "rounds" / sequence
  complete = json.loads((round_dir / "ROUND_COMPLETE.json").read_text())
  inventory = json.loads((round_dir / "ROUND_INVENTORY.json").read_text())
  records = [
      json.loads(line)
      for line in (round_dir / "pre-alignment.jsonl").read_text().splitlines()
      if line.strip()
  ]
  if complete.get("source_commit") != source:
    raise SystemExit(f"round {round_index} source commit drifted")
  if complete.get("diagnostic_round") != round_index:
    raise SystemExit(f"round {round_index} completion index drifted")
  if complete.get("status") != "sealed-and-verified":
    raise SystemExit(f"round {round_index} is not sealed")
  if complete.get("durability_profile") != "round-alignment-v1":
    raise SystemExit(f"round {round_index} durability profile drifted")
  archive = round_dir / "ROUND_ARCHIVE.tar"
  manifest = round_dir / "SHA256SUMS"
  if complete.get("archive_sha256") != hashlib.sha256(
      archive.read_bytes()).hexdigest():
    raise SystemExit(f"round {round_index} completion archive SHA drifted")
  if complete.get("manifest_sha256") != hashlib.sha256(
      manifest.read_bytes()).hexdigest():
    raise SystemExit(f"round {round_index} completion manifest SHA drifted")
  manifest_count = len([line for line in manifest.read_text().splitlines()
                        if line.strip()])
  if complete.get("logical_file_count") != manifest_count:
    raise SystemExit(f"round {round_index} logical file count drifted")
  if inventory.get("profile") != "alignment-only":
    raise SystemExit(f"round {round_index} inventory profile drifted")
  if inventory.get("diagnostic_round") != round_index:
    raise SystemExit(f"round {round_index} inventory index drifted")
  if inventory.get("pre_alignment_records") != 1:
    raise SystemExit(f"round {round_index} alignment record count drifted")
  for count_name in ("journal_records", "incident_records", "kv_records",
                     "seam_records", "tail_records", "terminal_records"):
    if inventory.get(count_name) != 0:
      raise SystemExit(f"round {round_index} observer count drifted: {count_name}")
  if len(records) != 1 or records[0].get("diagnostic_round") != round_index:
    raise SystemExit(f"round {round_index} pre-alignment scope drifted")
  record = records[0]
  a_b = record.get("boundaries", {}).get("S_decode_vs_S_prefill", {})
  b_c = record.get("boundaries", {}).get("S_prefill_vs_T_old", {})
  if (a_b.get("valid") is not True or a_b.get("finite") is not True or
      not isinstance(a_b.get("differing_bytes"), int)):
    raise SystemExit(f"round {round_index} A-B boundary is not admissible")
  if (b_c.get("valid") is not True or b_c.get("finite") is not True or
      b_c.get("differing_bytes") != 0):
    raise SystemExit(f"round {round_index} B-C boundary is not exact")
  capsule_present = (round_dir / "mismatch-capsule.npz").is_file()
  expected_capsule = a_b["differing_bytes"] != 0
  if inventory.get("capsule_present") is not expected_capsule:
    raise SystemExit(f"round {round_index} capsule inventory drifted")
  if capsule_present != expected_capsule:
    raise SystemExit(f"round {round_index} capsule payload drifted")
  rounds.append({
      "diagnostic_round": round_index,
      "N_action": record.get("N_action"),
      "a_b_differing_bytes": a_b["differing_bytes"],
      "a_b_differing_elements": a_b.get("differing_elements"),
      "a_b_max_abs": a_b.get("max_abs"),
      "b_c_differing_bytes": 0,
      "archive_sha256": complete.get("archive_sha256"),
  })

preflight = json.loads((root / "root" / "PREFLIGHT.json").read_text())
if (preflight.get("schema") != "canon-p38-gcs-preflight-v1" or
    preflight.get("source_commit") != source or
    preflight.get("status") != "writable"):
  raise SystemExit("root PREFLIGHT marker drifted")

for name, schema, status in (
    ("COLLECTED.json", "canon-p38-gcs-collection-v1", "collected"),
    ("COMPLETE.json", "canon-p38-gcs-completion-v1", "postflight-accepted"),
):
  marker_path = root / "root" / name
  if marker_path.is_file():
    marker = json.loads(marker_path.read_text())
    if (marker.get("schema") != schema or
        marker.get("source_commit") != source or
        marker.get("status") != status):
      raise SystemExit(f"root marker drifted: {name}")

if (root / "root" / "COMPLETE.json").is_file():
  complete_marker = json.loads((root / "root" / "COMPLETE.json").read_text())
  if complete_marker.get("manifest_sha256") != hashlib.sha256(
      (root / "root" / "SHA256SUMS").read_bytes()).hexdigest():
    raise SystemExit("root completion manifest SHA drifted")

all_exact = all(item["a_b_differing_bytes"] == 0 for item in rounds)
verdict = {
    "schema": "canon-p38s23r3-return-v1",
    "source_commit": source,
    "status": (
        "P38S23R3_FORWARD_EXACT_PASS" if all_exact
        else "P38S23R3_FIXED_LM_HEAD_INSUFFICIENT"
    ),
    "rounds": rounds,
    "claim_ceiling": (
        "three-round forward causal-repair candidate only; backward and optimizer untested"
        if all_exact else
        "fixed lm-head rejected as a sufficient forward repair; B-C remains exact"
    ),
}
(root / "verdict.json").write_text(
    json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(verdict, sort_keys=True))
PY

(
  cd "$output_dir"
  find . -type f ! -name RETURN_SHA256SUMS -printf '%P\n' \
    | LC_ALL=C sort \
    | xargs -r sha256sum > RETURN_SHA256SUMS
  sha256sum -c RETURN_SHA256SUMS --quiet
)
echo "[P38S23R3.RETURN] PASS output=$output_dir"
