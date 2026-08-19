#!/usr/bin/env bash
# Collect and mechanically classify the compact P38.2h result from stdout.
set -euo pipefail

usage() {
  echo "usage: collect_p38h_backward_return.sh --source-commit <sha> --head-log <file> --launch-dir <dir> --output-dir <new-abs-dir>" >&2
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
    *) echo "[P38.2H.RETURN] REFUSING: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[P38.2H.RETURN] REFUSING: invalid source commit" >&2; exit 2;
}
test -s "$head_log" && test -d "$launch_dir" || {
  echo "[P38.2H.RETURN] REFUSING: head log or launch dir absent" >&2; exit 2;
}
case "$output_dir" in /*) ;; *) echo "[P38.2H.RETURN] REFUSING: output must be absolute" >&2; exit 2;; esac
test ! -e "$output_dir" || {
  echo "[P38.2H.RETURN] REFUSING: output exists" >&2; exit 2;
}

repo="$(git rev-parse --show-toplevel)"
classifier="$repo/canon-zero-tim/tests/p33_workloads/classify_run.py"
mkdir -m 700 "$output_dir"
mkdir -m 700 "$output_dir/launch"
cp -- "$head_log" "$output_dir/head.full.log"
for name in source_commit.txt rendered.yaml render.txt semantic-preflight.txt \
    dry-run.txt apply.txt LAUNCH_SHA256SUMS; do
  test -s "$launch_dir/$name" || {
    echo "[P38.2H.RETURN] REFUSING: launch receipt absent: $name" >&2; exit 2;
  }
  cp -- "$launch_dir/$name" "$output_dir/launch/$name"
done
(cd "$output_dir/launch" && sha256sum -c LAUNCH_SHA256SUMS --quiet)

python3 - "$source_commit" "$output_dir" <<'PY'
import base64, hashlib, pathlib, re, sys
source = sys.argv[1]
root = pathlib.Path(sys.argv[2])
log = (root / "head.full.log").read_text(encoding="utf-8", errors="replace")
if (root / "launch/source_commit.txt").read_text().strip() != source:
  raise SystemExit("launch source receipt drifted")
if f"[sync] HEAD={source}" not in log:
  raise SystemExit("complete head log does not attest source commit")
if "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD" in log:
  raise SystemExit("historical precheck-only path executed")
if "[CANON_P38] CONTROLLED_EXIT" in log:
  raise SystemExit("historical controlled-exit path executed")
if log.count("[CANON_P33_DP16] backward_no_commit verdict=PASS commits=0") != 1:
  raise SystemExit("actual-model backward-no-commit marker is absent or repeated")
if "[CANON_P33_DP16] update_step_committed" in log:
  raise SystemExit("optimizer commit marker appeared")
if "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 semantic_M=4096 fixed_M=256 chunks=16 accumulation=lax.scan order=ascending" not in log:
  raise SystemExit("fixed-order lm-head VJP receipt is absent")
for semantic_m, chunks in (
    (16, 1), (32, 1), (64, 1), (128, 1), (256, 1),
    (4096, 16),
):
  pattern = rf"CANON_P38_FIXED_LM_HEAD=1 semantic_M={semantic_m}\b.*\bchunks={chunks}\b"
  if re.search(pattern, log) is None:
    raise SystemExit(f"fixed-lm-head receipt absent: M={semantic_m}")
pattern = re.compile(
    r"^\[CANON_P38H_ARTIFACT\] name=(pre-alignment|alignment|update) "
    r"sha256=([0-9a-f]{64}) encoding=base64 data=([A-Za-z0-9+/=]+)$",
    re.MULTILINE,
)
records = pattern.findall(log)
if len(records) != 3 or {x[0] for x in records} != {
    "pre-alignment", "alignment", "update"
}:
  raise SystemExit(f"compact evidence set invalid: {[(x[0], x[1]) for x in records]}")
filenames = {
    "pre-alignment": "pre_alignment.jsonl",
    "alignment": "alignment.jsonl",
    "update": "updates.json",
}
for name, expected_sha, encoded in records:
  payload = base64.b64decode(encoded, validate=True)
  actual_sha = hashlib.sha256(payload).hexdigest()
  if actual_sha != expected_sha:
    raise SystemExit(f"artifact SHA drifted: {name}")
  (root / filenames[name]).write_bytes(payload)
PY

JAX_PLATFORMS=cpu PYTHONPATH="$repo:${PYTHONPATH:-}" \
python3 "$classifier" \
  --workload frozenlake --dp-size 16 --tp-size 4 \
  --stage backward-no-commit \
  --run-log "$output_dir/head.full.log" \
  --pre-alignment-report "$output_dir/pre_alignment.jsonl" \
  --alignment-report "$output_dir/alignment.jsonl" \
  --update-report "$output_dir/updates.json" \
  --output "$output_dir/p33-recomputed.json" \
  > "$output_dir/classifier.txt"

python3 - "$source_commit" "$output_dir" <<'PY'
import hashlib, json, pathlib, sys
source = sys.argv[1]
root = pathlib.Path(sys.argv[2])
classification = json.loads((root / "p33-recomputed.json").read_text())
update = json.loads((root / "updates.json").read_text())
if classification.get("verdict") != "PASS":
  raise SystemExit(f"P33 classifier rejected result: {classification.get('reasons')}")
if update.get("commits") != 0 or update.get("verdict") != "PASS":
  raise SystemExit("no-commit report is not PASS with zero commits")
for key in ("model_changed_paths", "optimizer_changed_paths",
            "accumulator_changed_paths", "reference_changed_paths"):
  if update.get(key) != []:
    raise SystemExit(f"training state changed: {key}")
if not update.get("gradient_finite") or not any(update.get("gradient_activity", [])):
  raise SystemExit("actual-model gradient is nonfinite or has no signal")
files = [
    "head.full.log", "pre_alignment.jsonl", "alignment.jsonl", "updates.json",
    "p33-recomputed.json", "classifier.txt",
]
verdict = {
    "schema": "canon-p38h-backward-return-v1",
    "status": "P38H_FIXED_LM_HEAD_BACKWARD_NO_COMMIT_PASS",
    "source_commit": source,
    "claim_scope": "actual-model-backward-no-commit-only",
    "gradient_microsteps": update.get("microsteps"),
    "gradient_activity": update.get("gradient_activity"),
    "micro_gradient_norms": update.get("micro_gradient_norms"),
    "optimizer_commits": 0,
    "state_changed_paths": 0,
    "evidence_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in files
    },
}
(root / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
with (root / "RETURN_SHA256SUMS").open("w") as out:
  for name in sorted(files + ["verdict.json"]):
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    out.write(f"{digest}  {name}\n")
print(json.dumps(verdict, sort_keys=True))
PY
(cd "$output_dir" && sha256sum -c RETURN_SHA256SUMS --quiet)
echo "[P38.2H.RETURN] PASS output=$output_dir"
