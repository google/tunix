#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COLLECT="$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/collect_p38h_backward_return.sh"
SOURCE="$(printf 'b%.0s' {1..40})"
tmp="$(mktemp -d)"
trap 'rm -r "$tmp"' EXIT

mkdir -p "$tmp/launch"
for name in rendered.yaml render.txt semantic-preflight.txt dry-run.txt apply.txt; do
  printf '%s\n' "$name" > "$tmp/launch/$name"
done
printf '%s\n' "$SOURCE" > "$tmp/launch/source_commit.txt"
(
  cd "$tmp/launch"
  sha256sum source_commit.txt rendered.yaml render.txt semantic-preflight.txt \
    dry-run.txt apply.txt > LAUNCH_SHA256SUMS
)

python3 - "$ROOT" "$SOURCE" "$tmp/head.full.log" <<'PY'
import base64
import hashlib
import importlib.util
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source = sys.argv[2]
output = pathlib.Path(sys.argv[3])
test_path = root / "tests/p33_workloads/test_classify_run.py"
spec = importlib.util.spec_from_file_location("p38h_p33_fixture", test_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

pre = (json.dumps(module._pre_alignment(0), sort_keys=True) + "\n").encode()
alignment = "".join(
    json.dumps(module._alignment(i, optimizer_skipped=True), sort_keys=True) + "\n"
    for i in range(16)
).encode()
update = {
    "verdict": "PASS",
    "dp_axis": "data",
    "dp_size": 16,
    "tp_size": 4,
    "global_m": 4096,
    "mode": "backward-no-commit",
    "microsteps": 16,
    "commits": 0,
    "train_steps_before": 0,
    "train_steps_after": 0,
    "gradient_activity": [True] * 16,
    "gradient_finite": True,
    "alignment_hashes": [{"T_current": "a"}] * 16,
    "micro_gradient_norms": [1.0] * 16,
    "optimizer_memory_kinds_before": ["device"],
    "optimizer_placement": "device-resident",
    "model_changed_paths": [],
    "optimizer_changed_paths": [],
    "accumulator_changed_paths": [],
    "reference_changed_paths": [],
}
update_payload = (json.dumps(update, sort_keys=True) + "\n").encode()

lines = [
    f"[sync] HEAD={source}",
    "[CANON_P33_WANDB] ONLINE_RUN_PASS",
    "[CANON_P33_EVAL] DISABLED workload=frozenlake",
    "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0",
]
for semantic_m, chunks in (
    (8, 1), (16, 1), (32, 1), (64, 1), (128, 1), (256, 1),
    (4096, 16),
):
  lines.append(
      "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 "
      f"semantic_M={semantic_m} fixed_M=256 K=4096 local_N=37984 "
      f"fixed_N=38144 BM=128 BN=256 BK=256 chunks={chunks}"
  )
lines.extend([
    "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 semantic_M=4096 "
    "fixed_M=256 chunks=16 accumulation=lax.scan order=ascending",
    "[CANON_P33_DP16] backward_no_commit verdict=PASS commits=0 microsteps=16",
])
for name, payload in (
    ("pre-alignment", pre),
    ("alignment", alignment),
    ("update", update_payload),
):
  lines.append(
      f"[CANON_P38H_ARTIFACT] name={name} "
      f"sha256={hashlib.sha256(payload).hexdigest()} encoding=base64 "
      f"data={base64.b64encode(payload).decode()}"
  )
output.write_text("\n".join(lines) + "\n")
PY

bash "$COLLECT" --source-commit "$SOURCE" \
  --head-log "$tmp/head.full.log" --launch-dir "$tmp/launch" \
  --output-dir "$tmp/pass" > "$tmp/pass.stdout"
grep -q 'P38H_FIXED_LM_HEAD_BACKWARD_NO_COMMIT_PASS' "$tmp/pass/verdict.json"
(cd "$tmp/pass" && sha256sum -c RETURN_SHA256SUMS --quiet)

grep -v 'CANON_P38_FIXED_LM_HEAD_VJP=1' "$tmp/head.full.log" \
  > "$tmp/no-vjp.log"
if bash "$COLLECT" --source-commit "$SOURCE" \
    --head-log "$tmp/no-vjp.log" --launch-dir "$tmp/launch" \
    --output-dir "$tmp/no-vjp" > "$tmp/no-vjp.stdout" 2> "$tmp/no-vjp.stderr"; then
  echo "missing VJP receipt was accepted" >&2
  exit 1
fi
grep -q 'fixed-order lm-head VJP receipt is absent' "$tmp/no-vjp.stderr"

python3 - "$tmp/head.full.log" "$tmp/bad-sha.log" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
match = re.search(r"(name=update sha256=)[0-9a-f]{64}", text)
assert match
text = text[:match.start(0)] + match.group(1) + ("0" * 64) + text[match.end(0):]
pathlib.Path(sys.argv[2]).write_text(text)
PY
if bash "$COLLECT" --source-commit "$SOURCE" \
    --head-log "$tmp/bad-sha.log" --launch-dir "$tmp/launch" \
    --output-dir "$tmp/bad-sha" > "$tmp/bad-sha.stdout" 2> "$tmp/bad-sha.stderr"; then
  echo "corrupt compact artifact was accepted" >&2
  exit 1
fi
grep -q 'artifact SHA drifted: update' "$tmp/bad-sha.stderr"

python3 - "$tmp/head.full.log" "$tmp/mutated.log" <<'PY'
import base64
import hashlib
import json
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
pattern = re.compile(
    r"(\[CANON_P38H_ARTIFACT\] name=update )sha256=[0-9a-f]{64} "
    r"encoding=base64 data=([A-Za-z0-9+/=]+)"
)
match = pattern.search(text)
assert match
record = json.loads(base64.b64decode(match.group(2)))
record["model_changed_paths"] = ["params.layer0"]
payload = (json.dumps(record, sort_keys=True) + "\n").encode()
replacement = (
    match.group(1) + f"sha256={hashlib.sha256(payload).hexdigest()} "
    "encoding=base64 data=" + base64.b64encode(payload).decode()
)
pathlib.Path(sys.argv[2]).write_text(
    text[:match.start()] + replacement + text[match.end():]
)
PY
if bash "$COLLECT" --source-commit "$SOURCE" \
    --head-log "$tmp/mutated.log" --launch-dir "$tmp/launch" \
    --output-dir "$tmp/mutated" > "$tmp/mutated.stdout" 2> "$tmp/mutated.stderr"; then
  echo "mutating backward transaction was accepted" >&2
  exit 1
fi
grep -q '"verdict": "FAIL"' "$tmp/mutated/p33-recomputed.json"

echo "[P38.2H.OPERATOR] PASS positives=1 negatives=3"
