#!/usr/bin/env bash
# Render, verify, and optionally launch the P38.2y GSM8K full-training target.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: launch_p38y_gsm8k_full.sh \
  --source-commit <40-hex-sha> \
  --run-id <lowercase-dns-label> \
  --output-dir <new-absolute-dir> \
  --return-dir <new-absolute-dir> \
  [--apply]

Run from a clean checkout at the exact published source commit. Without
--apply, this performs render, semantic validation, and server dry-run only.
EOF
}

source_commit=""
run_id=""
output_dir=""
return_dir=""
apply=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-commit) source_commit="${2:-}"; shift 2 ;;
    --run-id) run_id="${2:-}"; shift 2 ;;
    --output-dir) output_dir="${2:-}"; shift 2 ;;
    --return-dir) return_dir="${2:-}"; shift 2 ;;
    --apply) apply=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[P38Y] REFUSING: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[P38Y] REFUSING: source commit must be 40 lowercase hex characters" >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^[a-z0-9]([-a-z0-9]{0,14}[a-z0-9])?$ ]]; then
  echo "[P38Y] REFUSING: run id must be a 1-16 character lowercase DNS label" >&2
  exit 2
fi
for path_value in "$output_dir" "$return_dir"; do
  case "$path_value" in
    /*) ;;
    *) echo "[P38Y] REFUSING: output and return dirs must be absolute" >&2; exit 2 ;;
  esac
  case "$path_value" in
    /|/home|/tmp) echo "[P38Y] REFUSING: directory target is too broad: $path_value" >&2; exit 2 ;;
  esac
  if [ -e "$path_value" ]; then
    echo "[P38Y] REFUSING: directory already exists: $path_value" >&2
    exit 2
  fi
done

repo="$(git rev-parse --show-toplevel)"
if [ "$(git -C "$repo" rev-parse HEAD)" != "$source_commit" ]; then
  echo "[P38Y] REFUSING: checkout HEAD does not equal source commit" >&2
  exit 2
fi
if [ -n "$(git -C "$repo" status --porcelain)" ]; then
  echo "[P38Y] REFUSING: checkout is dirty" >&2
  exit 2
fi

pkg="$repo/canon-zero-tim"
profile="$pkg/cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env"
mkdir -m 700 "$output_dir" "$return_dir"
printf '%s\n' "$source_commit" >"$return_dir/source_commit.txt"

# Verify the performance bundle from a clean environment. P47a has no flag;
# it is part of the source program. P52 batched reverse is deliberately absent
# because its DP16 grouped implementation has not been certified.
env -i PATH="$PATH" HOME="${HOME:-/tmp}" bash -c '
  set -euo pipefail
  source "$1"
  test "$CANON_P32_WORKLOAD" = gsm8k
  test "$CANON_OPT_STATE_RESIDENT" = 1
  test "$CANON_P30_OPT_STATE_OFFLOAD" = 0
  test "$CANON_BATCHED_EVIDENCE" = 1
  test "$CANON_P28_BATCHED_REPORT" = 1
  test "${CANON_P28_BATCHED_REVERSE:-0}" = 0
' _ "$profile"
echo "P38Y_PROFILE_PREFLIGHT_PASS resident=1 evidence=1 batched_report=1 batched_reverse=0" \
  | tee "$return_dir/profile-preflight.txt"

python3 - "$repo" <<'PY' | tee "$return_dir/sharding-preflight.txt"
from pathlib import Path
import sys

repo = Path(sys.argv[1])
demo = (repo / "examples/math_gsm8k/qwen3_grpo_demo.py").read_text()
workloads = (repo / "tunix/rl/dp_workloads.py").read_text()
runner = (repo / "canon-zero-tim/cluster/steps/90_run.sh").read_text()
linear = (repo / "canon-zero-tim/src/engine_shims/linear_p22xk.py").read_text()
classifier = (
    repo
    / "canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "classify_p38_fixed_lm_head_receipts.py"
).read_text()
assert "configure_model_sharding_for_mesh(config, mesh.axis_names)" in demo
assert "data_sharding_axis_for_mesh(\n              shared_mesh.axis_names" in demo
assert "def configure_model_sharding_for_mesh(" in workloads
assert '("dp", "tp"), ("data", "model")' in workloads
assert "GSM8K_FULL_ATTEMPT_EVIDENCE" in runner
assert 'attempt_evidence_dir="${CANON_STATE%/}/attempt-$JOBSET_RESTART_ATTEMPT"' in runner
assert "_p38_embed_module.JaxEmbed.decode = _p38_fixed_tied_head_decode" in linear
assert 'endpoint="tied_embed"' in linear
assert "gsm8k:full:cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env" in runner
assert "classify_p38_fixed_lm_head_receipts.py" in runner
assert "P38_FIXED_LM_HEAD_RECEIPTS_PASS" in classifier
print(
    "P38Y_SHARDING_PREFLIGHT_PASS "
    "model_axes=actual_mesh data_axis=actual_mesh restart_evidence=attempt_scoped "
    "tied_endpoint=static receipt_gate=terminal"
)
PY

python3 "$pkg/cluster/render_p33_jobsets.py" \
  --source-commit "$source_commit" \
  --run-id "$run_id" \
  --output-dir "$output_dir" \
  | tee "$return_dir/render.txt"

gsm="$output_dir/jobset-p33-gsm8k-full.yaml"
test -s "$gsm"
cp -- "$gsm" "$return_dir/rendered-gsm8k-full.yaml"
python3 - "$gsm" "$source_commit" <<'PY' | tee "$return_dir/semantic-preflight.txt"
import shlex
import sys
import yaml

path, source = sys.argv[1:]
document = yaml.safe_load(open(path, encoding="utf-8"))
assert document["metadata"]["labels"]["canon.zero-tim/workload"] == "gsm8k"
assert document["metadata"]["labels"]["canon.zero-tim/stage"] == "full"
assert document["spec"]["failurePolicy"]["maxRestarts"] == 3
head = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
worker = document["spec"]["replicatedJobs"][1]["template"]["spec"]["template"]["spec"]
assert head["priorityClassName"] == "very-high"
assert worker["priorityClassName"] == "very-high"
container = next(item for item in head["containers"] if item["name"] == "jax-tpu")
env = {item["name"]: str(item.get("value", "")) for item in container["env"]}
expected = {
    "CANON_EXPECT_COMMIT": source,
    "CANON_PROFILE_FILE": "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env",
    "CANON_P33_RUN_STAGE": "full",
    "CANON_P33_NO_COMMIT": "0",
    "CANON_P33_SHARED_MESH": "16,4",
    "CANON_OPT_STATE_RESIDENT": "1",
    "CANON_P30_OPT_STATE_OFFLOAD": "0",
    "CANON_GSM8K_ALIGNMENT_WARN_ONLY": "1",
    "CANON_GSM8K_AB_REPORT_ONLY": "0",
    "CANON_P38_FIXED_LM_HEAD": "1",
}
for name, value in expected.items():
  assert env.get(name) == value, (name, env.get(name), value)
command = shlex.split(env["CANON_RUN_CMD"])
for token in ("--mesh_dp=16", "--mesh_tp=4", "--max_steps=200"):
  assert command.count(token) == 1, (token, command)
assert not any(item.startswith("--eval_") or item.startswith("--num_test_batches") for item in command)
assert env.get("CANON_VLLM_ENABLE_PREFIX_CACHING", "0") in ("", "0")
for name in env:
  assert not name.startswith(("CANON_P38_SERVING_CAPTURE", "CANON_P38_SEAM", "CANON_P38_TAIL")), name
print("P38Y_SEMANTIC_PREFLIGHT_PASS steps=200 topology=DP16xTP4 fixed_lm_head=1 warning_only_ab=1")
PY

kubectl apply --dry-run=server -f "$gsm" | tee "$return_dir/dry-run.txt"
if [ "$apply" -eq 1 ]; then
  kubectl apply -f "$gsm" | tee "$return_dir/apply.txt"
  echo "[P38Y] LAUNCHED source=$source_commit run_id=$run_id"
else
  echo "[P38Y] PREFLIGHT_ONLY source=$source_commit run_id=$run_id"
fi
(
  cd "$return_dir"
  find . -maxdepth 1 -type f ! -name LAUNCH_SHA256SUMS -printf '%f\n' \
    | LC_ALL=C sort \
    | xargs -r sha256sum >LAUNCH_SHA256SUMS
  sha256sum -c LAUNCH_SHA256SUMS --quiet
)
echo "[P38Y] JOBSET=canon-p33-gsm8k-full-${run_id}-${source_commit:0:8}"
echo "[P38Y] RETURN_DIR=$return_dir"
