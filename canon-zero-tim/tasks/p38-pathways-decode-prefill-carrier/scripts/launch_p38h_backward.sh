#!/usr/bin/env bash
# Render, validate, and optionally launch the single P38.2h 64-TPU target.
set -euo pipefail

usage() {
  echo "usage: launch_p38h_backward.sh --source-commit <sha> --output-dir <abs> --return-dir <abs> [--apply]" >&2
}

source_commit=""
output_dir=""
return_dir=""
apply=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-commit) source_commit="${2:-}"; shift 2 ;;
    --output-dir) output_dir="${2:-}"; shift 2 ;;
    --return-dir) return_dir="${2:-}"; shift 2 ;;
    --apply) apply=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[P38.2H.LAUNCH] REFUSING: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[P38.2H.LAUNCH] REFUSING: source commit must be 40 lowercase hex" >&2
  exit 2
}
for path in "$output_dir" "$return_dir"; do
  case "$path" in
    /*) ;;
    *) echo "[P38.2H.LAUNCH] REFUSING: paths must be absolute" >&2; exit 2 ;;
  esac
  test ! -e "$path" || {
    echo "[P38.2H.LAUNCH] REFUSING: path exists: $path" >&2
    exit 2
  }
done

repo="$(git rev-parse --show-toplevel)"
test "$(git rev-parse HEAD)" = "$source_commit" || {
  echo "[P38.2H.LAUNCH] REFUSING: checkout is not source commit" >&2
  exit 2
}
test -z "$(git status --porcelain)" || {
  echo "[P38.2H.LAUNCH] REFUSING: checkout is dirty" >&2
  exit 2
}
mkdir -m 700 "$return_dir"
printf '%s\n' "$source_commit" > "$return_dir/source_commit.txt"

python3 "$repo/canon-zero-tim/cluster/render_p38_backward_jobset.py" \
  --source-commit "$source_commit" --run-id p38h1 \
  --output-dir "$output_dir" | tee "$return_dir/render.txt"
yaml="$output_dir/jobset-p38h-fixed-lm-head-backward.yaml"
cp -- "$yaml" "$return_dir/rendered.yaml"
python3 - "$yaml" "$source_commit" <<'PY' | tee "$return_dir/semantic-preflight.txt"
import pathlib, sys, yaml
path = pathlib.Path(sys.argv[1])
source = sys.argv[2]
d = yaml.safe_load(path.read_text())
assert d["metadata"]["name"] == f"canon-p38h-fl-bwd-p38h1-{source[:8]}"
assert d["spec"]["failurePolicy"]["maxRestarts"] == 0
pod = d["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
main = next(x for x in pod["containers"] if x["name"] == "jax-tpu")
env = {x["name"]: x.get("value") for x in main["env"]}
assert env["CANON_EXPECT_COMMIT"] == source
assert env["CANON_P38_FIXED_LM_HEAD"] == "1"
assert env["CANON_P33_RUN_STAGE"] == "backward-no-commit"
assert env["CANON_P33_NO_COMMIT"] == "1"
assert env["CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"] == "0"
assert env["CANON_P33_DISABLE_EVAL"] == "1"
for key in ("CANON_P38_PRECHECK_ONLY", "CANON_P38_CONTROLLED_EXIT",
            "CANON_P38_DIAGNOSTIC_ROUNDS", "CANON_P38_SERVING_CAPTURE_DIR",
            "CANON_P38_KV_OBSERVER_DIR", "CANON_P38_SEAM_OBSERVER",
            "CANON_MM_ALGO"):
  assert key not in env, key
print(f"P38H_SEMANTIC_PREFLIGHT_PASS source={source} job={d['metadata']['name']}")
PY
kubectl apply --dry-run=server -f "$yaml" | tee "$return_dir/dry-run.txt"
if [ "$apply" -eq 1 ]; then
  kubectl apply -f "$yaml" | tee "$return_dir/apply.txt"
else
  printf '%s\n' "NOT_APPLIED" > "$return_dir/apply.txt"
fi
(
  cd "$return_dir"
  sha256sum source_commit.txt rendered.yaml render.txt semantic-preflight.txt \
    dry-run.txt apply.txt > LAUNCH_SHA256SUMS
)
echo "[P38.2H.LAUNCH] PASS apply=$apply job=canon-p38h-fl-bwd-p38h1-${source_commit:0:8}"
