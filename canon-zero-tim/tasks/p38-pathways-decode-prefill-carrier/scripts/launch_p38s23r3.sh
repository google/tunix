#!/usr/bin/env bash
# Render, verify, and optionally launch the P38s23r3 fixed-lm-head target.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: launch_p38s23r3.sh \
  --source-commit <40-hex-sha> \
  --output-dir <new-absolute-dir> \
  --return-dir <new-absolute-dir> \
  [--apply]

Run this script from a clean checkout at the exact published source commit.
Without --apply it performs render, semantic validation, and server dry-run.
EOF
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
    *) echo "[P38S23R3] REFUSING: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[P38S23R3] REFUSING: source commit must be 40 lowercase hex characters" >&2
  exit 2
fi
for path_value in "$output_dir" "$return_dir"; do
  case "$path_value" in
    /*) ;;
    *) echo "[P38S23R3] REFUSING: output and return dirs must be absolute" >&2; exit 2 ;;
  esac
  case "$path_value" in
    /|/home|/tmp) echo "[P38S23R3] REFUSING: directory target is too broad: $path_value" >&2; exit 2 ;;
  esac
  if [ -e "$path_value" ]; then
    echo "[P38S23R3] REFUSING: directory already exists: $path_value" >&2
    exit 2
  fi
done

repo="$(git rev-parse --show-toplevel)"
if [ "$(git -C "$repo" rev-parse HEAD)" != "$source_commit" ]; then
  echo "[P38S23R3] REFUSING: checkout HEAD does not equal source commit" >&2
  exit 2
fi
if [ -n "$(git -C "$repo" status --porcelain)" ]; then
  echo "[P38S23R3] REFUSING: checkout is dirty" >&2
  exit 2
fi

run_id=p38s23r3
pkg="$repo/canon-zero-tim"
mkdir -m 700 "$output_dir" "$return_dir"
printf '%s\n' "$source_commit" > "$return_dir/source_commit.txt"

python3 "$pkg/cluster/render_p38_serving_jobsets.py" \
  --source-commit "$source_commit" \
  --run-id "$run_id" \
  --output-dir "$output_dir" \
  --stock-only \
  --max-concurrency 256 \
  --fixed-lm-head | tee "$return_dir/render.txt"

stock="$output_dir/jobset-p38-serving-stock.yaml"
test -s "$stock"
test ! -e "$output_dir/jobset-p38-serving-unified.yaml"
cp -- "$stock" "$return_dir/rendered-stock.yaml"
python3 - "$stock" "$source_commit" <<'PY' | tee "$return_dir/semantic-preflight.txt"
import shlex
import sys
import yaml

path, source = sys.argv[1:]
document = yaml.safe_load(open(path, encoding="utf-8"))
pod = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]
labels = pod["metadata"]["labels"]
container = next(
    item for item in pod["spec"]["containers"] if item["name"] == "jax-tpu"
)
env = {item["name"]: str(item.get("value", "")) for item in container["env"]}
assert env["CANON_EXPECT_COMMIT"] == source, env["CANON_EXPECT_COMMIT"]
assert env["CANON_P38_FIXED_LM_HEAD"] == "1", env
assert env["CANON_P38_DURABILITY_PROFILE"] == "round-alignment-v1", env
assert env["CANON_P38_DIAGNOSTIC_ROUNDS"] == "3", env
assert env["CANON_P38_PRECHECK_ONLY"] == "1", env
assert env["CANON_P38_CONTROLLED_EXIT"] == "1", env
assert env["CANON_KV_UNIFIED"] == "0", env
assert "CANON_MM_ALGO" not in env and "CANON_MM_ALGO_PRESET" not in env
for name in env:
  assert not name.startswith((
      "CANON_P38_KV_OBSERVER",
      "CANON_P38_SEAM",
      "CANON_P38_TAIL",
      "CANON_P38_TERMINAL",
  )), name
assert labels["canon.zero-tim/fixed-lm-head"] == "1", labels
assert labels["canon.zero-tim/durability-profile"] == "round-alignment-v1", labels
assert document["spec"]["failurePolicy"]["maxRestarts"] == 0
command = shlex.split(env["CANON_RUN_CMD"])
assert command.count("--max_concurrency=256") == 1, command
assert env.get("CANON_VLLM_ENABLE_PREFIX_CACHING", "0") in ("", "0"), env
print("P38S23R3_SEMANTIC_PREFLIGHT_PASS")
PY

kubectl apply --dry-run=server -f "$stock" | tee "$return_dir/dry-run.txt"
if [ "$apply" -eq 1 ]; then
  kubectl apply -f "$stock" | tee "$return_dir/apply.txt"
  echo "[P38S23R3] LAUNCHED source=$source_commit run_id=$run_id"
else
  echo "[P38S23R3] PREFLIGHT_ONLY source=$source_commit run_id=$run_id"
fi
(
  cd "$return_dir"
  find . -maxdepth 1 -type f ! -name LAUNCH_SHA256SUMS -printf '%f\n' \
    | LC_ALL=C sort \
    | xargs -r sha256sum > LAUNCH_SHA256SUMS
  sha256sum -c LAUNCH_SHA256SUMS --quiet
)
echo "[P38S23R3] RETURN_DIR=$return_dir"
