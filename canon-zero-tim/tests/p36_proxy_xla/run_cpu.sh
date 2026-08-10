#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="$(cd "$ROOT/.." && pwd)"
cd "$WORKTREE"

python3 -m py_compile \
  canon-zero-tim/cluster/render_p36_proxy_xla_jobset.py \
  canon-zero-tim/tests/p36_proxy_xla/test_render_p36_proxy_xla_jobset.py
python3 -m unittest \
  canon-zero-tim/tests/p36_proxy_xla/test_render_p36_proxy_xla_jobset.py

tmpdir="$(mktemp -d)"
trap 'rm -r "$tmpdir"' EXIT
python3 canon-zero-tim/cluster/render_p36_proxy_xla_jobset.py \
  --source-commit 1111111111111111111111111111111111111111 \
  --run-id local-gate \
  --output "$tmpdir/p36.yaml"
python3 - "$tmpdir/p36.yaml" <<'PY'
import sys
import yaml

document = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
head = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
proxy = next(item for item in head["initContainers"] if item["name"] == "pathways-proxy")
flags = [arg for arg in proxy["args"] if arg.startswith("--xla_allow_excess_precision=")]
assert flags == ["--xla_allow_excess_precision=false"], flags
main = next(item for item in head["containers"] if item["name"] == "jax-tpu")
env = {entry["name"]: entry.get("value") for entry in main["env"]}
assert env["CANON_MODE"] == "gate-only", env["CANON_MODE"]
assert env["CANON_GCS_CACHE_BUCKET"] == "", env["CANON_GCS_CACHE_BUCKET"]
print("[P36.PROXY_XLA] LOCAL_GATE_PASS flag_count=1 mode=gate-only cache=isolated")
PY
