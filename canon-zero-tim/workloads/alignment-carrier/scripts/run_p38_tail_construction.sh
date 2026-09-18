#!/usr/bin/env bash
# Run the bounded direct-attached P38 canonical-tail construction control.
set -euo pipefail

label="${1:?usage: run_p38_tail_construction.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P38.TAIL] REFUSING: invalid label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
evidence=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
raw="$evidence/p38_tail_${label}.raw.log"
report="$evidence/p38_tail_${label}.result.json"
container="p38_tail_${label}"

source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
for path in "$raw" "$report"; do
  if [ -e "$path" ]; then
    echo "[P38.TAIL] REFUSING: evidence path exists: $path" >&2
    exit 3
  fi
done

{
  echo "[P38.TAIL] source=$(git -C "$repo" rev-parse HEAD) image=$image"
  echo "[P38.TAIL] scope=direct-attached-construction-only mutation=none"
  sha256sum "$0" "$script_dir/probe_p38_tail_construction.py" \
    "$repo/tunix/rl/canonical_logsoftmax.py"
} >"$raw"

set +e
timeout --foreground --signal=TERM --kill-after=60s 900s \
sudo docker run --rm --privileged --net=host --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$repo":"$repo":ro \
  -e PYTHONPATH="$repo" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e WANDB_MODE=disabled \
  -e CANON_PALLAS_LOGSOFTMAX=1 \
  -e XLA_FLAGS="$XTRA_XLA" \
  -w "$repo" "$image" \
  python3 -u "$script_dir/probe_p38_tail_construction.py" --output "$report" \
  >>"$raw" 2>&1
docker_rc=$?
set -e

echo "[P38.TAIL] docker_exit=$docker_rc" >>"$raw"
test -s "$report"
python3 -c "import json; r=json.load(open('$report')); assert r['verdict']=='PASS_CONSTRUCTION_ONLY'; assert r['differing_elements']==0; assert r['negative_control_differing_elements']==1"
sha256sum "$raw" "$report" "$0"
