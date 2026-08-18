#!/usr/bin/env bash
# Run the real-weight P38 lm_head M=16/M=256 operator screen on one v5p host.
set -euo pipefail

label="${1:?usage: run_p38_lm_head_onehost.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P38.LM_HEAD] REFUSING: invalid label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
model_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
snapshot_sha="$(cat "$model_cache/refs/main")"
model="$model_cache/snapshots/$snapshot_sha"
evidence=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
raw="$evidence/p38_lm_head_${label}.raw.log"
report="$evidence/p38_lm_head_${label}.result.json"
container="p38_lm_head_${label}"

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -s "$model/config.json"
test -s "$model/model.safetensors.index.json"
for path in "$raw" "$report"; do
  if [ -e "$path" ]; then
    echo "[P38.LM_HEAD] REFUSING: evidence path exists: $path" >&2
    exit 3
  fi
done

{
  echo "[P38.LM_HEAD] source=$(git -C "$repo" rev-parse HEAD) image=$image"
  echo "[P38.LM_HEAD] snapshot=$snapshot_sha scope=operator-screen-only"
  echo "[P38.LM_HEAD] shapes=decode-M16,prefill-M256,K4096,V151936 topology=TP4"
  sha256sum "$0" "$script_dir/probe_p38_lm_head.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} >"$raw"

set +e
timeout --foreground --signal=TERM --kill-after=60s 1800s \
sudo docker run --rm --privileged --net=host --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$repo":"$repo":ro \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e WANDB_MODE=disabled \
  -e XLA_FLAGS="$XTRA_XLA" \
  -w "$repo" "$image" \
  python3 -u "$script_dir/probe_p38_lm_head.py" \
    --model "$model" --output "$report" --seeds 4 \
  >>"$raw" 2>&1
docker_rc=$?
set -e

echo "[P38.LM_HEAD] docker_exit=$docker_rc" >>"$raw"
test "$docker_rc" -eq 0
test -s "$report"
python3 - "$report" <<'PY'
import json, pathlib, sys
r = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert r["claim_scope"] == "onehost-real-weight-operator-screen-only"
assert r["negative_control_differing_elements"] == 1
assert len(r["seeds"]) == 4
assert r["verdict"] in {
    "ALGORITHM_ELIMINATES_OPERATOR_DRIFT",
    "BOTH_EXACT_OPERATOR_SCREEN_INCONCLUSIVE",
}
print(f"[P38.LM_HEAD] ONEHOST_PASS verdict={r['verdict']}")
PY
sha256sum "$raw" "$report" "$0" "$script_dir/probe_p38_lm_head.py"
