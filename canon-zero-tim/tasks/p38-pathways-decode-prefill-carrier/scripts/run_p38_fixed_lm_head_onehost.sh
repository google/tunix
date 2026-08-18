#!/usr/bin/env bash
# Run the real-weight P38.2x fixed lm_head construction on one v5p host.
set -euo pipefail

label="${1:?usage: run_p38_fixed_lm_head_onehost.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P38.FIXED_LM_HEAD] REFUSING: invalid label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
shim_dir="$repo/canon-zero-tim/src/engine_shims"
model_contract_dir="$shim_dir/models/qwen8b"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
model_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
snapshot_sha="$(cat "$model_cache/refs/main")"
model="$model_cache/snapshots/$snapshot_sha"
evidence=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
raw="$evidence/p38_fixed_lm_head_${label}.raw.log"
report="$evidence/p38_fixed_lm_head_${label}.result.json"
container="p38_fixed_lm_head_${label}"

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -s "$model/config.json"
test -s "$model/model.safetensors.index.json"
for path in "$raw" "$report"; do
  if [ -e "$path" ]; then
    echo "[P38.FIXED_LM_HEAD] REFUSING: evidence path exists: $path" >&2
    exit 3
  fi
done

{
  echo "[P38.FIXED_LM_HEAD] source=$(git -C "$repo" rev-parse HEAD) image=$image"
  echo "[P38.FIXED_LM_HEAD] snapshot=$snapshot_sha scope=construction-only"
  echo "[P38.FIXED_LM_HEAD] semantic_M=16,256 fixed_M=256 K=4096 local_N=37984 fixed_N=38144 tiles=128,256,256 topology=TP4"
  sha256sum "$0" "$script_dir/probe_p38_fixed_lm_head.py" \
    "$shim_dir/p38_fixed_lm_head.py" "$shim_dir/p22_pallas_matmul.py" \
    "$shim_dir/p22xi_padded_matmul.py" "$shim_dir/p22xk_vjp_ops.py" \
    "$model_contract_dir/p22xf_contract.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} >"$raw"

set +e
timeout --foreground --signal=TERM --kill-after=60s 3600s \
sudo docker run --rm --privileged --net=host --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$repo":"$repo":ro \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e WANDB_MODE=disabled \
  -e XLA_FLAGS="$XTRA_XLA" \
  -e PYTHONPATH="$model_contract_dir:$shim_dir:$script_dir" \
  -e CANON_P38_FIXED_LM_HEAD=1 \
  -e CANON_FIXED_AR=1 \
  -e CANON_FIXED_AR_EMBED=1 \
  -e CANON_PALLAS_ALL_PROJ=1 \
  -e CANON_PALLAS_ALL_RMSNORM=1 \
  -e CANON_PALLAS_SWIGLU=1 \
  -e CANON_PALLAS_MPAD=1 \
  -e CANON_PALLAS_SWIGLU_MPAD=1 \
  -e CANON_PALLAS_CANONICAL_VJP=1 \
  -w "$repo" "$image" \
  python3 -u "$script_dir/probe_p38_fixed_lm_head.py" \
    --model "$model" --output "$report" --seeds 4 \
  >>"$raw" 2>&1
docker_rc=$?
set -e

echo "[P38.FIXED_LM_HEAD] docker_exit=$docker_rc" >>"$raw"
test "$docker_rc" -eq 0
test -s "$report"
grep -Fq 'semantic_M=16 fixed_M=256 K=4096 local_N=37984 fixed_N=38144' "$raw"
grep -Fq 'semantic_M=256 fixed_M=256 K=4096 local_N=37984 fixed_N=38144' "$raw"
python3 - "$report" <<'PY'
import json, pathlib, sys
r = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert r["claim_scope"] == "onehost-real-weight-fixed-lm-head-construction-only"
assert r["verdict"] == "FIXED_LM_HEAD_ONEHOST_CONSTRUCTION_PASS"
assert r["negative_control_differing_elements"] == 1
assert r["semantic_m"] == [16, 256]
assert r["fixed_shape"] == [256, 4096, 38144]
assert r["local_vocab"] == 37984
assert r["tiles"] == {"BM": 128, "BN": 256, "BK": 256}
assert len(r["seeds"]) == 4
assert all(x["fixed_differing_elements"] == 0 for x in r["seeds"])
print(f"[P38.FIXED_LM_HEAD] ONEHOST_PASS verdict={r['verdict']}")
PY
sha256sum "$raw" "$report" "$0" "$script_dir/probe_p38_fixed_lm_head.py"
