#!/usr/bin/env bash
# Gate the Qwen3-1.7B TP4 fixed lm-head primal and VJP on one real v5p host.
set -euo pipefail

label="${1:?usage: run_gsm8k_fixed_lm_head_onehost.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P38.GSM8K.FIXED_LM_HEAD] REFUSING: invalid label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
shim_dir="$repo/canon-zero-tim/src/engine_shims"
model_contract_dir="$shim_dir/models/qwen1p7b"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
model=/mnt/disks/tunix-data/gsm8k_zero_tim/models
evidence="${CANON_P38_ONEHOST_EVIDENCE_DIR:-/mnt/disks/tunix-data/logp_probe_1host}"
image=tunix_frozenlake_image:vllm-tpu0.25.0
raw="$evidence/p38_gsm8k_fixed_lm_head_${label}.raw.log"
forward_report="$evidence/p38_gsm8k_fixed_lm_head_${label}.forward.json"
vjp_report="$evidence/p38_gsm8k_fixed_lm_head_${label}.vjp.json"

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test -s "$model/config.json"
test -s "$model/model.safetensors.index.json"
mkdir -p "$evidence"
for path in "$raw" "$forward_report" "$vjp_report"; do
  if [ -e "$path" ]; then
    echo "[P38.GSM8K.FIXED_LM_HEAD] REFUSING: evidence path exists: $path" >&2
    exit 3
  fi
done

{
  echo "[P38.GSM8K.FIXED_LM_HEAD] source=$(git -C "$repo" rev-parse HEAD) image=$image"
  echo "[P38.GSM8K.FIXED_LM_HEAD] model_path=$model scope=onehost-real-weight-construction-only"
  echo "[P38.GSM8K.FIXED_LM_HEAD] request_M=8,16,32,64,128,256 learner_M=4096 fixed_M=256 K=2048 V=151936 topology=TP4"
  sha256sum "$0" \
    "$script_dir/probe_p38_fixed_lm_head.py" \
    "$script_dir/probe_p38_fixed_lm_head_vjp.py" \
    "$script_dir/probe_p38_lm_head.py" \
    "$shim_dir/p38_fixed_lm_head.py" \
    "$shim_dir/p22_pallas_matmul.py" \
    "$shim_dir/p22xi_padded_matmul.py" \
    "$shim_dir/p22xk_vjp_ops.py" \
    "$model_contract_dir/p22xf_contract.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} >"$raw"

common_env=(
  -e PYTHONDONTWRITEBYTECODE=1
  -e WANDB_MODE=disabled
  -e XLA_FLAGS="$XTRA_XLA"
  -e PYTHONPATH="$model_contract_dir:$shim_dir:$script_dir"
  -e CANON_P38_FIXED_LM_HEAD=1
  -e CANON_QWEN3_HIDDEN_SIZE=2048
  -e CANON_QWEN3_INTERMEDIATE_SIZE=6144
  -e CANON_QWEN3_NUM_ATTENTION_HEADS=16
  -e CANON_QWEN3_NUM_KV_HEADS=8
  -e CANON_QWEN3_HEAD_DIM=128
  -e CANON_QWEN3_TP_SIZE=4
  -e CANON_FIXED_AR=1
  -e CANON_FIXED_AR_EMBED=1
  -e CANON_PALLAS_ALL_PROJ=1
  -e CANON_PALLAS_ALL_RMSNORM=1
  -e CANON_PALLAS_SWIGLU=1
  -e CANON_PALLAS_MPAD=1
  -e CANON_PALLAS_SWIGLU_MPAD=1
  -e CANON_PALLAS_CANONICAL_VJP=1
)

set +e
timeout --foreground --signal=TERM --kill-after=60s 7200s \
sudo docker run --rm --privileged --net=host \
  --name "p38_gsm8k_fixed_forward_${label}" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$evidence":"$evidence" \
  -v "$repo":"$repo":ro \
  "${common_env[@]}" \
  -w "$repo" "$image" \
  python3 -u "$script_dir/probe_p38_fixed_lm_head.py" \
    --model "$model" --output "$forward_report" --hidden-size 2048 \
  >>"$raw" 2>&1
forward_rc=$?
set -e
echo "[P38.GSM8K.FIXED_LM_HEAD] forward_docker_exit=$forward_rc" >>"$raw"
test "$forward_rc" -eq 0

set +e
timeout --foreground --signal=TERM --kill-after=60s 7200s \
sudo docker run --rm --privileged --net=host \
  --name "p38_gsm8k_fixed_vjp_${label}" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$evidence":"$evidence" \
  -v "$repo":"$repo":ro \
  "${common_env[@]}" \
  -w "$repo" "$image" \
  python3 -u "$script_dir/probe_p38_fixed_lm_head_vjp.py" \
    --model "$model" --output "$vjp_report" --hidden-size 2048 \
  >>"$raw" 2>&1
vjp_rc=$?
set -e
echo "[P38.GSM8K.FIXED_LM_HEAD] vjp_docker_exit=$vjp_rc" >>"$raw"
test "$vjp_rc" -eq 0

grep -Eq "semantic_M=16 fixed_M=256 K=2048 local_N=37984 fixed_N=38144 .*endpoint=direct_probe" "$raw"
grep -Eq "semantic_M=256 fixed_M=256 K=2048 local_N=37984 fixed_N=38144 .*endpoint=direct_probe" "$raw"
grep -Eq "semantic_M=4096 fixed_M=256 K=2048 local_N=37984 fixed_N=38144 .*endpoint=direct_probe" "$raw"
grep -Eq "semantic_M=4096 fixed_M=256 chunks=16 accumulation=lax.scan order=ascending K=2048 endpoint=direct_probe" "$raw"

python3 - "$forward_report" "$vjp_report" <<'PY'
import json
import pathlib
import sys

forward = json.loads(pathlib.Path(sys.argv[1]).read_text())
vjp = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert forward["verdict"] == "FIXED_LM_HEAD_ONEHOST_CONSTRUCTION_PASS"
assert forward["fixed_shape"] == [256, 2048, 38144]
assert forward["weight_shape"] == [2048, 151936]
assert forward["weight_source"] == "model.embed_tokens.weight"
assert all(row["fixed_differing_elements"] == 0 for row in forward["seeds"])
assert all(row["differing_elements"] == 0 for row in forward["learner_seeds"])
assert forward["negative_control_differing_elements"] == 1
assert vjp["verdict"] == "FIXED_LM_HEAD_ONEHOST_VJP_PASS"
assert vjp["weight_shape"] == [2048, 151936]
assert vjp["weight_source"] == "model.embed_tokens.weight"
for key in ("hidden", "weight", "repeat_hidden", "repeat_weight"):
  assert vjp[key] == {"differing_elements": 0, "max_abs": 0.0}
assert vjp["gradient_finite"] is True
assert vjp["hidden_gradient_nonzero"] > 0
assert vjp["weight_gradient_nonzero"] > 0
assert vjp["negative_control_differing_elements"] == 1
print("[P38.GSM8K.FIXED_LM_HEAD] ONEHOST_PASS forward=1 vjp=1")
PY

sha256sum "$raw" "$forward_report" "$vjp_report" "$0"
