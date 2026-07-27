#!/bin/bash
# Phase 3c Step 1 — 1-host COLOCATED kernel-isolation probe on a single v5p host (4 chips).
#
# Runs logp_diff_probe.py inside the SAME tunix_base_image container the GKE 256 run uses,
# but DIRECT (no Pathways proxy, no JobSet) on one TPU VM. vLLM(A/B) and tunix(C) share the
# SAME 4-chip mesh (fsdp1xtp4) and run SEQUENTIALLY (vLLM -> free HBM -> tunix). This removes
# the mesh confound so B-vs-C = the PURE kernel diff (the Phase 4 fix target), at real Qwen3-32B.
#
# Differences vs experimental/deepswe-256-mlperf.yaml (the 256 disaggregated run):
#   - NO Pathways: no JAX_PLATFORMS=proxy / JAX_BACKEND_TARGET, no pathwaysutils patch, single docker run.
#   - --colocated + tp4 + --vllm_server_mode=false (in-process LLM, not the Pathways server driver).
#   - KEEP the tpu_inference qwen3 ROPE hot-patch (scripts/logp_diff/patch_qwen3.py; A/B need it).
# Caveat: tp4 != tp8 -> the kernel-diff MAGNITUDE may differ from the 256 tp8 run; this validates
#   the method + gives a ballpark + attribution. The tp8 real value comes from Phase 3c Step 2 (256).
#
# Usage on the TPU VM (docker preinstalled):
#   gcloud auth configure-docker europe-west4-docker.pkg.dev   # one-time
#   HF_TOKEN=hf_xxx bash experimental/probe_v5p_1host_docker.sh
set -uo pipefail

IMAGE="${IMAGE:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_base_image:latest}"
BRANCH="${BRANCH:-yuxzhang/logp-diff-probe}"
MODEL_BASE="${MODEL_BASE:-/mnt/disks/linchai_data/models}"     # Qwen3-32B safetensors live here
OUT_DIR="${OUT_DIR:-/tmp/logp_probe_1host}"
N_PROMPT="${N_PROMPT:-2048}"
N_GEN="${N_GEN:-512}"
mkdir -p "$OUT_DIR"

# Self-contained input: a local .txt (probe tiles it to n_prompt tokens). Avoids HF load_dataset
# streaming as a failure mode; content barely matters for a kernel-reduction-order diff.
if [ ! -s "$OUT_DIR/sample.txt" ]; then
  for _ in $(seq 1 200); do
    echo "The quick brown fox jumps over the lazy dog. def solve(x): return sum(i*i for i in range(x))."
  done > "$OUT_DIR/sample.txt"
fi

# --privileged: TPU chip access.  --net=host: metadata-server ADC (gs:// if ever needed).
sudo docker run --rm --privileged --net=host \
  -v "$MODEL_BASE":"$MODEL_BASE" \
  -v /mnt/disks/linchai_data:/mnt/disks/linchai_data \
  -v "$OUT_DIR":"$OUT_DIR" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e BRANCH="$BRANCH" -e MODEL_BASE="$MODEL_BASE" -e OUT_DIR="$OUT_DIR" \
  -e N_PROMPT="$N_PROMPT" -e N_GEN="$N_GEN" \
  "$IMAGE" \
  bash -c '
    set -e
    git config --global --add safe.directory "$(pwd)"
    git init
    git remote set-url origin https://github.com/google/tunix.git 2>/dev/null \
      || git remote add origin https://github.com/google/tunix.git
    git fetch origin "$BRANCH"
    git reset --hard FETCH_HEAD

    # tpu_inference qwen3 ROPE hot-patch (checked-in file; A/B use tpu_inference).
    python3 scripts/logp_diff/patch_qwen3.py

    NEW_MODEL_DESIGN=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$PYTHONPATH:.:scripts/logp_diff" python3 scripts/logp_diff/logp_diff_probe.py \
      --model_path="$MODEL_BASE/Qwen3-32B" \
      --model_version=Qwen3-32B \
      --dataset="$OUT_DIR/sample.txt" \
      --n_prompt="$N_PROMPT" --n_gen="$N_GEN" \
      --colocated \
      --rollout_mesh_fsdp=1 --rollout_mesh_tp=4 \
      --train_mesh_fsdp=1 --train_mesh_tp=4 \
      --vllm_server_mode=false \
      --temperature=1.0 \
      --param_dtype=float32 --config_dtype=bfloat16 \
      --vllm_hbm_util=0.6 --vllm_max_num_seqs=64 --vllm_max_num_batched_tokens=8192 \
      --pairs=A-vs-C,A-vs-B,B-vs-C \
      --out="$OUT_DIR/report.json"
    echo "=== report at $OUT_DIR/report.json ==="
    cat "$OUT_DIR/report.json"
  '
