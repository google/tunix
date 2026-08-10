#!/usr/bin/env bash
# Run one bounded P35 serving-versus-adapter reproduction on the existing four-chip v5p host.
set -euo pipefail

LABEL="${1:?usage: run_p35_onehost.sh <unique-label>}"
ROOT=/home/yuxuan/code_rl_repro
REPO="$ROOT/sequence_packing/p34_scheduler_contract_0809"
PKG="$REPO/canon-zero-tim"
TASK="$PKG/tasks/p35-envelope-discriminator"
CANON_ENV=/mnt/disks/tunix-data/claude_work/canon_env.sh
ASSETS=/mnt/disks/tunix-data/gsm8k_zero_tim
MODEL="$ASSETS/models"
HF_MODEL_CACHE=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-1.7B
HF_SNAPSHOT_SHA="$(cat "$HF_MODEL_CACHE/refs/main")"
HF_SNAPSHOT="$HF_MODEL_CACHE/snapshots/$HF_SNAPSHOT_SHA"
EVIDENCE=/mnt/disks/tunix-data/logp_probe_1host
IMAGE=tunix_frozenlake_image:vllm-tpu0.25.0
SOURCE_SHA=cf4c12e4003199cd80c73603f8b54a0f80f49657
SP=/usr/local/lib/python3.12/site-packages/tpu_inference
STATE="$EVIDENCE/p35_onehost_${LABEL}"
RAW="$EVIDENCE/p35_onehost_${LABEL}.raw.log"
WRAPPER="$EVIDENCE/p35_onehost_${LABEL}.wrapper.log"
RESULT="$EVIDENCE/p35_onehost_${LABEL}.result.json"
ALIGN="$STATE/alignment.jsonl"
P35_REPORT="$STATE/p35_envelope.json"
P35_CLASSIFICATION="$STATE/p35_envelope.classification.json"
REPLAY_REPORT="$STATE/p35_exact_replay.json"
REPLAY_CLASSIFICATION="$STATE/p35_exact_replay.classification.json"
METADATA="$STATE/p35_metadata"
CANON_OUT="$STATE/canon"
CONTAINER="p35_onehost_${LABEL}"
TIMEOUT_SECONDS="${P35_ONEHOST_TIMEOUT_SECONDS:-2700}"

source "$CANON_ENV"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test "$(git -C "$REPO" rev-parse HEAD)" = "$SOURCE_SHA"
git -C "$REPO" diff --quiet "$SOURCE_SHA" -- \
  tunix examples tests canon-zero-tim/install.sh canon-zero-tim/src canon-zero-tim/patches
test -s "$MODEL/config.json"
[[ "$HF_SNAPSHOT_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "invalid Qwen3-1.7B cache ref: $HF_SNAPSHOT_SHA" >&2
  exit 1
}
test -s "$ASSETS/data/gsm8k/1.0.0/dataset_info.json"
test ! -e "$STATE"
test ! -e "$RAW"
test ! -e "$WRAPPER"
test ! -e "$RESULT"
mkdir -p "$STATE/wandb" "$STATE/logs" "$METADATA"

{
  echo "[P35.ONEHOST] source=$SOURCE_SHA host=$(hostname) image=$IMAGE"
  echo "[P35.ONEHOST] topology=DP1xTP4 prompts=2 generations=8 trajectories=16 local_m=256"
  echo "[P35.ONEHOST] mutation=none stop=before_backward wandb=disabled timeout=$TIMEOUT_SECONDS"
  sha256sum "$0" "$PKG/install.sh" \
    "$REPO/tunix/rl/envelope_probe.py" \
    "$REPO/tunix/rl/canonical_qwen3_adapter.py" \
    "$REPO/tunix/rl/agentic/agentic_grpo_learner.py" \
    "$REPO/tunix/rl/rollout/vllm_rollout.py" \
    "$REPO/examples/math_gsm8k/qwen3_grpo_demo.py" \
    "$MODEL/config.json" "$MODEL/model.safetensors.index.json"
} >"$RAW"

bash "$PKG/install.sh" "$CANON_OUT" --from-image "$IMAGE" --model qwen1p7b \
  >>"$RAW" 2>&1

set +e
started="$(date +%s)"
timeout --foreground --signal=TERM --kill-after=120s "${TIMEOUT_SECONDS}s" \
sudo docker run --rm --privileged --net=host --name "$CONTAINER" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$MODEL":"$HF_SNAPSHOT":ro \
  -v "$REPO":"$REPO":ro \
  -v "$CANON_OUT":"$CANON_OUT":ro \
  -v "$CANON_OUT/attn_iface_patched.py":"$SP/layers/common/attention_interface.py":ro \
  -v "$CANON_OUT/linear_p22xk.py":"$SP/layers/jax/linear.py":ro \
  -v "$CANON_OUT/embed_patched.py":"$SP/layers/jax/embed.py":ro \
  -v "$CANON_OUT/tpu_runner_p21_l30.py":"$SP/runner/tpu_runner.py":ro \
  -v "$CANON_OUT/qwen3_p22xk.py":"$SP/models/jax/qwen3.py":ro \
  -v "$CANON_OUT/qwen2_p22xk.py":"$SP/models/jax/qwen2.py":ro \
  -e PYTHONPATH="$CANON_OUT:$REPO" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$CANON_OUT" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e GSM8K_ARTIFACT_ROOT="$ASSETS" -e GSM8K_LOG_DIR="$STATE/logs" \
  -e WANDB_MODE=disabled -e WANDB_DIR="$STATE/wandb" \
  -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e CANON_GSM8K_TRAIN=1 -e CANON_GSM8K_L3=0 \
  -e CANON_GSM8K_UPDATE_CANARY=0 -e CANON_GSM8K_GRAD_PROBE=0 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=0 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=0 -e CANON_ALIGNMENT_TRAIN=1 \
  -e CANON_ALIGN_REPORT="$ALIGN" \
  -e CANON_P35_ENVELOPE=1 -e CANON_P35_ENVELOPE_REPORT="$P35_REPORT" \
  -e CANON_P35_METADATA_DIR="$METADATA" \
  -e CANON_P35_EXACT_REPLAY=1 -e CANON_P35_EXACT_REPLAY_REPORT="$REPLAY_REPORT" \
  -e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4 -e CANON_TARGET_M=256 \
  -e CANON_P32_TRAIN_ADMITTED=0 -e FL_SHARED_MESH=1,4 \
  -e XLA_FLAGS="$XTRA_XLA" \
  -e CANON_RPA_D=128,512,128,512 -e CANON_RPA_P=128,512,128,512 \
  -e CANON_RPA_M=128,512,128,512 -e MIN_TOKEN_BUCKET=256 \
  -e CANON_FIXED_AR=1 -e CANON_FIXED_AR_EMBED=1 \
  -e CANON_RPA_VJP2=1 -e CANON_VJP2_MAX_SEQS=1 \
  -e CANON_ENGINE_MODULE_C=1 -e CANON_PROMPT_PROCESSED_LOGPROBS=1 \
  -e CANON_LOGPROB_M=256 -e CANON_PALLAS_LOGSOFTMAX=1 \
  -e CANON_PALLAS_ALL_PROJ=1 -e CANON_PALLAS_ALL_RMSNORM=1 \
  -e CANON_PALLAS_SWIGLU=1 -e CANON_PALLAS_MPAD=1 \
  -e CANON_PALLAS_SWIGLU_MPAD=1 -e CANON_PALLAS_CANONICAL_VJP=1 \
  -e CANON_QWEN3_HIDDEN_SIZE=2048 -e CANON_QWEN3_INTERMEDIATE_SIZE=6144 \
  -e CANON_QWEN3_NUM_ATTENTION_HEADS=16 -e CANON_QWEN3_NUM_KV_HEADS=8 \
  -e CANON_QWEN3_HEAD_DIM=128 -e CANON_QWEN3_TP_SIZE=4 \
  -e CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3 \
  -e CANON_CUT= -e CANON_TAIL= -e CANON_POSTRPA_M= \
  -e CANON_PALLAS_MATMUL= -e CANON_PALLAS_MATERIALIZE= \
  -w "$REPO" "$IMAGE" bash -lc "
    set -e
    for pair in \
      'attn_iface_patched.py layers/common/attention_interface.py' \
      'linear_p22xk.py layers/jax/linear.py' \
      'embed_patched.py layers/jax/embed.py' \
      'tpu_runner_p21_l30.py runner/tpu_runner.py' \
      'qwen3_p22xk.py models/jax/qwen3.py' \
      'qwen2_p22xk.py models/jax/qwen2.py'; do
      set -- \$pair
      cmp '$CANON_OUT/'\"\$1\" '$SP/'\"\$2\"
    done
    echo '[P35.ONEHOST] OVERLAY_BYTE_IDENTITY PASS files=6'
    python3 - <<'PY'
import jax
devices = jax.devices()
print(f'[P35.ONEHOST] devices={len(devices)} ids={[d.id for d in devices]} platform={jax.default_backend()}')
if len(devices) != 4 or jax.default_backend() != 'tpu':
  raise SystemExit('one-host reproduction requires exactly four TPU devices')
PY
    exec python3 -u examples/math_gsm8k/qwen3_grpo_demo.py \
      --mesh_fsdp=1 --mesh_tp=4 --batch_size=2 --mini_batch_size=2 \
      --train_micro_batch_size=2 --compute_logps_micro_batch_size=2 \
      --train_trajectory_micro_batch_size=16 \
      --max_steps=1 --num_generations=8 \
      --max_prompt_length=1024 --max_response_length=256 \
      --max_concurrency=16 --rollout_vllm_hbm_utilization=0.20 \
      --rollout_vllm_max_num_seqs=16 \
      --rollout_vllm_max_num_batched_tokens=256 \
      --wandb_project=zero-tim-p35-onehost \
      --wandb_run_name=p35-onehost-$LABEL
  " >>"$RAW" 2>&1
docker_rc=$?
elapsed="$(( $(date +%s) - started ))"
set -e

echo "[P35.ONEHOST] docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$RAW"
canon_postflight "$RAW" 2>&1 | sed 's/^/[P35.ONEHOST.POSTFLIGHT] /' >>"$RAW" || true

n_ar="$(grep -ac 'CANON_FIXED_AR=1 fixed-order tree' "$RAW" || true)"
n_embed="$(grep -ac 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' "$RAW" || true)"
n_logprob="$(grep -ac 'CANON_LOGPROB_M on' "$RAW" || true)"
n_complete="$(grep -ac '\[CANON_P35\] REPORT_COMPLETE .*STOP_BEFORE_BACKWARD' "$RAW" || true)"
n_replay="$(grep -ac '\[CANON_P35.3\] REPLAY_COMPLETE' "$RAW" || true)"
# Count the terminal exception, not the source-code echo inside the traceback.
n_local_exact="$(grep -aEc '^\[rank0\]: tunix\.rl\.envelope_probe\.EnvelopeProbeError: known A-C red was not reproduced in the current batch$' "$RAW" || true)"
n_contract_red="$(grep -ac 'C7/C8 violation \[post-import\]' "$RAW" || true)"
printf '[P35.ONEHOST] PATHTRACE fixed_ar=%s embed=%s logprob=%s complete=%s replay=%s local_exact=%s contract_red=%s\n' \
  "$n_ar" "$n_embed" "$n_logprob" "$n_complete" "$n_replay" "$n_local_exact" "$n_contract_red" \
  >>"$RAW"

if [ "$n_contract_red" -ne 0 ] || [ "$n_ar" -eq 0 ] || [ "$n_embed" -eq 0 ] || [ "$n_logprob" -eq 0 ]; then
  verdict=VOID_CONTRACT
  final_rc=1
elif [ "$n_complete" -eq 1 ] && [ "$n_replay" -eq 1 ] && \
     [ -s "$P35_REPORT" ] && [ -s "$REPLAY_REPORT" ]; then
  JAX_PLATFORMS=cpu PYTHONPATH="$REPO" python3 "$PKG/tests/p35_envelope/classify_envelope.py" \
    --report "$P35_REPORT" --output "$P35_CLASSIFICATION"
  JAX_PLATFORMS=cpu PYTHONPATH="$REPO" python3 "$PKG/tests/p35_envelope/classify_exact_replay.py" \
    --report "$REPLAY_REPORT" --output "$REPLAY_CLASSIFICATION"
  verdict=LOCAL_REPRODUCED_AND_CLASSIFIED
  final_rc=0
elif [ "$n_local_exact" -eq 1 ] && [ "$n_complete" -eq 0 ] && [ "$n_replay" -eq 0 ]; then
  verdict=LOCAL_NOT_REPRODUCED
  final_rc=0
else
  verdict=INCONCLUSIVE
  final_rc=1
fi

python3 - "$RESULT" "$verdict" "$SOURCE_SHA" "$docker_rc" "$elapsed" \
  "$n_ar" "$n_embed" "$n_logprob" "$n_complete" "$n_replay" "$n_local_exact" <<'PY'
import json
import pathlib
import sys

path, verdict, source, docker_rc, elapsed, *counts = sys.argv[1:]
payload = {
    'schema_version': 1,
    'verdict': verdict,
    'source_commit': source,
    'topology': 'DP1xTP4-direct-attached',
    'prompts': 2,
    'generations': 8,
    'trajectories': 16,
    'local_m': 256,
    'docker_exit': int(docker_rc),
    'elapsed_seconds': int(elapsed),
    'counts': dict(zip(
        ('fixed_ar', 'fixed_ar_embed', 'logprob_m', 'p35_complete',
         'p35_replay', 'known_red_not_reproduced'),
        map(int, counts),
    )),
    'claims': {
        'backward_executed': False,
        'optimizer_update': False,
        'wandb_mutation': False,
        'pathways_target_verdict': False,
    },
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, sort_keys=True))
PY

evidence_files=("$RAW" "$RESULT" "$0")
for optional_path in \
  "$P35_REPORT" "$P35_CLASSIFICATION" "$REPLAY_REPORT" "$REPLAY_CLASSIFICATION"; do
  [ ! -s "$optional_path" ] || evidence_files+=("$optional_path")
done
sha256sum "${evidence_files[@]}" | tee -a "$WRAPPER"
echo "[P35.ONEHOST] verdict=$verdict result=$RESULT" | tee -a "$WRAPPER"
exit "$final_rc"
