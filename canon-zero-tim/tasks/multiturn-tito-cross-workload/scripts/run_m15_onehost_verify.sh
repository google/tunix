#!/usr/bin/env bash
# Run one strict APC-off M15 verify or exact TiTO arm on direct v5p.
set -euo pipefail

label="${1:?usage: run_m15_onehost_verify.sh <fresh-lowercase-label> [verify|exact]}"
mode="${2:-verify}"
if [[ ! "$label" =~ ^[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?$ ]]; then
  echo "[M15.TITO.ONEHOST] REFUSING invalid label" >&2
  exit 2
fi
if [ "$mode" != verify ] && [ "$mode" != exact ]; then
  echo "[M15.TITO.ONEHOST] REFUSING mode must be verify or exact" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
root="/mnt/disks/tunix-data/logp_probe_1host/m15_tito_${mode}_${label}"
raw="$root/raw.log"
report="$root/pre_alignment.jsonl"
classification="$root/classification.json"
contract="$root/RUN_CONTRACT.json"
round_file="$root/diagnostic_round"
source_diff="$root/source.diff"
canon_out="$root/canon"
container="m15_tito_${mode}_${label}"
hf_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
hf_snapshot_sha="$(head -n 1 "$hf_cache/refs/main")"
model="$hf_cache/snapshots/$hf_snapshot_sha"
data=/mnt/disks/tunix-data/frozenlake/data_convergence_v1
deps=/mnt/disks/tunix-data/logp_probe_1host/p38_incident_deps_nodeps
image=tunix_frozenlake_image:vllm-tpu0.25.0
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
timeout_seconds="${M15_TITO_ONEHOST_TIMEOUT_SECONDS:-3600}"

test "$(hostname)" = "${TIM_C2_EXPECT_HOSTNAME:-t1v-n-4a77ebd0-w-0}"
test ! -e "$root"
test -s "$model/config.json"
test -s "$model/model.safetensors.index.json"
test -s "$data/train.parquet"
test -s "$deps/gymnasium/__init__.py"

running_tpu="$(sudo docker ps --filter "ancestor=$image" --format '{{.Names}}')"
if [ -n "$running_tpu" ]; then
  echo "[M15.TITO.ONEHOST] REFUSING pinned-image TPU lane is busy" >&2
  exit 4
fi
if sudo docker ps --format '{{.Names}}' | grep -Eq '^(m15_|p3_apc_|p38_|p51_|p59_|p62_|p66_)'; then
  echo "[M15.TITO.ONEHOST] REFUSING another registered TPU carrier is active" >&2
  exit 4
fi

mkdir "$root"
mkdir "$root/tmp"
printf '0\n' > "$round_file"
git -C "$repo" diff --binary HEAD > "$source_diff"
git -C "$repo" ls-files --others --exclude-standard -z | \
  xargs -0 -r sha256sum > "$root/untracked.SHA256SUMS"
source_commit="$(git -C "$repo" rev-parse HEAD)"
source_diff_sha="$(sha256sum "$source_diff" | awk '{print $1}')"
image_id="$(sudo docker image inspect --format '{{.Id}}' "$image")"

{
  echo "[M15.TITO.ONEHOST] START label=$label source=$source_commit diff_sha256=$source_diff_sha"
  echo "[M15.TITO.ONEHOST] image_id=$image_id topology=DP1xTP4 rounds=3"
  echo "[M15.TITO.ONEHOST] mode=$mode apc=0 A=rollout B=full-reset-prefill C=trainer-old"
  echo "[M15.TITO.ONEHOST] backward=0 optimizer_commits=0 target_executed=0"
  sha256sum "$0" "$script_dir/classify_m15_onehost_verify.py" \
    "$pkg/install.sh" "$repo/tunix/rl/agentic/token_continuity.py" \
    "$repo/tunix/rl/agentic/trajectory/trajectory_collect_engine.py" \
    "$repo/tunix/rl/alignment.py" \
    "$repo/examples/frozenlake/train_frozenlake_qwen3.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} > "$raw"

TMPDIR="$root/tmp" bash "$pkg/install.sh" "$canon_out" \
  --from-image "$image" --model qwen8b \
  >> "$raw" 2>&1

xla_flags='--xla_cpu_max_isa=AVX2 --xla_allow_excess_precision=false'
set +e
started="$(date +%s)"
timeout --foreground --signal=TERM --kill-after=120s "${timeout_seconds}s" \
sudo docker run --rm --privileged --net=host --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$repo":"$repo":ro \
  -v "$canon_out":"$canon_out":ro \
  -v "$canon_out/attn_iface_patched.py":"$sp/layers/common/attention_interface.py":ro \
  -v "$canon_out/linear_p22xk.py":"$sp/layers/jax/linear.py":ro \
  -v "$canon_out/embed_patched.py":"$sp/layers/jax/embed.py":ro \
  -v "$canon_out/tpu_runner_p21_l30.py":"$sp/runner/tpu_runner.py":ro \
  -v "$canon_out/qwen3_p22xk.py":"$sp/models/jax/qwen3.py":ro \
  -v "$canon_out/qwen2_p22xk.py":"$sp/models/jax/qwen2.py":ro \
  -e PYTHONPATH="$deps:$canon_out:$repo" -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e MODEL_DOWNLOAD_DIR="$model" -e FROZENLAKE_DATA_DIR="$data" \
  -e WANDB_MODE=disabled -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e CANON_FROZENLAKE_L3=1 \
  -e CANON_VLLM_ENABLE_PREFIX_CACHING=0 \
  -e CANON_M15_TOKEN_CONTINUITY="$mode" \
  -e CANON_PROFILE_FILE= -e CANON_PROFILE= \
  -e CANON_APC_M15_TARGET_DEBUG= \
  -e CANON_P57_TIM_ARM= -e CANON_P57_RUN_KIND= \
  -e CANON_P57_EXPECTED_UPDATES= -e CANON_P57_STOP_AFTER_STEP= \
  -e CANON_FROZENLAKE_CKPT_MODE= \
  -e CANON_FROZENLAKE_CKPT_ROOT= -e CANON_FROZENLAKE_CKPT_TAG= \
  -e CANON_FROZENLAKE_CKPT_INTERVAL= \
  -e CANON_FROZENLAKE_CKPT_MAX_TO_KEEP= \
  -e CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL= \
  -e CANON_P38_DURABILITY_PROFILE= -e CANON_P38_SEAM_OBSERVER= \
  -e CANON_P38_TAIL_OBSERVER= \
  -e CANON_P32_WORKLOAD= -e CANON_V1_HP_FULL=0 \
  -e CANON_P57_WORKLOAD_CANDIDATE=m15 -e CANON_P57_DATA_SPLIT=main \
  -e CANON_P33_RUN_STAGE=backward-no-commit -e CANON_P33_NO_COMMIT=1 \
  -e CANON_P33_ENABLE_EVAL=0 -e CANON_P33_DISABLE_EVAL=1 \
  -e CANON_P31_ENABLE_EVAL=0 -e CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=0 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=1 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=0 -e CANON_ALIGNMENT_TRAIN=0 \
  -e CANON_PRE_ALIGN_GATE=1 -e CANON_PRE_ALIGN_REPORT="$report" \
  -e CANON_P38_PRECHECK_ONLY=1 -e CANON_P38_CONTROLLED_EXIT=1 \
  -e CANON_P38_DIAGNOSTIC_ROUNDS=3 \
  -e CANON_P38_ONEHOST_REHEARSAL=1 \
  -e CANON_P38_DIAGNOSTIC_ROUND_FILE="$round_file" \
  -e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4 -e CANON_TARGET_M=256 \
  -e CANON_P32_TRAIN_ADMITTED=0 -e FL_SHARED_MESH=1,4 \
  -e XLA_FLAGS="$xla_flags" \
  -e CANON_RPA_D=128,512,128,512 -e CANON_RPA_P=128,512,128,512 \
  -e CANON_RPA_M=128,512,128,512 -e MIN_TOKEN_BUCKET=256 \
  -e CANON_FIXED_AR=1 -e CANON_FIXED_AR_EMBED=1 \
  -e CANON_RPA_VJP2=1 -e CANON_VJP2_MAX_SEQS=1 \
  -e CANON_ENGINE_MODULE_C=1 -e CANON_PROMPT_PROCESSED_LOGPROBS=1 \
  -e CANON_LOGPROB_M=256 -e CANON_PALLAS_LOGSOFTMAX=1 \
  -e CANON_PALLAS_ALL_PROJ=1 -e CANON_PALLAS_ALL_RMSNORM=1 \
  -e CANON_PALLAS_SWIGLU=1 -e CANON_PALLAS_MPAD=1 \
  -e CANON_PALLAS_SWIGLU_MPAD=1 -e CANON_PALLAS_CANONICAL_VJP=1 \
  -e CANON_QWEN3_HIDDEN_SIZE=4096 -e CANON_QWEN3_INTERMEDIATE_SIZE=12288 \
  -e CANON_QWEN3_NUM_ATTENTION_HEADS=32 -e CANON_QWEN3_NUM_KV_HEADS=8 \
  -e CANON_QWEN3_HEAD_DIM=128 -e CANON_QWEN3_TP_SIZE=4 \
  -e CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3 \
  -e FL_VLLM_HBM_UTIL=0.56 -e FL_STATELESS_OPTIMIZER=1 \
  -w "$repo" "$image" bash -lc "
    set +e
    python3 -u examples/frozenlake/train_frozenlake_qwen3.py \
      --mesh_dp=1 --mesh_tp=4 \
      --batch_size=2 --mini_batch_size=2 \
      --train_trajectory_micro_batch_size=4 \
      --max_steps=1 --num_generations=2 --num_batches=3 \
      --max_prompt_length=4096 --max_response_length=512 \
      --max_concurrency=1 --vllm_max_num_seqs=4 \
      --vllm_max_num_batched_tokens=256 --env_max_steps=15 \
      --temperature=0.7 --top_k=0 --top_p=1.0 --seed=42 \
      --p57_workload_candidate=m15 --p57_data_split=main
    workload_rc=\$?
    chmod a+r '$report' '$round_file' 2>/dev/null || true
    exit \$workload_rc
  " >> "$raw" 2>&1
docker_rc=$?
elapsed=$(( $(date +%s) - started ))
set -e
echo "[M15.TITO.ONEHOST] docker_exit=$docker_rc elapsed_seconds=$elapsed" >> "$raw"

python3 - "$contract" "$source_commit" "$source_diff_sha" "$image_id" \
  "$docker_rc" "$elapsed" "$mode" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "canon.m15-onehost-token-continuity-contract.v2",
    "source_commit": sys.argv[2],
    "source_diff_sha256": sys.argv[3],
    "image_id": sys.argv[4],
    "docker_exit": int(sys.argv[5]),
    "elapsed_seconds": int(sys.argv[6]),
    "program_identity": (
        "m15-rendered-text-onehost-observer-v1"
        if sys.argv[7] == "verify"
        else "m15-exact-tito-onehost-v1"
    ),
    "m15_token_continuity": sys.argv[7],
    "apc": 0,
    "topology": "DP1xTP4",
    "rounds": 3,
    "backward": 0,
    "optimizer_commits": 0,
    "target_executed": False,
}, indent=2, sort_keys=True) + "\n")
PY

if [ "$docker_rc" -ne 42 ]; then
  echo "[M15.TITO.ONEHOST] INCONCLUSIVE docker_exit=$docker_rc root=$root" >&2
  sha256sum "$raw" "$contract" "$source_diff" > "$root/SHA256SUMS"
  exit 5
fi
test -s "$report"
test "$(grep -ac '^\[CANON_P38\] PRECHECK_ROUND_COMPLETE ' "$raw" || true)" -eq 3
test "$(tr -d '[:space:]' < "$round_file")" = 2

python3 "$script_dir/classify_m15_onehost_verify.py" \
  --raw "$raw" --report "$report" --output "$classification" \
  --mode "$mode" >> "$raw" 2>&1

# The classifier hashes raw.log before appending its terminal line. Rebind the
# classification to the final raw bytes before sealing the evidence set.
python3 - "$raw" "$classification" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

raw, classification = map(Path, sys.argv[1:])
result = json.loads(classification.read_text())
result["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
classification.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY

for name in RUN_CONTRACT.json classification.json diagnostic_round \
  pre_alignment.jsonl raw.log source.diff untracked.SHA256SUMS; do
  sha256sum "$root/$name"
done | sed "s#  $root/#  #" > "$root/SHA256SUMS"
(cd "$root" && sha256sum -c SHA256SUMS --quiet)
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$classification")"
echo "[M15.TITO.ONEHOST] COMPLETE status=$status topology=DP1xTP4 rounds=3 backward=0 optimizer_commits=0 root=$root"
