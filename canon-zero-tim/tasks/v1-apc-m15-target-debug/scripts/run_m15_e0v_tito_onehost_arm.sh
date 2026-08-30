#!/usr/bin/env bash
# Run one bounded three-round exact-TiTO M15 APC arm on a four-chip v5p host.
set -euo pipefail

arm="${1:?usage: run_m15_e0v_tito_onehost_arm.sh <off|on> <unique-label> <new-pair-root>}"
label="${2:?usage: run_m15_e0v_tito_onehost_arm.sh <off|on> <unique-label> <new-pair-root>}"
pair_root="${3:?usage: run_m15_e0v_tito_onehost_arm.sh <off|on> <unique-label> <new-pair-root>}"
case "$arm" in
  off) apc=0 ;;
  on) apc=1 ;;
  *) echo "[M15.E0V.ONEHOST.ARM] REFUSING arm must be off or on" >&2; exit 2 ;;
esac
case "$label" in
  *[!a-z0-9-]*|'')
    echo "[M15.E0V.ONEHOST.ARM] REFUSING label must be lowercase DNS text" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
state="$pair_root/$arm"
raw="$state/raw.log"
source_diff="$state/source.diff"
pre_report="$state/pre_alignment.jsonl"
round_file="$state/diagnostic_round"
alignment="$state/alignment.classification.json"
tito="$state/tito.classification.json"
contract="$state/RUN_CONTRACT.json"
canon_out="$state/canon"
container="m15_e0v_${label}_${arm}"
hf_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
hf_snapshot_sha="$(head -n 1 "$hf_cache/refs/main")"
model="$hf_cache/snapshots/$hf_snapshot_sha"
data=/mnt/disks/tunix-data/frozenlake/data_convergence_v1
deps=/mnt/disks/tunix-data/logp_probe_1host/p38_incident_deps_nodeps
image=tunix_frozenlake_image:vllm-tpu0.25.0
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
timeout_seconds="${M15_E0V_ONEHOST_TIMEOUT_SECONDS:-3600}"

test "$(hostname)" = "${TIM_C2_EXPECT_HOSTNAME:-t1v-n-4a77ebd0-w-0}"
test -d "$pair_root"
test ! -e "$state"
test -s "$model/config.json"
test -s "$model/model.safetensors.index.json"
test -s "$data/train.parquet"
test -s "$deps/gymnasium/__init__.py"

running_tpu="$(sudo docker ps --filter "ancestor=$image" --format '{{.Names}}')"
if [ -n "$running_tpu" ]; then
  echo "[M15.E0V.ONEHOST.ARM] REFUSING pinned-image TPU lane is busy" >&2
  exit 4
fi
if sudo docker ps --format '{{.Names}}' | grep -Eq '^(m15_e0v_|p3_apc_|p38_incident_|p51_|p59_|p62_|p66_)'; then
  echo "[M15.E0V.ONEHOST.ARM] REFUSING another registered TPU carrier is active" >&2
  exit 4
fi

mkdir "$state"
printf '0\n' > "$round_file"
git -C "$repo" diff --binary HEAD > "$source_diff"
source_commit="$(git -C "$repo" rev-parse HEAD)"
source_diff_sha="$(sha256sum "$source_diff" | awk '{print $1}')"
image_id="$(sudo docker image inspect --format '{{.Id}}' "$image")"
runner_sha="$(sha256sum "$0" | awk '{print $1}')"
token_sha="$(sha256sum "$repo/tunix/rl/agentic/token_continuity.py" | awk '{print $1}')"
rollout_sha="$(sha256sum "$repo/tunix/rl/rollout/vllm_rollout.py" | awk '{print $1}')"
pair_classifier_sha="$(sha256sum "$script_dir/classify_m15_e0v_onehost_pair.py" | awk '{print $1}')"
arm_classifier_sha="$(sha256sum "$script_dir/classify_m15_e0v_onehost_arm.py" | awk '{print $1}')"
tito_classifier_sha="$(sha256sum "$script_dir/classify_m15_apc_debug_tito.py" | awk '{print $1}')"

{
  echo "[M15.E0V.ONEHOST.ARM] source=$source_commit diff_sha256=$source_diff_sha arm=$arm"
  echo "[M15.E0V.ONEHOST.ARM] image_id=$image_id topology=DP1xTP4 rounds=3"
  echo "[M15.E0V.ONEHOST.ARM] A=rollout B=full-reset-prefill C=trainer-old"
  echo "[M15.E0V.ONEHOST.ARM] backward=0 optimizer_commits=0 target_executed=0"
  echo "[M15.E0V.ONEHOST] exact TITO enabled mode=exact arm=$arm topology=DP1xTP4 rounds=3"
  sha256sum "$0" "$script_dir/classify_m15_e0v_onehost_arm.py" \
    "$script_dir/classify_m15_apc_debug_tito.py" \
    "$script_dir/classify_m15_e0v_onehost_pair.py" \
    "$pkg/install.sh" "$repo/tunix/rl/agentic/token_continuity.py" \
    "$repo/tunix/rl/agentic/trajectory/trajectory_collect_engine.py" \
    "$repo/tunix/rl/alignment.py" \
    "$repo/examples/frozenlake/train_frozenlake_qwen3.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} > "$raw"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen8b \
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
  -e CANON_VLLM_ENABLE_PREFIX_CACHING="$apc" \
  -e CANON_M15_TOKEN_CONTINUITY=exact \
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
  -e CANON_PRE_ALIGN_GATE=1 -e CANON_PRE_ALIGN_REPORT="$pre_report" \
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
    python3 -u examples/frozenlake/train_frozenlake_qwen3.py \\
      --mesh_dp=1 --mesh_tp=4 \\
      --batch_size=2 --mini_batch_size=2 \\
      --train_trajectory_micro_batch_size=4 \\
      --max_steps=1 --num_generations=2 --num_batches=3 \\
      --max_prompt_length=4096 --max_response_length=512 \\
      --max_concurrency=1 --vllm_max_num_seqs=4 \\
      --vllm_max_num_batched_tokens=256 --env_max_steps=15 \\
      --temperature=0.7 --top_k=0 --top_p=1.0 --seed=42 \\
      --p57_workload_candidate=m15 --p57_data_split=main
    workload_rc=\$?
    chmod a+r '$pre_report' '$round_file' 2>/dev/null || true
    exit \$workload_rc
  " >> "$raw" 2>&1
docker_rc=$?
elapsed=$(( $(date +%s) - started ))
set -e
echo "[M15.E0V.ONEHOST.ARM] docker_exit=$docker_rc elapsed_seconds=$elapsed" >> "$raw"

python3 - "$contract" "$arm" "$apc" "$source_commit" "$source_diff_sha" \
  "$image_id" "$docker_rc" "$elapsed" "$runner_sha" "$token_sha" \
  "$rollout_sha" "$arm_classifier_sha" "$pair_classifier_sha" \
  "$tito_classifier_sha" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
report = {
    "schema": "m15-e0v-tito-onehost-arm-v1",
    "arm": sys.argv[2],
    "apc": int(sys.argv[3]),
    "source_commit": sys.argv[4],
    "source_diff_sha256": sys.argv[5],
    "image_id": sys.argv[6],
    "docker_exit": int(sys.argv[7]),
    "elapsed_seconds": int(sys.argv[8]),
    "runner_sha256": sys.argv[9],
    "token_continuity_sha256": sys.argv[10],
    "vllm_rollout_sha256": sys.argv[11],
    "arm_classifier_sha256": sys.argv[12],
    "pair_classifier_sha256": sys.argv[13],
    "tito_classifier_sha256": sys.argv[14],
    "program_identity": "m15-exact-tito-onehost-rehearsal-v1",
    "m15_token_continuity": "exact",
    "topology": "DP1xTP4",
    "rounds": 3,
    "backward": 0,
    "optimizer_commits": 0,
    "historical_1226_prefix_reused": False,
    "target_executed": False,
}
path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

if [ "$docker_rc" -ne 42 ]; then
  echo "[M15.E0V.ONEHOST.ARM] INCONCLUSIVE arm=$arm docker_exit=$docker_rc state=$state" >&2
  exit 5
fi
test "$(grep -ac '^\[CANON_P38\] PRECHECK_ROUND_COMPLETE ' "$raw" || true)" -eq 3
test "$(tr -d '[:space:]' < "$round_file")" = 2
test -s "$pre_report"

python3 "$script_dir/classify_m15_e0v_onehost_arm.py" \
  --raw "$raw" --report "$pre_report" --arm "$arm" \
  --output "$alignment" >> "$raw" 2>&1
python3 "$script_dir/classify_m15_apc_debug_tito.py" \
  --run-log "$raw" --arm "$arm" --scope onehost --expected-rounds 3 \
  --output "$tito" >> "$raw" 2>&1

# The classifiers append their terminal receipts to raw.log. Refresh their
# source binding to the final immutable raw bytes before sealing the arm.
python3 - "$raw" "$alignment" "$tito" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

raw, alignment_path, tito_path = map(Path, sys.argv[1:])
digest = hashlib.sha256(raw.read_bytes()).hexdigest()
alignment = json.loads(alignment_path.read_text())
tito = json.loads(tito_path.read_text())
alignment["raw_sha256"] = digest
tito["run_log_sha256"] = digest
alignment_path.write_text(json.dumps(alignment, sort_keys=True, indent=2) + "\n")
tito_path.write_text(json.dumps(tito, sort_keys=True, indent=2) + "\n")
PY

for name in RUN_CONTRACT.json alignment.classification.json diagnostic_round \
  pre_alignment.jsonl raw.log source.diff tito.classification.json; do
  sha256sum "$state/$name"
done | sed "s#  $state/#  #" > "$state/SHA256SUMS"
(cd "$state" && sha256sum -c SHA256SUMS --quiet)
arm_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$alignment")"
echo "[M15.E0V.ONEHOST.ARM] COMPLETE arm=$arm status=$arm_status rounds=3 B-C=0 tito_exact=1 backward=0 optimizer_commits=0 state=$state"
