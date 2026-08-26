#!/usr/bin/env bash
# Run one immutable P66 full-Qwen DP1xTP4 group-0 backward arm.
set -euo pipefail

arm="${1:?usage: run_onehost_tp4_arm.sh <tp4-serial|tp4-p59-old|tp4-p59|tp4-gather-off|tp4-vma-oracle> <unique-label>}"
label="${2:?usage: run_onehost_tp4_arm.sh <tp4-serial|tp4-p59-old|tp4-p59|tp4-gather-off|tp4-vma-oracle> <unique-label>}"
case "$arm" in
  tp4-serial) rank_parallel=0; check_vma=0; fixed_gather=1 ;;
  tp4-p59-old) rank_parallel=1; check_vma=0; fixed_gather=1 ;;
  tp4-p59) rank_parallel=1; check_vma=1; fixed_gather=1 ;;
  tp4-gather-off) rank_parallel=1; check_vma=1; fixed_gather=0 ;;
  tp4-vma-oracle) rank_parallel=1; check_vma=1; fixed_gather=1 ;;
  *) echo "[P66.TP4] invalid arm: $arm" >&2; exit 2 ;;
esac
case "$label" in
  *[!a-zA-Z0-9_-]*|'') echo "[P66.TP4] invalid label: $label" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
assets=/mnt/disks/tunix-data/gsm8k_zero_tim
model="$assets/models"
hf_model_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-1.7B
hf_snapshot_sha="$(head -n 1 "$hf_model_cache/refs/main")"
hf_snapshot="$hf_model_cache/snapshots/$hf_snapshot_sha"
evidence=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
root="$evidence/p66_tp4_${arm}_${label}"
canon_out="$root/canon"
state="$root/train"
raw="$state/raw.log"
pre="$state/pre_alignment.jsonl"
align="$state/alignment.jsonl"
update="$state/update.json"
classification="$state/classification.json"
driver="$root/driver.log"
container="p66_tp4_${arm}_${label}"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
timeout_seconds="${P66_TP4_TIMEOUT_SECONDS:-7200}"

# shellcheck disable=SC1090
source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -s "$model/config.json"
test -s "$assets/data/gsm8k/1.0.0/dataset_info.json"
[[ "$hf_snapshot_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[P66.TP4] invalid Qwen3-1.7B cache ref" >&2
  exit 2
}
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "${NETRC:-$HOME/.netrc}" ]; then
  WANDB_API_KEY="$(awk '
    $1 == "machine" { machine = $2 }
    machine == "api.wandb.ai" {
      for (i = 1; i <= NF; i++) {
        if ($i == "password" && i < NF) { print $(i + 1); exit }
      }
    }
  ' "${NETRC:-$HOME/.netrc}")"
  export WANDB_API_KEY
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[P66.TP4] WANDB_API_KEY or api.wandb.ai netrc entry is required" >&2
  exit 2
fi
active="$(sudo docker ps --format '{{.Names}}' | grep -E '^(p51_|p59_|p66_)' || true)"
if [ -n "$active" ]; then
  echo "[P66.TP4] REFUSING: another TPU diagnostic container is active" >&2
  printf '%s\n' "$active" >&2
  exit 2
fi
if [ -e "$root" ]; then
  echo "[P66.TP4] REFUSING: run label already exists: $root" >&2
  exit 2
fi
mkdir -p "$state/p66_backward" "$state/wandb" "$state/logs"

source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
{
  echo "[P66.TP4] source=$source_sha diff_sha256=$diff_sha image=$image"
  echo "[P66.TP4] arm=$arm rank_parallel=$rank_parallel check_vma=$check_vma fixed_ar_gather=$fixed_gather"
  echo "[P66.TP4] topology=DP1xTP4 prompts=2 generations=8 trajectories=16 groups=16 reverse_groups=1"
  echo "[P66.TP4] optimizer_commits=0 grad_norm_fatal_threshold=1e6 performance_eligible=0"
} >"$driver"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen1p7b \
  >>"$driver" 2>&1

{
  echo "[P66.TP4] RUN_BEGIN"
  sha256sum \
    "$repo/tunix/rl/agentic/agentic_rl_learner.py" \
    "$repo/tunix/rl/dp_workloads.py" \
    "$repo/tunix/rl/canonical_qwen3_adapter.py" \
    "$repo/tunix/rl/p66_vjp_oracle.py" \
    "$pkg/src/engine_shims/p38_fixed_lm_head.py" \
    "$pkg/src/engine_shims/linear_p22xf.py" \
    "$pkg/cluster/profiles/qwen3-1p7b-dp1-tp4-gsm8k-p66.env" \
    "$pkg/tests/p66_backward/classify_tp4_arm.py" \
    "$script_dir/run_tp4_inner.sh" "$0" \
    "$model/config.json" "$model/model.safetensors.index.json"
} >"$raw"

set +e
started="$(date +%s)"
timeout --signal=TERM --kill-after=180s "${timeout_seconds}s" \
sudo docker run --rm --privileged --net=host --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$model":"$hf_snapshot":ro \
  -v "$repo":"$repo":ro \
  -v "$canon_out":"$canon_out":ro \
  -v "$canon_out/attn_iface_patched.py":"$sp/layers/common/attention_interface.py":ro \
  -v "$canon_out/linear_p22xk.py":"$sp/layers/jax/linear.py":ro \
  -v "$canon_out/embed_patched.py":"$sp/layers/jax/embed.py":ro \
  -v "$canon_out/tpu_runner_p21_l30.py":"$sp/runner/tpu_runner.py":ro \
  -v "$canon_out/qwen3_p22xk.py":"$sp/models/jax/qwen3.py":ro \
  -v "$canon_out/qwen2_p22xk.py":"$sp/models/jax/qwen2.py":ro \
  -v "$canon_out/rpa_kernel_p66.py":"$sp/kernels/ragged_paged_attention/v3/kernel.py":ro \
  -e PYTHONPATH="$canon_out:$repo" -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e GSM8K_ARTIFACT_ROOT="$assets" -e GSM8K_LOG_DIR="$state/logs" \
  -e WANDB_API_KEY="$WANDB_API_KEY" -e WANDB_MODE=online \
  -e WANDB_DIR="$state/wandb" \
  -e P66_REPO="$repo" -e P66_XLA_FLAGS="$XTRA_XLA" \
  -e JOBSET_RESTART_ATTEMPT=0 \
  -e CANON_PROFILE_FILE=cluster/profiles/qwen3-1p7b-dp1-tp4-gsm8k-p66.env \
  -e CANON_P32_TRAIN_ADMITTED=1 -e CANON_P32_DP_REDUCTION_ADMITTED=1 \
  -e CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1 \
  -e CANON_P33_RUN_STAGE=backward-no-commit -e CANON_P33_NO_COMMIT=1 \
  -e CANON_OPT_STATE_RESIDENT=1 -e CANON_P30_OPT_STATE_OFFLOAD=0 \
  -e CANON_P60_DETERMINISTIC_AB=1 \
  -e CANON_P59_RANK_PARALLEL_BACKWARD="$rank_parallel" \
  -e CANON_P66_P59_CHECK_VMA="$check_vma" \
  -e CANON_FIXED_AR_GATHER="$fixed_gather" \
  -e CANON_P66_BACKWARD_ARM="$arm" \
  -e CANON_P66_BACKWARD_CAPTURE_DIR="$state/p66_backward" \
  -e CANON_P63_OVERFLOW_SAFE_CLIP=0 \
  -e CANON_WANDB_RUN_NAME="p66-${arm}-${label}" \
  -e CANON_PRE_ALIGN_REPORT="$pre" -e CANON_ALIGN_REPORT="$align" \
  -e CANON_UPDATE_REPORT="$update" \
  -e CANON_EXPECT_MODEL_MESH_IDS=0,1,2,3 \
  -e CANON_EXPECT_TRAIN_MESH_IDS=0,1,2,3 \
  -e CANON_XPROF_DIR= -e CANON_P59_XPROF_BACKWARD_DIR= \
  -w "$repo" "$image" bash "$script_dir/run_tp4_inner.sh" \
  >>"$raw" 2>&1 &
docker_wait_pid=$!
wait "$docker_wait_pid"
docker_rc=$?
set -e
if sudo docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
  echo "[P66.TP4] post_run_cleanup stopping lingering container" >>"$raw"
  sudo docker stop "$container" >/dev/null 2>&1 || true
fi
elapsed=$(( $(date +%s) - started ))
echo "[P66.TP4] RUN_END docker_exit=$docker_rc elapsed_seconds=$elapsed" >>"$raw"
chmod -R a+rX "$state" || true

set +e
python3 "$pkg/tests/p66_backward/classify_tp4_arm.py" \
  --arm "$arm" --run-log "$raw" --pre-alignment-report "$pre" \
  --alignment-report "$align" --update-report "$update" \
  --docker-exit "$docker_rc" --output "$classification" \
  >>"$driver" 2>&1
classifier_rc=$?
set -e

verdict=MISSING
if [ -s "$classification" ]; then
  verdict="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$classification")"
fi
case "$arm:$docker_rc:$classifier_rc:$verdict" in
  tp4-p59-old:*:0:EXPECTED_RED) ;;
  tp4-serial:0:0:PASS|tp4-p59:0:0:PASS|tp4-gather-off:0:0:PASS|tp4-vma-oracle:0:0:PASS) ;;
  *)
    echo "[P66.TP4] RED arm=$arm docker=$docker_rc classifier=$classifier_rc verdict=$verdict" >>"$driver"
    sha256sum "$raw" "$driver" "$classification" "$0" >"$root/SHA256SUMS" 2>/dev/null || true
    exit 1
    ;;
esac
echo "[P66.TP4] GREEN arm=$arm verdict=$verdict docker=$docker_rc classifier=$classifier_rc" >>"$driver"
manifest_inputs=("$raw" "$driver" "$classification" "$pre" "$0" "$script_dir/run_tp4_inner.sh")
for optional in "$align" "$update"; do
  [ ! -s "$optional" ] || manifest_inputs+=("$optional")
done
sha256sum "${manifest_inputs[@]}" >"$root/SHA256SUMS"
echo "P66_TP4_ARM_COMPLETE arm=$arm verdict=$verdict evidence=$root"
