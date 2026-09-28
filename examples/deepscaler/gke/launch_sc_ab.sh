#!/usr/bin/env bash
# DeepScaleR Score Centering A/B on GKE (Pathways, one v5p-32 JobSet per arm).
#
#   launch_sc_ab.sh up --image IMAGE [--arm both|sc-on|sc-off] [--smoke]
#                      [--run-tag TAG] [--code-tarball gs://...] [--no-ckpt]
#                      [--dry-run]
#                                            Render the JobSet(s) and create them.
#                                            --no-ckpt: no checkpoints; a crash
#                                            restart retrains from step 0.
#   launch_sc_ab.sh down [--arm ...] [--smoke]
#                                            Delete the JobSet(s).
#   launch_sc_ab.sh status                   JobSets / workloads / pods.
#   launch_sc_ab.sh code-tarball             Upload the working tree as a tarball
#                                            for `up --code-tarball` (hotfix
#                                            without an image rebuild).
#
# Arms (the only difference between them):
#   sc-on   --score_centering
#   sc-off  --score_centering_diagnostics  (logs score_centering/* only)
#
# Prerequisites: gcloud, kubectl pointed at the cluster, and a Secret holding
# the W&B API key under the key "api-key" (see README.md). Every setting below
# can be overridden through the environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

BUILD_PROJECT="${BUILD_PROJECT:-cloud-tpu-multipod-dev}"
IMAGE_REPO="${IMAGE_REPO:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/linchai-repo/tunix_base_image}"
PATHWAYS_TAG="${PATHWAYS_TAG:-20260911-jax_0.11.0}"
RESERVATION="${RESERVATION:-cloudtpu-20260902214500-1810493672}"
PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
BUCKET="${BUCKET:-gs://linchai-bucket-dev}"
WANDB_SECRET="${WANDB_SECRET:-linchai-wandb-api-key}"
WANDB_PROJECT="${WANDB_PROJECT:-tunix-deepscaler}"
WANDB_GROUP="${WANDB_GROUP:-ds-sc-ab-v5p32-300step}"
EXPERIMENT="${EXPERIMENT:-ds-sc-ab-v5p32}"
SEED="${SEED:-42}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
RENDER_DIR="${RENDER_DIR:-$HERE/rendered}"
TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-2}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
# The GKE node service account cannot read gs://tunix, so the recipe reads a
# byte-identical mirror of gs://tunix/{data,models}/... under $BUCKET.
DATA_DIR="${DATA_DIR:-$BUCKET/data}"
MODEL_DIR="${MODEL_DIR:-$BUCKET/models}"

# Sampling / loss / LR settings follow the linchai_deepscaler branch. The actor
# must be stored in fp32: at lr 1e-6 bf16 storage rounds ~98% of the AdamW
# updates to zero (r4 ran with bf16 and its policy barely moved).
# Both arms use token-level TIS (threshold 2.0), so the A/B is TIS vs TIS+SC
# (arXiv:2609.20807 Eq. 12), as in the FrozenLake validation.
# Trainer settings follow examples/frozenlake: decoder remat, splash attention
# (block 256), bf16 compute over fp32 actor storage (bf16 for the frozen
# reference), and log-probs in 2048-token chunks.
COMMON_ARGS=(
  --batch_size 128 --mini_batch_size 128 --num_generations 8
  --train_micro_batch_size "$TRAIN_MICRO_BATCH"
  --num_epochs 1 --seed "$SEED"
  --learning_rate 1e-6 --lr_schedule constant
  --max_prompt_length 1024 --max_response_length 8192
  --temperature 0.6 --top_p 1.0
  --shuffle_data
  --loss_agg_mode sequence-mean-token-mean
  --sampler_is token --sampler_is_threshold 2.0
  --score_centering_top_k 32
  --score_centering_eps 1e-6
  --rollout_devices 8 --rollout_dp 8
  --eval_every_n_steps 1000
  --model_dtype float32 --ref_model_dtype bfloat16
  --mixed_precision --remat decoder
  --flash_attention --flash_attention_block_size 256
  --compute_logps_chunk_size 2048
)
FULL_ARGS=(--num_batches 300)
SMOKE_ARGS=(--num_batches 2)

die() { echo "error: $*" >&2; exit 1; }

arm_flag() {
  case "$1" in
    sc-on) echo "--score_centering" ;;
    sc-off) echo "--score_centering_diagnostics" ;;
    *) die "unknown arm: $1" ;;
  esac
}

jobset_name() {  # ARM SMOKE
  local name="ds-$1-s$SEED"
  [[ "$2" == 1 ]] && name="$name-smoke"
  echo "$name"
}

arms_of() {
  case "$1" in
    both) echo "sc-on sc-off" ;;
    sc-on|sc-off) echo "$1" ;;
    *) die "--arm must be both, sc-on or sc-off" ;;
  esac
}

render() {  # TEMPLATE OUT  (placeholders come from R_* variables)
  python3 - "$1" "$2" <<'PY'
import os, re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
def sub(m):
  key = "R_" + m.group(1)
  if key not in os.environ:
    sys.exit(f"unset placeholder {m.group(1)}")
  return os.environ[key]
out = re.sub(r"\{\{([A-Z_]+)\}\}", sub, text)
open(dst, "w").write(out)
PY
}

cmd_tarball() {
  local sha ts out
  sha="$(git -C "$REPO" rev-parse --short=8 HEAD)"
  ts="$(date -u +%Y%m%d-%H%M%S)"
  out="$BUCKET/code/deepscaler-sc-ab/${sha}-${ts}.tar.gz"
  echo "[sc-ab] packaging working tree -> $out"
  local tmp
  tmp="$(mktemp -d)"
  git -C "$REPO" ls-files -z -co --exclude-standard \
    | (cd "$REPO" && tar --null -T - -czf "$tmp/code.tar.gz")
  gcloud storage cp "$tmp/code.tar.gz" "$out" >/dev/null
  rm -rf "$tmp"
  echo "$out"
}

cmd_up() {
  local image="" arm="both" smoke=0 dry_run=0 run_tag="" code_tarball="" no_ckpt=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --image) image="$2"; shift 2 ;;
      --arm) arm="$2"; shift 2 ;;
      --smoke) smoke=1; shift ;;
      --dry-run) dry_run=1; shift ;;
      --run-tag) run_tag="$2"; shift 2 ;;
      --code-tarball) code_tarball="$2"; shift 2 ;;
      --no-ckpt) no_ckpt=1; shift ;;
      -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  [[ -z "$image" ]] && die "--image is required (built image tag or digest)"

  local tpl="$HERE/deepscaler_sc_ab_v5p32.yaml"
  [[ -f "$tpl" ]] || die "template not found: $tpl"
  mkdir -p "$RENDER_DIR"

  for a in $(arms_of "$arm"); do
    local js_name; js_name="$(jobset_name "$a" "$smoke")"
    local run_id="$js_name"
    [[ -n "$run_tag" ]] && run_id="$run_id-$run_tag"
    local tags="sc-ab,$a,seed-$SEED"
    [[ "$smoke" == 1 ]] && tags="$tags,smoke"
    [[ -n "$run_tag" ]] && tags="$tags,$run_tag"
    tags="$tags,mb$TRAIN_MICRO_BATCH"

    local extra_args=("${COMMON_ARGS[@]}")
    if [[ "$smoke" == 1 ]]; then
      extra_args+=("${SMOKE_ARGS[@]}")
    else
      extra_args+=("${FULL_ARGS[@]}")
    fi
    extra_args+=("$(arm_flag "$a")")
    if [[ "$no_ckpt" == 1 ]]; then
      extra_args+=(--no_ckpt)
    fi
    if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
      # Intentionally unquoted so multiple flags can be passed.
      # shellcheck disable=SC2206
      extra_args+=($EXTRA_TRAIN_ARGS)
    fi

    local ckpt_dir=""
    if [[ "$no_ckpt" != 1 ]]; then
      ckpt_dir="$BUCKET/checkpoints/$EXPERIMENT/$run_id"
    fi

    export R_JOBSET_NAME="$js_name"
    export R_EXPERIMENT="$EXPERIMENT"
    export R_ARM="$a"
    export R_MAX_RESTARTS="$MAX_RESTARTS"
    export R_IMAGE="$image"
    export R_PATHWAYS_TAG="$PATHWAYS_TAG"
    export R_RESERVATION="$RESERVATION"
    export R_PRIORITY_CLASS="$PRIORITY_CLASS"
    export R_TRAIN_ARGS="${extra_args[*]}"
    export R_CODE_TARBALL="$code_tarball"
    export R_DATA_DIR="$DATA_DIR"
    export R_MODEL_DIR="$MODEL_DIR"
    export R_CKPT_DIR="$ckpt_dir"
    export R_TB_LOG_DIR="$BUCKET/tensorboard/$EXPERIMENT/$run_id"
    export R_WANDB_SECRET="$WANDB_SECRET"
    export R_WANDB_PROJECT="$WANDB_PROJECT"
    export R_WANDB_GROUP="$WANDB_GROUP"
    export R_WANDB_RUN_NAME="$run_id"
    export R_WANDB_RUN_ID="$run_id"
    export R_WANDB_TAGS="$tags"

    local out="$RENDER_DIR/${js_name}.yaml"
    render "$tpl" "$out"
    echo "[sc-ab] rendered $out"
    if [[ "$dry_run" == 1 ]]; then
      kubectl apply --server-side --dry-run=server -f "$out"
    else
      kubectl apply --server-side -f "$out"
    fi
  done
}

cmd_down() {
  local arm="both" smoke=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --arm) arm="$2"; shift 2 ;;
      --smoke) smoke=1; shift ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  for a in $(arms_of "$arm"); do
    local js_name; js_name="$(jobset_name "$a" "$smoke")"
    echo "[sc-ab] deleting JobSet $js_name"
    kubectl delete jobset "$js_name" --ignore-not-found
  done
}

cmd_status() {
  echo "=== JobSets ==="
  kubectl get jobset -l "experiment=$EXPERIMENT" -o wide || true
  echo "=== Pods ==="
  kubectl get pods -l "experiment=$EXPERIMENT" -o wide || true
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  down) shift; cmd_down "$@" ;;
  status) shift; cmd_status "$@" ;;
  code-tarball) shift; cmd_tarball "$@" ;;
  *) sed -n '2,24p' "$0"; exit 1 ;;
esac
