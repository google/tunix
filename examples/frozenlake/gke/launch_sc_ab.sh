#!/usr/bin/env bash
# FrozenLake Score Centering A/B on GKE (Pathways, one v5p-32 JobSet per arm).
#
#   launch_sc_ab.sh build                    Snapshot the working tree and start
#                                            an async Cloud Build (cloudbuild.yaml).
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
# 20k-prompt 5x5-12x12 train set (same generator as grid5to12, same test set),
# so 200 steps x 64 prompts never repeat a prompt.
DATA_DIR="${DATA_DIR:-$BUCKET/data/frozenlake/grid5to12_train20k}"
WANDB_SECRET="${WANDB_SECRET:-linchai-wandb-api-key}"
WANDB_PROJECT="${WANDB_PROJECT:-tunix-frozenlake}"
WANDB_GROUP="${WANDB_GROUP:-sc-ab-v5p32-g5to12-8k-bs64}"
EXPERIMENT="${EXPERIMENT:-sc-ab-v5p32}"
SEED="${SEED:-42}"
MAX_RESTARTS="${MAX_RESTARTS:-1}"
RENDER_DIR="${RENDER_DIR:-$HERE/rendered}"
# Prompts per forward+backward chunk (x 8 generations); memory only, the
# optimizer still sees the full 64-prompt mini-batch. Old/reference logps use
# the same micro batch (the agentic learner requires it). 8 is the largest
# divisor of 64 that fits a v5p trainer chip (95.7 GB HBM) at 2048+8192 tokens:
# probes needed 104.3 GB of HLO temporaries at 16 (sc-on) and 143.8 GB at 32.
TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-8}"
# Appended after every other flag (argparse: last one wins), e.g. for short
# memory probes: EXTRA_TRAIN_ARGS="--num_batches 2 --disable_eval".
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# Shared by both arms. Full runs: 200 steps of 64 prompts x 8 generations
# (200 distinct batches, one epoch), eval every 20 steps on the full 100-prompt
# test set (2 batches of 64), checkpoint every 20 steps.
COMMON_ARGS=(
  --model_version Qwen/Qwen3-1.7B
  --batch_size 64 --mini_batch_size 64 --num_generations 8
  --train_micro_batch_size "$TRAIN_MICRO_BATCH"
  --num_epochs 1 --seed "$SEED"
  --learning_rate 1e-6
  --max_prompt_length 2048 --max_response_length 8192
  --env_max_steps 15
  --score_centering_top_k 32
  --rollout_devices 8 --rollout_dp 2
  --vllm_hbm_utilization 0.7 --vllm_max_num_seqs 128
  # Keep every checkpoint (~16 GiB each, one per 20 steps): the runs then never
  # delete from GCS, and the intermediate policies stay available for later
  # evaluation. (The GKE node SA has object-only access on $BUCKET; the recipe
  # tolerates the resulting 403 on orbax's bucket-metadata lookup.)
  --max_to_keep 100
)
FULL_ARGS=(--num_batches 200 --eval_every_n_steps 20 --num_test_batches 2
           --save_interval_steps 20)
# Smoke: 3 steps, eval + checkpoint at step 2.
SMOKE_ARGS=(--num_batches 3 --eval_every_n_steps 2 --num_test_batches 1
            --save_interval_steps 2)

die() { echo "error: $*" >&2; exit 1; }

arm_flag() {
  case "$1" in
    sc-on) echo "--score_centering" ;;
    sc-off) echo "--score_centering_diagnostics" ;;
    *) die "unknown arm: $1" ;;
  esac
}

jobset_name() {  # ARM SMOKE
  local name="fl-$1-s$SEED"
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

cmd_build() {
  local ctx tag sha
  sha="$(git -C "$REPO" rev-parse --short=8 HEAD)"
  ctx="$(mktemp -d)"
  # Tracked + untracked (non-ignored) files, i.e. the working tree as is.
  git -C "$REPO" ls-files -z -co --exclude-standard \
    | (cd "$REPO" && tar --null -T - -cf -) | tar -xf - -C "$ctx"
  mkdir -p "$ctx/raiden_wheels" && touch "$ctx/raiden_wheels/.keep"
  local digest
  digest="$(cd "$ctx" && find . -type f ! -path './raiden_wheels/*' -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-8)"
  tag="${TAG:-sc-ab-$sha-$digest}"
  git -C "$REPO" status --short > "$ctx/BUILD_WORKTREE_STATUS.txt"
  echo "context: $ctx"
  echo "image:   $IMAGE_REPO:$tag"
  gcloud builds submit "$ctx" --async \
    --project "$BUILD_PROJECT" \
    --config "$ctx/examples/frozenlake/gke/cloudbuild.yaml" \
    --substitutions "_IMAGE=$IMAGE_REPO,_TAG=$tag"
}

cmd_code_tarball() {
  local sha dst
  sha="$(git -C "$REPO" rev-parse --short=8 HEAD)"
  dst="${CODE_TARBALL_DST:-$BUCKET/code/frozenlake-sc-ab/$sha-$(date -u +%Y%m%d-%H%M%S).tar.gz}"
  git -C "$REPO" ls-files -z -co --exclude-standard -- tunix examples \
    | (cd "$REPO" && tar --null -T - -czf -) | gcloud storage cp - "$dst"
  echo "$dst"
}

cmd_up() {
  local image="" arm=both smoke=0 dry_run=0 run_tag code_tarball="" no_ckpt=0
  run_tag="$(date -u +%Y%m%d-%H%M)"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --image) image="$2"; shift 2 ;;
      --arm) arm="$2"; shift 2 ;;
      --smoke) smoke=1; shift ;;
      --run-tag) run_tag="$2"; shift 2 ;;
      --code-tarball) code_tarball="$2"; shift 2 ;;
      --no-ckpt) no_ckpt=1; shift ;;
      --dry-run) dry_run=1; shift ;;
      *) die "unknown flag: $1" ;;
    esac
  done
  [[ -n "$image" ]] || die "--image is required"
  kubectl get secret "$WANDB_SECRET" -o name >/dev/null \
    || die "missing Secret $WANDB_SECRET (see README.md)"
  mkdir -p "$RENDER_DIR"

  local a name group kind ckpt_dir files=()
  for a in $(arms_of "$arm"); do
    name="$(jobset_name "$a" "$smoke")"
    if [[ "$smoke" == 1 ]]; then
      kind=smoke; group="$WANDB_GROUP-smoke"
      set -- "${COMMON_ARGS[@]}" "${SMOKE_ARGS[@]}" "$(arm_flag "$a")"
    else
      kind=full; group="$WANDB_GROUP"
      set -- "${COMMON_ARGS[@]}" "${FULL_ARGS[@]}" "$(arm_flag "$a")"
    fi
    # shellcheck disable=SC2086  # intentional word splitting
    set -- "$@" $EXTRA_TRAIN_ARGS
    # An empty CKPT_DIR disables checkpointing in the recipe; a JobSet restart
    # then retrains from step 0 as a new W&B run (see the JobSet template).
    ckpt_dir="$BUCKET/ckpt/frozenlake-sc-ab/$name-$run_tag"
    [[ "$no_ckpt" == 1 ]] && ckpt_dir=""
    R_JOBSET_NAME="$name" R_EXPERIMENT="$EXPERIMENT" R_ARM="$a" \
    R_IMAGE="$image" R_PATHWAYS_TAG="$PATHWAYS_TAG" \
    R_RESERVATION="$RESERVATION" R_PRIORITY_CLASS="$PRIORITY_CLASS" \
    R_MAX_RESTARTS="$MAX_RESTARTS" R_DATA_DIR="$DATA_DIR" \
    R_CKPT_DIR="$ckpt_dir" \
    R_TB_LOG_DIR="$BUCKET/tensorboard/grpo/$name-$run_tag" \
    R_CODE_TARBALL="$code_tarball" \
    R_WANDB_SECRET="$WANDB_SECRET" R_WANDB_PROJECT="$WANDB_PROJECT" \
    R_WANDB_GROUP="$group" R_WANDB_RUN_NAME="$name-$run_tag" \
    R_WANDB_RUN_ID="$name-$run_tag" \
    R_WANDB_TAGS="$EXPERIMENT,$a,$kind,grid5to12,train20k,bs64,mb$TRAIN_MICRO_BATCH,resp8k,envsteps15,qwen3-1.7b" \
    R_TRAIN_ARGS="$*" \
      render "$HERE/frozenlake_sc_ab_v5p32.yaml" "$RENDER_DIR/$name.yaml"
    echo "rendered: $RENDER_DIR/$name.yaml"
    echo "  args:   $*"
    echo "  ckpt:   ${ckpt_dir:-(disabled)}"
    files+=(-f "$RENDER_DIR/$name.yaml")
  done
  if [[ "$dry_run" == 1 ]]; then
    kubectl create --dry-run=server "${files[@]}"
  else
    kubectl create "${files[@]}"
  fi
}

cmd_down() {
  local arm=both smoke=0 names=() a
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --arm) arm="$2"; shift 2 ;;
      --smoke) smoke=1; shift ;;
      *) die "unknown flag: $1" ;;
    esac
  done
  for a in $(arms_of "$arm"); do names+=("$(jobset_name "$a" "$smoke")"); done
  kubectl delete jobset --ignore-not-found "${names[@]}"
}

cmd_status() {
  kubectl get jobset -l "experiment=$EXPERIMENT" -o wide || true
  kubectl get workloads -o wide 2>/dev/null | grep -E "NAME|fl-sc-" || true
  kubectl get pods -l "experiment=$EXPERIMENT" -o wide || true
}

case "${1:-}" in
  build) shift; cmd_build "$@" ;;
  code-tarball) shift; cmd_code_tarball "$@" ;;
  up) shift; cmd_up "$@" ;;
  down) shift; cmd_down "$@" ;;
  status) shift; cmd_status "$@" ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac
