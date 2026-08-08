#!/usr/bin/env bash
# Resolve the canonical configuration into one file that every later step sources.
#
# Replaces the environment that used to live inline in the YAML.  Two things it does that an
# inline block did not:
#   * strips whitespace from injected secrets -- Kubernetes secrets routinely carry a trailing
#     newline, and a token with "\n" fails authentication in a way that reads like a bad key
#   * refuses to continue on an incomplete canonical set, before anything expensive starts
set -euo pipefail

: "${CANON_PKG:?CANON_PKG unset -- run via entrypoint.sh}"
: "${CANON_STATE:?CANON_STATE unset -- run via entrypoint.sh}"
PROFILE="${CANON_PROFILE_FILE:-cluster/profiles/qwen3-1p7b.env}"
case "$PROFILE" in /*) PROFILE_ABS="$PROFILE";; *) PROFILE_ABS="$CANON_PKG/$PROFILE";; esac
[ -f "$PROFILE_ABS" ] || { echo "profile not found: $PROFILE_ABS" >&2; exit 1; }

# shellcheck disable=SC1090
set -a
source "$CANON_PKG/cluster/profiles/_canonical_engine.env"
source "$PROFILE_ABS"
set +a

# Secrets: keep out of the env file on disk; strip whitespace; never echo the value.
for k in HF_TOKEN WANDB_API_KEY; do
  inj="INJECTED_$k"
  if [ -n "${!inj:-}" ]; then
    v="$(printf '%s' "${!inj}" | tr -d '[:space:]')"
    export "$k=$v"
    echo "[env] $k inherited from $inj (len=${#v})"
  elif [ -n "${!k:-}" ]; then
    v="$(printf '%s' "${!k}" | tr -d '[:space:]')"
    export "$k=$v"
    echo "[env] $k inherited from environment (len=${#v})"
  else
    echo "[env] $k not set"
  fi
done

# Logging verbosity used by the existing cluster runs.
export TPU_MIN_LOG_LEVEL="${TPU_MIN_LOG_LEVEL:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export TPU_STDERR_LOG_LEVEL="${TPU_STDERR_LOG_LEVEL:-0}"
export PYTHONDONTWRITEBYTECODE=1
export CANON_TPU_INFERENCE_PATH="${CANON_TPU_INFERENCE_PATH:-/usr/local/lib/python3.12/site-packages/tpu_inference}"

# Pathways backend configuration is activated in Step 70/90 to keep preflight 00..60 100% CPU isolated.
export JAX_PLATFORMS="cpu"
export JAX_BACKEND_TARGET=""
export PATHWAYS_HEAD=""
export ENABLE_PATHWAYS_PERSISTENCE="${ENABLE_PATHWAYS_PERSISTENCE:-1}"
echo "[env] preflight mode: JAX_PLATFORMS=cpu (Pathways connection deferred to Step 70)"

# Preflight: refuse an incomplete canonical set rather than warn inside a log nobody reads.
fail=0
req() { [ -n "${!1:-}" ] || { echo "[env] MISSING: $1" >&2; fail=1; }; }
positive_int() {
  local key="$1" value="${!1:-}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "[env] INVALID positive integer: $key=${value@Q}" >&2
    fail=1
  }
}
validate_train_mesh_pin() {
  local value="$1"
  local -a ids=()
  local id unique_count
  IFS=',' read -r -a ids <<< "$value"
  if [ "${#ids[@]}" -ne "$CANON_TOTAL_DEVICES" ]; then
    echo "[env] P32 train mesh pin must contain exactly $CANON_TOTAL_DEVICES ids; got ${#ids[@]}" >&2
    fail=1
    return
  fi
  for id in "${ids[@]}"; do
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
      echo "[env] P32 train mesh pin contains a non-integer id" >&2
      fail=1
      return
    fi
  done
  unique_count="$(printf '%s\n' "${ids[@]}" | sort -un | wc -l | tr -d '[:space:]')"
  if [ "$unique_count" -ne "$CANON_TOTAL_DEVICES" ]; then
    echo "[env] P32 train mesh pin must contain $CANON_TOTAL_DEVICES unique ids; got $unique_count" >&2
    fail=1
    return
  fi
  echo "[env] P32 train mesh pin OK: $unique_count unique ids"
}
for k in CANON_FIXED_AR CANON_FIXED_AR_EMBED CANON_RPA_D CANON_RPA_P CANON_RPA_M \
         CANON_RPA_VJP2 CANON_VJP2_MAX_SEQS CANON_LOGPROB_M CANON_PROMPT_PROCESSED_LOGPROBS \
         MIN_TOKEN_BUCKET NEW_MODEL_DESIGN XLA_FLAGS CANON_PROFILE CANON_MODEL_DIR_NAME \
         CANON_QWEN3_HIDDEN_SIZE CANON_QWEN3_TP_SIZE; do req "$k"; done
case "${XLA_FLAGS:-}" in
  *--xla_allow_excess_precision=false*) ;;
  *) echo "[env] MISSING: XLA_FLAGS lacks --xla_allow_excess_precision=false" >&2; fail=1;;
esac
if [ -n "${CANON_RPA_VJP:-}" ] && [ "${CANON_RPA_VJP:-}" = "1" ]; then
  echo "[env] NOTE: CANON_RPA_VJP=1 is set alongside VJP2.  VJP2 wins in the engine, but if"
  echo "[env]       VJP2 were ever unset this would silently select the prefill-only contract"
  echo "[env]       whose kv gradients are identically zero.  See KNOWN_FOOTGUNS.md."
fi

if [ "${CANON_P32_DP_ADMISSION:-0}" = "1" ]; then
  for k in CANON_DP_SIZE CANON_TP_SIZE CANON_TOTAL_DEVICES CANON_ENGINE_DP_SIZE \
           CANON_GLOBAL_PROMPTS CANON_LOCAL_PROMPTS CANON_NUM_GENERATIONS \
           CANON_LOCAL_TRAJECTORIES CANON_GLOBAL_TRAJECTORIES \
           CANON_DP_PROBE_LOCAL_SAMPLES CANON_TARGET_M CANON_MAX_BATCHED; do
    req "$k"
    positive_int "$k"
  done
  req CANON_CANONICAL_DEPTHS
  req CANON_WAYCOUNT_WIDTHS
  req CANON_EXPECT_JAX_VERSION
  req CANON_EXPECT_PATHWAYS_RELEASE
  req CANON_TRAIN_DP_SHARDING
  case "${CANON_REQUIRE_TRAIN_MESH_PIN:-0}" in
    0|1) ;;
    *)
      echo "[env] CANON_REQUIRE_TRAIN_MESH_PIN must be 0 or 1" >&2
      fail=1
      ;;
  esac
  [ "${CANON_TRAIN_DP_SHARDING:-}" = "replicated-params" ] || {
    echo "[env] P32 requires replicated-params, got ${CANON_TRAIN_DP_SHARDING:-unset}" >&2
    fail=1
  }
  [ "$((CANON_DP_SIZE * CANON_TP_SIZE))" -eq "$CANON_TOTAL_DEVICES" ] || {
    echo "[env] P32 arithmetic FAIL: dp*tp != total devices" >&2; fail=1;
  }
  [ "$CANON_ENGINE_DP_SIZE" -eq "$CANON_DP_SIZE" ] || {
    echo "[env] P32 arithmetic FAIL: engine dp != trainer dp" >&2; fail=1;
  }
  [ "$((CANON_DP_SIZE * CANON_LOCAL_PROMPTS))" -eq "$CANON_GLOBAL_PROMPTS" ] || {
    echo "[env] P32 arithmetic FAIL: dp*local prompts != global prompts" >&2; fail=1;
  }
  [ "$((CANON_LOCAL_PROMPTS * CANON_NUM_GENERATIONS))" -eq "$CANON_LOCAL_TRAJECTORIES" ] || {
    echo "[env] P32 arithmetic FAIL: local prompts*generations != local trajectories" >&2; fail=1;
  }
  [ "$((CANON_GLOBAL_PROMPTS * CANON_NUM_GENERATIONS))" -eq "$CANON_GLOBAL_TRAJECTORIES" ] || {
    echo "[env] P32 arithmetic FAIL: global prompts*generations != global trajectories" >&2; fail=1;
  }
  [ "$CANON_DP_PROBE_LOCAL_SAMPLES" -eq "$CANON_LOCAL_TRAJECTORIES" ] || {
    echo "[env] P32 probe must measure one local trajectory batch" >&2; fail=1;
  }
  [ "$MIN_TOKEN_BUCKET" -eq "$((CANON_DP_SIZE * CANON_LOGPROB_M))" ] || {
    echo "[env] P32 bucket FAIL: MIN_TOKEN_BUCKET must equal dp*CANON_LOGPROB_M" >&2; fail=1;
  }
  [ "$CANON_CANONICAL_DEPTHS" = "1,2,4,8" ] || {
    echo "[env] P32 canonical-op depths must remain 1,2,4,8" >&2
    fail=1
  }
  [ "$CANON_WAYCOUNT_WIDTHS" = "2,4,8" ] || {
    echo "[env] P32 generic way-count widths must remain 2,4,8" >&2
    fail=1
  }
  [ "$CANON_EXPECT_JAX_VERSION" = "0.10.2" ] || {
    echo "[env] P32 canonical client JAX must remain 0.10.2" >&2
    fail=1
  }
  [ "$CANON_EXPECT_PATHWAYS_RELEASE" = "20260730-jax_0.10.2" ] || {
    echo "[env] P32 Pathways release must remain 20260730-jax_0.10.2" >&2
    fail=1
  }
  if [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "1" ]; then
    req CANON_P32_WORKLOAD
    for k in CANON_P32_DP_REDUCTION_ADMITTED CANON_P33_WORKLOAD_LAUNCH_ADMITTED \
             CANON_P32_DP16_SEGMENTED \
             CANON_P28_SEGMENTED_FORWARD CANON_P28_SEGMENTED_TRAIN \
             CANON_P28_G6_UPDATE CANON_P29_FULL_TRAIN \
             CANON_ALIGNMENT_GATE CANON_ALIGNMENT_TRAIN \
             CANON_P30_OPT_STATE_OFFLOAD CANON_P30_SPARSE_GRAD_ASSEMBLY \
             CANON_P30_REUSE_SEGMENTED_ENGINE \
             CANON_P30_RELEASE_CAPTURED_STATE \
             CANON_P30_RESHARD_ACCUMULATOR; do
      req "$k"
      [ "${!k:-}" = "1" ] || {
        echo "[env] admitted P33 training requires $k=1" >&2
        fail=1
      }
    done
    case "${CANON_P32_WORKLOAD:-}" in
      gsm8k|frozenlake) ;;
      *) echo "[env] admitted P33 training has invalid workload" >&2; fail=1 ;;
    esac
    if [ "${CANON_P32_WORKLOAD:-}" = "frozenlake" ]; then
      req CANON_P33_DISABLE_EVAL
      [ "${CANON_P33_DISABLE_EVAL:-0}" = "1" ] || {
        echo "[env] admitted P33 FrozenLake requires periodic evaluation disabled" >&2
        fail=1
      }
    fi
    [ "${CANON_P30_FUSED_PAIR_ACCUMULATION:-}" = "0" ] || {
      echo "[env] P33 rank-reduced groups require fused pair accumulation off" >&2
      fail=1
    }
    [ "${FL_SHARED_MESH:-}" = "16,4" ] || {
      echo "[env] admitted P33 training requires FL_SHARED_MESH=16,4" >&2
      fail=1
    }
    [ "${CANON_P32_DP_REDUCTION_ADMITTED:-0}" = "1" ] || {
      echo "[env] admitted P33 training requires the DP reduction gate" >&2
      fail=1
    }
    [ "${CANON_P33_WORKLOAD_LAUNCH_ADMITTED:-0}" = "1" ] || {
      echo "[env] admitted P33 training requires the workload launch gate" >&2
      fail=1
    }
    req CANON_P33_RUN_STAGE
    case "${CANON_P33_NO_COMMIT:-0}" in
      0|1) ;;
      *) echo "[env] CANON_P33_NO_COMMIT must be 0 or 1" >&2; fail=1 ;;
    esac
    case "${CANON_P33_RUN_STAGE:-}" in
      backward-no-commit)
        [ "${CANON_P33_NO_COMMIT:-0}" = "1" ] || {
          echo "[env] backward-no-commit stage requires CANON_P33_NO_COMMIT=1" >&2
          fail=1
        }
        ;;
      one-update|three-update|full)
        [ "${CANON_P33_NO_COMMIT:-0}" = "0" ] || {
          echo "[env] update/full stages require CANON_P33_NO_COMMIT=0" >&2
          fail=1
        }
        ;;
      *) echo "[env] invalid CANON_P33_RUN_STAGE" >&2; fail=1 ;;
    esac
    for k in CANON_WANDB_ONLINE_REQUIRED CANON_P31_MONOTONIC_METRICS \
             CANON_WANDB_PROJECT CANON_WANDB_GROUP CANON_WANDB_RUN_NAME \
             WANDB_MODE WANDB_API_KEY; do
      req "$k"
    done
    [ "${CANON_WANDB_ONLINE_REQUIRED:-0}" = "1" ] || {
      echo "[env] admitted P33 training requires online W&B" >&2
      fail=1
    }
    [ "${CANON_P31_MONOTONIC_METRICS:-0}" = "1" ] || {
      echo "[env] admitted P33 training requires monotonic W&B metrics" >&2
      fail=1
    }
    [ "${WANDB_MODE:-}" = "online" ] || {
      echo "[env] admitted P33 training requires WANDB_MODE=online" >&2
      fail=1
    }
  else
    [ "${FL_SHARED_MESH:-}" = "1,4" ] || {
      echo "[env] unadmitted P32 modes must keep the legacy trainer at TP4-only" >&2
      fail=1
    }
  fi
  if [ "${CANON_REQUIRE_TRAIN_MESH_PIN:-0}" = "1" ]; then
    req CANON_EXPECT_TRAIN_MESH_IDS
  fi
  if [ -n "${CANON_EXPECT_TRAIN_MESH_IDS:-}" ]; then
    validate_train_mesh_pin "$CANON_EXPECT_TRAIN_MESH_IDS"
  else
    echo "[env] P32 train mesh pin DISCOVERY: no release placement assertion"
  fi
  echo "[env] P32 admission arithmetic OK: DP${CANON_DP_SIZE}xTP${CANON_TP_SIZE}, "\
"${CANON_LOCAL_TRAJECTORIES} local / ${CANON_GLOBAL_TRAJECTORIES} global trajectories, "\
"global M=${MIN_TOKEN_BUCKET}"
fi
if [ "${CANON_MODE:-}" = "model-init-only" ]; then
  for k in CANON_P32_MODEL_INIT_ONLY CANON_P32_MODEL_STATE_KIND \
           CANON_P32_OPTIMIZER_MEMORY_KIND CANON_WANDB_PROJECT \
           CANON_WANDB_GROUP CANON_WANDB_RUN_NAME; do
    req "$k"
  done
  [ "${CANON_P32_DP_ADMISSION:-0}" = "1" ] || {
    echo "[env] model-init-only requires the P32 DP admission contract" >&2
    fail=1
  }
  [ "${CANON_P32_MODEL_INIT_ONLY:-0}" = "1" ] || {
    echo "[env] model-init-only requires CANON_P32_MODEL_INIT_ONLY=1" >&2
    fail=1
  }
  [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "0" ] || {
    echo "[env] model-init-only must keep training refused" >&2
    fail=1
  }
  [ "${CANON_P32_MODEL_STATE_KIND:-}" = "zero-structural" ] || {
    echo "[env] model-init-only state kind must remain zero-structural" >&2
    fail=1
  }
  [ "${CANON_P32_OPTIMIZER_MEMORY_KIND:-}" = "pinned_host" ] || {
    echo "[env] model-init-only requires pinned-host optimizer state" >&2
    fail=1
  }
  echo "[env] P32 model-init-only contract OK: structural state, zero commits"
fi
if [ "${CANON_MODE:-}" = "dp16-rc" ]; then
  for k in CANON_P32_RC CANON_P32_RC_STAGE CANON_P32_CHECKPOINT_DIR \
           CANON_P32_OPTIMIZER_MEMORY_KIND; do
    req "$k"
  done
  [ "${CANON_P32_DP_ADMISSION:-0}" = "1" ] || {
    echo "[env] dp16-rc requires the P32 DP admission contract" >&2
    fail=1
  }
  [ "${CANON_P32_RC:-0}" = "1" ] || {
    echo "[env] dp16-rc requires CANON_P32_RC=1" >&2
    fail=1
  }
  [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "0" ] || {
    echo "[env] dp16-rc must not admit production training" >&2
    fail=1
  }
  case "${CANON_P32_RC_STAGE:-}" in
    checkpoint-forward|backward|one-update|three-update) ;;
    *)
      echo "[env] invalid CANON_P32_RC_STAGE=${CANON_P32_RC_STAGE:-unset}" >&2
      fail=1
      ;;
  esac
  [ "${CANON_P32_OPTIMIZER_MEMORY_KIND:-}" = "pinned_host" ] || {
    echo "[env] dp16-rc requires pinned-host optimizer state" >&2
    fail=1
  }
  echo "[env] P32 dp16-rc contract OK: stage=${CANON_P32_RC_STAGE} production_training=refused"
fi
if [ "${CANON_MODE:-}" = "workload-contract-only" ]; then
  for k in CANON_P32_WORKLOAD CANON_P32_DP_REDUCTION_ADMITTED \
           CANON_P33_WORKLOAD_LAUNCH_ADMITTED \
           CANON_WANDB_PROJECT CANON_WANDB_GROUP CANON_WANDB_RUN_NAME; do
    req "$k"
  done
  [ "${CANON_P32_DP_ADMISSION:-0}" = "1" ] || {
    echo "[env] workload contract requires the P32 DP admission contract" >&2
    fail=1
  }
  [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "0" ] || {
    echo "[env] workload contract must keep production training refused" >&2
    fail=1
  }
  case "${CANON_P32_WORKLOAD:-}" in
    gsm8k|frozenlake) ;;
    *) echo "[env] invalid CANON_P32_WORKLOAD=${CANON_P32_WORKLOAD:-unset}" >&2; fail=1 ;;
  esac
  if [ "${CANON_P32_WORKLOAD:-}" = "frozenlake" ]; then
    req CANON_P33_DISABLE_EVAL
    [ "${CANON_P33_DISABLE_EVAL:-0}" = "1" ] || {
      echo "[env] P33 FrozenLake contract requires periodic evaluation disabled" >&2
      fail=1
    }
  fi
  [ "${CANON_P32_DP_REDUCTION_ADMITTED:-}" = "0" ] || {
    echo "[env] contract-only mode must keep DP reduction unadmitted" >&2
    fail=1
  }
  [ "${CANON_P33_WORKLOAD_LAUNCH_ADMITTED:-}" = "0" ] || {
    echo "[env] contract-only mode must keep workload launch unadmitted" >&2
    fail=1
  }
  echo "[env] P33 workload contract OK: workload=${CANON_P32_WORKLOAD} launch=refused"
fi
[ "$fail" = 0 ] || { echo "[env] REFUSING TO CONTINUE: canonical set incomplete" >&2; exit 1; }

# Emit the resolved configuration.  Secrets are re-exported by later steps from the process
# environment, never written here.
{
  echo "# generated by cluster/steps/00_env.sh -- do not edit"
  for k in $(compgen -e | grep -E '^(CANON_|MIN_TOKEN_BUCKET|NEW_MODEL_DESIGN|VLLM_|ROLLOUT_ENGINE|XLA_FLAGS|JAX_|FL_SHARED_MESH|TPU_|TF_CPP|ENABLE_PATHWAYS|PYTHONDONTWRITEBYTECODE)' | sort); do
    printf 'export %s=%q\n' "$k" "${!k}"
  done
} > "$CANON_STATE/env.sh"

echo "[env] profile=$CANON_PROFILE model_dir=$CANON_MODEL_DIR_NAME"
echo "[env] resolved configuration written to $CANON_STATE/env.sh ($(wc -l < "$CANON_STATE/env.sh") lines)"
