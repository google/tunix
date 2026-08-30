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

# Preserve contradictory values supplied by the caller before the profile
# derives reward-only invariants. Without this snapshot, sourcing the profile
# could silently turn an explicit trainer/alignment request back to zero and
# make a contradictory JobSet appear valid.
_CANON_P46_INPUT_CONTRADICTIONS=()
for _canon_p46_key in CANON_P46_DEEPSWE_TRAIN CANON_P34_DEEPSWE \
    CANON_P34_TRAJECTORY_ADMITTED CANON_P34_UPDATE_ADMITTED \
    CANON_P32_TRAIN_ADMITTED CANON_P33_WORKLOAD_LAUNCH_ADMITTED \
    CANON_PROMPT_PROCESSED_LOGPROBS CANON_PALLAS_LOGSOFTMAX \
    CANON_ENGINE_MODULE_C CANON_RPA_VJP2 CANON_ALIGNMENT_GATE \
    CANON_ALIGNMENT_GATE_ONLY CANON_ALIGNMENT_UPDATE_CANARY \
    CANON_ALIGNMENT_TRAIN CANON_PRE_ALIGN_GATE \
    CANON_DEEPSWE_ALIGNMENT_WARN_ONLY CANON_P28_SEGMENTED_FORWARD \
    CANON_P28_SEGMENTED_VJP CANON_P28_SEGMENTED_TRAIN \
    CANON_P28_G6_UPDATE CANON_P29_FULL_TRAIN CANON_OPT_STATE_RESIDENT \
    CANON_P30_SPARSE_GRAD_ASSEMBLY CANON_P30_FUSED_PAIR_ACCUMULATION \
    CANON_P30_REUSE_SEGMENTED_ENGINE CANON_P30_RELEASE_CAPTURED_STATE \
    CANON_P30_RESHARD_ACCUMULATOR CANON_P59_RANK_PARALLEL_BACKWARD; do
  if [[ -v "$_canon_p46_key" && "${!_canon_p46_key}" != "0" ]]; then
    _CANON_P46_INPUT_CONTRADICTIONS+=(
      "${_canon_p46_key}=${!_canon_p46_key}"
    )
  fi
done
unset _canon_p46_key

# A Native GSM8K manifest must not smuggle an active Zero selector through the
# raw JobSet and rely on the stock profile to unset it. Snapshot contradictions
# before profile resolution so hand-edited mixed arms fail closed.
_CANON_GSM8K_NATIVE_INPUT_CONTRADICTIONS=()
if [ "${CANON_PROFILE_FILE:-}" = \
     "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env" ] && \
   [ "${CANON_GSM8K_TRAIN:-}" = "1" ] && \
   [ "${CANON_GSM8K_VANILLA:-}" = "1" ]; then
  for _canon_gsm_native_key in CANON_P32_WORKLOAD CANON_ENGINE_MODULE_C \
      CANON_ALIGNMENT_GATE CANON_ALIGNMENT_GATE_ONLY \
      CANON_ALIGNMENT_UPDATE_CANARY CANON_ALIGNMENT_TRAIN \
      CANON_PRE_ALIGN_GATE CANON_GSM8K_AB_REPORT_ONLY \
      CANON_GSM8K_ALIGNMENT_WARN_ONLY CANON_P38_FIXED_LM_HEAD \
      CANON_P59_RANK_PARALLEL_BACKWARD CANON_P59_CHECKED_VMA \
      CANON_V1_HP_FULL CANON_V1_HP_FIRST_UPDATE_GATE \
      CANON_P63_OVERFLOW_SAFE_CLIP CANON_DP_COMPARE_MODE \
      CANON_DP_DISTINCT_SCHEDULE CANON_DP_FINITE_FETCH CANON_P71_SCAN \
      CANON_DP_COLLECTIVE_REDUCE CANON_P67_P66_VMA_P59_ONLY; do
    if [[ -v "$_canon_gsm_native_key" && \
          -n "${!_canon_gsm_native_key}" && \
          "${!_canon_gsm_native_key}" != "0" ]]; then
      _CANON_GSM8K_NATIVE_INPUT_CONTRADICTIONS+=(
        "${_canon_gsm_native_key}=${!_canon_gsm_native_key}"
      )
    fi
  done
  unset _canon_gsm_native_key
fi

# shellcheck disable=SC1090
set -a
export ENABLE_PATHWAYS_PERSISTENCE="${ENABLE_PATHWAYS_PERSISTENCE:-1}"
source "$CANON_PKG/cluster/profiles/_canonical_engine.env"
source "$PROFILE_ABS"
set +a

P57_STOCK_FAST=0
if [ "${CANON_PROFILE_FILE:-}" = \
     "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env" ] && \
   [ "${CANON_P57_INFERENCE_REGIME:-}" = "stock-fast" ]; then
  P57_STOCK_FAST=1
fi
P57_STOCK_TRAIN=0
if [ "$P57_STOCK_FAST" = "1" ] && \
   [ "${CANON_P57_RUN_KIND:-}" = "train" ]; then
  case "${CANON_P57_TIM_ARM:-}" in
    mismatch|is) P57_STOCK_TRAIN=1 ;;
  esac
fi
P57_STOCK_EVAL=0
if [ "$P57_STOCK_FAST" = "1" ] && \
   [ "${CANON_P57_RUN_KIND:-}" = "eval" ]; then
  case "${CANON_P57_TIM_ARM:-}" in
    mismatch|is) P57_STOCK_EVAL=1 ;;
  esac
fi
P58_NATIVE=0
if [ "${CANON_PROFILE_FILE:-}" = \
     "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env" ] && \
   [ "${CANON_P58_DEEPSWE_TIM:-}" = "1" ] && \
   [ "${CANON_P58_TIM_ARM:-}" = "native" ]; then
  P58_NATIVE=1
fi
GSM8K_NATIVE=0
if [ "${CANON_PROFILE_FILE:-}" = \
     "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env" ] && \
   [ "${CANON_GSM8K_TRAIN:-}" = "1" ] && \
   [ "${CANON_GSM8K_VANILLA:-}" = "1" ]; then
  GSM8K_NATIVE=1
fi

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
export CANON_TPU_INFERENCE_PATH="${CANON_TPU_INFERENCE_PATH:-}"

# Pathways & gRPC peer communication timeouts & buffers (mitigating worker-to-worker pipe timeouts during long JIT compilation)
export PATHWAYS_PIPE_UNREACHABLE_TIMEOUT="${PATHWAYS_PIPE_UNREACHABLE_TIMEOUT:-300s}"
export JAX_PATHWAYS_PIPE_UNREACHABLE_TIMEOUT="${JAX_PATHWAYS_PIPE_UNREACHABLE_TIMEOUT:-300s}"
export GRPC_KEEPALIVE_TIME_MS="${GRPC_KEEPALIVE_TIME_MS:-10000}"
export GRPC_KEEPALIVE_TIMEOUT_MS="${GRPC_KEEPALIVE_TIMEOUT_MS:-30000}"
export GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS="${GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS:-1}"
export TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-8589934592}"

# Pathways backend configuration is activated in Step 70/90 to keep preflight 00..60 100% CPU isolated.
export JAX_PLATFORMS="cpu"
export JAX_BACKEND_TARGET=""
export PATHWAYS_HEAD=""
export ENABLE_PATHWAYS_PERSISTENCE="${ENABLE_PATHWAYS_PERSISTENCE:-1}"
echo "[env] preflight mode: JAX_PLATFORMS=cpu (Pathways connection deferred to Step 70)"

# Preflight: refuse an incomplete canonical set rather than warn inside a log nobody reads.
fail=0
req() { [ -n "${!1:-}" ] || { echo "[env] MISSING: $1" >&2; fail=1; }; }
case "${CANON_APC_M15_TARGET_DEBUG:-}" in
  "") APC_M15_TARGET_DEBUG=0 ;;
  off|on)
    APC_M15_TARGET_DEBUG=1
    [ "${CANON_PROFILE_FILE:-}" = \
        "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env" ] || {
      echo "[env] M15 APC target debug requires its exact profile" >&2
      fail=1
    }
    ;;
  *)
    APC_M15_TARGET_DEBUG=0
    echo "[env] CANON_APC_M15_TARGET_DEBUG must be unset, off, or on" >&2
    fail=1
    ;;
esac
case "${CANON_P58_SEAM_LOCALIZATION:-}" in
  "") P58_SEAM_LOCALIZATION=0 ;;
  coarse)
    P58_SEAM_LOCALIZATION=1
    [ "${CANON_PROFILE_FILE:-}" = \
        "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env" ] && \
    [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ] && \
    [ "${CANON_P58_TIM_ARM:-}" = "zero" ] || {
      echo "[env] P58 seam localization requires its exact Zero-HP profile" >&2
      fail=1
    }
    ;;
  *)
    P58_SEAM_LOCALIZATION=0
    echo "[env] CANON_P58_SEAM_LOCALIZATION must be unset or coarse" >&2
    fail=1
    ;;
esac
case "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" in
  0) ;;
  1)
    [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "1" ] && \
    [ "${CANON_P32_DP16_SEGMENTED:-0}" = "1" ] && \
    [ "${CANON_P30_SPARSE_GRAD_ASSEMBLY:-0}" = "1" ] || {
      echo "[env] P59 rank-parallel backward requires admitted DP segmented training with sparse assembly" >&2
      fail=1
    }
    ;;
  *)
    echo "[env] CANON_P59_RANK_PARALLEL_BACKWARD must be exactly 0 or 1" >&2
    fail=1
    ;;
esac
case "${CANON_V1_FL_TP8_AB_ARM:-}" in
  "") ;;
  p66-off)
    [ "${CANON_PROFILE_FILE:-}" = \
      "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug.env" ] && \
    [ "${CANON_PROFILE:-}" = \
      "qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug" ] && \
    [ "${CANON_P32_WORKLOAD:-}" = "frozenlake-dp8-tp8" ] && \
    [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ] && \
    [ "${CANON_P33_NO_COMMIT:-}" = "1" ] && \
    [ "${CANON_P38_PRECHECK_ONLY:-}" = "1" ] && \
    [ "${CANON_P38_CONTROLLED_EXIT:-}" = "1" ] && \
    [ "${CANON_P38_DIAGNOSTIC_ROUNDS:-}" = "1" ] && \
    [ "${CANON_P59_CHECKED_VMA:-0}" = "0" ] && \
    [ "${CANON_P66_P59_CHECK_VMA:-0}" = "0" ] && \
    [ "${CANON_P67_P66_VMA_P59_ONLY:-0}" = "0" ] && \
    [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ] && \
    [ "${CANON_V1_HP_FULL:-0}" = "0" ] || {
      echo "[env] V1 FrozenLake TP8 A/B p66-off contract drifted" >&2
      fail=1
    }
    case "${CANON_P57_WORKLOAD_CANDIDATE:-}:${CANON_P57_DATA_SPLIT:-}" in
      :|m15:main) ;;
      *)
        echo "[env] V1 FrozenLake TP8 A/B workload identity drifted" >&2
        fail=1
        ;;
    esac
    ;;
  serving-scope)
    [ "${CANON_PROFILE_FILE:-}" = \
      "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug.env" ] && \
    [ "${CANON_PROFILE:-}" = \
      "qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug" ] && \
    [ "${CANON_P32_WORKLOAD:-}" = "frozenlake-dp8-tp8" ] && \
    [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ] && \
    [ "${CANON_P33_NO_COMMIT:-}" = "1" ] && \
    [ "${CANON_P38_PRECHECK_ONLY:-}" = "1" ] && \
    [ "${CANON_P38_CONTROLLED_EXIT:-}" = "1" ] && \
    [ "${CANON_P38_DIAGNOSTIC_ROUNDS:-}" = "1" ] && \
    [ "${CANON_P59_CHECKED_VMA:-0}" = "1" ] && \
    [ "${CANON_P66_P59_CHECK_VMA:-0}" = "1" ] && \
    [ "${CANON_P67_P66_VMA_P59_ONLY:-0}" = "1" ] && \
    [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ] && \
    [ "${CANON_V1_HP_FULL:-0}" = "0" ] || {
      echo "[env] V1 FrozenLake TP8 A/B serving-scope contract drifted" >&2
      fail=1
    }
    case "${CANON_P57_WORKLOAD_CANDIDATE:-}:${CANON_P57_DATA_SPLIT:-}" in
      :|m15:main) ;;
      *)
        echo "[env] V1 FrozenLake TP8 A/B workload identity drifted" >&2
        fail=1
        ;;
    esac
    ;;
  *)
    echo "[env] CANON_V1_FL_TP8_AB_ARM must be unset, p66-off, or serving-scope" >&2
    fail=1
    ;;
esac
case "${CANON_P58_CHECKED_VMA_DIAGNOSTIC:-}" in
  "") ;;
  off|on)
    if [ "${CANON_P58_CHECKED_VMA_DIAGNOSTIC}" = "off" ]; then
      _canon_p58_vma_tuple="0:0:0:0:0"
    else
      _canon_p58_vma_tuple="1:1:1:0:0"
    fi
    [ "${CANON_PROFILE_FILE:-}" = \
      "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env" ] && \
    [ "${CANON_PROFILE:-}" = \
      "qwen3-4b-dp8-tp8-deepswe-v1-hp" ] && \
    [ "${CANON_P34_DEEPSWE:-0}" = "1" ] && \
    [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ] && \
    [ "${CANON_P58_TIM_ADMITTED:-0}" = "1" ] && \
    [ "${CANON_P58_TIM_ARM:-}" = "zero" ] && \
    [ "${CANON_P34_RUN_STAGE:-}" = "full" ] && \
    [ "${CANON_P34_NO_COMMIT:-1}" = "0" ] && \
    [ "${CANON_P58_EXPECTED_UPDATES:-}" = "1000" ] && \
    [ "${CANON_V1_HP_FULL:-0}" = "1" ] && \
    [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "1" ] && \
    [ "${CANON_P38_PRECHECK_ONLY:-0}" = "1" ] && \
    [ "${CANON_P38_CONTROLLED_EXIT:-0}" = "1" ] && \
    [ "${CANON_P38_DIAGNOSTIC_ROUNDS:-0}" = "1" ] && \
    [ -n "${CANON_P38_DIAGNOSTIC_ROUND_FILE:-}" ] && \
    [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ] && \
    [ "${CANON_P59_CHECKED_VMA:-unset}:${CANON_P66_P59_CHECK_VMA:-unset}:${CANON_P67_P66_VMA_P59_ONLY:-unset}:${CANON_V1_HP_FIRST_UPDATE_GATE:-unset}:${CANON_P63_OVERFLOW_SAFE_CLIP:-unset}" = "$_canon_p58_vma_tuple" ] && \
    [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-1}" = "0" ] || {
      echo "[env] P58 checked-VMA-${CANON_P58_CHECKED_VMA_DIAGNOSTIC} diagnostic contract drifted" >&2
      fail=1
    }
    echo "[env] P58 checked-VMA-${CANON_P58_CHECKED_VMA_DIAGNOSTIC} precheck admitted: DP8xTP8 roles, backward=0 optimizer_commits=0"
    unset _canon_p58_vma_tuple
    ;;
  *)
    echo "[env] CANON_P58_CHECKED_VMA_DIAGNOSTIC must be unset, off, or on" >&2
    fail=1
    ;;
esac
if [ "$P58_SEAM_LOCALIZATION" = "1" ]; then
  [ -z "${CANON_P58_CHECKED_VMA_DIAGNOSTIC:-}" ] && \
  [ "${CANON_P34_RUN_STAGE:-}" = "full" ] && \
  [ "${CANON_P58_EXPECTED_UPDATES:-}" = "1000" ] && \
  [ "${CANON_P38_PRECHECK_ONLY:-0}" = "1" ] && \
  [ "${CANON_P38_CONTROLLED_EXIT:-0}" = "1" ] && \
  [ "${CANON_P38_DIAGNOSTIC_ROUNDS:-0}" = "3" ] && \
  [ "${CANON_P38_DURABILITY_PROFILE:-}" = "p58-seam-v1" ] || {
    echo "[env] P58 coarse seam three-round carrier drifted" >&2
    fail=1
  }
  echo "[env] P58 coarse seam precheck admitted: DP8xTP8 roles rounds=3 backward=0 optimizer_commits=0"
fi
case "${CANON_P67_P66_VMA_P59_ONLY:-0}" in
  0) ;;
  1)
    _canon_p67_context=""
    case "${CANON_V1_FL_TP8_AB_ARM:-}" in
      serving-scope)
        _canon_p67_context=v1-fl-serving-scope
        ;;
      "")
        if [ "${CANON_PROFILE_FILE:-}" = \
               "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env" ] && \
           [ "${CANON_PROFILE:-}" = \
               "qwen3-8b-dp8-tp8-frozenlake-v1-hp" ] && \
           [ "${CANON_P32_WORKLOAD:-}" = "frozenlake-dp8-tp8" ] && \
           [ "${CANON_P33_SHARED_MESH:-}" = "8,8" ] && \
           [ "${CANON_P57_RUN_KIND:-}" = "train" ] && \
           [ "${CANON_P57_TIM_ARM:-}" = "zero" ] && \
           [ "${CANON_P57_EXPECTED_UPDATES:-}" = "300" ] && \
           [ "${CANON_P33_RUN_STAGE:-}" = "full" ] && \
           [ "${CANON_P33_NO_COMMIT:-1}" = "0" ] && \
           [ "${CANON_V1_HP_FULL:-0}" = "1" ] && \
           [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ] && \
           [ "${CANON_P59_CHECKED_VMA:-0}" = "1" ]; then
          _canon_p67_context=v1-fl-full
          case "${CANON_P57_WORKLOAD_CANDIDATE:-}:${CANON_P57_DATA_SPLIT:-}" in
            :|m15:main) ;;
            *)
              echo "[env] P67 FrozenLake full workload identity drifted" >&2
              fail=1
              ;;
          esac
        elif [ "${CANON_PROFILE_FILE:-}" = \
                 "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env" ] && \
             [ "${CANON_PROFILE:-}" = \
                 "qwen3-4b-dp8-tp8-deepswe-v1-hp" ] && \
             [ "${CANON_P34_DEEPSWE:-0}" = "1" ] && \
             [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ] && \
             [ "${CANON_P58_TIM_ADMITTED:-0}" = "1" ] && \
             [ "${CANON_P58_TIM_ARM:-}" = "zero" ] && \
             [ "${CANON_P34_RUN_STAGE:-}" = "full" ] && \
             [ "${CANON_P34_NO_COMMIT:-1}" = "0" ] && \
             [ "${CANON_P58_EXPECTED_UPDATES:-}" = "1000" ] && \
             [ "${CANON_V1_HP_FULL:-0}" = "1" ] && \
             [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "1" ] && \
             [ "${CANON_P59_CHECKED_VMA:-0}" = "1" ] && \
             [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ] && \
             [ "${CANON_P34_DISABLE_SAMPLER_IS:-0}" = "1" ] && \
             [ "${CANON_P34_DISABLE_TIS:-0}" = "1" ] && \
             [ "${CANON_PROMPT_PROCESSED_LOGPROBS:-0}" = "1" ] && \
             [ "${CANON_ENGINE_MODULE_C:-0}" = "1" ] && \
             [ "${CANON_OPT_STATE_RESIDENT:-0}" = "1" ] && \
             [ "${CANON_P30_OPT_STATE_OFFLOAD:-1}" = "0" ] && \
             [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-1}" = "0" ]; then
          _canon_p67_context=p58
        fi
        ;;
      *) ;;
    esac
    [ -n "$_canon_p67_context" ] || {
      echo "[env] P67 VMA scoping is restricted to exact FrozenLake or P58 Zero-HP contexts" >&2
      fail=1
    }
    unset _canon_p67_context
    ;;
  *)
    echo "[env] CANON_P67_P66_VMA_P59_ONLY must be exactly 0 or 1" >&2
    fail=1
    ;;
esac
case "${CANON_P59_CHECKED_VMA:-0}" in
  0) ;;
  1)
    _canon_p59_checked_context=""
    case "${CANON_PROFILE_FILE:-}:${CANON_PROFILE:-}:${CANON_P32_WORKLOAD:-}" in
      cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env:qwen3-1p7b-dp16-tp4-gsm8k-v1-hp:gsm8k|\
      cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env:qwen3-8b-dp8-tp8-frozenlake-v1-hp:frozenlake-dp8-tp8)
        _canon_p59_checked_context=phase4
        ;;
      cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env:qwen3-4b-dp8-tp8-deepswe-v1-hp:)
        _canon_p59_checked_context=p58
        ;;
      cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug.env:qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug:frozenlake-dp8-tp8)
        _canon_p59_checked_context=v1-fl-serving-scope
        ;;
      *)
        echo "[env] P59 checked VMA is restricted to registered full profiles" >&2
        fail=1
        ;;
    esac
    case "$_canon_p59_checked_context" in
      phase4)
        [ "${CANON_P33_RUN_STAGE:-}" = "full" ] && \
        [ "${CANON_P33_NO_COMMIT:-1}" = "0" ] || {
          echo "[env] P59 checked VMA Phase4 stage contract changed" >&2
          fail=1
        }
        ;;
      p58)
        [ "${CANON_P34_DEEPSWE:-0}" = "1" ] && \
        [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ] && \
        [ "${CANON_P58_TIM_ADMITTED:-0}" = "1" ] && \
        [ "${CANON_P58_TIM_ARM:-}" = "zero" ] && \
        [ "${CANON_P34_RUN_STAGE:-}" = "full" ] && \
        [ "${CANON_P34_NO_COMMIT:-1}" = "0" ] && \
        [ "${CANON_P58_EXPECTED_UPDATES:-}" = "1000" ] && \
        [ "${CANON_P34_DISABLE_SAMPLER_IS:-0}" = "1" ] && \
        [ "${CANON_P34_DISABLE_TIS:-0}" = "1" ] && \
        [ "${CANON_PROMPT_PROCESSED_LOGPROBS:-0}" = "1" ] && \
        [ "${CANON_ENGINE_MODULE_C:-0}" = "1" ] && \
        [ "${CANON_OPT_STATE_RESIDENT:-0}" = "1" ] && \
        [ "${CANON_P30_OPT_STATE_OFFLOAD:-1}" = "0" ] && \
        [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-1}" = "0" ] || {
          echo "[env] P59 checked VMA P58 Zero-HP contract changed" >&2
          fail=1
        }
        ;;
      v1-fl-serving-scope)
        [ "${CANON_V1_FL_TP8_AB_ARM:-}" = "serving-scope" ] && \
        [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ] && \
        [ "${CANON_P33_NO_COMMIT:-}" = "1" ] && \
        [ "${CANON_P38_PRECHECK_ONLY:-}" = "1" ] && \
        [ "${CANON_P67_P66_VMA_P59_ONLY:-0}" = "1" ] && \
        [ "${CANON_V1_HP_FULL:-0}" = "0" ] || {
          echo "[env] P59 checked VMA serving-scope diagnostic changed" >&2
          fail=1
        }
        ;;
    esac
    if [ "$_canon_p59_checked_context" != "v1-fl-serving-scope" ]; then
      [ "${CANON_V1_HP_FULL:-0}" = "1" ] && \
      [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ] || {
        echo "[env] P59 checked VMA requires exact committed P59 full training" >&2
        fail=1
      }
    fi
    case "${CANON_P66_P59_CHECK_VMA:-}" in
      ""|1) export CANON_P66_P59_CHECK_VMA=1 ;;
      *)
        echo "[env] P59 checked VMA compatibility alias conflicts" >&2
        fail=1
        ;;
    esac
    echo "[env] P59 checked VMA backward enabled compatibility_alias=CANON_P66_P59_CHECK_VMA"
    unset _canon_p59_checked_context
    ;;
  *)
    echo "[env] CANON_P59_CHECKED_VMA must be exactly 0 or 1" >&2
    fail=1
    ;;
esac
case "${CANON_V1_HP_FIRST_UPDATE_GATE:-0}" in
  0) ;;
  1)
    [ "${CANON_V1_HP_FULL:-0}" = "1" ] && \
    [ "${CANON_P59_CHECKED_VMA:-0}" = "1" ] || {
      echo "[env] V1 first-update gate requires exact checked-VMA committed full training" >&2
      fail=1
    }
    if [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ]; then
      [ "${CANON_P34_RUN_STAGE:-}" = "full" ] && \
      [ "${CANON_P34_NO_COMMIT:-1}" = "0" ] && \
      [ "${CANON_P58_TIM_ARM:-}" = "zero" ] || {
        echo "[env] P58 first-update gate requires strict Zero full" >&2
        fail=1
      }
    else
      [ "${CANON_P33_RUN_STAGE:-}" = "full" ] && \
      [ "${CANON_P33_NO_COMMIT:-1}" = "0" ] || {
        echo "[env] Phase4 first-update gate requires committed full training" >&2
        fail=1
      }
    fi
    echo "[env] V1 first-update precommit gate enabled stable_norm_max=1000000"
    ;;
  *)
    echo "[env] CANON_V1_HP_FIRST_UPDATE_GATE must be exactly 0 or 1" >&2
    fail=1
    ;;
esac
case "${CANON_P63_OVERFLOW_SAFE_CLIP:-0}" in
  0) ;;
  1)
    case "${CANON_PROFILE_FILE:-}:${CANON_PROFILE:-}:${CANON_P32_WORKLOAD:-}" in
      cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env:qwen3-1p7b-dp16-tp4-gsm8k-v1-hp:gsm8k)
        [ "${CANON_GSM8K_ALIGNMENT_WARN_ONLY:-1}" = "0" ] || {
          echo "[env] P63 GSM8K overflow-safe clip requires strict alignment" >&2
          fail=1
        }
        ;;
      cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env:qwen3-8b-dp8-tp8-frozenlake-v1-hp:frozenlake-dp8-tp8)
        [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-1}" = "0" ] || {
          echo "[env] P63 FrozenLake overflow-safe clip requires strict alignment" >&2
          fail=1
        }
        ;;
      cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env:qwen3-4b-dp8-tp8-deepswe-v1-hp:)
        [ "${CANON_P34_DEEPSWE:-0}" = "1" ] && \
        [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ] && \
        [ "${CANON_P58_TIM_ADMITTED:-0}" = "1" ] && \
        [ "${CANON_P58_TIM_ARM:-}" = "zero" ] && \
        [ "${CANON_P34_DISABLE_SAMPLER_IS:-0}" = "1" ] && \
        [ "${CANON_P34_DISABLE_TIS:-0}" = "1" ] && \
        [ "${CANON_PROMPT_PROCESSED_LOGPROBS:-0}" = "1" ] && \
        [ "${CANON_ENGINE_MODULE_C:-0}" = "1" ] && \
        [ "${CANON_OPT_STATE_RESIDENT:-0}" = "1" ] && \
        [ "${CANON_P30_OPT_STATE_OFFLOAD:-1}" = "0" ] && \
        [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-1}" = "0" ] || {
          echo "[env] P63 P58 overflow-safe clip requires strict Zero-HP" >&2
          fail=1
        }
        ;;
      *)
        echo "[env] P63 overflow-safe clip is restricted to registered full profiles" >&2
        fail=1
        ;;
    esac
    if [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ]; then
      [ "${CANON_P34_RUN_STAGE:-}" = "full" ] && \
      [ "${CANON_P34_NO_COMMIT:-1}" = "0" ] || {
        echo "[env] P63 P58 clip requires committed full training" >&2
        fail=1
      }
    else
      [ "${CANON_P33_RUN_STAGE:-}" = "full" ] && \
      [ "${CANON_P33_NO_COMMIT:-1}" = "0" ] || {
        echo "[env] P63 Phase4 clip requires committed full training" >&2
        fail=1
      }
    fi
    [ "${CANON_V1_HP_FULL:-0}" = "1" ] && \
    [ "${CANON_P33_WORKLOAD_LAUNCH_ADMITTED:-0}" = "1" ] && \
    [ "${CANON_P28_SEGMENTED_TRAIN:-0}" = "1" ] && \
    [ "${CANON_P28_G6_UPDATE:-0}" = "1" ] && \
    [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ] && \
    [ "${CANON_P59_CHECKED_VMA:-0}" = "1" ] && \
    [ "${CANON_V1_HP_FIRST_UPDATE_GATE:-0}" = "1" ] && \
    [ "${CANON_VLLM_ENABLE_PREFIX_CACHING:-1}" = "0" ] || {
      echo "[env] P63 overflow-safe clip requires strict committed P59 full training with APC off" >&2
      fail=1
    }
    echo "[env] P63 hybrid overflow-safe global-norm clipping enabled"
    ;;
  *)
    echo "[env] CANON_P63_OVERFLOW_SAFE_CLIP must be exactly 0 or 1" >&2
    fail=1
    ;;
esac
case "${CANON_P59_DP4_TAIL8:-0}" in
  0) ;;
  1)
    [ "${CANON_P32_WORKLOAD:-}" = "gsm8k-p59-dp4-tp1" ] && \
    [ "${CANON_P33_RUN_STAGE:-}" = "p59-eight-update" ] && \
    [ "${CANON_P33_NO_COMMIT:-0}" = "0" ] || {
      echo "[env] CANON_P59_DP4_TAIL8=1 requires committed P59 DP4 eight-update stage" >&2
      fail=1
    }
    ;;
  *)
    echo "[env] CANON_P59_DP4_TAIL8 must be exactly 0 or 1" >&2
    fail=1
    ;;
esac
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
for k in MIN_TOKEN_BUCKET NEW_MODEL_DESIGN XLA_FLAGS CANON_PROFILE \
         CANON_MODEL_DIR_NAME CANON_QWEN3_HIDDEN_SIZE \
         CANON_QWEN3_TP_SIZE; do req "$k"; done
if [ "$P57_STOCK_FAST" = "1" ] || [ "$P58_NATIVE" = "1" ] || \
   [ "$GSM8K_NATIVE" = "1" ]; then
  stock_label="P57 stock-fast"
  [ "$P58_NATIVE" = "0" ] || stock_label="P58 native"
  [ "$GSM8K_NATIVE" = "0" ] || stock_label="GSM8K native"
  for k in CANON_FIXED_AR CANON_FIXED_AR_EMBED \
           CANON_RPA_D CANON_RPA_P CANON_RPA_M CANON_LOGPROB_M \
           CANON_PALLAS_ALL_PROJ CANON_PALLAS_ALL_RMSNORM \
           CANON_PALLAS_SWIGLU CANON_PALLAS_MPAD \
           CANON_PALLAS_SWIGLU_MPAD CANON_PALLAS_CANONICAL_VJP; do
    if [[ -v "$k" ]]; then
      echo "[env] $stock_label requires $k absent, got ${!k@Q}" >&2
      fail=1
    fi
  done
  for k in CANON_RPA_VJP2 CANON_VJP2_MAX_SEQS \
           CANON_PROMPT_PROCESSED_LOGPROBS \
           CANON_PALLAS_LOGSOFTMAX \
           CANON_ENGINE_MODULE_C CANON_KV_UNIFIED \
           CANON_P32_DP_ADMISSION CANON_P32_TRAIN_ADMITTED \
           CANON_P32_DP_REDUCTION_ADMITTED \
           CANON_P33_WORKLOAD_LAUNCH_ADMITTED CANON_P32_DP16_SEGMENTED \
           CANON_FROZENLAKE_L3 CANON_FROZENLAKE_P27 \
           CANON_P28_SEGMENTED_FORWARD CANON_P28_SEGMENTED_VJP \
           CANON_P28_SEGMENTED_TRAIN CANON_P28_G6_UPDATE \
           CANON_P28_BATCHED_REPORT CANON_P28_BATCHED_REVERSE \
           CANON_BATCHED_EVIDENCE CANON_P29_FULL_TRAIN \
           CANON_P30_SPARSE_GRAD_ASSEMBLY \
           CANON_P30_FUSED_PAIR_ACCUMULATION \
           CANON_P30_REUSE_SEGMENTED_ENGINE \
           CANON_P30_RELEASE_CAPTURED_STATE \
           CANON_P30_RESHARD_ACCUMULATOR \
           CANON_ALIGNMENT_GATE CANON_ALIGNMENT_GATE_ONLY \
           CANON_ALIGNMENT_UPDATE_CANARY CANON_ALIGNMENT_TRAIN \
           CANON_PRE_ALIGN_GATE CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY \
           CANON_P38_FIXED_LM_HEAD; do
    stock_expected=0
    if [ "$P57_STOCK_TRAIN" = "1" ]; then
      case "$k" in
        CANON_PROMPT_PROCESSED_LOGPROBS|CANON_P32_TRAIN_ADMITTED|CANON_P33_WORKLOAD_LAUNCH_ADMITTED|CANON_ALIGNMENT_GATE|CANON_ALIGNMENT_TRAIN|CANON_PRE_ALIGN_GATE|CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY)
          stock_expected=1
          ;;
      esac
    elif [ "$P57_STOCK_EVAL" = "1" ]; then
      case "$k" in
        CANON_P33_WORKLOAD_LAUNCH_ADMITTED)
          stock_expected=1
          ;;
      esac
    fi
    if [ "$P58_NATIVE" = "1" ]; then
      case "$k" in
        CANON_P32_TRAIN_ADMITTED|CANON_P33_WORKLOAD_LAUNCH_ADMITTED|CANON_ALIGNMENT_GATE|CANON_ALIGNMENT_TRAIN|CANON_PRE_ALIGN_GATE)
          stock_expected=1
          ;;
      esac
    fi
    if [ "${!k:-}" != "$stock_expected" ]; then
      echo "[env] $stock_label requires $k=$stock_expected, got ${!k:-unset}" >&2
      fail=1
    fi
  done
  if [ "$P58_NATIVE" = "1" ] && \
     [ "${CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER:-}" != "1" ]; then
    echo "[env] P58 native requires its independent stock prompt observer" >&2
    fail=1
  fi
  if [ "$P57_STOCK_FAST" = "1" ] && \
     [ "${CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER:-0}" != "0" ]; then
    echo "[env] P57 stock-fast forbids the P58 stock prompt observer" >&2
    fail=1
  fi
  if [ "$GSM8K_NATIVE" = "1" ]; then
    if [ "${#_CANON_GSM8K_NATIVE_INPUT_CONTRADICTIONS[@]}" -ne 0 ]; then
      echo "[env] GSM8K native caller contradictions: ${_CANON_GSM8K_NATIVE_INPUT_CONTRADICTIONS[*]}" >&2
      fail=1
    fi
    [ "${CANON_PROFILE:-}" = \
      "qwen3-1p7b-dp16-tp4-gsm8k-native" ] && \
    [ "${CANON_GSM8K_TRAIN:-}" = "1" ] && \
    [ "${CANON_GSM8K_VANILLA:-}" = "1" ] && \
    [ -z "${CANON_P32_WORKLOAD:-}" ] && \
    [ "${CANON_P33_RUN_STAGE:-}" = "full" ] && \
    [ "${CANON_P33_NO_COMMIT:-1}" = "0" ] && \
    [ "${CANON_P32_TRAIN_ADMITTED:-1}" = "0" ] && \
    [ "${CANON_P32_DP_REDUCTION_ADMITTED:-1}" = "0" ] && \
    [ "${CANON_P33_WORKLOAD_LAUNCH_ADMITTED:-1}" = "0" ] && \
    [ "${CANON_ALIGNMENT_GATE:-1}" = "0" ] && \
    [ "${CANON_ALIGNMENT_TRAIN:-1}" = "0" ] && \
    [ "${CANON_PRE_ALIGN_GATE:-1}" = "0" ] && \
    [ "${CANON_GSM8K_ALIGNMENT_WARN_ONLY:-1}" = "0" ] && \
    [ "${CANON_P38_FIXED_LM_HEAD:-1}" = "0" ] && \
    [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-1}" = "0" ] && \
    [ "${CANON_V1_HP_FULL:-1}" = "0" ] || {
      echo "[env] GSM8K native requires stock vanilla full training with P32, alignment, P59, and V1 disabled" >&2
      fail=1
    }
    for _canon_gsm_native_arg in \
      examples/math_gsm8k/qwen3_grpo_demo.py \
      --mesh_dp=16 --mesh_tp=4 --max_steps=200; do
      case " ${CANON_RUN_CMD:-} " in
        *" $_canon_gsm_native_arg "*) ;;
        *)
          echo "[env] GSM8K native command lacks $_canon_gsm_native_arg" >&2
          fail=1
          ;;
      esac
    done
    unset _canon_gsm_native_arg
  fi
  unset stock_expected
  case "${XLA_FLAGS:-}" in
    *--xla_allow_excess_precision=false*)
      echo "[env] $stock_label forbids the canonical excess-precision pin" >&2
      fail=1
      ;;
  esac
  unset stock_label
else
  for k in CANON_FIXED_AR CANON_FIXED_AR_EMBED CANON_RPA_D CANON_RPA_P \
           CANON_RPA_M CANON_RPA_VJP2 CANON_VJP2_MAX_SEQS \
           CANON_LOGPROB_M CANON_PROMPT_PROCESSED_LOGPROBS; do req "$k"; done
  case "${XLA_FLAGS:-}" in
    *--xla_allow_excess_precision=false*) ;;
    *) echo "[env] MISSING: XLA_FLAGS lacks --xla_allow_excess_precision=false" >&2; fail=1;;
  esac
fi

case "${CANON_RUN_P38_AVAL:-0}" in
  0) ;;
  1)
    req CANON_P38_AVAL_REPORT
    [ "${CANON_MODE:-}" = "gate-only" ] || {
      echo "[env] P38 aval probe requires CANON_MODE=gate-only" >&2
      fail=1
    }
    [ "${CANON_P32_DP_ADMISSION:-0}" = "1" ] || {
      echo "[env] P38 aval probe requires the DP16 topology profile" >&2
      fail=1
    }
    echo "[env] P38 model-free aval probe enabled"
    ;;
  *)
    echo "[env] CANON_RUN_P38_AVAL must be 0 or 1" >&2
    fail=1
    ;;
esac

case "${CANON_KV_UNIFIED:-0}" in
  0|1) ;;
  *) echo "[env] CANON_KV_UNIFIED must be 0 or 1" >&2; fail=1 ;;
esac
if [ -n "${CANON_P38_SERVING_CAPTURE_DIR:-}" ]; then
  for k in CANON_P38_SERVING_CAPTURE_MAX_CALLS \
           CANON_P38_SERVING_CAPTURE_MIN_PREFIX \
           CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS \
           CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER \
           CANON_P38_SERVING_CAPTURE_EXPECTED_PATH \
           CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS \
           CANON_P38_REQUEST_JOURNAL \
           CANON_P38_INCIDENT_LEDGER \
           CANON_P38_INCIDENT_MIN_PREFIX \
           CANON_P38_INCIDENT_MAX_PREFIX \
           CANON_P38_INCIDENT_MAX_BYTES \
           CANON_P38_DURABILITY_PROFILE \
           CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS \
           CANON_P38_LIVE_SNAPSHOT_STOP_FILE \
           CANON_P38_LIVE_SNAPSHOT_WORKER_LOG \
           CANON_P38_LIVE_COLLECT_REQUEST_FILE \
           CANON_P38_LIVE_COLLECT_ACK_FILE \
           CANON_P38_LIVE_COMPLETE_REQUEST_FILE \
           CANON_P38_LIVE_COMPLETE_ACK_FILE \
           CANON_P38_SERVING_CAPTURE_CLASSIFICATION \
           CANON_P38_SERVING_CAPTURE_ARCHIVE \
           CANON_P38_GCS_PREFIX \
           CANON_P38_MISMATCH_CAPSULE \
           CANON_P38_PRECHECK_ONLY \
           CANON_P38_CONTROLLED_EXIT \
           CANON_P38_DIAGNOSTIC_ROUNDS \
           CANON_P38_DIAGNOSTIC_ROUND_FILE \
           CANON_P38_ROUND_SEAL_REQUEST_DIR \
           CANON_P38_ROUND_SEAL_ACK_DIR \
           CANON_P38_MIN_ACTION_KV; do
    req "$k"
  done
  if [ "$APC_M15_TARGET_DEBUG" = "1" ]; then
    req CANON_APC_M15_REPLAY_LEDGER
  fi
  { [ "$APC_M15_TARGET_DEBUG" = "0" ] && \
    [ "$P58_SEAM_LOCALIZATION" = "0" ] && \
    [ "${CANON_P32_WORKLOAD:-}" = "frozenlake" ]; } || \
  { [ "$APC_M15_TARGET_DEBUG" = "1" ] && \
    [ "${CANON_P32_WORKLOAD:-}" = "frozenlake-dp8-tp8" ]; } || \
  { [ "$P58_SEAM_LOCALIZATION" = "1" ] && \
    [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ]; } || {
    echo "[env] P38 serving capture workload identity drifted" >&2
    fail=1
  }
  { [ "$P58_SEAM_LOCALIZATION" = "0" ] && \
    [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ] && \
    [ "${CANON_P33_NO_COMMIT:-}" = "1" ]; } || \
  { [ "$P58_SEAM_LOCALIZATION" = "1" ] && \
    [ "${CANON_P34_RUN_STAGE:-}" = "full" ] && \
    [ "${CANON_P34_NO_COMMIT:-1}" = "0" ]; } || {
    echo "[env] P38 serving capture requires backward-no-commit" >&2
    fail=1
  }
  [ "${CANON_P38_SERVING_CAPTURE_MAX_CALLS:-}" = "4" ] && \
  [ "${CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS:-}" = "4" ] || {
    echo "[env] P38 serving capture must retain exactly four strata" >&2
    fail=1
  }
  [ "${CANON_P38_PRECHECK_ONLY:-}" = "1" ] || {
    echo "[env] P38 serving capture must stop after an exact precheck" >&2
    fail=1
  }
  [ "${CANON_P38_CONTROLLED_EXIT:-}" = "1" ] || {
    echo "[env] P38 serving capture requires controlled diagnostic exit" >&2
    fail=1
  }
  expected_p38_rounds=3
  if [ "$APC_M15_TARGET_DEBUG" = "1" ]; then
    expected_p38_rounds=1
    if [ "${CANON_P38_DURABILITY_PROFILE:-}" = "m15-wide-v1" ] || \
       [ "${CANON_P38_DURABILITY_PROFILE:-}" = "m15-e0-kv-v1" ]; then
      expected_p38_rounds=3
    fi
  fi
  [ "${CANON_P38_DIAGNOSTIC_ROUNDS:-}" = "$expected_p38_rounds" ] || {
    echo "[env] P38 diagnostic round count drifted: expected=$expected_p38_rounds" >&2
    fail=1
  }
  [ "${CANON_P38_ONEHOST_REHEARSAL:-0}" = "0" ] || {
    echo "[env] P38 one-host rehearsal flag is forbidden on target" >&2
    fail=1
  }
  case "${CANON_P38_FIXED_LM_HEAD:-0}" in
    0|1) ;;
    *)
      echo "[env] CANON_P38_FIXED_LM_HEAD must be unset, 0, or 1" >&2
      fail=1
      ;;
  esac
  case "${CANON_P38_DURABILITY_PROFILE:-}" in
    full-v1|round-alignment-v1|m15-wide-v1|m15-e0-kv-v1|p58-seam-v1) ;;
    *)
      echo "[env] P38 durability profile is not admitted" >&2
      fail=1
      ;;
  esac
  if [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "1" ]; then
    [ "${CANON_KV_UNIFIED:-0}" = "0" ] || {
      echo "[env] P38 fixed lm-head requires the stock KV path" >&2
      fail=1
    }
    { [ "$APC_M15_TARGET_DEBUG" = "0" ] && \
      [ "$P58_SEAM_LOCALIZATION" = "0" ] && \
      [ "${CANON_PROFILE_FILE:-}" = \
        "cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env" ]; } || \
    { [ "$APC_M15_TARGET_DEBUG" = "1" ] && \
      [ "${CANON_PROFILE_FILE:-}" = \
        "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env" ]; } || \
    { [ "$P58_SEAM_LOCALIZATION" = "1" ] && \
      [ "${CANON_PROFILE_FILE:-}" = \
        "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env" ]; } || {
      echo "[env] P38 fixed lm-head profile identity drifted" >&2
      fail=1
    }
    [ -z "${CANON_MM_ALGO:-}" ] || {
      echo "[env] P38 fixed lm-head conflicts with CANON_MM_ALGO" >&2
      fail=1
    }
    if [ "$P58_SEAM_LOCALIZATION" = "1" ]; then
      [ "${CANON_P38_DURABILITY_PROFILE:-}" = "p58-seam-v1" ] && \
      [ "${CANON_P38_SEAM_OBSERVER:-}" = "layer" ] && \
      [ "${CANON_P38_TAIL_OBSERVER:-0}" = "1" ] || {
        echo "[env] P58 fixed lm-head seam run requires p58-seam-v1" >&2
        fail=1
      }
    elif [ "$APC_M15_TARGET_DEBUG" = "1" ] && \
       [ -n "${CANON_P38_SEAM_OBSERVER:-}" ]; then
      [ "${CANON_P38_DURABILITY_PROFILE:-}" = "m15-wide-v1" ] || {
        echo "[env] M15 seam observer requires m15-wide-v1 durability" >&2
        fail=1
      }
      [ -z "${CANON_P38_KV_OBSERVER_DIR:-}${CANON_P38_KV_OBSERVER_CLASSIFICATION:-}${CANON_P38_TERMINAL_DISCRIMINATOR:-}" ] || {
        echo "[env] M15 fixed lm-head seam runs must not attach KV or terminal-discriminator observers" >&2
        fail=1
      }
    elif [ "$APC_M15_TARGET_DEBUG" = "1" ] && \
       [ -n "${CANON_P38_KV_OBSERVER_DIR:-}" ]; then
      { { [ "${CANON_P38_DURABILITY_PROFILE:-}" = "round-alignment-v1" ] && \
          [ "${CANON_P38_DIAGNOSTIC_ROUNDS:-}" = "1" ]; } || \
        { [ "${CANON_P38_DURABILITY_PROFILE:-}" = "m15-e0-kv-v1" ] && \
          [ "${CANON_P38_DIAGNOSTIC_ROUNDS:-}" = "3" ]; }; } && \
      [ -z "${CANON_P38_SEAM_OBSERVER:-}${CANON_P38_TAIL_OBSERVER:-}${CANON_P38_TERMINAL_DISCRIMINATOR:-}" ] || {
        echo "[env] M15 targeted KV observer durability contract drifted" >&2
        fail=1
      }
    else
      [ "${CANON_P38_DURABILITY_PROFILE:-}" = "round-alignment-v1" ] || {
        echo "[env] fixed lm-head without M15 seam observation requires round-alignment-v1 durability" >&2
        fail=1
      }
    fi
    if { [ "$APC_M15_TARGET_DEBUG" != "1" ] || \
         [ -z "${CANON_P38_SEAM_OBSERVER:-}${CANON_P38_KV_OBSERVER_DIR:-}" ]; } && \
       { [ "$P58_SEAM_LOCALIZATION" != "1" ] || \
         [ -z "${CANON_P38_SEAM_OBSERVER:-}" ]; } && \
       [ -n "${CANON_P38_KV_OBSERVER_DIR:-}${CANON_P38_KV_OBSERVER_CLASSIFICATION:-}${CANON_P38_SEAM_OBSERVER:-}${CANON_P38_TAIL_OBSERVER:-}${CANON_P38_TERMINAL_DISCRIMINATOR:-}" ]; then
      echo "[env] P38 fixed lm-head diagnostic observers are not admitted on this carrier" >&2
      fail=1
    fi
    for k in CANON_FIXED_AR CANON_FIXED_AR_EMBED \
             CANON_PALLAS_ALL_PROJ CANON_PALLAS_ALL_RMSNORM \
             CANON_PALLAS_SWIGLU CANON_PALLAS_MPAD \
             CANON_PALLAS_SWIGLU_MPAD CANON_PALLAS_CANONICAL_VJP; do
      [ "${!k:-}" = "1" ] || {
        echo "[env] P38 fixed lm-head requires $k=1" >&2
        fail=1
      }
    done
  elif [ "${CANON_P38_DURABILITY_PROFILE:-}" != "full-v1" ]; then
    echo "[env] specialized durability profiles are exclusive to fixed lm-head" >&2
    fail=1
  fi
  [ "${CANON_P38_DIAGNOSTIC_ROUND_FILE:-}" = \
      "${CANON_STATE%/}/p38_diagnostic_round" ] || {
    echo "[env] P38 diagnostic round path drifted" >&2
    fail=1
  }
  [ "${CANON_P38_ROUND_SEAL_REQUEST_DIR:-}" = \
      "${CANON_STATE%/}/p38_round_seal_requests" ] || {
    echo "[env] P38 round-seal request directory drifted" >&2
    fail=1
  }
  [ "${CANON_P38_ROUND_SEAL_ACK_DIR:-}" = \
      "${CANON_STATE%/}/p38_round_seal_acks" ] || {
    echo "[env] P38 round-seal acknowledgement directory drifted" >&2
    fail=1
  }
  expected_p38_gcs_root=p38
  [ "$P58_SEAM_LOCALIZATION" = "0" ] || expected_p38_gcs_root=p58
  expected_p38_gcs_prefix="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/$expected_p38_gcs_root/${CANON_STATE##*/}/attempt-0"
  [ "${CANON_P38_GCS_PREFIX:-}" = "$expected_p38_gcs_prefix" ] || {
    echo "[env] P38 GCS evidence prefix drifted: expected=$expected_p38_gcs_prefix" >&2
    fail=1
  }
  expected_p38_min_action_kv=1686
  [ "$P58_SEAM_LOCALIZATION" = "0" ] || expected_p38_min_action_kv=1686
  [ "${CANON_P38_MIN_ACTION_KV:-}" = "$expected_p38_min_action_kv" ] || {
    echo "[env] P38 serving capture depth-sufficiency contract drifted" >&2
    fail=1
  }
  [[ "${CANON_P38_SERVING_CAPTURE_MIN_PREFIX:-}" =~ ^[0-9]+$ ]] || {
    echo "[env] P38 serving capture minimum prefix must be non-negative" >&2
    fail=1
  }
  expected_p38_capture_min=1536
  expected_p38_capture_bounds=1536,1664,1792,1920,2048
  if [ "$APC_M15_TARGET_DEBUG" = "1" ]; then
    expected_p38_capture_min=1152
    expected_p38_capture_bounds=1152,1216,1280,1408,1696
  fi
  if [ "$P58_SEAM_LOCALIZATION" = "1" ]; then
    expected_p38_capture_min=1686
    expected_p38_capture_bounds=1686,2512,3072,3584,4096
  fi
  [ "${CANON_P38_SERVING_CAPTURE_MIN_PREFIX:-}" = \
      "$expected_p38_capture_min" ] && \
  [ "${CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS:-}" = \
      "$expected_p38_capture_bounds" ] || {
    echo "[env] P38 serving capture prefix strata drifted" >&2
    fail=1
  }
  [ "${CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER:-}" = "5" ] || {
    echo "[env] P38 serving capture requires the five-times free-space guard" >&2
    fail=1
  }
  [ "${CANON_P38_SERVING_CAPTURE_EXPECTED_PATH:-}" = "standard" ] || {
    echo "[env] P38 serving capture must target the standard runner path" >&2
    fail=1
  }
  [ "${CANON_P38_REQUEST_JOURNAL:-}" = \
      "${CANON_P38_SERVING_CAPTURE_DIR%/}/p38_request_journal.jsonl" ] || {
    echo "[env] P38 request journal must live in the capture directory" >&2
    fail=1
  }
  [ "${CANON_P38_INCIDENT_LEDGER:-}" = \
      "${CANON_P38_SERVING_CAPTURE_DIR%/}/p38_incident_ledger.jsonl" ] || {
    echo "[env] P38 incident ledger must live in the capture directory" >&2
    fail=1
  }
  if [ "$APC_M15_TARGET_DEBUG" = "1" ] && \
     [ "${CANON_APC_M15_REPLAY_LEDGER:-}" != \
       "${CANON_P38_SERVING_CAPTURE_DIR%/}/m15_replay_envelope.jsonl" ]; then
    echo "[env] M15 replay ledger must live in the capture directory" >&2
    fail=1
  fi
  expected_p38_incident_min=1400
  expected_p38_incident_max=3072
  expected_p38_incident_bytes=134217728
  if [ "$APC_M15_TARGET_DEBUG" = "1" ]; then
    expected_p38_incident_min=1152
    expected_p38_incident_max=7168
    expected_p38_incident_bytes=2147483648
  fi
  if [ "$P58_SEAM_LOCALIZATION" = "1" ]; then
    expected_p38_incident_min=1686
    expected_p38_incident_max=4096
    expected_p38_incident_bytes=134217728
  fi
  [ "${CANON_P38_INCIDENT_MIN_PREFIX:-}" = \
      "$expected_p38_incident_min" ] && \
  [ "${CANON_P38_INCIDENT_MAX_PREFIX:-}" = \
      "$expected_p38_incident_max" ] && \
  [ "${CANON_P38_INCIDENT_MAX_BYTES:-}" = \
      "$expected_p38_incident_bytes" ] || {
    echo "[env] P38 incident ledger bounds drifted" >&2
    fail=1
  }
  if [ -n "${CANON_P38_SEAM_OBSERVER:-}" ]; then
    for k in CANON_P38_SEAM_OBSERVER_DIR \
             CANON_P38_SEAM_MIN_POSITION \
             CANON_P38_SEAM_MAX_POSITION \
             CANON_P38_SEAM_MAX_BYTES \
             CANON_P38_SEAM_CLASSIFICATION; do
      req "$k"
    done
    if [ "$APC_M15_TARGET_DEBUG" = "1" ]; then
      req CANON_APC_M15_SEAM_BUNDLE
      [ "${CANON_APC_M15_SEAM_BUNDLE:-}" = \
          "${CANON_STATE%/}/m15_wide_seam_bundle.tar" ] || {
        echo "[env] M15 wide seam bundle path drifted" >&2
        fail=1
      }
    elif [ -n "${CANON_APC_M15_SEAM_BUNDLE:-}" ]; then
      echo "[env] M15 wide seam bundle is valid only on its target carrier" >&2
      fail=1
    fi
    [ "${CANON_KV_UNIFIED:-0}" = "0" ] || {
      echo "[env] P38 seam observer requires the stock arm" >&2
      fail=1
    }
    [ "${CANON_P38_SEAM_OBSERVER_DIR%/}" = \
        "${CANON_P38_SERVING_CAPTURE_DIR%/}" ] || {
      echo "[env] P38 seam observer must share the capture directory" >&2
      fail=1
    }
    [ "${CANON_P38_SEAM_OBSERVER:-}" = "layer" ] || \
    [ "${CANON_P38_SEAM_OBSERVER:-}" = "full" ] || {
      echo "[env] P38 seam observer mode must be layer or full" >&2
      fail=1
    }
    expected_p38_seam_min=1400
    expected_p38_seam_max=3072
    expected_p38_seam_bytes=4294967296
    if [ "$APC_M15_TARGET_DEBUG" = "1" ]; then
      expected_p38_seam_min=960
      expected_p38_seam_max=4096
      expected_p38_seam_bytes=8589934592
    fi
    if [ "$P58_SEAM_LOCALIZATION" = "1" ]; then
      expected_p38_seam_min=1686
      expected_p38_seam_max=4096
      expected_p38_seam_bytes=4294967296
    fi
    [ "${CANON_P38_SEAM_MIN_POSITION:-}" = "$expected_p38_seam_min" ] && \
    [ "${CANON_P38_SEAM_MAX_POSITION:-}" = "$expected_p38_seam_max" ] && \
    [ "${CANON_P38_SEAM_MAX_BYTES:-}" = "$expected_p38_seam_bytes" ] || {
      echo "[env] P38 seam observer bounds drifted" >&2
      fail=1
    }
    if [ "${CANON_P38_SEAM_OBSERVER:-}" = "full" ]; then
      case "${CANON_P38_SEAM_LAYER:-}" in
        ''|*[!0-9]*) echo "[env] P38 full seam observer requires a numeric layer" >&2; fail=1;;
      esac
      if [[ "${CANON_P38_SEAM_LAYER:-}" =~ ^[0-9]+$ ]] && \
         [ "$CANON_P38_SEAM_LAYER" -ge 36 ]; then
        echo "[env] P38 seam layer is outside Qwen3-8B" >&2
        fail=1
      fi
    elif [ -n "${CANON_P38_SEAM_LAYER:-}" ]; then
      echo "[env] P38 seam layer is valid only in full mode" >&2
      fail=1
    fi
    if [ -n "${CANON_P38_KV_OBSERVER_DIR:-}" ]; then
      echo "[env] P38 seam and KV observers may not share one target run" >&2
      fail=1
    fi
    expected_p38_concurrency=256
    [ "$P58_SEAM_LOCALIZATION" = "0" ] || expected_p38_concurrency=128
    case " ${CANON_RUN_CMD:-} " in
      *" --max_concurrency=$expected_p38_concurrency "*) ;;
      *) echo "[env] P38 seam observer concurrency drifted" >&2; fail=1;;
    esac
    expected_p38_seam_classification="${CANON_STATE%/}/p38_seam.classification.json"
    [ "$P58_SEAM_LOCALIZATION" = "0" ] || \
      expected_p38_seam_classification="${CANON_STATE%/}/p58_seam.classification.json"
    [ "${CANON_P38_SEAM_CLASSIFICATION:-}" = \
        "$expected_p38_seam_classification" ] || {
      echo "[env] P38 seam classification path drifted" >&2
      fail=1
    }
    if [ -n "${CANON_P38_TAIL_OBSERVER:-}" ]; then
      req CANON_P38_TAIL_MAX_BYTES
      expected_p38_tail_bytes=268435456
      [ "$P58_SEAM_LOCALIZATION" = "0" ] || expected_p38_tail_bytes=67108864
      [ "${CANON_P38_SEAM_OBSERVER:-}" = "layer" ] && \
      [ "${CANON_P38_TAIL_OBSERVER:-}" = "1" ] && \
      [ "${CANON_P38_TAIL_MAX_BYTES:-}" = "$expected_p38_tail_bytes" ] || {
        echo "[env] P38 terminal tail requires bounded layer seam mode" >&2
        fail=1
      }
    elif [ -n "${CANON_P38_TAIL_MAX_BYTES:-}" ]; then
      echo "[env] P38 terminal-tail byte bound is set without the observer" >&2
      fail=1
    fi
    if [ -n "${CANON_P38_TERMINAL_DISCRIMINATOR:-}" ]; then
      req CANON_P38_TERMINAL_MAX_BYTES
      req CANON_P38_TERMINAL_CLASSIFICATION
      [ "${CANON_P38_TERMINAL_DISCRIMINATOR:-}" = "1" ] && \
      [ "${CANON_P38_TAIL_OBSERVER:-}" = "1" ] && \
      [ "${CANON_P38_TERMINAL_MAX_BYTES:-}" = "4294967296" ] && \
      [ "${CANON_P38_TERMINAL_CLASSIFICATION:-}" = \
          "${CANON_STATE%/}/p38_terminal.classification.json" ] || {
        echo "[env] P38 terminal discriminator contract drifted" >&2
        fail=1
      }
    elif [ -n "${CANON_P38_TERMINAL_MAX_BYTES:-}${CANON_P38_TERMINAL_CLASSIFICATION:-}" ]; then
      echo "[env] P38 terminal discriminator fields are set without the observer" >&2
      fail=1
    fi
  elif [ "${CANON_KV_UNIFIED:-0}" = "0" ] && \
       [ -n "${CANON_P38_KV_OBSERVER_DIR:-}" ]; then
    for k in CANON_P38_KV_OBSERVER_DIR \
             CANON_P38_KV_OBSERVER_MAX_CANDIDATES \
             CANON_P38_KV_OBSERVER_MAX_PAGES \
             CANON_P38_KV_OBSERVER_MAX_BYTES \
             CANON_P38_KV_OBSERVER_MAX_READ_BYTES \
             CANON_P38_KV_OBSERVER_CLASSIFICATION; do
      req "$k"
    done
    [ "${CANON_P38_KV_OBSERVER_DIR%/}" = \
        "${CANON_P38_SERVING_CAPTURE_DIR%/}" ] || {
      echo "[env] P38 KV observer must share the capture directory" >&2
      fail=1
    }
    if [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "1" ]; then
      for k in CANON_P38_KV_OBSERVER_LAYER \
               CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256 \
               CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS; do
        req "$k"
      done
      [ "$APC_M15_TARGET_DEBUG" = "1" ] && \
      [ "${CANON_P38_KV_OBSERVER_MAX_CANDIDATES:-}" = "8" ] && \
      [ "${CANON_P38_KV_OBSERVER_MAX_PAGES:-}" = "96" ] && \
      [ "${CANON_P38_KV_OBSERVER_MAX_BYTES:-}" = "134217728" ] && \
      [ "${CANON_P38_KV_OBSERVER_MAX_READ_BYTES:-}" = "671088640" ] && \
      [ "${CANON_P38_KV_OBSERVER_LAYER:-}" = "0" ] && \
      [ "${CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS:-}" = "1226" ] && \
      [ "${CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256:-}" = \
          "546ee92f36038198c4e4056078cb05ec77ef9b653f64cd1a8de49f7812e9e75d" ] || {
        echo "[env] M15 targeted KV observer bounds drifted" >&2
        fail=1
      }
    else
      [ -z "${CANON_P38_KV_OBSERVER_LAYER:-}${CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256:-}${CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS:-}" ] && \
      [ "${CANON_P38_KV_OBSERVER_MAX_CANDIDATES:-}" = "3" ] && \
      [ "${CANON_P38_KV_OBSERVER_MAX_PAGES:-}" = "16" ] && \
      [ "${CANON_P38_KV_OBSERVER_MAX_BYTES:-}" = "134217728" ] && \
      [ "${CANON_P38_KV_OBSERVER_MAX_READ_BYTES:-}" = "671088640" ] || {
        echo "[env] P38 KV observer bounds drifted" >&2
        fail=1
      }
    fi
    [ "${CANON_P38_KV_OBSERVER_CLASSIFICATION:-}" = \
        "${CANON_STATE%/}/p38_kv_observer.classification.json" ] || {
      echo "[env] P38 KV observer classification path drifted" >&2
      fail=1
    }
  fi
  [ "${CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS:-}" = "30" ] || {
    echo "[env] P38 live snapshot interval drifted" >&2
    fail=1
  }
  [ "${CANON_P38_LIVE_SNAPSHOT_STOP_FILE:-}" = \
      "${CANON_STATE%/}/p38_live.stop" ] || {
    echo "[env] P38 live snapshot stop path drifted" >&2
    fail=1
  }
  [ "${CANON_P38_LIVE_SNAPSHOT_WORKER_LOG:-}" = \
      "${CANON_STATE%/}/p38_live_worker.log" ] || {
    echo "[env] P38 live snapshot worker log path drifted" >&2
    fail=1
  }
  for live_contract in \
      "CANON_P38_LIVE_COLLECT_REQUEST_FILE:p38_collect.request" \
      "CANON_P38_LIVE_COLLECT_ACK_FILE:p38_collect.ack" \
      "CANON_P38_LIVE_COMPLETE_REQUEST_FILE:p38_complete.request" \
      "CANON_P38_LIVE_COMPLETE_ACK_FILE:p38_complete.ack"; do
    live_key="${live_contract%%:*}"
    live_name="${live_contract#*:}"
    if [ "${!live_key:-}" != "${CANON_STATE%/}/$live_name" ]; then
      echo "[env] P38 live worker control path drifted: $live_key" >&2
      fail=1
    fi
  done
  echo "[env] P38 serving capture enabled: kv_unified=${CANON_KV_UNIFIED:-0} path=${CANON_P38_SERVING_CAPTURE_EXPECTED_PATH:-missing}"
elif [ -n "${CANON_P58_CHECKED_VMA_DIAGNOSTIC:-}" ]; then
  for k in CANON_P38_PRECHECK_ONLY CANON_P38_CONTROLLED_EXIT \
           CANON_P38_DIAGNOSTIC_ROUNDS CANON_P38_DIAGNOSTIC_ROUND_FILE; do
    req "$k"
  done
  [ "${CANON_KV_UNIFIED:-0}" = "0" ] && \
  [ "${CANON_P38_ONEHOST_REHEARSAL:-0}" = "0" ] && \
  [ "${CANON_VLLM_ENABLE_PREFIX_CACHING:-0}" = "0" ] && \
  [ "${CANON_P38_DIAGNOSTIC_ROUND_FILE:-}" = \
      "${CANON_STATE%/}/p38_diagnostic_round" ] || {
    echo "[env] P58 checked-VMA-${CANON_P58_CHECKED_VMA_DIAGNOSTIC} precheck carrier drifted" >&2
    fail=1
  }
  echo "[env] P58 checked-VMA-${CANON_P58_CHECKED_VMA_DIAGNOSTIC} P38 precheck admitted"
elif [ -n "${CANON_V1_FL_TP8_AB_ARM:-}" ]; then
  for k in CANON_P38_PRECHECK_ONLY CANON_P38_CONTROLLED_EXIT \
           CANON_P38_DIAGNOSTIC_ROUNDS CANON_P38_DIAGNOSTIC_ROUND_FILE \
           CANON_P38_MIN_ACTION_KV; do
    req "$k"
  done
  [ "${CANON_KV_UNIFIED:-0}" = "0" ] && \
  [ "${CANON_P38_ONEHOST_REHEARSAL:-0}" = "0" ] && \
  [ "${CANON_VLLM_ENABLE_PREFIX_CACHING:-0}" = "0" ] || {
    echo "[env] V1 FrozenLake TP8 A/B carrier drifted" >&2
    fail=1
  }
  echo "[env] V1 FrozenLake TP8 A/B precheck admitted arm=$CANON_V1_FL_TP8_AB_ARM"
elif [ "${CANON_KV_UNIFIED:-0}" = "1" ]; then
  echo "[env] CANON_KV_UNIFIED is admitted only with bounded P38 serving capture" >&2
  fail=1
elif [ -n "${CANON_P38_SERVING_CAPTURE_MAX_CALLS:-}${CANON_P38_SERVING_CAPTURE_MIN_PREFIX:-}${CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS:-}${CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER:-}${CANON_P38_SERVING_CAPTURE_EXPECTED_PATH:-}${CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS:-}${CANON_P38_REQUEST_JOURNAL:-}${CANON_P38_INCIDENT_LEDGER:-}${CANON_APC_M15_REPLAY_LEDGER:-}${CANON_APC_M15_SEAM_BUNDLE:-}${CANON_P38_INCIDENT_MIN_PREFIX:-}${CANON_P38_INCIDENT_MAX_PREFIX:-}${CANON_P38_INCIDENT_MAX_BYTES:-}${CANON_P38_DURABILITY_PROFILE:-}${CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS:-}${CANON_P38_LIVE_SNAPSHOT_STOP_FILE:-}${CANON_P38_LIVE_SNAPSHOT_WORKER_LOG:-}${CANON_P38_LIVE_COLLECT_REQUEST_FILE:-}${CANON_P38_LIVE_COLLECT_ACK_FILE:-}${CANON_P38_LIVE_COMPLETE_REQUEST_FILE:-}${CANON_P38_LIVE_COMPLETE_ACK_FILE:-}${CANON_P38_SERVING_CAPTURE_CLASSIFICATION:-}${CANON_P38_SERVING_CAPTURE_ARCHIVE:-}${CANON_P38_GCS_PREFIX:-}${CANON_P38_PRECHECK_ONLY:-}${CANON_P38_CONTROLLED_EXIT:-}${CANON_P38_DIAGNOSTIC_ROUNDS:-}${CANON_P38_DIAGNOSTIC_ROUND_FILE:-}${CANON_P38_ROUND_SEAL_REQUEST_DIR:-}${CANON_P38_ROUND_SEAL_ACK_DIR:-}${CANON_P38_ONEHOST_REHEARSAL:-}${CANON_P38_MIN_ACTION_KV:-}${CANON_P38_KV_OBSERVER_DIR:-}${CANON_P38_KV_OBSERVER_MAX_CANDIDATES:-}${CANON_P38_KV_OBSERVER_MAX_PAGES:-}${CANON_P38_KV_OBSERVER_MAX_BYTES:-}${CANON_P38_KV_OBSERVER_MAX_READ_BYTES:-}${CANON_P38_KV_OBSERVER_LAYER:-}${CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256:-}${CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS:-}${CANON_P38_KV_OBSERVER_CLASSIFICATION:-}${CANON_P38_SEAM_OBSERVER:-}${CANON_P38_SEAM_OBSERVER_DIR:-}${CANON_P38_SEAM_MIN_POSITION:-}${CANON_P38_SEAM_MAX_POSITION:-}${CANON_P38_SEAM_MAX_BYTES:-}${CANON_P38_SEAM_LAYER:-}${CANON_P38_SEAM_CLASSIFICATION:-}${CANON_P38_TAIL_OBSERVER:-}${CANON_P38_TAIL_MAX_BYTES:-}" ]; then
  echo "[env] partial P38 serving-capture configuration is not admitted" >&2
  fail=1
fi
if [ "$APC_M15_TARGET_DEBUG" = "1" ] && \
   [ -z "${CANON_P38_SERVING_CAPTURE_DIR:-}" ]; then
  echo "[env] M15 APC target debug requires bounded serving capture" >&2
  fail=1
fi

case "${CANON_P38_FIXED_LM_HEAD:-0}" in
  0|1) ;;
  *)
    echo "[env] CANON_P38_FIXED_LM_HEAD must be unset, 0, or 1" >&2
    fail=1
    ;;
esac
if [ "${CANON_PROFILE_FILE:-}" = \
     "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env" ] || \
   [ "${CANON_PROFILE_FILE:-}" = \
     "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env" ]; then
  [[ "${CANON_P57_EXPECTED_UPDATES:-}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[env] P57 requires a positive expected-update horizon" >&2
    fail=1
  }
  p57_expected_milestone_interval=0
  p57_expected_checkpoint_interval=10
  p57_checkpoint_disabled=0
  p57_expected_eval_enabled=0
  case "${CANON_P57_WORKLOAD_CANDIDATE:-}:${CANON_P57_DATA_SPLIT:-}:${CANON_P57_EXPECTED_UPDATES:-}" in
    ::300|m15:main:300)
      p57_expected_eval_enabled=1
      if [ "${CANON_P57_TIM_ARM:-}" = "zero" ]; then
        p57_expected_eval_enabled=0
        [ "${CANON_P57_RUN_KIND:-}" != "train" ] || p57_checkpoint_disabled=1
      fi
      p57_expected_checkpoint_interval=300
      ;;
  esac
  p57_expected_eval_disabled=$((1 - p57_expected_eval_enabled))
  if [ "$p57_checkpoint_disabled" = "1" ]; then
    [ "${CANON_FROZENLAKE_CKPT_MODE:-}" = "disabled" ] && \
    [ -z "${CANON_FROZENLAKE_CKPT_ROOT:-}${CANON_FROZENLAKE_CKPT_TAG:-}${CANON_FROZENLAKE_CKPT_INTERVAL:-}${CANON_FROZENLAKE_CKPT_MAX_TO_KEEP:-}${CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL:-}" ] || {
      echo "[env] P57 optimized zero checkpoint-disabled contract drifted" >&2
      fail=1
    }
  else
    [ "${CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL:-0}" = \
      "$p57_expected_milestone_interval" ] || {
      echo "[env] P57 checkpoint milestone retention drifted" >&2
      fail=1
    }
    [ "${CANON_FROZENLAKE_CKPT_INTERVAL:-}" = \
      "$p57_expected_checkpoint_interval" ] && \
    [ "${CANON_FROZENLAKE_CKPT_MAX_TO_KEEP:-}" = "1" ] || {
      echo "[env] P57 checkpoint cadence drifted" >&2
      fail=1
    }
  fi
  if [ -n "${CANON_P57_WORKLOAD_CANDIDATE:-}" ] || \
     [ -n "${CANON_P57_DATA_SPLIT:-}" ]; then
    case "${CANON_P57_WORKLOAD_CANDIDATE:-}:${CANON_P57_DATA_SPLIT:-}" in
      l0:calibration|l0:selection|l0:main|\
      m10:calibration|m10:selection|m10:main|\
      m15:calibration|m15:selection|m15:main|\
      m20:calibration|m20:selection|m20:main) ;;
      *)
        echo "[env] P57 materialized workload fields drifted" >&2
        fail=1
        ;;
    esac
    if [ "${CANON_P57_DATA_SPLIT:-}" != "main" ] && \
       [ "${CANON_P57_TIM_ARM:-}" != "mismatch" ]; then
      echo "[env] P57 calibration/selection workloads are stock-only" >&2
      fail=1
    fi
  fi
  case "${CANON_P57_RUN_KIND:-}:${CANON_P57_TIM_ARM:-}" in
    calibration:mismatch)
      [ "$P57_STOCK_FAST" = "1" ] && \
      [ "${CANON_P57_INFERENCE_REGIME:-}" = "stock-fast" ] && \
      [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "0" ] && \
      [ "${CANON_P57_CALIBRATION_MODE:-}" = "stochastic" ] && \
      [ "${CANON_P57_CALIBRATION_RECIPES:-}" = "m10,m15,m20" ] && \
      [ -z "${CANON_P57_WORKLOAD_CANDIDATE:-}${CANON_P57_DATA_SPLIT:-}" ] && \
      [ -z "${CANON_P57_EVAL_CHECKPOINT_STEP:-}${CANON_P57_EVAL_OUTPUT:-}" ] && \
      [[ "${CANON_P57_CALIBRATION_OUTPUT:-}" = /* ]] || {
        echo "[env] P57 stock-fast calibration contract drifted" >&2
        fail=1
      }
      ;;
    train:zero)
      [ -z "${CANON_P57_INFERENCE_REGIME:-}" ] && \
      [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "1" ] && \
      [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" = "0" ] && \
      [ "${CANON_P33_ENABLE_EVAL:-}" = "$p57_expected_eval_enabled" ] && \
      [ "${CANON_P33_DISABLE_EVAL:-}" = "$p57_expected_eval_disabled" ] && \
      [ "${CANON_P31_ENABLE_EVAL:-}" = "$p57_expected_eval_enabled" ] && \
      [ "${CANON_P57_STOP_AFTER_STEP:-}" = \
        "${CANON_P57_EXPECTED_UPDATES:-}" ] && \
      [ -z "${CANON_P57_EVAL_CHECKPOINT_STEP:-}${CANON_P57_EVAL_OUTPUT:-}" ] || {
        echo "[env] P57 zero training contract drifted" >&2
        fail=1
      }
      ;;
    train:mismatch|train:is)
      [ "${CANON_P57_INFERENCE_REGIME:-}" = "stock-fast" ] && \
      [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "0" ] && \
      [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" = "1" ] && \
      [ "${CANON_P33_ENABLE_EVAL:-}" = "$p57_expected_eval_enabled" ] && \
      [ "${CANON_P33_DISABLE_EVAL:-}" = "$p57_expected_eval_disabled" ] && \
      [ "${CANON_P31_ENABLE_EVAL:-}" = "$p57_expected_eval_enabled" ] && \
      [[ "${CANON_P57_STOP_AFTER_STEP:-}" =~ ^[1-9][0-9]*$ ]] && \
      [ $((CANON_P57_STOP_AFTER_STEP % 50)) -eq 0 ] && \
      [ "$CANON_P57_STOP_AFTER_STEP" -le "$CANON_P57_EXPECTED_UPDATES" ] && \
      { [ "$p57_expected_checkpoint_interval" != "300" ] || \
        [ "$CANON_P57_STOP_AFTER_STEP" = "$CANON_P57_EXPECTED_UPDATES" ]; } && \
      [ -z "${CANON_P57_EVAL_CHECKPOINT_STEP:-}${CANON_P57_EVAL_OUTPUT:-}" ] || {
        echo "[env] P57 native training contract drifted" >&2
        fail=1
      }
      ;;
    eval:zero|eval:mismatch|eval:is)
      if [ "${CANON_P57_TIM_ARM}" != "zero" ]; then
        [ "$P57_STOCK_EVAL" = "1" ] && \
        [ "${CANON_P57_INFERENCE_REGIME:-}" = "stock-fast" ] || {
          echo "[env] P57 native eval requires the stock-fast runtime" >&2
          fail=1
        }
      elif [ -n "${CANON_P57_INFERENCE_REGIME:-}" ]; then
        echo "[env] P57 zero eval forbids an inference-regime override" >&2
        fail=1
      fi
      p57_expected_fixed=0
      [ "${CANON_P57_TIM_ARM}" = "zero" ] && p57_expected_fixed=1
      [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "$p57_expected_fixed" ] && \
      [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" = "0" ] && \
      [[ "${CANON_P57_EVAL_CHECKPOINT_STEP:-}" =~ ^(0|[1-9][0-9]*)$ ]] && \
      { [ "$p57_expected_milestone_interval" = "50" ] && \
        [ $((CANON_P57_EVAL_CHECKPOINT_STEP % 50)) -eq 0 ] || \
        { [ "$CANON_P57_EVAL_CHECKPOINT_STEP" = "0" ] || \
          [ "$CANON_P57_EVAL_CHECKPOINT_STEP" = "$CANON_P57_EXPECTED_UPDATES" ]; }; } && \
      [ "$CANON_P57_EVAL_CHECKPOINT_STEP" -le "$CANON_P57_EXPECTED_UPDATES" ] && \
      [[ "${CANON_P57_EVAL_OUTPUT:-}" = /* ]] || {
        echo "[env] P57 isolated evaluation contract drifted" >&2
        fail=1
      }
      p57_expected_mode=resume
      [ "${CANON_P57_EVAL_CHECKPOINT_STEP:-}" = "0" ] && p57_expected_mode=new
      [ "${CANON_FROZENLAKE_CKPT_MODE:-}" = "$p57_expected_mode" ] || {
        echo "[env] P57 evaluation checkpoint mode drifted" >&2
        fail=1
      }
      unset p57_expected_fixed
      unset p57_expected_mode
      ;;
    *)
      echo "[env] P57 run kind/arm contract is invalid" >&2
      fail=1
      ;;
  esac
  unset p57_expected_milestone_interval
  unset p57_expected_checkpoint_interval
  unset p57_checkpoint_disabled
  unset p57_expected_eval_enabled
  unset p57_expected_eval_disabled
elif [ "${CANON_PROFILE_FILE:-}" = \
       "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env" ]; then
  # The bounded APC carrier is not a P57 train/eval arm, but it reuses the
  # materialized M15 workload.  Admit only its exact signed workload identity;
  # all training-horizon/arm fields remain forbidden.
  [ "${CANON_P57_WORKLOAD_CANDIDATE:-}" = "m15" ] && \
  [ "${CANON_P57_DATA_SPLIT:-}" = "main" ] && \
  [ -z "${CANON_P57_TIM_ARM:-}${CANON_P57_RUN_KIND:-}${CANON_P57_INFERENCE_REGIME:-}${CANON_P57_EXPECTED_UPDATES:-}${CANON_P57_STOP_AFTER_STEP:-}${CANON_P57_EVAL_CHECKPOINT_STEP:-}${CANON_P57_EVAL_OUTPUT:-}${CANON_P57_CALIBRATION_MODE:-}${CANON_P57_CALIBRATION_OUTPUT:-}${CANON_P57_CALIBRATION_RECIPES:-}" ] || {
    echo "[env] M15 APC target debug P57 identity drifted" >&2
    fail=1
  }
elif [ "${CANON_PROFILE_FILE:-}" = \
       "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug.env" ]; then
  case "${CANON_P57_WORKLOAD_CANDIDATE:-}:${CANON_P57_DATA_SPLIT:-}" in
    :|m15:main) ;;
    *)
      echo "[env] V1 FrozenLake TP8 A/B P57 identity drifted" >&2
      fail=1
      ;;
  esac
  [ -z "${CANON_P57_TIM_ARM:-}${CANON_P57_RUN_KIND:-}${CANON_P57_INFERENCE_REGIME:-}${CANON_P57_EXPECTED_UPDATES:-}${CANON_P57_STOP_AFTER_STEP:-}${CANON_P57_EVAL_CHECKPOINT_STEP:-}${CANON_P57_EVAL_OUTPUT:-}${CANON_P57_CALIBRATION_MODE:-}${CANON_P57_CALIBRATION_OUTPUT:-}${CANON_P57_CALIBRATION_RECIPES:-}" ] || {
    echo "[env] V1 FrozenLake TP8 A/B forbids P57 train/eval state" >&2
    fail=1
  }
elif [ -n "${CANON_P57_TIM_ARM:-}${CANON_P57_RUN_KIND:-}${CANON_P57_INFERENCE_REGIME:-}${CANON_P57_EXPECTED_UPDATES:-}${CANON_P57_STOP_AFTER_STEP:-}${CANON_P57_EVAL_CHECKPOINT_STEP:-}${CANON_P57_EVAL_OUTPUT:-}${CANON_P57_CALIBRATION_MODE:-}${CANON_P57_CALIBRATION_OUTPUT:-}${CANON_P57_CALIBRATION_RECIPES:-}${CANON_P57_WORKLOAD_CANDIDATE:-}${CANON_P57_DATA_SPLIT:-}" ]; then
  echo "[env] P57 fields require the P57 profile" >&2
  fail=1
fi
if [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "1" ] && \
   [ -z "${CANON_P38_SERVING_CAPTURE_DIR:-}" ]; then
  if [ "${CANON_P34_DEEPSWE:-0}" = "1" ]; then
    case "${CANON_P34_RUN_STAGE:-}:${CANON_PROFILE_FILE:-}" in
      backward-no-commit:cluster/profiles/qwen3-32b-dp16-tp8-deepswe.env|\
      one-update:cluster/profiles/qwen3-32b-dp16-tp8-deepswe.env|\
      three-update:cluster/profiles/qwen3-32b-dp16-tp8-deepswe.env|\
      full:cluster/profiles/qwen3-32b-dp16-tp8-deepswe.env|\
      full:cluster/profiles/qwen3-32b-dp-parity-deepswe-full.env|\
      one-update:cluster/profiles/qwen3-4b-dp-parity-deepswe-debug.env|\
      three-update:cluster/profiles/qwen3-4b-dp-parity-deepswe-debug.env)
        [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-0}" = "1" ] || {
          echo "[env] fixed lm-head DeepSWE training requires warning-only A-B reporting" >&2
          fail=1
        }
        [ "${CANON_P46_EVALUATION:-0}" = "0" ] || {
          echo "[env] fixed lm-head is forbidden in the P46 evaluation lane" >&2
          fail=1
        }
        echo "[env] P38.2y2 fixed lm-head DeepSWE training enabled"
        ;;
      full:cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env)
        [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ] && \
        [ "${CANON_P58_TIM_ADMITTED:-0}" = "1" ] && \
        [ "${CANON_P58_TIM_ARM:-}" = "zero" ] && \
        [ "${CANON_V1_HP_FULL:-0}" = "1" ] && \
        [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-1}" = "0" ] && \
        [ "${CANON_P46_EVALUATION:-0}" = "0" ] || {
          echo "[env] P58 v1-hp fixed lm-head requires strict Zero full training" >&2
          fail=1
        }
        echo "[env] P58 v1-hp Qwen3-4B TP8 fixed lm-head enabled"
        ;;
      *)
        echo "[env] fixed lm-head is not admitted for this DeepSWE stage/profile" >&2
        fail=1
        ;;
    esac
  else
    case "${CANON_P32_WORKLOAD:-}:${CANON_P33_RUN_STAGE:-}:${CANON_P33_NO_COMMIT:-}:${CANON_PROFILE_FILE:-}" in
    frozenlake:backward-no-commit:1:cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env)
      [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" = "0" ] && \
      [ "${CANON_P33_ENABLE_EVAL:-0}" = "0" ] && \
      [ "${CANON_P33_DISABLE_EVAL:-}" = "1" ] || {
        echo "[env] fixed lm-head backward requires strict alignment and evaluation off" >&2
        fail=1
      }
      echo "[env] P38.2h fixed lm-head backward-no-commit enabled"
      ;;
    frozenlake-dp8-tp8:full:0:cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env|\
    frozenlake-dp8-tp8:full:0:cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env)
      case "${CANON_P57_RUN_KIND:-}:${CANON_P57_TIM_ARM:-}" in
        train:zero)
          [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" = "0" ] && \
          [ "${CANON_P33_ENABLE_EVAL:-}" = "0" ] && \
          [ "${CANON_P33_DISABLE_EVAL:-}" = "1" ] && \
          [ "${CANON_P31_ENABLE_EVAL:-}" = "0" ] || {
            echo "[env] P57 zero arm requires strict full training with in-process evaluation disabled" >&2
            fail=1
          }
          echo "[env] P57 zero-TIM fixed lm-head training enabled"
          ;;
        eval:zero)
          [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" = "0" ] && \
          [ "${CANON_P33_ENABLE_EVAL:-0}" = "0" ] && \
          [ "${CANON_P33_DISABLE_EVAL:-}" = "1" ] || {
            echo "[env] P57 zero evaluation requires strict isolated resume" >&2
            fail=1
          }
          echo "[env] P57 zero-TIM fixed lm-head evaluation enabled"
          ;;
        *)
          echo "[env] P57 fixed lm-head is admitted only for the zero arm" >&2
          fail=1
          ;;
      esac
      ;;
    gsm8k:full:0:cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env)
      [ "${CANON_GSM8K_ALIGNMENT_WARN_ONLY:-0}" = "1" ] || {
        echo "[env] fixed lm-head GSM8K full requires warning-only A-B reporting" >&2
        fail=1
      }
      echo "[env] P38.2y fixed lm-head GSM8K full enabled"
      ;;
    gsm8k:full:0:cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env)
      [ "${CANON_GSM8K_ALIGNMENT_WARN_ONLY:-0}" = "0" ] && \
      [ "${CANON_V1_HP_FULL:-0}" = "1" ] || {
        echo "[env] V1 high-performance GSM8K requires strict alignment" >&2
        fail=1
      }
      echo "[env] V1 high-performance fixed lm-head GSM8K full enabled"
      ;;
    gsm8k:backward-no-commit:1:cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-p62-debug.env)
      [ "${CANON_P62_BACKWARD_NUMERIC_DEBUG:-0}" = "1" ] && \
      [ "${CANON_GSM8K_ALIGNMENT_WARN_ONLY:-0}" = "0" ] && \
      [ "${CANON_P33_NO_COMMIT:-0}" = "1" ] || {
        echo "[env] P62 numeric debug requires strict alignment and zero optimizer commits" >&2
        fail=1
      }
      echo "[env] P62 numeric debug fixed lm-head enabled"
      ;;
    frozenlake-dp8-tp8:backward-no-commit:1:cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-p64-debug.env)
      [ "${CANON_P64_P45_NUMERIC_DEBUG:-0}" = "1" ] && \
      [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" = "0" ] && \
      [ "${CANON_P33_NO_COMMIT:-0}" = "1" ] || {
        echo "[env] P64 numeric debug requires strict alignment and zero optimizer commits" >&2
        fail=1
      }
      echo "[env] P64 numeric debug fixed lm-head enabled"
      ;;
    frozenlake-dp8-tp8:backward-no-commit:1:cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug.env)
      [ -n "${CANON_V1_FL_TP8_AB_ARM:-}" ] && \
      [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" = "0" ] || {
        echo "[env] V1 FrozenLake TP8 A/B fixed-head contract drifted" >&2
        fail=1
      }
      echo "[env] V1 FrozenLake TP8 A/B fixed lm-head enabled"
      ;;
    *)
      echo "[env] fixed lm-head is not admitted for this workload/stage/profile" >&2
      fail=1
      ;;
    esac
  fi
  [ -n "${CANON_V1_FL_TP8_AB_ARM:-}" ] || \
  [ -n "${CANON_P58_CHECKED_VMA_DIAGNOSTIC:-}" ] || \
  [ -z "${CANON_MM_ALGO:-}${CANON_P38_PRECHECK_ONLY:-}${CANON_P38_CONTROLLED_EXIT:-}${CANON_P38_DIAGNOSTIC_ROUNDS:-}${CANON_P38_KV_OBSERVER_DIR:-}${CANON_P38_SEAM_OBSERVER:-}${CANON_P38_TAIL_OBSERVER:-}${CANON_P38_TERMINAL_DISCRIMINATOR:-}" ] || {
    echo "[env] fixed lm-head backward conflicts with diagnostic/algorithm env" >&2
    fail=1
  }
fi

if [ "${CANON_P35_ENVELOPE:-0}" = "1" ]; then
  for k in CANON_P35_ENVELOPE_REPORT CANON_P35_METADATA_DIR \
           CANON_P35_CLASSIFICATION CANON_DP_SIZE CANON_LOGPROB_M; do
    req "$k"
  done
  [ "${CANON_P32_WORKLOAD:-}" = "gsm8k" ] || {
    echo "[env] P35 envelope-short requires the GSM8K workload" >&2; fail=1;
  }
  [ "${CANON_P33_RUN_STAGE:-}" = "envelope-short" ] && \
  [ "${CANON_P33_NO_COMMIT:-}" = "1" ] || {
    echo "[env] P35 requires envelope-short with no-commit=1" >&2; fail=1;
  }
  [ "${CANON_DP_SIZE:-}" = "16" ] && \
  [ "${CANON_LOGPROB_M:-}" = "256" ] || {
    echo "[env] P35 requires DP16 and canonical local M256" >&2; fail=1;
  }
  case " ${CANON_RUN_CMD:-} " in
    *" --max_response_length=256 "*)
      echo "[env] P35 envelope contract OK: gsm8k DP16 local-M256 response-256"
      ;;
    *) echo "[env] P35 command must pin max_response_length=256" >&2; fail=1;;
  esac
  if [ "${CANON_P35_EXACT_REPLAY:-0}" = "1" ]; then
    for k in CANON_P35_EXACT_REPLAY_REPORT \
             CANON_P35_EXACT_REPLAY_CLASSIFICATION; do
      req "$k"
    done
    echo "[env] P35.3 exact-input replay enabled"
  fi
  case "${CANON_P35_REPLAY_STAGE_PROBE:-0}" in
    0) ;;
    1)
      [ "${CANON_P35_EXACT_REPLAY:-0}" = "1" ] || {
        echo "[env] P35.3c stage probe requires exact replay" >&2; fail=1;
      }
      for k in CANON_P35_REPLAY_STAGE_REPORT \
               CANON_P35_REPLAY_STAGE_CLASSIFICATION; do
        req "$k"
      done
      echo "[env] P35.3c first-record stage probe enabled; no numerical verdict"
      ;;
    *)
      echo "[env] CANON_P35_REPLAY_STAGE_PROBE must be 0 or 1" >&2
      fail=1
      ;;
  esac
fi
if [ -n "${CANON_RPA_VJP:-}" ] && [ "${CANON_RPA_VJP:-}" = "1" ]; then
  echo "[env] NOTE: CANON_RPA_VJP=1 is set alongside VJP2.  VJP2 wins in the engine, but if"
  echo "[env]       VJP2 were ever unset this would silently select the prefill-only contract"
  echo "[env]       whose kv gradients are identically zero.  See KNOWN_FOOTGUNS.md."
fi

if [ "${CANON_P34_DEEPSWE:-0}" = "1" ]; then
  for k in CANON_P34_TOPOLOGY_ADMITTED CANON_P34_TP8_ADMITTED \
           CANON_P34_TRAJECTORY_ADMITTED CANON_P34_UPDATE_ADMITTED \
           CANON_P34_RUN_STAGE CANON_P34_NO_COMMIT \
           CANON_DP_SIZE CANON_TP_SIZE CANON_TOTAL_DEVICES \
           CANON_ENGINE_DP_SIZE CANON_GLOBAL_PROMPTS \
           CANON_NUM_GENERATIONS CANON_LOCAL_TRAJECTORIES \
           CANON_GLOBAL_TRAJECTORIES CANON_TARGET_M CANON_P34_ABCPROD \
           CANON_P34_PREFIX_CACHE CANON_P34_MAX_NUM_SEQS \
           CANON_P34_MAX_BATCHED_TOKENS CANON_P34_STRICT_CLI \
           CANON_P34_DISABLE_SAMPLER_IS CANON_P34_DISABLE_TIS \
           CANON_PRE_ALIGN_GATE \
           CANON_P34_TRAJECTORY_CAPTURE \
           CANON_P34_DATASET_NAME CANON_P34_DATASET_REVISION \
           CANON_P34_DATASET_SPLIT CANON_P34_DATASET_ROWS \
           CANON_P34_CLEAN_ROWS \
           CANON_P39_64CHIP_PILOT CANON_P39_PILOT_ADMITTED \
           CANON_P43_DEEPSWE_DEBUG CANON_P43_DEBUG_ADMITTED \
           CANON_P43_ROLLOUT_ONLY \
           CANON_P44_DEEPSWE_PARITY CANON_P44_PARITY_ADMITTED \
           CANON_P44_TOPOLOGY CANON_P44_ROLLOUT_ONLY \
           CANON_P46_DEEPSWE_TRAIN CANON_P46_EVALUATION \
           CANON_P46_TOPOLOGY CANON_P58_DEEPSWE_TIM \
           CANON_P58_TIM_ADMITTED CANON_P58_TIM_ARM \
           CANON_P58_EXPECTED_UPDATES \
           CANON_OPT_STATE_RESIDENT CANON_P30_OPT_STATE_OFFLOAD \
           CANON_DEEPSWE_ALIGNMENT_WARN_ONLY \
           CANON_TRAIN_DP_SHARDING FL_SHARED_MESH \
           CANON_P34_WHITELIST CANON_P34_WHITELIST_SHA256; do
    req "$k"
  done
  for k in CANON_DP_SIZE CANON_TP_SIZE CANON_TOTAL_DEVICES \
           CANON_ENGINE_DP_SIZE CANON_GLOBAL_PROMPTS \
           CANON_NUM_GENERATIONS CANON_LOCAL_TRAJECTORIES \
           CANON_GLOBAL_TRAJECTORIES CANON_TARGET_M CANON_P34_ABCPROD \
           CANON_P34_MAX_NUM_SEQS CANON_P34_MAX_BATCHED_TOKENS \
           CANON_P34_DATASET_ROWS; do
    positive_int "$k"
  done
  case "${CANON_P43_DEEPSWE_DEBUG:-}" in
    0) ;;
    1) ;;
    *)
      echo "[env] CANON_P43_DEEPSWE_DEBUG must be exactly 0 or 1" >&2
      fail=1
      ;;
  esac
  case "${CANON_P44_DEEPSWE_PARITY:-}" in
    0) ;;
    1) ;;
    *)
      echo "[env] CANON_P44_DEEPSWE_PARITY must be exactly 0 or 1" >&2
      fail=1
      ;;
  esac
  case "${CANON_P46_DEEPSWE_TRAIN:-}" in
    0|1) ;;
    *)
      echo "[env] CANON_P46_DEEPSWE_TRAIN must be exactly 0 or 1" >&2
      fail=1
      ;;
  esac
  case "${CANON_P58_DEEPSWE_TIM:-}" in
    0|1) ;;
    *)
      echo "[env] CANON_P58_DEEPSWE_TIM must be exactly 0 or 1" >&2
      fail=1
      ;;
  esac
  if [ "${CANON_P44_DEEPSWE_PARITY:-}" != "1" ]; then
    [ "${CANON_P44_PARITY_ADMITTED:-}" = "0" ] || {
      echo "[env] non-P44 runs require CANON_P44_PARITY_ADMITTED=0" >&2
      fail=1
    }
    [ "${CANON_P44_TOPOLOGY:-}" = "none" ] || {
      echo "[env] non-P44 runs require CANON_P44_TOPOLOGY=none" >&2
      fail=1
    }
    [ "${CANON_P44_ROLLOUT_ONLY:-}" = "0" ] || {
      echo "[env] non-P44 runs require CANON_P44_ROLLOUT_ONLY=0" >&2
      fail=1
    }
  fi
  if [ "${CANON_P58_DEEPSWE_TIM:-}" = "1" ]; then
    [ "${CANON_P39_64CHIP_PILOT:-}:${CANON_P39_PILOT_ADMITTED:-}" = "0:0" ] && \
    [ "${CANON_P43_DEEPSWE_DEBUG:-}:${CANON_P43_DEBUG_ADMITTED:-}" = "0:0" ] && \
    [ "${CANON_P44_DEEPSWE_PARITY:-}:${CANON_P44_PARITY_ADMITTED:-}" = "0:0" ] && \
    [ "${CANON_P46_DEEPSWE_TRAIN:-}" = "0" ] || {
      echo "[env] P58 cannot overlap P39/P43/P44/P46" >&2
      fail=1
    }
    [ "${CANON_P58_TIM_ADMITTED:-}" = "1" ] && \
    [ "${CANON_P34_TRAJECTORY_CAPTURE:-}" = "0" ] && \
    [ "${CANON_P34_CLEAN_ROWS:-}" = "1012" ] || {
      echo "[env] P58 requires admission, its own journal, and 1012 clean rows" >&2
      fail=1
    }
    case "${CANON_P34_RUN_STAGE:-}:${CANON_P58_EXPECTED_UPDATES:-}" in
      three-update:3|full:1000) ;;
      *)
        echo "[env] P58 stage/update horizon mismatch" >&2
        fail=1
        ;;
    esac
    p34_expected_dp=8
    p34_expected_devices=64
    p34_expected_prompts=8
    p34_expected_generations=16
    p34_expected_global_trajectories=128
    p34_expected_local_trajectories=16
    p34_expected_global_m=2048
    p34_expected_max_seqs=16
    p34_expected_mesh=8,8
    [ "${CANON_OPT_STATE_RESIDENT:-}:${CANON_P30_OPT_STATE_OFFLOAD:-}" = "1:0" ] || {
      echo "[env] P58 requires a TPU-resident optimizer" >&2
      fail=1
    }
    case "${CANON_P58_TIM_ARM:-}:${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-}:${CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER:-}" in
      native:1:1|zero:0:0) ;;
      *)
        echo "[env] P58 arm/alignment/stock-observer treatment drifted" >&2
        fail=1
        ;;
    esac
    case "${CANON_P58_DEBUG_DIR:-}" in
      /*) ;;
      *) echo "[env] P58 trajectory directory must be absolute" >&2; fail=1 ;;
    esac
  elif [ "${CANON_P46_DEEPSWE_TRAIN:-}" = "1" ]; then
    [ "${CANON_P39_64CHIP_PILOT:-}:${CANON_P39_PILOT_ADMITTED:-}" = "0:0" ] && \
    [ "${CANON_P43_DEEPSWE_DEBUG:-}:${CANON_P43_DEBUG_ADMITTED:-}" = "0:0" ] && \
    [ "${CANON_P44_DEEPSWE_PARITY:-}:${CANON_P44_PARITY_ADMITTED:-}" = "0:0" ] || {
      echo "[env] P46 Qwen3-32B training cannot overlap P39/P43/P44" >&2
      fail=1
    }
    [ "${CANON_P34_RUN_STAGE:-}" = "full" ] && \
    [ "${CANON_P34_TRAJECTORY_CAPTURE:-}" = "1" ] && \
    [ "${CANON_P34_CLEAN_ROWS:-}" = "1851" ] || {
      echo "[env] P46 Qwen3-32B training requires full capture on 1851 clean rows" >&2
      fail=1
    }
    p34_expected_prompts=8
    p34_expected_generations=8
    p34_expected_global_trajectories=64
    case "${CANON_P46_TOPOLOGY:-}" in
      64)
        p34_expected_dp=4
        p34_expected_devices=32
        p34_expected_local_trajectories=16
        p34_expected_global_m=1024
        p34_expected_max_seqs=16
        p34_expected_mesh=4,8
        ;;
      256)
        p34_expected_dp=16
        p34_expected_devices=128
        p34_expected_local_trajectories=4
        p34_expected_global_m=4096
        p34_expected_max_seqs=4
        p34_expected_mesh=16,8
        ;;
      *)
        echo "[env] P46 Qwen3-32B training requires topology 64 or 256" >&2
        fail=1
        p34_expected_dp=0
        p34_expected_devices=0
        p34_expected_local_trajectories=0
        p34_expected_global_m=0
        p34_expected_max_seqs=0
        p34_expected_mesh=invalid
        ;;
    esac
    [ "${CANON_OPT_STATE_RESIDENT:-}:${CANON_P30_OPT_STATE_OFFLOAD:-}" = "1:0" ] || {
      echo "[env] P46 training requires device-resident optimizer state" >&2
      fail=1
    }
    [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-}" = "1" ] || {
      echo "[env] P46 training requires finite alignment warning-only" >&2
      fail=1
    }
    case "${CANON_P34_DEBUG_DIR:-}" in
      /*) ;;
      *) echo "[env] P46 training artifact directory must be absolute" >&2; fail=1 ;;
    esac
  elif [ "${CANON_P44_DEEPSWE_PARITY:-}" = "1" ]; then
    [ "${CANON_P39_64CHIP_PILOT:-}:${CANON_P39_PILOT_ADMITTED:-}" = "0:0" ] && \
    [ "${CANON_P43_DEEPSWE_DEBUG:-}:${CANON_P43_DEBUG_ADMITTED:-}" = "0:0" ] && \
    [ "${CANON_P43_ROLLOUT_ONLY:-}" = "0" ] || {
      echo "[env] P44 parity cannot overlap P39 or P43" >&2
      fail=1
    }
    [ "${CANON_P44_PARITY_ADMITTED:-}" = "1" ] || {
      echo "[env] P44 parity requires CANON_P44_PARITY_ADMITTED=1" >&2
      fail=1
    }
    p34_expected_prompts=4
    p34_expected_generations=4
    p34_expected_global_trajectories=16
    case "${CANON_P44_TOPOLOGY:-}" in
      64)
        p34_expected_dp=4
        p34_expected_devices=32
        p34_expected_local_trajectories=4
        p34_expected_global_m=1024
        p34_expected_max_seqs=4
        p34_expected_mesh=4,8
        ;;
      128)
        p34_expected_dp=8
        p34_expected_devices=64
        p34_expected_local_trajectories=2
        p34_expected_global_m=2048
        p34_expected_max_seqs=2
        p34_expected_mesh=8,8
        ;;
      *)
        echo "[env] P44 parity requires topology 64 or 128" >&2
        fail=1
        p34_expected_dp=0
        p34_expected_devices=0
        p34_expected_local_trajectories=0
        p34_expected_global_m=0
        p34_expected_max_seqs=0
        p34_expected_mesh=invalid
        ;;
    esac
    [ "${CANON_OPT_STATE_RESIDENT:-}:${CANON_P30_OPT_STATE_OFFLOAD:-}" = "1:0" ] || {
      echo "[env] P44 parity requires device-resident optimizer state" >&2
      fail=1
    }
    [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-}" = "1" ] || {
      echo "[env] P44 parity requires the preregistered alignment warning policy" >&2
      fail=1
    }
    case "${CANON_P44_DEBUG_DIR:-}" in
      /*) ;;
      *) echo "[env] P44 parity artifact directory must be absolute" >&2; fail=1 ;;
    esac
  elif [ "${CANON_P43_DEEPSWE_DEBUG:-}" = "1" ]; then
    [ "${CANON_P39_64CHIP_PILOT:-}:${CANON_P39_PILOT_ADMITTED:-}" = "0:0" ] || {
      echo "[env] P43 debug cannot overlap the P39 pilot" >&2
      fail=1
    }
    [ "${CANON_P43_DEBUG_ADMITTED:-}" = "1" ] || {
      echo "[env] P43 debug requires CANON_P43_DEBUG_ADMITTED=1" >&2
      fail=1
    }
    p34_expected_dp=4
    p34_expected_devices=32
    p34_expected_prompts=4
    p34_expected_generations=4
    p34_expected_global_trajectories=16
    p34_expected_local_trajectories=4
    p34_expected_global_m=1024
    p34_expected_max_seqs=4
    p34_expected_mesh=4,8
    [ "${CANON_OPT_STATE_RESIDENT:-}:${CANON_P30_OPT_STATE_OFFLOAD:-}" = "1:0" ] || {
      echo "[env] P43 debug requires device-resident optimizer state" >&2
      fail=1
    }
    [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-}" = "1" ] || {
      echo "[env] P43 debug requires the preregistered alignment warning policy" >&2
      fail=1
    }
    case "${CANON_P43_DEBUG_DIR:-}" in
      /*) ;;
      *) echo "[env] P43 debug artifact directory must be absolute" >&2; fail=1 ;;
    esac
  else
    [ "${CANON_P43_DEBUG_ADMITTED:-}" = "0" ] || {
      echo "[env] non-P43 runs require CANON_P43_DEBUG_ADMITTED=0" >&2
      fail=1
    }
    [ "${CANON_P43_ROLLOUT_ONLY:-}" = "0" ] || {
      echo "[env] non-P43 runs require CANON_P43_ROLLOUT_ONLY=0" >&2
      fail=1
    }
    case "${CANON_P39_64CHIP_PILOT:-}" in
      0)
        p34_expected_dp=16
        p34_expected_devices=128
        p34_expected_prompts=8
        p34_expected_generations=8
        p34_expected_global_trajectories=64
        p34_expected_local_trajectories=4
        p34_expected_global_m=4096
        p34_expected_max_seqs=4
        p34_expected_mesh=16,8
        [ "${CANON_P39_PILOT_ADMITTED:-}" = "0" ] || {
          echo "[env] production P34 requires CANON_P39_PILOT_ADMITTED=0" >&2
          fail=1
        }
        [ "${CANON_OPT_STATE_RESIDENT:-}:${CANON_P30_OPT_STATE_OFFLOAD:-}" = "1:0" ] || {
          echo "[env] production P34 requires device-resident optimizer state" >&2
          fail=1
        }
        if [ "${CANON_P34_RUN_STAGE:-}" = "full" ]; then
          [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-}" = "1" ] || {
            echo "[env] P34 full requires finite alignment warning-only policy" >&2
            fail=1
          }
          [ "${CANON_P34_TRAJECTORY_CAPTURE:-}" = "1" ] && \
          [ "${CANON_P34_CLEAN_ROWS:-}" = "1851" ] || {
            echo "[env] P34 full requires durable capture and 1851 clean rows" >&2
            fail=1
          }
          case "${CANON_P34_DEBUG_DIR:-}" in
            /*) ;;
            *) echo "[env] P34 full artifact directory must be absolute" >&2; fail=1 ;;
          esac
        else
          [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-}" = "1" ] && \
          [ "${CANON_P34_TRAJECTORY_CAPTURE:-}" = "0" ] && \
          [ "${CANON_P34_CLEAN_ROWS:-}" = "0" ] || {
            echo "[env] P34 short diagnostic requires warning-only alignment without production capture" >&2
            fail=1
          }
        fi
        ;;
      1)
        p34_expected_dp=4
        p34_expected_devices=32
        p34_expected_prompts=8
        p34_expected_generations=8
        p34_expected_global_trajectories=64
        p34_expected_local_trajectories=16
        p34_expected_global_m=1024
        p34_expected_max_seqs=16
        p34_expected_mesh=4,8
        [ "${CANON_P39_PILOT_ADMITTED:-}" = "1" ] || {
          echo "[env] P39 pilot requires CANON_P39_PILOT_ADMITTED=1" >&2
          fail=1
        }
        [ "${CANON_OPT_STATE_RESIDENT:-}:${CANON_P30_OPT_STATE_OFFLOAD:-}" = "1:0" ] || {
          echo "[env] P39 pilot requires device-resident optimizer state" >&2
          fail=1
        }
        [ "${CANON_DEEPSWE_ALIGNMENT_WARN_ONLY:-}" = "1" ] || {
          echo "[env] P39 pilot requires the preregistered alignment warning policy" >&2
          fail=1
        }
        ;;
      *)
        echo "[env] CANON_P39_64CHIP_PILOT must be exactly 0 or 1" >&2
        fail=1
        p34_expected_dp=0
        p34_expected_devices=0
        p34_expected_prompts=0
        p34_expected_generations=0
        p34_expected_global_trajectories=0
        p34_expected_local_trajectories=0
        p34_expected_global_m=0
        p34_expected_max_seqs=0
        p34_expected_mesh=invalid
        ;;
    esac
  fi
  [ "${CANON_DP_SIZE:-}" = "$p34_expected_dp" ] && \
  [ "${CANON_TP_SIZE:-}" = "8" ] && \
  [ "${CANON_TOTAL_DEVICES:-}" = "$p34_expected_devices" ] || {
    echo "[env] P34 role topology does not match the selected contract" >&2
    fail=1
  }
  [ "$((CANON_DP_SIZE * CANON_TP_SIZE))" -eq "$CANON_TOTAL_DEVICES" ] || {
    echo "[env] P34 arithmetic FAIL: dp*tp != role devices" >&2; fail=1;
  }
  [ "${CANON_GLOBAL_PROMPTS:-}" = "$p34_expected_prompts" ] && \
  [ "${CANON_NUM_GENERATIONS:-}" = "$p34_expected_generations" ] && \
  [ "${CANON_GLOBAL_TRAJECTORIES:-}" = "$p34_expected_global_trajectories" ] && \
  [ "${CANON_LOCAL_TRAJECTORIES:-}" = "$p34_expected_local_trajectories" ] || {
    echo "[env] P34 trajectory geometry does not match the selected contract" >&2
    fail=1
  }
  [ "$((CANON_DP_SIZE * CANON_LOCAL_TRAJECTORIES))" -eq \
      "$CANON_GLOBAL_TRAJECTORIES" ] || {
    echo "[env] P34 trajectory arithmetic does not close" >&2; fail=1;
  }
  if [ "$P58_NATIVE" = "1" ]; then
    [ ! -v CANON_LOGPROB_M ] && \
    [ "${CANON_TARGET_M:-}" = "256" ] && \
    [ "${CANON_P34_ABCPROD:-}" = "256" ] && \
    [ "${MIN_TOKEN_BUCKET:-}" = "$p34_expected_global_m" ] || {
      echo "[env] P58 native local/global capacity contract changed" >&2
      fail=1
    }
    [ "$MIN_TOKEN_BUCKET" -eq "$((CANON_DP_SIZE * CANON_TARGET_M))" ] || {
      echo "[env] P58 native bucket FAIL: global M != dp*target M" >&2
      fail=1
    }
    [ "${CANON_VJP2_MAX_SEQS:-}" = "0" ] || {
      echo "[env] P58 native forbids the canonical grouped VJP" >&2
      fail=1
    }
  else
    [ "${CANON_LOGPROB_M:-}" = "256" ] && \
    [ "${CANON_TARGET_M:-}" = "256" ] && \
    [ "${CANON_P34_ABCPROD:-}" = "256" ] && \
    [ "${MIN_TOKEN_BUCKET:-}" = "$p34_expected_global_m" ] || {
      echo "[env] P34 local/global M does not match the selected contract" >&2
      fail=1
    }
    [ "$MIN_TOKEN_BUCKET" -eq "$((CANON_DP_SIZE * CANON_LOGPROB_M))" ] || {
      echo "[env] P34 bucket FAIL: global M != dp*local M" >&2; fail=1;
    }
    [ "${CANON_VJP2_MAX_SEQS:-}" = "1" ] || {
      echo "[env] P34 grouped model_fn requires CANON_VJP2_MAX_SEQS=1" >&2; fail=1;
    }
  fi
  [ "${CANON_P34_PREFIX_CACHE:-}" = "0" ] && \
  [ "${CANON_P34_MAX_NUM_SEQS:-}" = "$p34_expected_max_seqs" ] && \
  [ "${CANON_P34_MAX_BATCHED_TOKENS:-}" = "256" ] || {
    echo "[env] P34 rollout scheduler contract changed" >&2; fail=1;
  }
  [ "$((CANON_P34_MAX_NUM_SEQS * CANON_DP_SIZE))" -eq \
      "$CANON_GLOBAL_TRAJECTORIES" ] || {
    echo "[env] P34 global scheduler request capacity changed" >&2; fail=1;
  }
  [ "$((CANON_P34_MAX_BATCHED_TOKENS * CANON_DP_SIZE))" -eq \
      "$MIN_TOKEN_BUCKET" ] || {
    echo "[env] P34 global scheduler token capacity changed" >&2; fail=1;
  }
  [ "${CANON_TRAIN_DP_SHARDING:-}" = "replicated-params" ] && \
  [ "${FL_SHARED_MESH:-}" = "$p34_expected_mesh" ] || {
    echo "[env] P34 requires DP-replicated parameters on the selected mesh" >&2
    fail=1
  }
  [ "${CANON_P34_STRICT_CLI:-}" = "1" ] && \
  [ "${CANON_PRE_ALIGN_GATE:-}" = "1" ] || {
    echo "[env] P34 strict CLI and pre-backward gate are mandatory" >&2; fail=1;
  }
  if [ "$P58_NATIVE" = "1" ]; then
    case "${CANON_P34_DISABLE_SAMPLER_IS:-}:${CANON_P34_DISABLE_TIS:-}" in
      1:1|0:0) ;;
      *)
        echo "[env] P58 native sampler recipe must be exactly raw 1/1 or token-IS 0/0" >&2
        fail=1
        ;;
    esac
  elif [ "${CANON_P34_DISABLE_SAMPLER_IS:-}" != "1" ] || \
       [ "${CANON_P34_DISABLE_TIS:-}" != "1" ]; then
    echo "[env] P34 non-native-P58 workloads require neutral importance paths" >&2
    fail=1
  fi
  [ "${CANON_P34_DATASET_NAME:-}" = "R2E-Gym/R2E-Gym-Subset" ] && \
  [ "${CANON_P34_DATASET_REVISION:-}" = "2e8108ff942f24fcb5686badfaf7f9a8808566d5" ] && \
  [ "${CANON_P34_DATASET_SPLIT:-}" = "train" ] && \
  [ "${CANON_P34_DATASET_ROWS:-}" = "4578" ] || {
    echo "[env] P34 dataset source pin changed" >&2; fail=1;
  }
  [ -z "${CANON_EXPECT_MODEL_MESH_IDS:-}" ] || {
    echo "[env] P34 must not inherit a one-host model mesh ID assertion" >&2
    fail=1
  }
  if [[ ! "${CANON_P34_WHITELIST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[env] P34 whitelist SHA-256 is malformed" >&2
    fail=1
  elif [ ! -f "${CANON_P34_WHITELIST:-}" ]; then
    echo "[env] P34 whitelist file is missing: ${CANON_P34_WHITELIST:-unset}" >&2
    fail=1
  elif ! printf '%s  %s\n' "$CANON_P34_WHITELIST_SHA256" \
      "$CANON_P34_WHITELIST" | sha256sum -c - --quiet; then
    echo "[env] P34 whitelist SHA-256 mismatch" >&2
    fail=1
  else
    echo "[env] P34 whitelist SHA256 OK: $CANON_P34_WHITELIST_SHA256"
  fi
  case "${CANON_P34_RUN_STAGE:-}" in
    rollout-only)
      [ "${CANON_P34_NO_COMMIT:-}" = "1" ] || {
        echo "[env] P43 rollout-only requires no-commit=1" >&2; fail=1;
      } ;;
    backward-no-commit)
      [ "${CANON_P34_NO_COMMIT:-}" = "1" ] || {
        echo "[env] P34 backward-no-commit requires no-commit=1" >&2; fail=1;
      } ;;
    one-update|three-update|full)
      [ "${CANON_P34_NO_COMMIT:-}" = "0" ] || {
        echo "[env] P34 update stages require no-commit=0" >&2; fail=1;
      } ;;
    *) echo "[env] invalid CANON_P34_RUN_STAGE" >&2; fail=1 ;;
  esac
  if [ "${CANON_P39_64CHIP_PILOT:-}" = "1" ]; then
    case "${CANON_P34_RUN_STAGE:-}" in
      one-update|three-update) ;;
      *)
        echo "[env] P39 pilot admits only one-update or three-update" >&2
        fail=1
        ;;
    esac
  fi
  if [ "${CANON_P46_DEEPSWE_TRAIN:-}" = "1" ] && \
     [ "${CANON_P34_RUN_STAGE:-}" != "full" ]; then
    echo "[env] P46 Qwen3-32B training admits only full" >&2
    fail=1
  fi
  if [ "${CANON_P43_DEEPSWE_DEBUG:-}" = "1" ]; then
    case "${CANON_P34_RUN_STAGE:-}:${CANON_P43_ROLLOUT_ONLY:-}" in
      rollout-only:1|one-update:0|three-update:0) ;;
      *)
        echo "[env] P43 admits rollout-only, one-update, or three-update with exact rollout flag" >&2
        fail=1
        ;;
    esac
  elif [ "${CANON_P44_DEEPSWE_PARITY:-}" = "1" ]; then
    case "${CANON_P34_RUN_STAGE:-}:${CANON_P44_ROLLOUT_ONLY:-}" in
      rollout-only:1|one-update:0|three-update:0) ;;
      *)
        echo "[env] P44 admits rollout-only, one-update, or three-update with exact rollout flag" >&2
        fail=1
        ;;
    esac
  elif [ "${CANON_P34_RUN_STAGE:-}" = "rollout-only" ]; then
    echo "[env] rollout-only is admitted only for P43/P44 debug" >&2
    fail=1
  fi
  p34_admitted=0
  [ "${CANON_MODE:-}" = "run" ] && p34_admitted=1
  for k in CANON_P34_TOPOLOGY_ADMITTED CANON_P34_TP8_ADMITTED \
           CANON_P34_TRAJECTORY_ADMITTED CANON_P34_UPDATE_ADMITTED; do
    [ "${!k:-}" = "$p34_admitted" ] || {
      echo "[env] P34 admission mismatch: $k must equal $p34_admitted" >&2
      fail=1
    }
  done
  for k in CANON_P32_TRAIN_ADMITTED CANON_P33_WORKLOAD_LAUNCH_ADMITTED; do
    [ "${!k:-}" = "$p34_admitted" ] || {
      echo "[env] P34 inherited admission mismatch: $k must equal $p34_admitted" >&2
      fail=1
    }
  done
  p34_reduction_admitted="$p34_admitted"
  if [ "$P58_NATIVE" = "1" ]; then
    # Native is intentionally the stock JAX-sharded trainer.  It admits
    # training, but it must not claim the zero arm's explicit fixed-tree DP
    # reduction contract.
    p34_reduction_admitted=0
  fi
  [ "${CANON_P32_DP_REDUCTION_ADMITTED:-}" = \
    "$p34_reduction_admitted" ] || {
    echo "[env] P34 inherited admission mismatch: CANON_P32_DP_REDUCTION_ADMITTED must equal $p34_reduction_admitted" >&2
    fail=1
  }
  unset p34_reduction_admitted
  if [ "$p34_admitted" = "1" ]; then
    for k in CANON_RUN_ID CANON_RUN_CMD CANON_RUN_LOG CANON_P34_WEIGHT_REPORT \
             CANON_PRE_ALIGN_REPORT \
             CANON_ALIGN_REPORT \
             CANON_UPDATE_REPORT CANON_WANDB_PROJECT CANON_WANDB_GROUP \
             CANON_WANDB_RUN_NAME WANDB_API_KEY; do
      req "$k"
    done
    if [ "${CANON_P34_TRAJECTORY_CAPTURE:-0}" = "1" ]; then
      req CANON_P34_DEBUG_DIR
    fi
    [ "${WANDB_MODE:-}" = "online" ] || {
      echo "[env] P34 requires WANDB_MODE=online" >&2; fail=1;
    }
  fi
  echo "[env] P34 contract OK: DP${CANON_DP_SIZE}xTP${CANON_TP_SIZE} per role, local M256, global M${MIN_TOKEN_BUCKET}"
fi

case "${CANON_P46_EVALUATION:-0}" in
  0) ;;
  1)
    for k in CANON_P46_TOPOLOGY CANON_EXPECT_COMMIT CANON_CLIENT_IMAGE \
             CANON_RUN_CMD CANON_RUN_LOG CANON_P46_OUTPUT_DIR \
             CANON_P46_RESUME_TAG \
             CANON_P46_SAMPLING_SOURCE_COMMIT \
             CANON_P46_GOLD_JSONL CANON_P46_GOLD_JSONL_SHA256 \
             CANON_P46_MODEL_BASE_DIR CANON_P46_LOGICAL_SHARD_INDEX \
             CANON_P46_PHYSICAL_SHARD_INDEX CANON_P46_EVALUATION_MODE \
             CANON_P46_PARITY_CANARY CANON_P46_FULL_CAMPAIGN \
             CANON_P46_CENSUS_FIRST_PASS; do
      req "$k"
    done
    [ "${CANON_MODE:-}" = "run" ] && \
    [ "${CANON_P34_DEEPSWE:-0}" = "0" ] && \
    [ "${CANON_P46_DEEPSWE_TRAIN:-0}" = "0" ] && \
    [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "0" ] && \
    [ "${CANON_P33_WORKLOAD_LAUNCH_ADMITTED:-0}" = "0" ] || {
      echo "[env] P46 evaluation must not admit a trainer" >&2
      fail=1
    }
    case "${CANON_P46_EVALUATION_MODE:-}" in
      reward_only) ;;
      logprob_observer)
        [ "${CANON_P46_PARITY_CANARY:-0}" = "1" ] && \
        [ "${CANON_P46_TOPOLOGY:-}" = "64" ] || {
          echo "[env] logprob_observer requires the 64-chip parity canary" >&2
          fail=1
        }
        ;;
      *)
        echo "[env] unsupported P46 evaluation_mode" >&2
        fail=1
        ;;
    esac
    case "${CANON_P46_PARITY_CANARY:-}" in 0|1) ;; *)
      echo "[env] CANON_P46_PARITY_CANARY must be exactly 0 or 1" >&2
      fail=1 ;;
    esac
    case "${CANON_P46_FULL_CAMPAIGN:-}" in 0|1) ;; *)
      echo "[env] CANON_P46_FULL_CAMPAIGN must be exactly 0 or 1" >&2
      fail=1 ;;
    esac
    case "${CANON_P46_CENSUS_FIRST_PASS:-}" in 0|1) ;; *)
      echo "[env] CANON_P46_CENSUS_FIRST_PASS must be exactly 0 or 1" >&2
      fail=1 ;;
    esac
    [[ "${CANON_P46_RESUME_TAG:-}" =~ ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$ ]] || {
      echo "[env] CANON_P46_RESUME_TAG must be lowercase and Kubernetes-safe" >&2
      fail=1
    }
    [[ "${CANON_P46_SAMPLING_SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || {
      echo "[env] CANON_P46_SAMPLING_SOURCE_COMMIT must be a lowercase SHA" >&2
      fail=1
    }
    if [ -n "${CANON_P46_LEGACY_IMPORT_ID:-}" ]; then
      [[ "${CANON_P46_LEGACY_IMPORT_ID}" =~ ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$ ]] || {
        echo "[env] CANON_P46_LEGACY_IMPORT_ID must be lowercase and Kubernetes-safe" >&2
        fail=1
      }
      [ "${CANON_P46_FULL_CAMPAIGN:-0}" = "1" ] || {
        echo "[env] P46 legacy import requires a full campaign" >&2
        fail=1
      }
      p46_import_root="${CANON_P46_OUTPUT_DIR%/outputs}/imports/${CANON_P46_LEGACY_IMPORT_ID}"
      [ -f "$p46_import_root/SHA256SUMS" ] || {
        echo "[env] P46 frozen legacy snapshot is missing SHA256SUMS: $p46_import_root" >&2
        fail=1
      }
      [ -f "$p46_import_root/legacy_source_contract.json" ] || {
        echo "[env] P46 frozen legacy snapshot is missing legacy_source_contract.json: $p46_import_root" >&2
        fail=1
      }
      [ ! -e "$p46_import_root/resume_contract.json" ] || {
        echo "[env] P46 legacy-v5 snapshot must not contain resume_contract.json; use a fresh v5-only staging copy or the frozen-v6 import path" >&2
        fail=1
      }
    fi
    if [ -n "${CANON_P46_FROZEN_V6_IMPORT_ID:-}" ]; then
      [[ "${CANON_P46_FROZEN_V6_IMPORT_ID}" =~ ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$ ]] || {
        echo "[env] CANON_P46_FROZEN_V6_IMPORT_ID must be lowercase and Kubernetes-safe" >&2
        fail=1
      }
      [ "${CANON_P46_FULL_CAMPAIGN:-0}" = "1" ] || {
        echo "[env] P46 frozen v6 import requires a full campaign" >&2
        fail=1
      }
      p46_v6_import_root="${CANON_P46_OUTPUT_DIR%/outputs}/imports/${CANON_P46_FROZEN_V6_IMPORT_ID}"
      [ -f "$p46_v6_import_root/SHA256SUMS" ] && \
      [ -f "$p46_v6_import_root/resume_contract.json" ] || {
        echo "[env] P46 frozen v6 snapshot is missing SHA256SUMS or resume_contract.json: $p46_v6_import_root" >&2
        fail=1
      }
    fi
    if [ -n "${CANON_P46_LEGACY_IMPORT_ID:-}" ] && \
       [ -n "${CANON_P46_FROZEN_V6_IMPORT_ID:-}" ]; then
      echo "[env] P46 permits only one frozen resume import" >&2
      fail=1
    fi
    if [ "${CANON_P46_FULL_CAMPAIGN:-0}" = "1" ]; then
      [ "${CANON_P46_PARITY_CANARY:-0}" = "0" ] && \
      [ "${CANON_P46_LOGICAL_SHARD_INDEX:-}" = "0" ] && \
      [ "${CANON_P46_PHYSICAL_SHARD_INDEX:-}" = "0" ] || {
        echo "[env] P46 full campaign owns all shards and rejects parity" >&2
        fail=1
      }
    fi
    if [ "${CANON_P46_CENSUS_FIRST_PASS:-0}" = "1" ]; then
      [ "${CANON_P46_FULL_CAMPAIGN:-0}" = "1" ] && \
      [ "${CANON_P46_PARITY_CANARY:-0}" = "0" ] && \
      [ "${CANON_P46_EVALUATION_MODE:-}" = "reward_only" ] || {
        echo "[env] P46 first-pass census requires a full reward-only campaign" >&2
        fail=1
      }
    fi
    if [ "${CANON_P46_PARITY_CANARY:-0}" = "1" ] && \
       [ "${CANON_P46_TOPOLOGY:-}" != "64" ]; then
      echo "[env] P46 parity canary requires topology 64" >&2
      fail=1
    fi
    if [ "${#_CANON_P46_INPUT_CONTRADICTIONS[@]}" -ne 0 ]; then
      echo "[env] P46 evaluation caller contradictions: ${_CANON_P46_INPUT_CONTRADICTIONS[*]}" >&2
      fail=1
    fi
    for k in CANON_P34_TRAJECTORY_CAPTURE \
             CANON_PROMPT_PROCESSED_LOGPROBS CANON_PALLAS_LOGSOFTMAX \
             CANON_ENGINE_MODULE_C CANON_RPA_VJP2 CANON_ALIGNMENT_GATE \
             CANON_ALIGNMENT_GATE_ONLY CANON_ALIGNMENT_UPDATE_CANARY \
             CANON_ALIGNMENT_TRAIN CANON_PRE_ALIGN_GATE \
             CANON_DEEPSWE_ALIGNMENT_WARN_ONLY CANON_P28_SEGMENTED_FORWARD \
             CANON_P28_SEGMENTED_VJP CANON_P28_SEGMENTED_TRAIN \
             CANON_P28_G6_UPDATE CANON_P29_FULL_TRAIN \
             CANON_OPT_STATE_RESIDENT CANON_P30_SPARSE_GRAD_ASSEMBLY \
             CANON_P30_FUSED_PAIR_ACCUMULATION \
             CANON_P30_REUSE_SEGMENTED_ENGINE \
             CANON_P30_RELEASE_CAPTURED_STATE \
             CANON_P30_RESHARD_ACCUMULATOR; do
      [ "${!k:-}" = "0" ] || {
        echo "[env] P46 evaluation contradicts $k=${!k:-unset}" >&2
        fail=1
      }
    done
    case "${CANON_P46_TOPOLOGY:-}" in 64|128) ;; *)
      echo "[env] P46 evaluation topology must be 64 or 128" >&2; fail=1 ;;
    esac
    case " ${CANON_RUN_CMD:-} " in
      *" examples/deepswe/eval_deepswe.py "*) ;;
      *) echo "[env] P46 evaluation command drifted" >&2; fail=1 ;;
    esac
    echo "[env] P46 evaluation contract OK: topology=${CANON_P46_TOPOLOGY} logical=${CANON_P46_LOGICAL_SHARD_INDEX} physical=${CANON_P46_PHYSICAL_SHARD_INDEX} mode=${CANON_P46_EVALUATION_MODE} parity=${CANON_P46_PARITY_CANARY} campaign=${CANON_P46_FULL_CAMPAIGN} census=${CANON_P46_CENSUS_FIRST_PASS} resume_tag=${CANON_P46_RESUME_TAG} sampled_by=stock@${CANON_P46_SAMPLING_SOURCE_COMMIT} harness=${CANON_EXPECT_COMMIT}"
    ;;
  *)
    echo "[env] CANON_P46_EVALUATION must be exactly 0 or 1" >&2
    fail=1
    ;;
esac

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
  if [ "$P57_STOCK_FAST" = "1" ]; then
    [ "$MIN_TOKEN_BUCKET" -eq "$((CANON_DP_SIZE * CANON_TARGET_M))" ] || {
      echo "[env] P57 stock-fast bucket FAIL: MIN_TOKEN_BUCKET must equal dp*CANON_TARGET_M" >&2; fail=1;
    }
  else
    [ "$MIN_TOKEN_BUCKET" -eq "$((CANON_DP_SIZE * CANON_LOGPROB_M))" ] || {
      echo "[env] P32 bucket FAIL: MIN_TOKEN_BUCKET must equal dp*CANON_LOGPROB_M" >&2; fail=1;
    }
  fi
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
             CANON_P30_SPARSE_GRAD_ASSEMBLY \
             CANON_P30_REUSE_SEGMENTED_ENGINE \
             CANON_P30_RELEASE_CAPTURED_STATE \
             CANON_P30_RESHARD_ACCUMULATOR; do
      req "$k"
      [ "${!k:-}" = "1" ] || {
        echo "[env] admitted P33 training requires $k=1" >&2
        fail=1
      }
    done
    req CANON_OPT_STATE_RESIDENT
    req CANON_P30_OPT_STATE_OFFLOAD
    case "${CANON_OPT_STATE_RESIDENT:-}:${CANON_P30_OPT_STATE_OFFLOAD:-}" in
      0:1|1:0) ;;
      *)
        echo "[env] admitted P33 training requires exactly one optimizer placement: resident or offload" >&2
        fail=1
        ;;
    esac
    if [ "${CANON_P34_DEEPSWE:-0}" != "1" ]; then
      req CANON_PRE_ALIGN_GATE
      [ "${CANON_PRE_ALIGN_GATE:-0}" = "1" ] || {
        echo "[env] admitted P33 training requires CANON_PRE_ALIGN_GATE=1" >&2
        fail=1
      }
    fi
    case "${CANON_P32_WORKLOAD:-}" in
      gsm8k|gsm8k-p59-dp4-tp1|frozenlake|frozenlake-dp8-tp8) ;;
      *) echo "[env] admitted P33 training has invalid workload" >&2; fail=1 ;;
    esac
    case "${CANON_P32_WORKLOAD:-}" in
    frozenlake|frozenlake-dp8-tp8)
      req CANON_P33_ENABLE_EVAL
      req CANON_P33_DISABLE_EVAL
      req CANON_P31_ENABLE_EVAL
      case "${CANON_P33_ENABLE_EVAL:-}:${CANON_P33_DISABLE_EVAL:-}:${CANON_P31_ENABLE_EVAL:-}" in
        0:1:0) ;;
        1:0:1)
          [ "${CANON_P33_RUN_STAGE:-}" = "full" ] && \
          [ "${CANON_P33_NO_COMMIT:-}" = "0" ] || {
            echo "[env] FrozenLake evaluation requires committed full training" >&2
            fail=1
          } ;;
        *)
          echo "[env] FrozenLake evaluation selection is inconsistent" >&2
          fail=1 ;;
      esac
      case "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" in
        0|1) ;;
        *) echo "[env] CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY must be 0 or 1" >&2; fail=1 ;;
      esac
      if [ "${CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY:-0}" = "1" ] && \
         { [ "${CANON_P33_RUN_STAGE:-}" != "full" ] || \
           [ "${CANON_P33_NO_COMMIT:-}" != "0" ]; }; then
        echo "[env] FrozenLake warning-only policy requires committed full training" >&2
        fail=1
      fi
      ;;
    esac
    [ "${CANON_P30_FUSED_PAIR_ACCUMULATION:-}" = "0" ] || {
      echo "[env] P33 rank-reduced groups require fused pair accumulation off" >&2
      fail=1
    }
    [ "${FL_SHARED_MESH:-}" = "${CANON_DP_SIZE:-},${CANON_TP_SIZE:-}" ] || {
      echo "[env] admitted P33 training requires FL_SHARED_MESH=CANON_DP_SIZE,CANON_TP_SIZE" >&2
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
    case "${CANON_P62_BACKWARD_NUMERIC_DEBUG:-0}" in
      0|1) ;;
      *) echo "[env] CANON_P62_BACKWARD_NUMERIC_DEBUG must be 0 or 1" >&2; fail=1 ;;
    esac
    case "${CANON_P64_P45_NUMERIC_DEBUG:-0}" in
      0|1) ;;
      *) echo "[env] CANON_P64_P45_NUMERIC_DEBUG must be 0 or 1" >&2; fail=1 ;;
    esac
    if [ "${CANON_P62_BACKWARD_NUMERIC_DEBUG:-0}" = "1" ] && \
       [ "${CANON_P64_P45_NUMERIC_DEBUG:-0}" = "1" ]; then
      echo "[env] P62 and P64 numerical observers are mutually exclusive" >&2
      fail=1
    fi
    case "${CANON_GSM8K_AB_REPORT_ONLY:-0}" in
      0|1) ;;
      *) echo "[env] CANON_GSM8K_AB_REPORT_ONLY must be 0 or 1" >&2; fail=1 ;;
    esac
    if [ "${CANON_P62_BACKWARD_NUMERIC_DEBUG:-0}" = "1" ]; then
      [ "${CANON_P32_WORKLOAD:-}" = "gsm8k" ] && \
      [ "${CANON_DP_SIZE:-}" = "16" ] && \
      [ "${CANON_TP_SIZE:-}" = "4" ] && \
      [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ] && \
      [ "${CANON_P33_NO_COMMIT:-}" = "1" ] && \
      [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-}" = "1" ] && \
      [ "${CANON_P38_FIXED_LM_HEAD:-}" = "1" ] && \
      [ "${CANON_V1_HP_FULL:-0}" = "0" ] && \
      [ "${CANON_PROFILE_FILE:-}" = "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-p62-debug.env" ] || {
        echo "[env] P62 requires exact GSM8K DP16xTP4 fixed-head backward-no-commit profile" >&2
        fail=1
      }
    fi
    if [ "${CANON_P64_P45_NUMERIC_DEBUG:-0}" = "1" ]; then
      [ "${CANON_P32_WORKLOAD:-}" = "frozenlake-dp8-tp8" ] && \
      [ "${CANON_DP_SIZE:-}" = "8" ] && \
      [ "${CANON_TP_SIZE:-}" = "8" ] && \
      [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ] && \
      [ "${CANON_P33_NO_COMMIT:-}" = "1" ] && \
      [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-}" = "1" ] && \
      [ "${CANON_P38_FIXED_LM_HEAD:-}" = "1" ] && \
      [ "${CANON_V1_HP_FULL:-0}" = "0" ] && \
      [ "${CANON_PROFILE_FILE:-}" = "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-p64-debug.env" ] || {
        echo "[env] P64 requires exact P45 DP8xTP8 fixed-head backward-no-commit profile" >&2
        fail=1
      }
      [ "${CANON_P64_TRAINING_CAPSULE:-}" = \
        "${CANON_STATE%/}/p64_training_capsule.npz" ] || {
        echo "[env] P64 capsule path must be isolated under CANON_STATE" >&2
        fail=1
      }
      [[ "${CANON_P64_TRAINING_CAPSULE_GCS_URI:-}" =~ ^gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p64/[a-z0-9][a-z0-9-]*/training-capsule\.npz$ ]] || {
        echo "[env] P64 capsule GCS URI is outside the evidence root" >&2
        fail=1
      }
      case "${CANON_P64_TRAINING_CAPSULE_MODE:-}" in
        capture)
          [ -z "${CANON_P64_TRAINING_CAPSULE_SHA256:-}" ] && \
          [ -z "${CANON_P64_MODEL_BINDING_SHA256:-}" ] || {
            echo "[env] P64 capture mode forbids replay hashes" >&2
            fail=1
          }
          ;;
        replay)
          [[ "${CANON_P64_TRAINING_CAPSULE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] && \
          [[ "${CANON_P64_MODEL_BINDING_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || {
            echo "[env] P64 replay requires capsule and model-binding hashes" >&2
            fail=1
          }
          ;;
        *)
          echo "[env] P64 capsule mode must be capture or replay" >&2
          fail=1
          ;;
      esac
    fi
    case "${CANON_GSM8K_ALIGNMENT_WARN_ONLY:-0}" in
      0|1) ;;
      *) echo "[env] CANON_GSM8K_ALIGNMENT_WARN_ONLY must be 0 or 1" >&2; fail=1 ;;
    esac
    if [ "${CANON_GSM8K_AB_REPORT_ONLY:-0}" = "1" ] && \
       [ "${CANON_GSM8K_ALIGNMENT_WARN_ONLY:-0}" = "1" ]; then
      echo "[env] GSM8K bounded and warning-only policies are mutually exclusive" >&2
      fail=1
    fi
  case "${CANON_P33_RUN_STAGE:-}" in
    envelope-short)
      [ "${CANON_P33_NO_COMMIT:-}" = "1" ] || {
        echo "[env] P33 envelope-short requires no-commit=1" >&2; fail=1;
      } ;;
    alignment-short|backward-no-commit)
        [ "${CANON_P33_NO_COMMIT:-0}" = "1" ] || {
          echo "[env] diagnostic no-commit stage requires CANON_P33_NO_COMMIT=1" >&2
          fail=1
        }
        ;;
      one-update|three-update|full)
        [ "${CANON_P33_NO_COMMIT:-0}" = "0" ] || {
          echo "[env] update/full stages require CANON_P33_NO_COMMIT=0" >&2
          fail=1
        }
        ;;
      p59-eight-update)
        [ "${CANON_P33_NO_COMMIT:-0}" = "0" ] && \
        [ "${CANON_P32_WORKLOAD:-}" = "gsm8k-p59-dp4-tp1" ] && \
        [ "${CANON_P59_DP4_TAIL8:-0}" = "1" ] || {
          echo "[env] p59-eight-update requires committed P59 DP4 tail admission" >&2
          fail=1
        }
        ;;
      *) echo "[env] invalid CANON_P33_RUN_STAGE" >&2; fail=1 ;;
    esac
    if [ "${CANON_GSM8K_AB_REPORT_ONLY:-0}" = "1" ]; then
      [ "${CANON_P32_WORKLOAD:-}" = "gsm8k" ] && \
      [ "${CANON_P33_RUN_STAGE:-}" = "full" ] && \
      [ "${CANON_P33_NO_COMMIT:-}" = "0" ] && \
      [ "${CANON_ALIGNMENT_TRAIN:-}" = "1" ] || {
        echo "[env] A/B report policy is admitted only for committed GSM8K full training" >&2
        fail=1
      }
      echo "[env] GSM8K full A/B policy: bounded drift is report-only; zero-TIM claim disabled"
    fi
    if [ "${CANON_GSM8K_ALIGNMENT_WARN_ONLY:-0}" = "1" ]; then
      [ "${CANON_P32_WORKLOAD:-}" = "gsm8k" ] && \
      [ "${CANON_P33_RUN_STAGE:-}" = "full" ] && \
      [ "${CANON_P33_NO_COMMIT:-}" = "0" ] && \
      [ "${CANON_ALIGNMENT_TRAIN:-}" = "1" ] || {
        echo "[env] alignment warning-only policy is admitted only for committed GSM8K full training" >&2
        fail=1
      }
      echo "[env] GSM8K full alignment policy: finite numerical drift is warning-only; claim=convergence-only"
    fi
    for k in CANON_WANDB_ONLINE_REQUIRED CANON_P31_MONOTONIC_METRICS \
             CANON_WANDB_PROJECT CANON_WANDB_GROUP CANON_WANDB_RUN_NAME \
             WANDB_MODE WANDB_API_KEY CANON_RUN_CMD CANON_RUN_LOG \
             CANON_ALIGN_REPORT \
             CANON_UPDATE_REPORT; do
      req "$k"
    done
    if [ "${CANON_P34_DEEPSWE:-0}" != "1" ]; then
      req CANON_PRE_ALIGN_REPORT
      case "${CANON_P33_SHORT_ALIGNMENT:-}" in
        0|1) ;;
        *) echo "[env] CANON_P33_SHORT_ALIGNMENT must be 0 or 1" >&2; fail=1 ;;
      esac
      if [ "${CANON_P33_RUN_STAGE:-}" = "alignment-short" ]; then
        [ "${CANON_P33_SHORT_ALIGNMENT:-0}" = "1" ] || {
          echo "[env] alignment-short requires CANON_P33_SHORT_ALIGNMENT=1" >&2
          fail=1
        }
      elif [ "${CANON_P33_SHORT_ALIGNMENT:-0}" != "0" ]; then
        echo "[env] only alignment-short may enable CANON_P33_SHORT_ALIGNMENT" >&2
        fail=1
      fi
      if { [ "${CANON_P32_WORKLOAD:-}" = "frozenlake" ] || \
           { [ "$APC_M15_TARGET_DEBUG" = "1" ] && \
             [ "${CANON_P32_WORKLOAD:-}" = "frozenlake-dp8-tp8" ]; } || \
           { [ -n "${CANON_V1_FL_TP8_AB_ARM:-}" ] && \
             [ "${CANON_P32_WORKLOAD:-}" = "frozenlake-dp8-tp8" ]; }; } && \
         [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ]; then
        req CANON_P38_MISMATCH_CAPSULE
        expected_p38_capsule_rows=2
        if [ -n "${CANON_P38_SERVING_CAPTURE_DIR:-}" ]; then
          expected_p38_capsule_rows=256
        fi
        [ "${CANON_P38_MISMATCH_CAPSULE_MAX_ROWS:-}" = \
          "$expected_p38_capsule_rows" ] || {
          echo "[env] FrozenLake replay capsule row bound drifted: expected=$expected_p38_capsule_rows" >&2
          fail=1
        }
      elif [ "${CANON_P64_P45_NUMERIC_DEBUG:-0}" = "1" ] && \
           [ "${CANON_P32_WORKLOAD:-}" = "frozenlake-dp8-tp8" ] && \
           [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ]; then
        req CANON_P64_TRAINING_CAPSULE
        req CANON_P64_TRAINING_CAPSULE_GCS_URI
      elif [ -n "${CANON_P38_MISMATCH_CAPSULE:-}" ]; then
        echo "[env] mismatch capsule is admitted only for FrozenLake backward-no-commit" >&2
        fail=1
      fi
    fi
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
    if [ "$P57_STOCK_FAST" = "1" ]; then
      [ "${FL_SHARED_MESH:-}" = "8,8" ] || {
        echo "[env] P57 stock-fast calibration requires the DP8xTP8 carrier mesh" >&2
        fail=1
      }
    else
      [ "${FL_SHARED_MESH:-}" = "1,4" ] || {
        echo "[env] unadmitted P32 modes must keep the legacy trainer at TP4-only" >&2
        fail=1
      }
    fi
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
    req CANON_P33_ENABLE_EVAL
    req CANON_P33_DISABLE_EVAL
    req CANON_P31_ENABLE_EVAL
    [ "${CANON_P33_ENABLE_EVAL:-0}:${CANON_P33_DISABLE_EVAL:-0}:${CANON_P31_ENABLE_EVAL:-0}" = "0:1:0" ] || {
      echo "[env] contract-only FrozenLake must keep evaluation disabled" >&2
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
if [ "$P57_STOCK_FAST" = "1" ]; then
  if [ "$P57_STOCK_TRAIN" = "1" ]; then
    echo "[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=train absent=12 observer=train processed_b=on"
  elif [ "$P57_STOCK_EVAL" = "1" ]; then
    echo "[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=eval absent=12 observer=off"
  else
    echo "[P57.STOCK_FAST] ZERO_TIM_OFF_PASS absent=12 zero=25"
  fi
fi
if [ "$GSM8K_NATIVE" = "1" ]; then
  echo "[GSM8K.NATIVE] ZERO_TIM_OFF_PASS p32=absent canonical_engine=off alignment=off p59=off v1=off"
fi

# Emit the resolved configuration as an authoritative snapshot.  00_env.sh runs in a child
# process, so an `unset` performed by a profile cannot mutate the parent entrypoint.  Clear
# every namespace managed by this file when the snapshot is sourced before exporting the
# resolved values; otherwise presence-sensitive switches from the raw JobSet environment can
# leak back into later steps.  Secrets are deliberately outside that reset: they are
# re-exported by later steps from the process environment and are never written here.
{
  echo "# generated by cluster/steps/00_env.sh -- do not edit"
  cat <<'EOF'
# Replace the managed environment instead of layering it over the caller's raw JobSet env.
for canon_env_key in $(compgen -e); do
  case "$canon_env_key" in
    CANON_*|R2E_*|WANDB_*|HF_*|MIN_TOKEN_BUCKET|NEW_MODEL_DESIGN|VLLM_*|ROLLOUT_ENGINE|XLA_FLAGS|JAX_*|FL_SHARED_MESH|TPU_*|TF_CPP*|ENABLE_PATHWAYS|PYTHONDONTWRITEBYTECODE|PATHWAYS_*|GRPC_*)
      case "$canon_env_key" in
        HF_TOKEN|WANDB_API_KEY|INJECTED_*) ;;
        *) unset "$canon_env_key" ;;
      esac
      ;;
  esac
done
unset canon_env_key
EOF
  for k in $(compgen -e | grep -E '^(CANON_|R2E_|WANDB_|HF_|MIN_TOKEN_BUCKET|NEW_MODEL_DESIGN|VLLM_|ROLLOUT_ENGINE|XLA_FLAGS|JAX_|FL_SHARED_MESH|TPU_|TF_CPP|ENABLE_PATHWAYS|PYTHONDONTWRITEBYTECODE|PATHWAYS_|GRPC_)' | sort); do
    case "$k" in
      HF_TOKEN|WANDB_API_KEY|INJECTED_*) continue ;;
    esac
    printf 'export %s=%q\n' "$k" "${!k}"
  done
} > "$CANON_STATE/env.sh"

echo "[env] profile=$CANON_PROFILE model_dir=$CANON_MODEL_DIR_NAME"
echo "[env] resolved configuration written to $CANON_STATE/env.sh ($(wc -l < "$CANON_STATE/env.sh") lines)"
