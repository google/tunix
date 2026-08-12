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
           CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS \
           CANON_P38_SERVING_CAPTURE_CLASSIFICATION \
           CANON_P38_SERVING_CAPTURE_ARCHIVE \
           CANON_P38_MISMATCH_CAPSULE \
           CANON_P38_PRECHECK_ONLY; do
    req "$k"
  done
  [ "${CANON_P32_WORKLOAD:-}" = "frozenlake" ] || {
    echo "[env] P38 serving capture requires the FrozenLake workload" >&2
    fail=1
  }
  [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ] && \
  [ "${CANON_P33_NO_COMMIT:-}" = "1" ] || {
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
  [[ "${CANON_P38_SERVING_CAPTURE_MIN_PREFIX:-}" =~ ^[0-9]+$ ]] || {
    echo "[env] P38 serving capture minimum prefix must be non-negative" >&2
    fail=1
  }
  [ "${CANON_P38_SERVING_CAPTURE_MIN_PREFIX:-}" = "1536" ] && \
  [ "${CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS:-}" = \
      "1536,1792,2048,2304,2560" ] || {
    echo "[env] P38 serving capture prefix strata drifted" >&2
    fail=1
  }
  [ "${CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER:-}" = "5" ] || {
    echo "[env] P38 serving capture requires the five-times free-space guard" >&2
    fail=1
  }
  echo "[env] P38 serving capture enabled: kv_unified=${CANON_KV_UNIFIED:-0}"
elif [ "${CANON_KV_UNIFIED:-0}" = "1" ]; then
  echo "[env] CANON_KV_UNIFIED is admitted only with bounded P38 serving capture" >&2
  fail=1
elif [ -n "${CANON_P38_SERVING_CAPTURE_MAX_CALLS:-}${CANON_P38_SERVING_CAPTURE_MIN_PREFIX:-}${CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS:-}${CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER:-}${CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS:-}${CANON_P38_SERVING_CAPTURE_CLASSIFICATION:-}${CANON_P38_SERVING_CAPTURE_ARCHIVE:-}${CANON_P38_PRECHECK_ONLY:-}" ]; then
  echo "[env] partial P38 serving-capture configuration is not admitted" >&2
  fail=1
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
           CANON_P39_64CHIP_PILOT CANON_P39_PILOT_ADMITTED \
           CANON_P43_DEEPSWE_DEBUG CANON_P43_DEBUG_ADMITTED \
           CANON_P43_ROLLOUT_ONLY \
           CANON_P44_DEEPSWE_PARITY CANON_P44_PARITY_ADMITTED \
           CANON_P44_TOPOLOGY CANON_P44_ROLLOUT_ONLY \
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
           CANON_P34_MAX_NUM_SEQS CANON_P34_MAX_BATCHED_TOKENS; do
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
  if [ "${CANON_P44_DEEPSWE_PARITY:-}" = "1" ]; then
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
      256)
        p34_expected_dp=16
        p34_expected_devices=128
        p34_expected_local_trajectories=1
        p34_expected_global_m=4096
        p34_expected_max_seqs=1
        p34_expected_mesh=16,8
        ;;
      *)
        echo "[env] P44 parity requires topology 64 or 256" >&2
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
  [ "${CANON_P34_DISABLE_SAMPLER_IS:-}" = "1" ] && \
  [ "${CANON_P34_DISABLE_TIS:-}" = "1" ] && \
  [ "${CANON_PRE_ALIGN_GATE:-}" = "1" ] || {
    echo "[env] P34 strict CLI, neutral importance paths and pre-backward gate are mandatory" >&2; fail=1;
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
  for k in CANON_P32_TRAIN_ADMITTED CANON_P32_DP_REDUCTION_ADMITTED \
           CANON_P33_WORKLOAD_LAUNCH_ADMITTED; do
    [ "${!k:-}" = "$p34_admitted" ] || {
      echo "[env] P34 inherited admission mismatch: $k must equal $p34_admitted" >&2
      fail=1
    }
  done
  if [ "$p34_admitted" = "1" ]; then
    for k in CANON_RUN_ID CANON_RUN_CMD CANON_RUN_LOG CANON_P34_WEIGHT_REPORT \
             CANON_PRE_ALIGN_REPORT \
             CANON_ALIGN_REPORT \
             CANON_UPDATE_REPORT CANON_WANDB_PROJECT CANON_WANDB_GROUP \
             CANON_WANDB_RUN_NAME WANDB_API_KEY; do
      req "$k"
    done
    [ "${WANDB_MODE:-}" = "online" ] || {
      echo "[env] P34 requires WANDB_MODE=online" >&2; fail=1;
    }
  fi
  echo "[env] P34 contract OK: DP${CANON_DP_SIZE}xTP${CANON_TP_SIZE} per role, local M256, global M${MIN_TOKEN_BUCKET}"
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
      gsm8k|frozenlake|frozenlake-dp8-tp8) ;;
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
    case "${CANON_GSM8K_AB_REPORT_ONLY:-0}" in
      0|1) ;;
      *) echo "[env] CANON_GSM8K_AB_REPORT_ONLY must be 0 or 1" >&2; fail=1 ;;
    esac
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
      if [ "${CANON_P32_WORKLOAD:-}" = "frozenlake" ] && \
         [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ]; then
        req CANON_P38_MISMATCH_CAPSULE
        [ "${CANON_P38_MISMATCH_CAPSULE_MAX_ROWS:-}" = "2" ] || {
          echo "[env] FrozenLake replay capsule must retain exactly two rows" >&2
          fail=1
        }
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

# Emit the resolved configuration.  Secrets are re-exported by later steps from the process
# environment, never written here.
{
  echo "# generated by cluster/steps/00_env.sh -- do not edit"
  for k in $(compgen -e | grep -E '^(CANON_|WANDB_|HF_|MIN_TOKEN_BUCKET|NEW_MODEL_DESIGN|VLLM_|ROLLOUT_ENGINE|XLA_FLAGS|JAX_|FL_SHARED_MESH|TPU_|TF_CPP|ENABLE_PATHWAYS|PYTHONDONTWRITEBYTECODE)' | sort); do
    case "$k" in
      HF_TOKEN|WANDB_API_KEY|INJECTED_*) continue ;;
    esac
    printf 'export %s=%q\n' "$k" "${!k}"
  done
} > "$CANON_STATE/env.sh"

echo "[env] profile=$CANON_PROFILE model_dir=$CANON_MODEL_DIR_NAME"
echo "[env] resolved configuration written to $CANON_STATE/env.sh ($(wc -l < "$CANON_STATE/env.sh") lines)"
