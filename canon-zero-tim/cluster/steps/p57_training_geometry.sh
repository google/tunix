#!/usr/bin/env bash
# Closed full-training geometry constants. No environment masking/DP aliasing.
# Python's examples.frozenlake.training_geometry is the paired authority;
# host tests compare the complete shell/Python tuple.
p57_training_geometry_init() {
  _p57_full_dp=8
  _p57_full_prompts=32
  _p57_full_trajectories=256
  _p57_full_devices=64
  _p57_full_global_m=2048
  if [[ -v CANON_P57_TRAIN_GEOMETRY ]]; then
    if [[ "$CANON_P57_TRAIN_GEOMETRY" != dp4-tp8-b128 ||
          "${CANON_PROFILE_FILE:-}" != cluster/profiles/qwen3-8b-dp4-tp8-frozenlake-v1-hp.env ||
          "${CANON_V1_HP_FULL:-}" != 1 ||
          "${CANON_P57_RUN_KIND:-}" != train ||
          "${CANON_P57_TIM_ARM:-}" != zero ||
          "${CANON_P57_EXPECTED_UPDATES:-}" != 300 ||
          "${CANON_P33_RUN_STAGE:-}" != full ||
          "${CANON_P33_NO_COMMIT:-}" != 0 ]]; then
      echo "[P57.GEOMETRY] FATAL: DP4xTP8/B128 requires its exact Zero-HP full identity" >&2
      return 1
    fi
    case "${CANON_P57_WORKLOAD_CANDIDATE:-}:${CANON_P57_DATA_SPLIT:-}" in
      :|m15:main) ;;
      *) echo "[P57.GEOMETRY] FATAL: unsupported workload" >&2; return 1 ;;
    esac
    _p57_full_dp=4
    _p57_full_prompts=16
    _p57_full_trajectories=128
    _p57_full_devices=32
    _p57_full_global_m=1024
  elif [[ "${CANON_PROFILE_FILE:-}" == cluster/profiles/qwen3-8b-dp4-tp8-frozenlake-v1-hp.env ]]; then
    echo "[P57.GEOMETRY] FATAL: new profile requires the explicit geometry selector" >&2
    return 1
  fi
  _p57_full_workload="frozenlake-dp${_p57_full_dp}-tp8"
  _p57_full_profile="qwen3-8b-dp${_p57_full_dp}-tp8-frozenlake-v1-hp"
  _p57_full_profile_file="cluster/profiles/${_p57_full_profile}.env"
  _p57_full_mesh="${_p57_full_dp},8"
}

p57_training_geometry_check_raw() {
  [[ -v CANON_P57_TRAIN_GEOMETRY ]] || return 0
  local pair key expected
  for pair in \
      "CANON_P32_WORKLOAD=$_p57_full_workload" \
      "CANON_PROFILE=$_p57_full_profile" \
      "CANON_DP_SIZE=$_p57_full_dp" "CANON_TP_SIZE=8" \
      "CANON_ENGINE_DP_SIZE=$_p57_full_dp" \
      "CANON_TOTAL_DEVICES=$_p57_full_devices" \
      "CANON_GLOBAL_PROMPTS=$_p57_full_prompts" "CANON_LOCAL_PROMPTS=4" \
      "CANON_NUM_GENERATIONS=8" "CANON_LOCAL_TRAJECTORIES=32" \
      "CANON_GLOBAL_TRAJECTORIES=$_p57_full_trajectories" \
      "CANON_DP_PROBE_LOCAL_SAMPLES=32" "CANON_LOGPROB_M=256" \
      "CANON_TARGET_M=256" "CANON_MAX_BATCHED=256" \
      "MIN_TOKEN_BUCKET=$_p57_full_global_m" \
      "CANON_P33_SHARED_MESH=$_p57_full_mesh" "FL_SHARED_MESH=$_p57_full_mesh"; do
    key="${pair%%=*}"; expected="${pair#*=}"
    if [[ -v "$key" && "${!key}" != "$expected" ]]; then
      echo "[P57.GEOMETRY] FATAL: contradictory raw $key" >&2
      return 1
    fi
  done
}
