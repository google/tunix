#!/bin/bash
# ==============================================================================
# uBench Reporting
# ==============================================================================
# When UBENCH_REPORTING=true, mlperf_base.sh runs this file in a subshell,
# so that uBench can report the run to its dashboard after the run finishes.
# uBench does not launch, monitor, or clean up the run.
#
# This file doesn't change the recipe's variables and never stops the launch.
# For runs that uBench can't report (non-tpu7x slices, or the default
# JOB_PREFIX), it prints a one-line note. If something else is wrong, it prints
# a warning, writes nothing, and returns 1.
#
# uBench uses MAXTEXT_OUTPUT_DIR as its run directory, so it must end with
# /${JOB_PREFIX}. uBench reads the MLPerf log from ${MAXTEXT_OUTPUT_DIR}/mllog
# (the default METRIC_LOGGER_DIR). It reads the step metrics from TensorBoard
# in ${MAXTEXT_OUTPUT_DIR}/tensorboard if LOG_DIR is set to that path, and
# from Cloud Logging otherwise.
#
# On `start`, this file writes to MAXTEXT_OUTPUT_DIR:
#   logs/rl_topology.yaml      Trainer and rollout topologies and chip counts.
#   workload_timestamps.yaml   Launch time. uBench ignores older pods.
#   logs/resolved_config.yaml  uBench config for this run. Written last.
# Then it prints the command that reports the run after it finishes.
# ==============================================================================

# Messages go to stderr so that dry-run YAML on stdout stays valid.
_ubench_log() {
  echo "UBENCH_REPORTING: $*" >&2
}

# Returns the number of chips in a topology such as 4x4x4.
_ubench_num_chips() {
  local chips=1 dim
  local -a dims
  IFS=x read -ra dims <<< "$1"
  for dim in "${dims[@]}"; do
    chips=$((chips * dim))
  done
  echo "${chips}"
}

# Streams stdin to a GCS object ("-" reads stdin). Prints only errors, not
# gcloud's progress lines.
_ubench_upload() {
  gcloud storage cp - "$1" --no-user-output-enabled
}

# `set -e` has no effect in here, because mlperf_base.sh runs this file as
# `( ... ) || ...`. So each command that can fail is checked.
_ubench_main() {
  if [[ "${1:-start}" != "start" || "${MLPERF_NO_LAUNCH:-0}" == "1" ]]; then
    return 0
  fi
  local dry_run="${DRY_RUN:-false}" arg
  for arg in "$@"; do
    case "${arg}" in
      --dry-run|--dry_run|--render) dry_run="true" ;;
    esac
  done

  if [[ "${TRAINER_TPU_SLICE:-}" != tpu7x:* ||
        "${ROLLOUT_TPU_SLICE:-}" != tpu7x:* ]]; then
    _ubench_log "Not recording this run for uBench, because only tpu7x" \
      "slices are supported."
    return 0
  fi
  # uBench finds the run's pods by name prefix, and pod names start with
  # JOB_PREFIX. So each run needs its own JOB_PREFIX. Also don't run two jobs
  # at the same time when one name starts with the other (run-1 and run-10).
  if [[ "${JOB_PREFIX:-}" == "${USER:-}" ]]; then
    _ubench_log "Not recording this run for uBench, because JOB_PREFIX is" \
      "the default (\$USER). To record runs, use a new JOB_PREFIX for each" \
      "run."
    return 0
  fi
  if [[ ! "${JOB_PREFIX:-}" =~ ^[a-z][a-z0-9-]*$ ]]; then
    _ubench_log "JOB_PREFIX '${JOB_PREFIX:-}' must start with a lowercase" \
      "letter and use only lowercase letters, digits, and '-'."
    return 1
  fi
  # The topologies and ROLLOUT_REPLICAS are used in arithmetic below, so check
  # their format first. Leading zeros are rejected because bash reads them as
  # octal numbers.
  local slice_re='^tpu7x:[1-9][0-9]*(x[1-9][0-9]*)*$'
  if [[ ! "${TRAINER_TPU_SLICE:-}" =~ ${slice_re} ||
        ! "${ROLLOUT_TPU_SLICE:-}" =~ ${slice_re} ]]; then
    _ubench_log "Only tpu7x slices with a topology such as tpu7x:4x4x4 are" \
      "supported. Got TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-}," \
      "ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-}."
    return 1
  fi
  local rollout_replicas="${ROLLOUT_REPLICAS:-1}"
  if [[ ! "${rollout_replicas}" =~ ^[1-9][0-9]*$ ]]; then
    _ubench_log "ROLLOUT_REPLICAS must be a positive integer. Got" \
      "'${rollout_replicas}'."
    return 1
  fi
  # The uBench run directory is <root>/<run name>, and the run name is
  # JOB_PREFIX.
  local run_dir="${MAXTEXT_OUTPUT_DIR:-}"
  run_dir="${run_dir%/}"
  if [[ "${run_dir}" != gs://*/"${JOB_PREFIX}" ]]; then
    _ubench_log "MAXTEXT_OUTPUT_DIR must be a gs:// path that ends with" \
      "/${JOB_PREFIX}, because uBench uses it as the run directory. Got" \
      "'${MAXTEXT_OUTPUT_DIR:-}'."
    return 1
  fi
  local root="${run_dir%/*}"
  # Same values as k8s_launcher.sh accepts.
  case "${RCP_LOGGING:-false}" in
    1|true|True)
      if [[ "${METRIC_LOGGER_DIR:-}" != "${run_dir}/mllog"* ]]; then
        _ubench_log "Warning: uBench reads the MLPerf log from" \
          "${run_dir}/mllog, but METRIC_LOGGER_DIR is" \
          "'${METRIC_LOGGER_DIR:-}'. uBench won't report MLPerf metrics."
      fi
      ;;
    *)
      _ubench_log "Warning: RCP_LOGGING is off, so the run writes no MLPerf" \
        "log, and uBench won't report MLPerf metrics."
      ;;
  esac

  local trainer_topology="${TRAINER_TPU_SLICE#tpu7x:}"
  local rollout_topology="${ROLLOUT_TPU_SLICE#tpu7x:}"
  local trainer_chips rollout_chips
  trainer_chips="$(_ubench_num_chips "${trainer_topology}")"
  rollout_chips="$(( $(_ubench_num_chips "${rollout_topology}") * rollout_replicas ))"

  if [[ "${dry_run}" == "true" ]]; then
    _ubench_log "Dry run. Not writing files to ${run_dir}."
    return 0
  fi
  if gcloud storage ls "${run_dir}/logs/resolved_config.yaml" > /dev/null 2>&1; then
    _ubench_log "${run_dir} already has uBench files from an earlier run." \
      "Use a new JOB_PREFIX."
    return 1
  fi

  # uBench copies this into the metrics_other BigQuery field. `chips` is the
  # total for the role: chips per replica times replicas.
  local rl_topology
  rl_topology="$(cat <<EOF
trainer:
  accelerator: tpu7x
  topology: "${trainer_topology}"
  replicas: 1
  chips: ${trainer_chips}
rollout:
  accelerator: tpu7x
  topology: "${rollout_topology}"
  replicas: ${rollout_replicas}
  chips: ${rollout_chips}
total_chips: $((trainer_chips + rollout_chips))
EOF
)"
  local resolved_config
  resolved_config="$(cat <<EOF
# Written by tunix/experimental/examples/recipes/ubench_reporting.sh.
benchmark_type: DISTRIBUTED
run_name: "${JOB_PREFIX}"
cluster:
  gke:
    project: "${PROJECT:-}"
    zone: "${REGION:-}"
    cluster_name: "${CLUSTER:-}"
distributed:
  recipe:
    generator_type: HELM
    artifacts_gcs_root_path: "${root}"
  workload:
    type: TUNIX_RL
    model: "${MODEL_ID:-}"
    hardware: V7X_${trainer_topology//x/X}
    num_steps: ${MAX_STEPS:-}
    image: "${TUNIX_IMAGE:-}"
    recipe_type: MLPERF
    # Same as the uBench default: skip the first (warmup) step.
    metrics_config:
      start_step: 1
EOF
)"
  # Each file is streamed to GCS, so no temporary files are left behind if an
  # upload fails. resolved_config.yaml goes last: the check above looks for it,
  # so a failed upload doesn't block a retry.
  if ! printf '%s\n' "${rl_topology}" |
         _ubench_upload "${run_dir}/logs/rl_topology.yaml" ||
     ! echo "job_start_time: '$(date -u +%Y-%m-%dT%H:%M:%SZ)'" |
         _ubench_upload "${run_dir}/workload_timestamps.yaml" ||
     ! printf '%s\n' "${resolved_config}" |
         _ubench_upload "${run_dir}/logs/resolved_config.yaml"; then
    _ubench_log "Could not write the uBench files to ${run_dir}."
    return 1
  fi

  _ubench_log "After the run finishes, report it with:"
  echo "  ubench benchmark report --run-name ${JOB_PREFIX} --gcs-root-path ${root}" >&2
}

_ubench_main "$@"
