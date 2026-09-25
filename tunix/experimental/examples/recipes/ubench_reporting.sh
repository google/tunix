#!/bin/bash
# ==============================================================================
# Optional uBench Reporting
# ==============================================================================
# mlperf_base.sh sources this file when UBENCH_REPORTING=true, so that uBench
# can report the run to its dashboard after the run finishes. uBench does not
# launch, monitor, or clean up the run.
#
# For every command, TensorBoard events and the MLPerf log go to the uBench run
# directory ${UBENCH_GCS_ROOT_PATH}/${JOB_PREFIX}.
#
# On `start`, this file also writes the files that uBench reads:
#   logs/resolved_config.yaml  uBench config for this run.
#   logs/rl_topology.yaml      Trainer and rollout topologies and chip counts.
#   workload_timestamps.yaml   Launch time. uBench ignores older pods.
#
# After the run finishes, report it with:
#   ubench benchmark report --run-name ${JOB_PREFIX} \
#     --gcs-root-path ${UBENCH_GCS_ROOT_PATH}
# ==============================================================================

UBENCH_GCS_ROOT_PATH="${UBENCH_GCS_ROOT_PATH:-gs://ubench-logs/rl}"
# Remove a trailing "/", if any, so that the paths below don't contain "//".
UBENCH_GCS_ROOT_PATH="${UBENCH_GCS_ROOT_PATH%/}"
UBENCH_RUN_DIR="${UBENCH_GCS_ROOT_PATH}/${JOB_PREFIX}"

export LOG_DIR="${UBENCH_RUN_DIR}/tensorboard"
# uBench reads the MLPerf log from <run dir>/mllog_*.log.
export METRIC_LOGGER_DIR="${UBENCH_RUN_DIR}/mllog_tunix_rl.log"
# Messages go to stderr so that dry-run YAML on stdout stays valid.
echo "UBENCH_REPORTING: LOG_DIR=${LOG_DIR}" >&2
echo "UBENCH_REPORTING: METRIC_LOGGER_DIR=${METRIC_LOGGER_DIR}" >&2

_ubench_fail() {
  echo "UBENCH_REPORTING: $*" >&2
  exit 1
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

_ubench_command="${1:-start}"
_ubench_dry_run="${DRY_RUN:-false}"
for _ubench_arg in "$@"; do
  case "${_ubench_arg}" in
    --dry-run|--dry_run|--render) _ubench_dry_run="true" ;;
  esac
done

if [[ "${_ubench_command}" == "start" && "${MLPERF_NO_LAUNCH:-0}" != "1" ]]; then
  # uBench finds the run's pods by name prefix, and pod names start with
  # JOB_PREFIX. So each run needs its own JOB_PREFIX. Also don't run two jobs
  # at the same time when one name starts with the other (run-1 and run-10).
  if [[ ! "${JOB_PREFIX}" =~ ^[a-z][a-z0-9-]*$ ]]; then
    _ubench_fail "JOB_PREFIX '${JOB_PREFIX}' must start with a lowercase" \
      "letter and use only lowercase letters, digits, and '-'."
  fi
  if [[ "${JOB_PREFIX}" == "${USER:-}" ]]; then
    _ubench_fail "Set JOB_PREFIX to a new name for each run. The default" \
      "(\$USER) is the same for all of your runs."
  fi
  # The topologies and ROLLOUT_REPLICAS are used in arithmetic below, so check
  # their format first. Leading zeros are rejected because bash reads them as
  # octal numbers.
  _ubench_slice_re='^tpu7x:[1-9][0-9]*(x[1-9][0-9]*)*$'
  if [[ ! "${TRAINER_TPU_SLICE}" =~ ${_ubench_slice_re} ||
        ! "${ROLLOUT_TPU_SLICE}" =~ ${_ubench_slice_re} ]]; then
    _ubench_fail "Only tpu7x slices with a topology such as tpu7x:4x4x4 are" \
      "supported. Got TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE}," \
      "ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE}."
  fi
  _ubench_rollout_replicas="${ROLLOUT_REPLICAS:-1}"
  if [[ ! "${_ubench_rollout_replicas}" =~ ^[1-9][0-9]*$ ]]; then
    _ubench_fail "ROLLOUT_REPLICAS must be a positive integer. Got" \
      "'${_ubench_rollout_replicas}'."
  fi

  _ubench_trainer_topology="${TRAINER_TPU_SLICE#tpu7x:}"
  _ubench_rollout_topology="${ROLLOUT_TPU_SLICE#tpu7x:}"
  _ubench_trainer_chips="$(_ubench_num_chips "${_ubench_trainer_topology}")"
  _ubench_rollout_chips="$(( $(_ubench_num_chips "${_ubench_rollout_topology}") * _ubench_rollout_replicas ))"

  if [[ "${_ubench_dry_run}" == "true" ]]; then
    echo "UBENCH_REPORTING: Dry run. Not writing files to ${UBENCH_RUN_DIR}." >&2
  else
    if gcloud storage ls "${UBENCH_RUN_DIR}/" > /dev/null 2>&1; then
      _ubench_fail "${UBENCH_RUN_DIR} already exists. Use a new JOB_PREFIX," \
        "or delete the old run first: gcloud storage rm -r ${UBENCH_RUN_DIR}"
    fi

    # Each file is streamed to GCS ("-" reads stdin), so no temporary files are
    # left behind if an upload fails. With `set -e`, a failed upload stops the
    # launch. This file is sourced, so it doesn't set an EXIT trap, which would
    # replace a trap of the caller.
    {
      echo "# Written by tunix/experimental/examples/recipes/ubench_reporting.sh."
      echo "benchmark_type: DISTRIBUTED"
      echo "run_name: \"${JOB_PREFIX}\""
      echo "cluster:"
      echo "  gke:"
      echo "    project: \"${PROJECT}\""
      echo "    zone: \"${REGION}\""
      echo "    cluster_name: \"${CLUSTER}\""
      echo "distributed:"
      echo "  recipe:"
      echo "    generator_type: HELM"
      echo "    artifacts_gcs_root_path: \"${UBENCH_GCS_ROOT_PATH}\""
      echo "  workload:"
      echo "    type: TUNIX_RL"
      echo "    model: \"${MODEL_ID}\""
      echo "    hardware: V7X_${_ubench_trainer_topology//x/X}"
      echo "    num_steps: ${MAX_STEPS}"
      echo "    image: \"${TUNIX_IMAGE}\""
      echo "    recipe_type: MLPERF"
      # Same as the uBench default: skip the first (warmup) step.
      echo "    metrics_config:"
      echo "      start_step: 1"
    } | gcloud storage cp - "${UBENCH_RUN_DIR}/logs/resolved_config.yaml"
    # uBench copies this into the metrics_other BigQuery field. `chips` is
    # the total for the role: chips per replica times replicas.
    gcloud storage cp - "${UBENCH_RUN_DIR}/logs/rl_topology.yaml" <<EOF
trainer:
  accelerator: tpu7x
  topology: "${_ubench_trainer_topology}"
  replicas: 1
  chips: ${_ubench_trainer_chips}
rollout:
  accelerator: tpu7x
  topology: "${_ubench_rollout_topology}"
  replicas: ${_ubench_rollout_replicas}
  chips: ${_ubench_rollout_chips}
total_chips: $((_ubench_trainer_chips + _ubench_rollout_chips))
EOF
    echo "job_start_time: '$(date -u +%Y-%m-%dT%H:%M:%SZ)'" |
      gcloud storage cp - "${UBENCH_RUN_DIR}/workload_timestamps.yaml"

    echo "UBENCH_REPORTING: After the run finishes, report it with:" >&2
    echo "  ubench benchmark report --run-name ${JOB_PREFIX} --gcs-root-path ${UBENCH_GCS_ROOT_PATH}" >&2
  fi
fi
