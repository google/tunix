#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ROLLOUT_ID=${ROLLOUT_ID:-${USER}-roll-single}
export ROLLOUT_JOBSET_YAML=${ROLLOUT_JOBSET_YAML:-jobset.pathways.yaml}

if [[ "${ROLLOUT_JOBSET_YAML}" == "jobset.tpu.yaml" ]]; then
  export ROLLOUT_COMPLETIONS=${ROLLOUT_COMPLETIONS:-1}
  export ROLLOUT_PARALLELISM=${ROLLOUT_PARALLELISM:-1}
else
  unset ROLLOUT_COMPLETIONS
  unset ROLLOUT_PARALLELISM
fi

echo "Launching single rollout proc jobset ${ROLLOUT_ID}"
echo "Using target ${1:-inference-v5e}"
echo "Using rollout jobset ${ROLLOUT_JOBSET_YAML}"

bash "${SCRIPT_DIR}/k8s_launcher.sh" \
  --target="${1:-inference-v5e}" \
  --command=rollout