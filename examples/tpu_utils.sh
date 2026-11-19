# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
# tpu_utils.sh

# Helper function: extracts dimensions from a mesh tuple and calculates the product
calc_mesh_tpus() {
  local input="$1"
  if [[ -z "$input" || "$input" == "null" ]]; then
    echo 0
    return 0
  fi
  # Remove parens and spaces, then replace commas with spaces
  local dims="${input//[() ]/}"
  dims="${dims//,/ }"
  local product=1
  for d in $dims; do
    if [[ -n "$d" ]]; then
      product=$(( product * d ))
    fi
  done
  echo $product
}

# Main validation function
# Usage: validate_mesh_allocation <total_tpus> <trainer_mesh> <rollout_mesh> <reference_mesh>
validate_mesh_allocation() {
  if [[ $# -ne 4 ]]; then
    echo "Error: validate_mesh_allocation requires exactly 4 arguments: <total_tpus> <trainer_mesh> <rollout_mesh> <reference_mesh>" >&2
    echo "Got: $@" >&2
    return 1
  fi

  local total_tpus="$1"
  local trainer_mesh="$2"
  local rollout_mesh="$3"
  local reference_mesh="$4"

  local trainer_tpus
  local rollout_tpus
  local reference_tpus
  local required_tpus

  trainer_tpus=$(calc_mesh_tpus "$trainer_mesh")
  rollout_tpus=$(calc_mesh_tpus "$rollout_mesh")
  reference_tpus=$(calc_mesh_tpus "$reference_mesh")
  required_tpus=$(( trainer_tpus + rollout_tpus + reference_tpus ))

  if (( required_tpus > total_tpus )); then
    # Print errors to standard error (stderr) using >&2
    echo "Error: Required TPUs ($required_tpus) exceeds total_tpus ($total_tpus)." >&2
    echo "  Trainer needs: $trainer_tpus (mesh: $trainer_mesh)" >&2
    echo "  Rollout needs: $rollout_tpus (mesh: $rollout_mesh)" >&2
      echo "  Reference needs: $reference_tpus (mesh: $reference_mesh)" >&2
    return 1 # Return failure so the caller can handle it
  fi

  return 0 # Return success
}
