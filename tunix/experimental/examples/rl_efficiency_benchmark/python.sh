#!/usr/bin/env bash
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

set -euo pipefail

args=("$@")
if [[ ${args[0]:-} == -m && ${args[1]:-} == tunix.experimental.distributed.runtime.main ]]; then
  for index in "${!args[@]}"; do
    case ${args[index]} in
      --process_main=tunix.experimental.examples.common.run_trainer_node.main)
        args[index]=--process_main=tunix.experimental.examples.rl_efficiency_benchmark.runtime.trainer_main ;;
      --process_main=tunix.experimental.examples.common.run_rollout_node.main)
        args[index]=--process_main=tunix.experimental.examples.rl_efficiency_benchmark.runtime.rollout_main ;;
      --process_main=tunix.experimental.examples.frozenlake_dist.run_frozenlake_dist.main)
        args[index]=--process_main=tunix.experimental.examples.rl_efficiency_benchmark.runtime.dist_main ;;
    esac
  done
fi
exec "$BENCHMARK_PYTHON" "${args[@]}"
