#!/bin/bash
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

# Launches the MLPerf DeepSWE recipe (`mlperf_35b_128_v5p.yaml`: Qwen3.5-35B-A3B
# MoE on 128 TPU v5p chips) on GKE through the shared launcher
# (`tunix/dist_rl/launch/`). Pass the per-run settings as `section.key=value`
# arguments; see `README.md`.

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared by the trainer container (`LIBTPU_INIT_ARGS`), the Pathways workers,
# and the Pathways proxy (its `--xla_*` flags).
trainer_libtpu_flags=(
  --DANGEROUS_tpu_runtime_abi_verification_disabled=true
  --xla_tpu_use_tc_device_shape_on_sc=true
  --xla_sc_disable_megacore_partitioning=true
  --xla_tpu_enable_offloading_gather_to_sparsecore=true
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true
  --xla_tpu_enable_concurrent_sparse_core_offloading=true
  --xla_tpu_aggressive_opt_barrier_removal=true
  --xla_tpu_scoped_vmem_limit_kib=65536
  --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false
)
trainer_libtpu_args="${trainer_libtpu_flags[*]}"
xla_flags=""
for flag in "${trainer_libtpu_flags[@]}"; do
  if [[ "${flag}" == --xla_* ]]; then
    xla_flags+="${flag} "
  fi
done

# `yaml_generator` reads the Pathways worker and proxy settings from the
# launcher's environment; they have no config keys. The template sets
# TPU_RAIDEN_DATA_NICS on the workers itself.
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV:-\
LIBTPU_INIT_ARGS=${trainer_libtpu_args} --megascale_port=-1 \
--xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true
RAIDEN_BROADCAST_HOST_RATIO=1.0
RAIDEN_BROADCAST_PIPELINE_STAGES=4}"
export PATHWAYS_PROXY_EXTRA_ARGS="${PATHWAYS_PROXY_EXTRA_ARGS:-${xla_flags% }}"
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"
export TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES="${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES:-16777216}"

exec "${DIR}/../../../dist_rl/launch/common/launch.sh" \
  "override_config_file=${DIR}/mlperf_35b_128_v5p.yaml" \
  "cluster_config.role_env.trainer.LIBTPU_INIT_ARGS=${trainer_libtpu_args}" \
  "$@"
