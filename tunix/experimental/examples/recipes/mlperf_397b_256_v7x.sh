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

# Launches the MLPerf DeepSWE 397B recipe (`mlperf_397b_256_v7x.yaml`:
# Qwen3.5-397B-A17B MoE on 256 TPU v7x chips with single-host FP8 MoE rollout)
# on GKE through the shared launcher (`tunix/dist_rl/launch/`). Pass per-run
# settings as `section.key=value` arguments (and export `TPU_RESERVATION` when
# targeting a reserved TPU7x node pool); see `README.md`.

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared by the trainer container (`LIBTPU_INIT_ARGS`), the Pathways workers,
# and the Pathways proxy (its `--xla_*` flags).
# Note: TPU v7x caps scoped VMEM at 67,043,328 bytes (65472 KiB); 65536 KiB is
# rejected with `INVALID_ARGUMENT` in `pathways-rm`.
trainer_libtpu_flags=(
  --DANGEROUS_tpu_runtime_abi_verification_disabled=true
  --xla_tpu_use_tc_device_shape_on_sc=true
  --xla_sc_disable_megacore_partitioning=true
  --xla_tpu_enable_offloading_gather_to_sparsecore=false
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true
  --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=true
  --xla_tpu_enable_concurrent_sparse_core_offloading=true
  --xla_tpu_aggressive_opt_barrier_removal=true
  --xla_tpu_scoped_vmem_limit_kib=65472
  --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false
  --xla_tpu_enable_offloading_copy_to_sparsecore=false
  --xla_tpu_dvfs_p_state=7
)
# Reserve ~10 GiB of TPU v7x HBM headroom for Raiden FFI staging buffers by
# default; set TRAINER_MAX_HBM_SIZE_MIB=0 to disable the cap.
: "${TRAINER_MAX_HBM_SIZE_MIB:=84992}"
if [[ -n "${TRAINER_MAX_HBM_SIZE_MIB}" && "${TRAINER_MAX_HBM_SIZE_MIB}" != "0" ]]; then
  trainer_libtpu_flags+=(
    "--xla_tpu_max_hbm_size_mib=${TRAINER_MAX_HBM_SIZE_MIB}"
  )
fi
trainer_libtpu_args="${trainer_libtpu_flags[*]}"
xla_flags=""
for flag in "${trainer_libtpu_flags[@]}"; do
  if [[ "${flag}" == --xla_* ]]; then
    xla_flags+="${flag} "
  fi
done

# `yaml_generator` reads the Pathways worker and proxy settings from the
# launcher's environment; they have no config keys. `jobset.pathways.yaml`
# already defines `ENABLE_MULTI_NUMA` and `TPU_RAIDEN_DATA_NICS` directly.
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV:-\
LIBTPU_INIT_ARGS=${trainer_libtpu_args} --megascale_port=-1 \
--xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true
RAIDEN_PARALLELISM=16
RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864
RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16
RAIDEN_BROADCAST_HOST_RATIO=1.0
RAIDEN_BROADCAST_PIPELINE_STAGES=4
TPU_RAIDEN_CONTROL_PLANE_BACKEND=tcp
TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=120000
TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=5}"
export PATHWAYS_PROXY_EXTRA_ARGS="${PATHWAYS_PROXY_EXTRA_ARGS:-${xla_flags% }}"
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"
export TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES="${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES:-16777216}"

exec "${DIR}/../../../dist_rl/launch/common/launch.sh" \
  "override_config_file=${DIR}/mlperf_397b_256_v7x.yaml" \
  "cluster_config.role_env.trainer.LIBTPU_INIT_ARGS=${trainer_libtpu_args}" \
  "$@"
