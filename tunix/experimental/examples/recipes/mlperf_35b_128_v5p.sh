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

# MLPerf DeepSWE recipe on the shared launcher (`tunix/dist_rl/launch/`):
# - Model: Qwen3.5-35B-A3B MoE on 128 TPU v5p chips
# - Trainer: 64 chips (tpuv5:4x4x4, MaxText FSDP=32, CP=2)
# - Rollout: 64 chips (16 x tpuv5:2x2x1, vLLM EP=4, TP=1)
# - Sandboxes: OpenHands CodeAct on Kubernetes Agent Sandbox warm pools

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: bash mlperf_35b_128_v5p.sh \
  [start|stop|orchestrator|trainer|rollout|otel|stop-otel] [--dry-run] \
  --image=<tunix image> --bucket=<gs://bucket> \
  --maxtext-ckpt=<gs://path> --dataset-path=<gs://path> \
  [--pathways-server-image=<image>] [--pathways-proxy-image=<image>] \
  [section.key=value ...]
EOF
}

require_opt_arg() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Error: $1 requires a non-empty value." >&2
    exit 1
  fi
}

tunix_image=""
namespace="default"
kueue_queue_name=""
service_account="xpk-sa"
priority_class="medium"
cpu_nodepool=""
head_excluded_nodepools=""
sandbox_node_selector_key="cloud.google.com/gke-nodepool"
sandbox_node_selector_val=""
image_rewrite_prefix=""
openhands_server_image=""
pathways_server_image=""
pathways_proxy_image=""
bucket=""
maxtext_ckpt=""
dataset_path=""
if [[ "${MLPERF_SKIP_INTERNAL_ARGS:-0}" != "1" && \
  -f "${DIR}/google/mlperf_35b_internal_args.sh" ]]; then
  # shellcheck source=/dev/null
  source "${DIR}/google/mlperf_35b_internal_args.sh"
fi

tunix_image="${TUNIX_IMAGE:-${tunix_image}}"
command="start"
launch_args=()
user_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --command|--image|--bucket|--maxtext-ckpt|--dataset-path|--pathways-server-image|--pathways-proxy-image)
      require_opt_arg "$@"
      set -- "$1=$2" "${@:3}"
      ;;
    start|stop|orchestrator|trainer|rollout|otel|stop-otel)
      command="$1"; launch_args+=("$1"); shift ;;
    --command=*)
      command="${1#*=}"; launch_args+=("$1"); shift ;;
    --dry-run|--render)
      launch_args+=("$1"); shift ;;
    -h|--help)
      usage; exit 0 ;;
    --image=*) tunix_image="${1#*=}"; shift ;;
    --bucket=*) bucket="${1#*=}"; shift ;;
    --maxtext-ckpt=*) maxtext_ckpt="${1#*=}"; shift ;;
    --dataset-path=*) dataset_path="${1#*=}"; shift ;;
    --pathways-server-image=*) pathways_server_image="${1#*=}"; shift ;;
    --pathways-proxy-image=*) pathways_proxy_image="${1#*=}"; shift ;;
    cluster_config.k8s.tunix_image=*)
      tunix_image="${1#*=}"; user_args+=("$1"); shift ;;
    actor_model_config.model_path=*)
      maxtext_ckpt="${1#*=}"; user_args+=("$1"); shift ;;
    data_config.dataset_path=*)
      dataset_path="${1#*=}"; user_args+=("$1"); shift ;;
    *)
      user_args+=("$1"); shift ;;
  esac
done

if [[ "${command}" != "stop" && "${command}" != "otel" && \
  "${command}" != "stop-otel" ]]; then
  missing_opts=()
  [[ -z "${tunix_image}" ]] && missing_opts+=("--image (or TUNIX_IMAGE)")
  [[ -z "${bucket}" ]] && missing_opts+=("--bucket")
  [[ -z "${maxtext_ckpt}" ]] && missing_opts+=("--maxtext-ckpt")
  [[ -z "${dataset_path}" ]] && missing_opts+=("--dataset-path")
  if (( ${#missing_opts[@]} > 0 )); then
    echo "Error: missing required option(s): ${missing_opts[*]}." >&2
    usage >&2
    exit 1
  fi
else
  maxtext_ckpt="${maxtext_ckpt:-unused}"
fi

job_prefix="${JOB_PREFIX:-${USER}}"
deepswe_pkg="tunix.experimental.examples.deepswe_dist"

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

maxtext_extra_flags=(
  remat_policy=custom use_gdn_kernel=true gdn_cp_mode=head gdn_chunk_size=64
  decoder_layer_input=offload context=remat gdn=remat gdn_conv=remat
  gdn_states=remat
  megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true
  use_gmm_v2_heuristic_tiling=true merge_gating_gmm=false
  use_ragged_sort=false use_custom_sort_vjp=false
  use_tokamax_splash=true use_splash_scheduler=true
  sa_block_q=1024 sa_block_kv=4096 sa_block_kv_compute=512
  sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=512
  sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3
  context_parallel_strategy=ring context_parallel_load_balance=false
  allow_split_physical_axes=true context_parallel_attention_load_balance=true
  num_vocab_tiling=16 use_iota_embed=false mu_dtype=float32 grad_dtype=float32
  checkpoint_storage_concurrent_gb=96
  checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false
  packing=True optimizer_memory_host_offload=true
  # The trainer has no keys for these; keep them equal to the rollout's.
  float32_gate_logits=true float32_logits=true
)

mlperf_args=(
  "override_config_file=${DIR}/mlperf_35b_128_v5p.yaml"
  cluster_config.run_id="${job_prefix}" cluster_config.debug=false
  cluster_config.k8s.orchestrator_id="${job_prefix}-orch"
  cluster_config.k8s.trainer_id="${job_prefix}-train"
  cluster_config.k8s.rollout_id="${job_prefix}-roll"
  cluster_config.k8s.namespace="${namespace}"
  cluster_config.k8s.kueue_queue_name="${kueue_queue_name}"
  cluster_config.k8s.service_account="${service_account}"
  cluster_config.k8s.priority_class="${priority_class}"
  cluster_config.k8s.cpu_machine="n2-standard-64"
  cluster_config.k8s.cpu_nodepool="${cpu_nodepool}"
  cluster_config.k8s.cpu_memory="240G"
  cluster_config.k8s.tunix_image="${tunix_image}"
  ${head_excluded_nodepools:+cluster_config.k8s.head_excluded_nodepools="${head_excluded_nodepools}"}
  ${pathways_server_image:+cluster_config.k8s.pathways_server_image="${pathways_server_image}"}
  ${pathways_proxy_image:+cluster_config.k8s.pathways_proxy_image="${pathways_proxy_image}"}

  rl_training_config.metrics_logging_options.project_name="trellis-deepswe"
  rl_training_config.metrics_logging_options.run_name="${job_prefix}-mlperf-35b"
  rl_training_config.metrics_logging_options.flush_every_n_steps=1
  dist_training_config.trajectory_logging=false
  cluster_config.k8s.jax_cache.local_dir="/tmp/jax_cache"
  cluster_config.k8s.pathways_proxy_memory="16G"
  cluster_config.k8s.pathways_proxy_memory_limit="160G"
  cluster_config.k8s.user_container_memory="260G"
  cluster_config.k8s.user_container_memory_limit="260G"
  cluster_config.k8s.pathways_worker_memory="100G"
  cluster_config.k8s.pathways_rm_memory="4G"

  model_config.model_name="Qwen3.5-35B-A3B"
  model_config.model_id="Qwen/Qwen3.5-35B-A3B"
  model_config.model_source="maxtext" model_config.remat_config="none"
  model_config.use_flash_attention=false
  model_config.kwargs.maxtext_model_name="qwen3.5-35b-a3b"
  tokenizer_config.tokenizer_path="Qwen/Qwen3.5-35B-A3B"

  cluster_config.k8s.trainer.jobset_yaml="jobset.pathways.yaml"
  cluster_config.k8s.trainer.tpu_slice="tpuv5:4x4x4"
  actor_model_config.model_path="${maxtext_ckpt}"
  actor_model_config.mesh.shape="(32,1,1)"
  actor_model_config.mesh.axis_names="('fsdp','tp','expert')"
  actor_model_config.kwargs.mesh_context=2
  actor_model_config.kwargs.attention="flash"
  actor_model_config.kwargs.base_num_kv_heads=2
  actor_model_config.kwargs.prefuse_moe_weights=true
  actor_model_config.kwargs.warmup_steps_fraction=0.0
  dist_training_config.precompile_trainer=true
  dist_training_config.maxtext_extra_flags="${maxtext_extra_flags[*]}"
  cluster_config.role_env.trainer.LIBTPU_INIT_ARGS="${trainer_libtpu_args}"

  cluster_config.k8s.rollout.jobset_yaml="jobset.tpu.yaml"
  cluster_config.k8s.rollout.tpu_slice="tpuv5:2x2x1"
  cluster_config.rollout_replicas=16 rollout_engine="vllm"
  rollout_model_config.mesh.shape="(1,1,4)"
  rollout_model_config.mesh.axis_names="('fsdp','tp','expert')"
  rollout_model_config.kwargs.attention="vllm_rpa"
  rollout_model_config.kwargs.float32_gate_logits=true
  rollout_model_config.kwargs.float32_logits=true
  vllm_config.skip_jax_precompile=false vllm_config.max_model_len=65536
  vllm_config.max_num_seqs=16 vllm_config.max_num_batched_tokens=2048
  vllm_config.async_scheduling=true vllm_config.hbm_utilization=0.9
  vllm_config.enable_prefix_caching=true vllm_config.return_routed_experts=true
  vllm_config.routed_experts_encode_mode="auto"
  vllm_config.routed_experts_num_experts=256
  rollout_config.eos_tokens="248046,248044"

  rl_training_config.max_seq_token_per_tpu=65536
  rl_training_config.max_segments_per_packed_row=16
  dist_training_config.segment_alignment_boundary=64
  rl_training_config.max_steps=50 batch_size=16
  rl_training_config.mini_batch_size=16 agentic_grpo_config.num_generations=16
  rl_training_config.train_micro_batch_size=64
  rl_training_config.compute_logps_micro_batch_size=1
  rl_training_config.compute_logps_chunk_size=512
  rollout_config.max_prompt_length=8192
  rollout_config.total_generation_steps=57344
  rollout_config.temperature=1.0 rollout_config.top_p=1.0 rollout_config.top_k=-1
  rl_training_config.actor_optimizer_config.learning_rate=1.0e-6
  rl_training_config.actor_optimizer_config.b1=0.9
  rl_training_config.actor_optimizer_config.b2=0.999
  rl_training_config.actor_optimizer_config.eps=1.0e-8
  rl_training_config.actor_optimizer_config.weight_decay=0.0
  rl_training_config.actor_optimizer_config.opt_chain_type=clip_by_global_norm
  rl_training_config.actor_optimizer_config.chain_kwargs.max_norm=0.125
  agentic_grpo_config.max_turns=30 agentic_grpo_config.off_policy_steps=1
  agentic_grpo_config.trajectory_group_order="trajectory_completion"
  agentic_grpo_config.max_version_span=2 agentic_grpo_config.beta=0.0
  agentic_grpo_config.epsilon=0.2 agentic_grpo_config.epsilon_high=0.28
  agentic_grpo_config.advantage_estimator="grpo-loo"
  agentic_grpo_config.loss_agg_mode="token-mean"
  agentic_grpo_config.sampler_is="token"
  agentic_grpo_config.sampler_is_threshold=null
  agentic_grpo_config.sampler_rs="geometric"
  agentic_grpo_config.sampler_rs_min=0.999
  agentic_grpo_config.sampler_rs_max=1.002
  agentic_grpo_config.seq_logprob_error_threshold=2.0
  agentic_grpo_config.use_rollout_logps=true
  agentic_grpo_config.overlong_filter=false

  weight_sync_config.mode="raiden" weight_sync_config.use_weight_converter=true
  weight_sync_config.prefuse_moe_weights=true
  weight_sync_config.in_flight_weight_updates=true
  weight_sync_config.verify_weights=false
  weight_sync_config.raiden_data_nics="eth0"
  rl_training_config.checkpointing_options.save_interval_steps=0
  rl_training_config.checkpointing_options.max_to_keep=35
  dist_training_config.ckpt_d2h_concurrent_gb=32
  cluster_config.k8s.enable_pathways_persistence=true

  data_config.dataset_path="${dataset_path}" data_config.split="train"
  data_config.shuffle=false data_config.seed=42
  data_config.cache_dir="/tmp/hf_datasets_cache"

  cluster_config.registry_module="${deepswe_pkg}.deepswe"
  env_class_path="deepswe_env" env_kwargs.max_steps=30
  env_kwargs.step_timeout=60 env_kwargs.reward_timeout=60
  env_kwargs.backend="kubernetes-sandbox" env_kwargs.use_agent_sandbox=true
  env_kwargs.scaffold="openhands"
  agent_class_path="codeact_agent" agent_kwargs.scaffold="openhands"
  agentic_grpo_config.max_concurrency=256
  agentic_grpo_config.episode_timeout=1800
  kubernetes_config.node_selector_key="${sandbox_node_selector_key}"
  kubernetes_config.node_selector_val="${sandbox_node_selector_val}"
  kubernetes_config.image_rewrite_prefix="${image_rewrite_prefix}"
  prompt_items_wrapper="${deepswe_pkg}.prewarm.prewarm_sandboxes"
  prompt_items_wrapper_kwargs.max_concurrency=1024
  prompt_items_wrapper_kwargs.max_warmpool_replicas=16
  cluster_config.rpc_timeout_s=10800 cluster_config.init_timeout_s=7200
)

for entry in "RUN_ID='${job_prefix}'" \
  ${openhands_server_image:+"OPENHANDS_SERVER_IMAGE='${openhands_server_image}'"}; do
  mlperf_args+=(
    "cluster_config.role_env.orchestrator.${entry}"
    "cluster_config.role_env.rollout.${entry}"
  )
done

if [[ -n "${bucket}" ]]; then
  rollout_jax_cache_dir="${bucket}/jax_cache/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep4_tp1_8fb2c88823"
  mlperf_args+=(
    rl_training_config.metrics_logging_options.log_dir="${bucket}/logs/${job_prefix}"
    trajectory_store_config.enable=true trajectory_store_config.backend="file"
    trajectory_store_config.dir="${bucket}/trajectories/${job_prefix}/store"
    rl_training_config.checkpoint_root_directory="${bucket}/checkpoints/${job_prefix}"
    actor_model_config.kwargs.maxtext_output_dir="${bucket}/maxtext/${job_prefix}"
    cluster_config.k8s.gcs_scratch_location="${bucket}/pathways-staging/${job_prefix}"
    cluster_config.k8s.jax_cache.rollout_gcs_dir="${rollout_jax_cache_dir}"
    cluster_config.k8s.jax_cache.save=true
  )
fi

DEEPSWE_SKIP_INTERNAL_ARGS=1 \
  exec "${DIR}/../../../dist_rl/examples/deepswe_dist/k8s_launcher_v2.sh" \
  "${launch_args[@]}" "${mlperf_args[@]}" "${user_args[@]}"
