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

# Fails fast on settings meant for the old per-example launchers, sourced by
# `launch.sh`.
#
# The old `examples/*/launcher.sh` and `examples/*/k8s_launcher.sh` scripts, and
# the recipes on top of them, read their settings from environment variables.
# `launch.sh` reads settings only from `override_config_file=` and
# `section.key=value` arguments, so it would silently ignore these variables;
# some (e.g. `DRY_RUN`, `RUN_ID`) are even overwritten by the resolver's
# exports. `fail_on_legacy_launcher_env` turns a set variable into an error.
#
# The list leaves out the variables the drivers still read from the caller's
# environment (`_ENV_PASSTHROUGH` in `launch_test.py`) and the ones read by the
# runtime itself: JAX, libtpu, TFDS, W&B, Python, `enter_kube_context.sh`, and
# the Tunix libraries (e.g. `VERIFY_WEIGHTS`, `CKPT_D2H_CONCURRENT_GB`).

LEGACY_LAUNCHER_ENV_VARS=(
  ADAM_B1
  ADAM_B2
  ADAM_EPS
  ADVANTAGE_ESTIMATOR
  ARTIFACT_ROOT
  BATCH_SIZE
  BETA
  CHAT_PARSER
  CHECKPOINT_MAX_TO_KEEP
  CHECKPOINT_ROOT_DIRECTORY
  CHECKPOINT_SAVE_INTERVAL_STEPS
  COMPUTE_LOGPS_CHUNK_SIZE
  COMPUTE_LOGPS_MICRO_BATCH_SIZE
  CPU_MACHINE
  DATASET_CACHE_DIR
  DATASET_NAME
  DATASET_PATH
  DATASET_SIZE
  DATASET_SPLIT
  DEBUG
  DRY_RUN
  ENABLE_PREFIX_CACHING
  ENABLE_TPU_PROFILING_OPTIONS
  ENV_BACKEND
  EOS_TOKENS
  EPISODE_TIMEOUT_SECS
  EPSILON
  EPSILON_HIGH
  EVAL_EVERY_N_STEPS
  EVAL_JAX_CACHE_GCS_DIR
  FLASH_ATTENTION_BLOCK_SIZE
  FLUSH_EVERY_N_STEPS
  FLUSH_METRICS_EVERY_N_STEPS
  FORCE_ARM_SECS
  GCS_SCRATCH_LOCATION
  GRID_SIZE_RANGE
  INFERENCE_ADDR
  INFERENCE_PORT
  INFERENCE_TPU_CHIPS
  IN_FLIGHT_WEIGHT_UPDATES
  IS_SLIPPERY
  KL_LOSS_MODE
  LEARNING_RATE
  LOCAL_JAX_CACHE_DIR
  LOG_DIR
  LORA_ALPHA
  LORA_RANK
  LOSS_AGG_MODE
  LOSS_ALGO
  LR_DECAY_STEPS
  LR_END_VALUE
  LR_INIT_VALUE
  LR_PEAK_VALUE
  MAXTEXT_ATTENTION
  MAXTEXT_CKPT
  MAXTEXT_OUTPUT_DIR
  MAX_GRAD_NORM
  MAX_PROMPT_LENGTH
  MAX_RESPONSE_LENGTH
  MAX_SEGMENTS_PER_PACKED_ROW
  MAX_SEQ_TOKEN_PER_TPU
  MAX_STALENESS
  MAX_STEPS
  MAX_TURNS
  MINI_BATCH_SIZE
  MODEL_DIR
  MODEL_DOWNLOAD_DIR
  MODEL_DTYPE
  MODEL_ID
  MODEL_LOAD_DTYPE
  MODEL_NAME
  NAMESPACE
  NUM_BATCHES
  NUM_EPOCHS
  NUM_GENERATIONS
  NUM_ITERATIONS
  OPT_CHAIN_TYPE
  ORCHESTRATOR_PORT
  ORCHESTRATOR_PROCESS_MAIN
  PATHWAYS_PROXY_IMAGE
  PATHWAYS_PROXY_MEMORY
  PATHWAYS_PROXY_MEMORY_LIMIT
  PATHWAYS_RM_MEMORY
  PATHWAYS_SERVER_IMAGE
  PATHWAYS_WORKER_MEMORY
  PREFUSE_MOE_WEIGHTS
  PROFILER_PERIOD
  PROFILER_STEPS
  RETURN_ROUTED_EXPERTS
  REWARD_MODE
  REWARD_TIMEOUT_SECS
  ROLLOUT_AGENT_CONFIG_JSON
  ROLLOUT_AGENT_NAME
  ROLLOUT_ENV_NAME
  ROLLOUT_FSDP
  ROLLOUT_ID
  ROLLOUT_JOBSET_YAML
  ROLLOUT_MAXTEXT_ATTENTION
  ROLLOUT_MAX_CONCURRENCY
  ROLLOUT_MESH_FSDP
  ROLLOUT_MESH_TP
  ROLLOUT_PORT
  ROLLOUT_PROCESS_MAIN
  ROLLOUT_REGISTRY_MODULE
  ROLLOUT_REPLICAS
  ROLLOUT_SKIP_JAX_PRECOMPILE
  ROLLOUT_TP
  ROLLOUT_TPU_CHIPS
  ROLLOUT_TPU_SLICE
  ROLLOUT_USE_BATCHED_RPA
  ROUTED_EXPERTS_ENCODE_MODE
  ROUTED_EXPERTS_NUM_EXPERTS
  RUN_ID
  RUN_INFERENCE_NODE
  SAMPLER
  SAMPLER_IS
  SAMPLER_IS_THRESHOLD
  SANDBOX_NAMESPACE
  SANDBOX_NODE_SELECTOR_KEY
  SANDBOX_NODE_SELECTOR_VAL
  SCAFFOLD
  SCHEDULE_TYPE
  SEED
  SEGMENT_ALIGNMENT_BOUNDARY
  SHUFFLE
  SHUTDOWN_GRACE_SECS
  SKIP_FIRST_N_PROFILER_STEPS
  STEP_TIMEOUT_SECS
  TEMPERATURE
  TOKENIZER_PATH
  TOP_K
  TOP_P
  TRAINER_BACKEND
  TRAINER_EXTRA_ENV
  TRAINER_FSDP
  TRAINER_ID
  TRAINER_JOBSET_YAML
  TRAINER_MESH_EXPERT
  TRAINER_MESH_FSDP
  TRAINER_MESH_TP
  TRAINER_PADDED_MOE_MLP_DIM
  TRAINER_PATHWAYS_LOCAL_INIT
  TRAINER_PORT
  TRAINER_PRECOMPILE
  TRAINER_PROCESS_MAIN
  TRAINER_SHUTDOWN_GRACE_SECS
  TRAINER_TP
  TRAINER_TPU_CHIPS
  TRAINER_TPU_SLICE
  TRAIN_MICRO_BATCH_SIZE
  TRAJECTORY_GROUP_ORDER
  TRAJECTORY_LOG_DIR
  TRAJECTORY_STORE_BACKEND
  TRAJECTORY_STORE_DIR
  TRAJECTORY_STORE_ENABLED
  UPLOAD_ALL_PROFILER_RESULTS
  USER_CONTAINER_MEMORY
  USER_CONTAINER_MEMORY_LIMIT
  USE_AGENT_SANDBOX
  USE_LORA
  USE_MULTISTEP_PROMPT
  USE_ROLLOUT_LOGPS
  USE_WEIGHT_CONVERTER
  VLLM_HBM_UTILIZATION
  VLLM_MAX_MODEL_LEN
  VLLM_MAX_NUM_BATCHED_TOKENS
  VLLM_MAX_NUM_SEQS
  WAIT_DEBUG_EVERY_POLLS
  WAIT_LOG_TAIL_LINES
  WAIT_POLL_SECS
  WAIT_TIMEOUT_SECS
  WARMUP_STEPS
  WEIGHT_DECAY
  WORKER_SHUTDOWN_GRACE_SECS
)

# Exits with status 1, naming every variable in `LEGACY_LAUNCHER_ENV_VARS` that
# is set to a non-empty value.
fail_on_legacy_launcher_env() {
  local name
  local -a found=()
  for name in "${LEGACY_LAUNCHER_ENV_VARS[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      found+=("${name}")
    fi
  done
  if (( ${#found[@]} > 0 )); then
    echo "Error: environment variables of the old example launchers are set:" \
      "${found[*]}." >&2
    echo "launch.sh does not read them. Unset them and pass the settings as" \
      "section.key=value arguments instead (e.g. MAX_STEPS=2 becomes" \
      "rl_training_config.max_steps=2); see base_dist_config.yaml and the" \
      "example YAML for the keys." >&2
    exit 1
  fi
}
