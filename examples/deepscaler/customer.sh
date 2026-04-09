WANDB_PROJECT=tunix
WANDB_ENTITY=lancewang-google

export PROJECT="cloud-tpu-multipod-dev"
export ZONE="europe-west4-b"
export REGION="europe-west4"
export CLUSTER="lance-v5p-16"
export TPU_TYPE="v5p-16"
export NUM_SLICES=1

export WORKLOAD_NAME="lance-$(date +%m%d%H%M)"
export DOCKER_IMAGE="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/linchai-repo/tunix_base_image:new_head_vllm_seq_token_mean"

# export GCS_BUCKET="${PROJECT}-bucket-${REGION}"
export GCS_BUCKET="lancewang-dev-supercomputer-testing/tunix/customer_issue"
# gcloud storage buckets describe gs://${GCS_BUCKET} > /dev/null 2>&1 || \
# gcloud storage buckets create gs://${GCS_BUCKET} --location=${REGION} --project=${PROJECT}

export MESH_SHAPE="(8,1)" # Example 4x4 mesh for 16 devices
export MESH_AXES="('fsdp','tp')"
# export PER_DEVICE_BATCH_SIZE=8
export PER_DEVICE_BATCH_SIZE=128
export GLOBAL_BATCH_SIZE=128
export TRAIN_MICRO_BATCH_SIZE=8
export MINI_BATCH_SIZE=${GLOBAL_BATCH_SIZE}

python -m tunix.cli.grpo_main base_config.yaml \
model_config.model_name=Qwen3-4B-base \
model_config.model_id=Qwen/Qwen3-4B-base \
model_config.model_source=huggingface \
model_config.intermediate_ckpt_dir=gs://${GCS_BUCKET}/intermediate_ckpt/${WORKLOAD_NAME} \
"model_config.mesh.shape=${MESH_SHAPE}" \
"model_config.mesh.axis_names=${MESH_AXES}" \
model_config.rng_seed=42 \
actor_model_config.lora_config.enabled=True \
actor_model_config.lora_config.rank=64 \
actor_model_config.lora_config.alpha=64.0 \
"actor_model_config.lora_config.module_path=.*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj" \
"actor_model_config.mesh.shape=${MESH_SHAPE}" \
"actor_model_config.mesh.axis_names=${MESH_AXES}" \
"rollout_model_config.mesh.shape=${MESH_SHAPE}" \
"rollout_model_config.mesh.axis_names=${MESH_AXES}" \
actor_model_config.model_name=Qwen3-4B-base \
actor_model_config.model_id=Qwen/Qwen3-4B-base \
rollout_model_config.model_name=Qwen3-4B-base \
rollout_model_config.model_id=Qwen/Qwen3-4B-base \
reference_model_config.model_name=Qwen3-4B-base \
reference_model_config.model_id=Qwen/Qwen3-4B-base \
tokenizer_config.tokenizer_path=Qwen/Qwen3-4B-base \
tokenizer_config.tokenizer_type=huggingface \
tokenizer_config.add_bos=false \
data_source=huggingface \
dataset_name=nvidia/OpenMathInstruct-2 \
batch_size=${PER_DEVICE_BATCH_SIZE} \
num_batches=5000 \
num_train_epochs=1 \
num_test_batches=10 \
rl_training_config.mini_batch_size=${MINI_BATCH_SIZE} \
rl_training_config.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
rl_training_config.actor_optimizer_config.opt_type=adamw \
rl_training_config.actor_optimizer_config.peak_value=3e-6 \
rl_training_config.actor_optimizer_config.schedule_type=warmup_cosine_decay_schedule \
rl_training_config.actor_optimizer_config.init_value=0.0 \
rl_training_config.actor_optimizer_config.end_value=0.0 \
rl_training_config.actor_optimizer_config.warmup_ratio=0.1 \
rl_training_config.actor_optimizer_config.warmup_steps=200 \
rl_training_config.actor_optimizer_config.decay_steps=5000 \
rl_training_config.actor_optimizer_config.b1=0.9 \
rl_training_config.actor_optimizer_config.b2=0.99 \
rl_training_config.actor_optimizer_config.weight_decay=0.1 \
rl_training_config.actor_optimizer_config.max_grad_norm=1.0 \
rl_training_config.eval_every_n_steps=500 \
rl_training_config.max_steps=5000 \
rl_training_config.metrics_logging_options.log_dir=gs://${GCS_BUCKET}/tensorboard/${WORKLOAD_NAME} \
rl_training_config.metrics_logging_options.run_name="${WORKLOAD_NAME}" \
rl_training_config.metrics_logging_options.flush_every_n_steps=20 \
rl_training_config.checkpointing_options.root_dir=gs://${GCS_BUCKET}/checkpoints/${WORKLOAD_NAME} \
rl_training_config.checkpointing_options.save_interval_steps=1000 \
rl_training_config.checkpointing_options.max_to_keep=4 \
"rl_training_config.profiler_options={}" \
rollout_config.total_generation_steps=512 \
rollout_config.max_prompt_length=512 \
rollout_config.temperature=0.9 \
rollout_config.top_p=1.0 \
rollout_config.top_k=50 \
rollout_engine=vanilla \
offload_to_cpu=false \
grpo_config.num_generations=4 \
grpo_config.num_iterations=1 \
grpo_config.beta=0.15 \
grpo_config.epsilon=0.2 \
reward_functions=['tunix/cli/reward_fn/openmath_util_reward.py'] \
rl_training_config.metrics_logging_options.use_wandb=True \
rl_training_config.metrics_logging_options.wandb_entity=${WANDB_ENTITY} \
rl_training_config.metrics_logging_options.wandb_project=${WANDB_PROJECT}

