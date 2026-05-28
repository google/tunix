# Copyright 2025 Google LLC
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


set -x # Enable xtrace

# specify at cmd line to override defaults, e.g.
model_name=${model_name:-"Qwen3-1.7B-base"}
batch_size=${batch_size:-8}
train_micro_batch_size=${train_micro_batch_size:-8}
compute_logps_micro_batch_size=${compute_logps_micro_batch_size:-8}
rollout_micro_batch_size=${rollout_micro_batch_size:-8}
num_generations=${num_generations:-4}
num_train_epochs=${num_train_epochs:-1}
warmup_ratio=${warmup_ratio:-0.1}
train_fraction=${train_fraction:-0.8}
actor_mesh_shape=${actor_mesh_shape:-"(4,1)"}
rollout_mesh_shape=${rollout_mesh_shape:-"(4,1)"}
checkpoint_dir=${checkpoint_dir-"/tmp/grpo_checkpoints/${model_name}"}
vllm_max_num_seqs=${vllm_max_num_seqs:-$((batch_size * num_generations))}
vllm_max_num_batched_tokens=${vllm_max_num_batched_tokens:-$((vllm_max_num_seqs * 256))}

if [[ -n "$checkpoint_dir" ]]; then
  checkpoint_args=(
    rl_training_config.checkpoint_root_directory=${checkpoint_dir}
    rl_training_config.checkpointing_options.save_interval_steps=500
    rl_training_config.checkpointing_options.max_to_keep=4
  )
else
  checkpoint_args=(
    rl_training_config.checkpoint_root_directory=null
    rl_training_config.checkpointing_options=null
  )
fi

# 128, 32768
# 256, 65536
echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Num Epochs: $num_train_epochs"
echo "  Warmup Ratio: $warmup_ratio"
echo "  Train Fraction: $train_fraction"
echo "  Actor Mesh Shape: $actor_mesh_shape"
echo "  Rollout Mesh Shape: $rollout_mesh_shape"
echo "  Checkpoint Directory: $checkpoint_args"
echo "  vLLM Max Num Seqs: $vllm_max_num_seqs"
echo "  vLLM Max Num Batched Tokens: $vllm_max_num_batched_tokens"

grpo_args=(
  base_config.yaml
  model_config.model_name=${model_name}
  model_config.model_id=Qwen/${model_name}
  model_config.model_source=huggingface
  model_config.use_flash_attention=true
  model_config.flash_attention_block_size=256
  model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/${model_name}"
  model_config.rng_seed=42
  actor_model_config.mesh.shape=${actor_mesh_shape}
  actor_model_config.mesh.axis_names="('fsdp','tp')"
  reference_model_config.mesh=null
  reference_model_config.same_mesh_as="actor"
  rollout_model_config.mesh.shape=${rollout_mesh_shape}
  rollout_model_config.mesh.axis_names="('fsdp','tp')"
  tokenizer_config.tokenizer_path=Qwen/${model_name}
  tokenizer_config.tokenizer_type=huggingface
  tokenizer_config.add_bos=false
  dataset_name="gsm8k"
  batch_size=$batch_size
  num_test_batches=100
  num_train_epochs=$num_train_epochs
  rl_training_config.actor_optimizer_config.opt_type="adamw"
  rl_training_config.actor_optimizer_config.peak_value=3e-6
  rl_training_config.actor_optimizer_config.schedule_type="warmup_cosine_decay_schedule"
  rl_training_config.actor_optimizer_config.init_value=0.0
  rl_training_config.actor_optimizer_config.end_value=0.0
  rl_training_config.actor_optimizer_config.warmup_ratio=$warmup_ratio
  rl_training_config.actor_optimizer_config.b1=0.9
  rl_training_config.actor_optimizer_config.b2=0.99
  rl_training_config.actor_optimizer_config.weight_decay=0.1
  rl_training_config.actor_optimizer_config.max_grad_norm=0.1
  rl_training_config.eval_every_n_steps=10
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/${model_name}"
  rl_training_config.metrics_logging_options.flush_every_n_steps=20
  rl_training_config.train_micro_batch_size=$train_micro_batch_size
  rl_training_config.mini_batch_size=$batch_size
  rl_training_config.compute_logps_micro_batch_size=$compute_logps_micro_batch_size
  rl_training_config.rollout_micro_batch_size=$rollout_micro_batch_size
  "${checkpoint_args[@]}"
  rl_training_config.perf_metrics_options.enable_perf_v1=true
  rollout_config.total_generation_steps=768
  rollout_config.max_prompt_length=256
  rollout_config.temperature=0.9
  rollout_config.top_p=1.0
  rollout_config.top_k=50
  rollout_engine="vllm"
  vllm_config.server_mode=false
  vllm_config.async_scheduling=true
  vllm_config.max_num_seqs=${vllm_max_num_seqs}
  vllm_config.max_num_batched_tokens=${vllm_max_num_batched_tokens}
  offload_to_cpu=false
  grpo_config.num_generations=$num_generations
  grpo_config.num_iterations=1
  grpo_config.beta=0.08
  grpo_config.epsilon=0.2
  reward_functions="['tunix/cli/reward_fn/gsm8k.py']"
)

python3 -m tunix.cli.grpo_main "${grpo_args[@]}"
