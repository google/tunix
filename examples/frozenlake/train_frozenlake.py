import argparse
import json
import logging
import os
import sys
from absl import logging as absl_logging
import datasets as datasets_lib
from datasets import load_dataset
from flax import nnx
import grain
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix
from transformers import AutoTokenizer
from tunix.cli.utils import data as data_lib

Dataset = datasets_lib.Dataset
# ====== Logging Configuration ======
# 1. Force absl to use python logging
absl_logging.use_python_logging()

# 2. Configure the root logger
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# 3. Explicitly set levels for relevant loggers
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.INFO)

# 4. Set absl verbosity to INFO so they actually print
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")

import faulthandler

# Enable the fault handler to dump tracebacks on crashes
faulthandler.enable()

# ==========================================
# 0. Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(description="Frozenlake Training")

# General Config
parser.add_argument("--models_base_dir", type=str, default="models")
parser.add_argument("--seed", type=int, default=42)

# Data & Training Flow
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--mini_batch_size", type=int, default=1)
parser.add_argument("--train_micro_batch_size", type=int, default=1)
parser.add_argument("--num_batches", type=int, default=20)
parser.add_argument("--num_test_batches", type=int, default=50)
parser.add_argument("--train_fraction", type=float, default=1.0)
parser.add_argument("--eval_every_n_steps", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--enable_remat", type=bool, default=True)

# GRPO Config
parser.add_argument("--num_generations", type=int, default=2)
parser.add_argument("--num_iterations", type=int, default=1)
parser.add_argument("--beta", type=float, default=0.001)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--epsilon_high", type=float, default=0.28)
parser.add_argument("--max_turns", type=int, default=2)

# Rollout Config
parser.add_argument("--max_prompt_length", type=int, default=1024)
parser.add_argument("--max_response_length", type=int, default=2048)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument("--rollout_engine", type=str, default="vllm")
parser.add_argument("--vllm_utilization", type=float, default=0.4)

# Optimizer Config
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.99)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--max_grad_norm", type=float, default=0.1)
parser.add_argument("--warmup_ratio", type=float, default=0.1)

# Checkpointing
parser.add_argument("--ckpt_dir", type=str, default="/tmp/cp/deepswe_ckpt/01")
parser.add_argument("--max_to_keep", type=int, default=100)
parser.add_argument("--save_interval_steps", type=int, default=5)

args, _ = parser.parse_known_args()


try:
  import pathwaysutils
except ImportError as e:
  print(f"pathwaysutils not available: {e}")

if pathwaysutils is not None and os.getenv("JAX_PLATFORMS", None) == "proxy":
  pathwaysutils.initialize()

# %%
# ==========================================
# 2. Imports from Custom Modules
# ==========================================
from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.models.gemma4 import model as model_lib
from tunix.sft import utils as sft_utils
from tunix.sft import metrics_logger
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.parser.chat_template_parser import parser as template_parser
from tunix.rl.agentic.rewards.reward_types import RewardOutput
from examples.frozenlake.agent import (
    SYSTEM_PROMPT,
    MULTI_SHOT_SYSTEM_PROMPT,
    FrozenLakeAgent,
)
from examples.frozenlake.env import FrozenLakeEnv

# ==========================================
# 4. Model & Training Hyperparameters
# ==========================================
MODEL_PATH = "/mnt/disks/linchai-data/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"

print(f"Looking for local model at: {MODEL_PATH}...")

# ====== Data ======
TRAIN_FRACTION = args.train_fraction

# ====== Reproducibility ======
SEED = args.seed

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = args.max_prompt_length
MAX_RESPONSE_LENGTH = args.max_response_length
TEMPERATURE = args.temperature
TOP_P = args.top_p
TOP_K = args.top_k
NUM_GENERATIONS = args.num_generations  # This corresponds to `G` in Algorithm 1

# === other GRPO configs ===
NUM_ITERATIONS = args.num_iterations
BETA = args.beta
EPSILON = args.epsilon
EPSILON_HIGH = args.epsilon_high

# ====== Training ======
ENABLE_REMAT = args.enable_remat
BATCH_SIZE = args.batch_size
MINI_BATCH_SIZE = args.mini_batch_size
NUM_BATCHES = args.num_batches
NUM_TEST_BATCHES = args.num_test_batches
TRAIN_MICRO_BATCH_SIZE = args.train_micro_batch_size

EVAL_EVERY_N_STEPS = args.eval_every_n_steps
NUM_EPOCHS = args.num_epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)
KV_CACHE_SIZE = MAX_PROMPT_LENGTH + (
    MAX_RESPONSE_LENGTH
)
print(f"kv_cache_size (Capped): {KV_CACHE_SIZE}")
MAX_CONCURRENCY = 1
# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = args.learning_rate
B1 = args.b1
B2 = args.b2
WEIGHT_DECAY = args.weight_decay
WARMUP_STEPS = int(args.warmup_ratio * MAX_STEPS)
MAX_GRAD_NORM = args.max_grad_norm
# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = args.save_interval_steps
MAX_TO_KEEP = args.max_to_keep

# ====== Rollout ======
ROLLOUT_ENGINE = args.rollout_engine
CKPT_DIR = args.ckpt_dir

# Max number of sequences to be processed in parallel by vllm.
VLLM_MAX_NUM_SEQS =  NUM_GENERATIONS * BATCH_SIZE

VLLM_UTILIZATION = args.vllm_utilization


MAX_TURNS = args.max_turns

# Max number of tokens to be processed in parallel by vllm.
# Divide by 8 for on policy, 1 step off divide by 4

# VLLM_MAX_BATCHED_TOKENS = (VLLM_MAX_NUM_SEQS * KV_CACHE_SIZE) // 8
VLLM_MAX_BATCHED_TOKENS = 256
print(f"vllm_max_batched_tokens: {VLLM_MAX_BATCHED_TOKENS}")
# %%
# ==========================================
# 5. JAX Device & Mesh Setup
# ==========================================
import jax
import jax.numpy as jnp
from tunix.models.automodel import call_model_config

config = model_lib.ModelConfig.gemma4_e2b()

if ENABLE_REMAT:
  config.remat_config = model_lib.RematConfig.BLOCK


devices = jax.devices()
rollout_devices = np.array(devices[0:1]).reshape(1, 1)
reference_devices = np.array(devices[1:2]).reshape(1, 1)
train_devices = np.array(devices[2:3]).reshape(1,1)
rollout_mesh = Mesh(rollout_devices, axis_names=("fsdp", "tp"))
reference_mesh = Mesh(reference_devices, axis_names=("fsdp", "tp"))
train_mesh = Mesh(train_devices, axis_names=("fsdp", "tp"))

# %%
# ==========================================
# 6. Model Initialization
# ==========================================

gemma4_reference = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, config, mesh=reference_mesh, dtype=jnp.bfloat16
)

gemma4_actor = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, config, mesh=train_mesh, dtype=jnp.bfloat16
)
sft_utils.show_hbm_usage()

# %%
# ==========================================
# 7. Tokenizer & Parser
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True, trust_remote_code=True
)

chat_parser = template_parser.DefaultChatTemplateParser(tokenizer)


# %%
# ==========================================
# 8. Data Loading
# ==========================================
print("Loading Dataset...")

import pandas as pd
dataset_path = "/mnt/disks/linchai-data/tunix/examples/frozenlake/data/frozenlake"
train_dataset = Dataset.from_pandas(pd.read_parquet(os.path.join(dataset_path, "train.parquet")))
test_dataset = Dataset.from_pandas(pd.read_parquet(os.path.join(dataset_path, "test.parquet")))

train_dataset = grain.MapDataset.source(train_dataset)
test_dataset = grain.MapDataset.source(test_dataset)

def add_prompt(item):
    item["prompt"] = ""
    return item
train_dataset = train_dataset.map(add_prompt)
test_dataset = test_dataset.map(add_prompt)

print(f"train_dataset length: {len(train_dataset)}, First 10 items in train_dataset:")
for i in range(min(10, len(train_dataset))):
    print(train_dataset[i])

train_dataset, _ = data_lib.post_init_dataset(
    train_dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_batches=NUM_BATCHES,
    max_prompt_length=MAX_PROMPT_LENGTH,
    fraction=TRAIN_FRACTION,
    num_epochs=NUM_EPOCHS,
    prompt_key="prompt",
)

for item in train_dataset:
    print(item)

# %%
# ==========================================
# 9. Optimizer & Checkpointing
# ==========================================
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/grpo", flush_every_n_steps=2
)

optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)


# %%
# ==========================================
# 10. RL Cluster Setup
# ==========================================

base_rollout_dict = {
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "kv_cache_size": KV_CACHE_SIZE,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "eos_tokens": [tokenizer.encode("<|im_end|>")[0]],
    "return_logprobs": True,
    "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
}

sglang_jax_rollout_dict = {
    "rollout_sglang_jax_model_version": "google/gemma-4-E2B-it",
    "rollout_sglang_jax_mem_fraction_static": 0.9,
    "rollout_sglang_jax_init_with_random_weights": True,
    "rollout_sglang_jax_disable_radix_cache": False,
    "rollout_sglang_jax_enable_deterministic_sampling": False,
    "rollout_sglang_jax_chunked_prefill_size": 2048,
    "rollout_sglang_jax_max_running_requests": MAX_CONCURRENCY,
    "rollout_sglang_jax_page_size": 128,
}

vllm_rollout_dict = {
    "rollout_vllm_model_version": "google/gemma-4-E2B-it",
    "rollout_vllm_hbm_utilization": 0.6,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_init_with_random_weights": True,
    "rollout_vllm_async_scheduling": True,
    "tensor_parallel_size": rollout_mesh.shape["tp"],
    "data_parallel_size": rollout_mesh.shape["fsdp"],
    "rollout_vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
    "rollout_vllm_max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": True,
    },
}


if ROLLOUT_ENGINE == "sglang_jax":
  rollout_engine_config = base_rollout.RolloutConfig(
      **base_rollout_dict, **sglang_jax_rollout_dict
  )
elif ROLLOUT_ENGINE == "vllm":
  os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
  rollout_engine_config = base_rollout.RolloutConfig(
      **base_rollout_dict, **vllm_rollout_dict
  )
elif ROLLOUT_ENGINE == "vanilla":
  rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict)
else:
  raise ValueError(f"Unsupported rollout engine: {ROLLOUT_ENGINE}")

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: train_mesh,
        rl_cluster_lib.Role.REFERENCE: train_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
    },
    rollout_engine=ROLLOUT_ENGINE,
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=MINI_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=None,
        checkpointing_options=None,
    ),
    rollout_config=rollout_engine_config,
)
sft_utils.show_hbm_usage()

rl_cluster = rl_cluster_lib.RLCluster(
    actor=gemma4_actor,
    reference=gemma4_reference,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)


# %%
# ==========================================
# 11. Learner & Agent Setup
# ==========================================
use_multistep_prompt = False
grpo_config = agentic_grpo_learner.GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    max_response_length=MAX_RESPONSE_LENGTH,
    beta=BETA,
    epsilon=EPSILON,
    system_prompt=SYSTEM_PROMPT if not use_multistep_prompt else MULTI_SHOT_SYSTEM_PROMPT,
    max_concurrency=MAX_CONCURRENCY,
    epsilon_high=EPSILON_HIGH,
    off_policy_steps=0,
)


agentic_grpo_learner = agentic_grpo_learner.GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=None,
    agent_class=FrozenLakeAgent,
    agent_kwargs={},
    env_class=FrozenLakeEnv,
    env_kwargs={"max_steps": MAX_TURNS},
    algo_config=grpo_config,
    chat_parser=chat_parser,
)


print("Starting training...")
agentic_grpo_learner.train(train_dataset=train_dataset)


# %%
