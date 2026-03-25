# %%
# [WIP] Reproduction of [DeepSWE](https://www.together.ai/blog/deepswe)
# with Multi-turn Agentic framework.

# %%
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
from kubernetes import client, config as k8s_config
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix
from transformers import AutoTokenizer
from tunix.cli.utils import data as data_lib
from tunix.utils import compat
import vllm  # pytype: disable=import-error

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

# ==========================================
# 0. Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(
    description="DeepSWE Training with Multi-turn Agentic Framework"
)

# General Config
parser.add_argument("--models_base_dir", type=str, default="models")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_version", type=str, default="Qwen3-32B")
parser.add_argument("--node_selector_val", type=str, default="deepswe-cpu-pool")

# Data & Training Flow
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--mini_batch_size", type=int, default=1)
parser.add_argument("--num_batches", type=int, default=20)
parser.add_argument("--num_test_batches", type=int, default=50)
parser.add_argument("--train_fraction", type=float, default=1.0)
parser.add_argument("--max_steps", type=int, default=10)
parser.add_argument("--eval_every_n_steps", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--enable_remat", type=bool, default=True)

# LoRA
# LoRA Config
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--alpha", type=float, default=64.0)
parser.add_argument("--train_with_lora", type=bool, default=False)

# GRPO Config
parser.add_argument("--num_generations", type=int, default=2)
parser.add_argument("--num_iterations", type=int, default=1)
parser.add_argument("--beta", type=float, default=0.001)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--epsilon_high", type=float, default=0.28)

# Rollout Config
parser.add_argument("--max_prompt_length", type=int, default=4096)
parser.add_argument("--max_response_length", type=int, default=8192)
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
parser.add_argument("--max_to_keep", type=int, default=4)
parser.add_argument("--save_interval_steps", type=int, default=500)

# Microbatch Sizes
parser.add_argument("--train_micro_batch_size", type=int, default=1)
parser.add_argument("--rollout_micro_batch_size", type=int, default=1)
parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)

# DeepSWE Agentic Specifics
parser.add_argument("--max_turns", type=int, default=20)
parser.add_argument("--per_turn_timeout_secs", type=int, default=300)
parser.add_argument("--max_concurrency", type=int, default=1)
parser.add_argument("--context_ratio", type=int, default=2)


# Other
parser.add_argument("--do_mem_profiling", type=bool, default=False)
parser.add_argument("--model_dtype", type=str, default="bfloat16",choices=["bfloat16", "float16", "float32"], # Restrict to valid inputs
    help="Data type for the model (e.g., bfloat16, float32)")

args, _ = parser.parse_known_args()
MODEL_VERSION = args.model_version
NODE_SELECTOR_VAL = args.node_selector_val

# %%
# ==========================================
# 1. Path Setup
# ==========================================
# Use the current working directory as ROOT folder
workdir = os.getcwd()
tunix_root = os.path.join(workdir, "tunix")
pathways_root = os.path.join(workdir, "pathways-utils")
r2egym_root = os.path.join(workdir, "r2egym")

for root in [workdir, tunix_root, pathways_root, r2egym_root]:
  if root not in sys.path:
    sys.path.insert(0, root)

# Verification
try:
  import tunix
  import pathwaysutils
  import r2egym  # pytype: disable=import-error

  print("✅ tunix pathways-utils, r2egym are successfully mapped.")
except ImportError as e:
  print(f"❌ Still missing a module: {e}")

if pathwaysutils is not None and os.getenv("JAX_PLATFORMS", None) == "proxy":
  pathwaysutils.initialize()

# %%
# ==========================================
# 2. Imports from Custom Modules
# ==========================================
from tunix.models.qwen3 import params as params_lib
from tunix.models.qwen3 import model as model_lib
from tunix.sft import utils as sft_utils
from tunix.sft import metrics_logger
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.parser.chat_template_parser import parser as template_parser
from tunix.rl.agentic.rewards.reward_types import RewardOutput
from examples.deepswe.swe_agent import (
    SWE_SYSTEM_PROMPT,
    SWE_SYSTEM_PROMPT_FN_CALL,
    SWE_USER_PROMPT,
    SWE_USER_PROMPT_FN_CALL,
    SWEAGENT_SYSTEM_PROMPT,
    SWEAGENT_USER_PROMPT,
)

# Assumed custom imports based on usage
from examples.deepswe.swe_agent import SWEAgent
from examples.deepswe.swe_env import SWEEnv

# %%
# ==========================================
# 3. Environment Configuration
# ==========================================
DATASET_CACHE = os.getenv(
    "DATASET_CACHE", os.path.join(workdir, "dataset_cache")
)
os.makedirs(DATASET_CACHE, exist_ok=True)

os.environ["KUBECONFIG"] = "~/.kube/config"
os.environ["NODE_SELECTOR_KEY"] = "cloud.google.com/gke-nodepool"
os.environ["NODE_SELECTOR_VAL"] = (
    NODE_SELECTOR_VAL  # NB: change based on your node pool name
)
print(
    "Using Kubernetes node selector:"
    f" {os.environ['NODE_SELECTOR_KEY']}={os.environ['NODE_SELECTOR_VAL']}"
)


# Kubernetes Setup
try:
  k8s_config.load_kube_config()
  k8s_client = client.CoreV1Api()
  # k8s_client.list_namespace(timeout_seconds=5)
except Exception as e:
  print(f"Warning: Kubernetes config loading failed: {e}")


# %%
# ==========================================
# 4. Model & Training Hyperparameters
# ==========================================
MODELS_BASE_DIR = os.path.join(workdir, args.models_base_dir)
MODEL_PATH = os.path.join(MODELS_BASE_DIR, MODEL_VERSION)

print(f"Looking for local model at: {MODEL_PATH}...")

# Check if directory exists and is not empty
if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
  print(f"Model not found locally. Starting download to {MODEL_PATH}...")
  os.makedirs(MODEL_PATH, exist_ok=True)

  # Assumes "Qwen/" organization prefix for HF download. Adjust if using other models.
  snapshot_download(
      repo_id=f"Qwen/{MODEL_VERSION}",
      local_dir=MODEL_PATH,
      local_dir_use_symlinks=False,
  )
  print("Download complete!")
else:
  print(f"✅ Found existing local model at {MODEL_PATH}")

# ====== Data ======
TRAIN_FRACTION = args.train_fraction

# ====== Reproducibility ======
SEED = args.seed

# ====== LoRA ======
RANK = args.rank
ALPHA = args.alpha
TRAIN_WITH_LORA = args.train_with_lora

# ====== Sharding ======
# MESH = [(4, 2), ("fsdp", "tp")]


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
DTYPE_MAP = {
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
    "int32": jnp.int32,
}
MODEL_DTYPE = DTYPE_MAP[args.model_dtype]
ENABLE_REMAT = args.enable_remat
BATCH_SIZE = args.batch_size
MINI_BATCH_SIZE = args.mini_batch_size
NUM_BATCHES = args.num_batches
NUM_TEST_BATCHES = args.num_test_batches

COMPUTE_LOGPS_MICRO_BATCH_SIZE = args.compute_logps_micro_batch_size
TRAIN_MICRO_BATCH_SIZE = args.train_micro_batch_size
ROLLOUT_MICRO_BATCH_SIZE = args.rollout_micro_batch_size

EVAL_EVERY_N_STEPS = args.eval_every_n_steps
NUM_EPOCHS = args.num_epochs

# Number of training steps.
MAX_STEPS = args.max_steps

# Max turns in mult-agent interaction (set to 1 for single-turn)
MAX_TURNS = args.max_turns
PER_TURN_TIMEOUT_SECS = args.per_turn_timeout_secs

MAX_CONCURRENCY = args.max_concurrency
CONTEXT_RATIO = args.context_ratio  # Context length can be up to 2x responselength in DeepSWE due to multi-turn interactions and long responses, so we set context ratio to 2 to accommodate this.
KV_CACHE_SIZE = MAX_PROMPT_LENGTH + (
    MAX_RESPONSE_LENGTH * CONTEXT_RATIO * MAX_TURNS
)
print(f"kv_cache_size (Capped): {KV_CACHE_SIZE}")
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
DO_MEM_PROFILING = args.do_mem_profiling

# ====== Rollout ======
ROLLOUT_ENGINE = args.rollout_engine
CKPT_DIR = args.ckpt_dir

# Max number of sequences to be processed in parallel by vllm.
VLLM_MAX_NUM_SEQS = ROLLOUT_MICRO_BATCH_SIZE * NUM_GENERATIONS

VLLM_UTILIZATION = args.vllm_utilization


# 2. Max number of sequences to be processed in parallel by vllm.
VLLM_MAX_NUM_SEQS = ROLLOUT_MICRO_BATCH_SIZE * NUM_GENERATIONS  # 1 * 2 = 2
# Max number of tokens to be processed in parallel by vllm.
# Divide by 8 for on policy, 1 step off divide by 4

VLLM_MAX_BATCHED_TOKENS = (VLLM_MAX_NUM_SEQS * KV_CACHE_SIZE) // 4
print(f"vllm_max_batched_tokens: {VLLM_MAX_BATCHED_TOKENS}")
# %%
# ==========================================
# 5. JAX Device & Mesh Setup
# ==========================================
import jax
import jax.numpy as jnp
from tunix.models.automodel import call_model_config

config = call_model_config(MODEL_VERSION)

if ENABLE_REMAT:
  config.remat_config = model_lib.RematConfig.BLOCK


devices = jax.devices()
split = int(len(devices) / 2)

# Favor TP for now.
# TODO(sizhi): Experiment with DP vs TP for rollout.
rollout_tp = np.gcd(split, config.num_kv_heads)
rollout_fsdp = split // rollout_tp
rollout_devices = np.array(devices[:split]).reshape(rollout_fsdp, rollout_tp)

train_fsdp = np.gcd(split, TRAIN_MICRO_BATCH_SIZE * NUM_GENERATIONS)
train_tp = split // train_fsdp
train_devices = np.array(devices[split:]).reshape(train_fsdp, train_tp)

rollout_mesh = Mesh(rollout_devices, axis_names=("fsdp", "tp"))
train_mesh = Mesh(train_devices, axis_names=("fsdp", "tp"))

# %%
# ==========================================
# 6. Model Initialization
# ==========================================

qwen_reference = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, config, mesh=train_mesh, dtype=MODEL_DTYPE
)


def get_lora_model(base_model, model_mesh):
  lora_provider = qwix.LoraProvider(
      module_path=(
          ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
          ".*gate_proj|.*down_proj|.*up_proj"
      ),
      rank=RANK,
      alpha=ALPHA,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  with compat.set_mesh(model_mesh):
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model


if TRAIN_WITH_LORA:
  qwen_actor = get_lora_model(qwen_reference, train_mesh)
else:
  graph_def, params = nnx.split(qwen_reference)
  qwen_actor = nnx.merge(
      graph_def,
      jax.tree.map(jnp.copy, params),
  )
sft_utils.show_hbm_usage()

# %%
# ==========================================
# 7. Tokenizer & Parser
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True, trust_remote_code=True
)

chat_parser = template_parser.QwenChatTemplateParser(tokenizer)


# %%
# ==========================================
# 8. Data Loading
# ==========================================
print("Loading Dataset...")

dataset = load_dataset(
    "R2E-Gym/R2E-Gym-V1",
    split="train",
    cache_dir=DATASET_CACHE,
    trust_remote_code=True,
)

def transform(entry):
  for k, v in entry.items():
    if isinstance(v, list):
      entry[k] = json.dumps(v)
  return entry

dataset = dataset.map(
    transform,
    keep_in_memory=True,
)

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
    "rollout_sglang_jax_model_version": MODEL_PATH,  # Uses local absolute path
    "rollout_sglang_jax_mem_fraction_static": 0.9,
    "rollout_sglang_jax_init_with_random_weights": True,
    "rollout_sglang_jax_disable_radix_cache": False,
    "rollout_sglang_jax_enable_deterministic_sampling": False,
    "rollout_sglang_jax_chunked_prefill_size": 2048,
    "rollout_sglang_jax_max_running_requests": MAX_CONCURRENCY,
    "rollout_sglang_jax_page_size": 128,
}

vllm_rollout_dict = {
    "rollout_vllm_model_version": MODEL_PATH,  # Uses local absolute path
    "rollout_vllm_hbm_utilization": 0.4,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
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
  # Currently, vllm does not support LoRA properly.
  if TRAIN_WITH_LORA:
    vllm_rollout_dict["rollout_vllm_lora_config"] = {
        "max_lora_rank": RANK,
    }
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
        compute_logps_micro_batch_size=COMPUTE_LOGPS_MICRO_BATCH_SIZE,
        rollout_micro_batch_size=ROLLOUT_MICRO_BATCH_SIZE,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=None,
        checkpointing_options=None,
    ),
    rollout_config=rollout_engine_config,
)
sft_utils.show_hbm_usage()

rl_cluster = rl_cluster_lib.RLCluster(
    actor=qwen_actor,
    reference=qwen_reference,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)


# %%
# ==========================================
# 11. Learner & Agent Setup
# ==========================================
grpo_config = agentic_grpo_learner.GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    max_response_length=MAX_RESPONSE_LENGTH,
    beta=None,  # no kl penalty
    epsilon=EPSILON,
    advantage_estimator="rloo",
    loss_agg_mode="sequence-mean-token-scale",
    policy_loss_fn="agentic_grpo",
    system_prompt=SWE_SYSTEM_PROMPT,
    max_concurrency=MAX_CONCURRENCY,
    epsilon_high=EPSILON_HIGH,
    compact_filter=True,
    filter_statuses={},  # use default filter statuses
    off_policy_steps=0,
    episode_timeout=PER_TURN_TIMEOUT_SECS * MAX_TURNS,
)


agentic_grpo_learner = agentic_grpo_learner.GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=None,
    agent_class=SWEAgent,
    agent_kwargs={},
    env_class=SWEEnv,
    env_kwargs={"max_steps": MAX_TURNS},
    algo_config=grpo_config,
    chat_parser=chat_parser,
)


# %%
# ==========================================
# 11. process dataset and start training
# ==========================================

dataset = dataset.shuffle(seed=SEED)
grain_dataset = grain.MapDataset.source(dataset)


def mixed_type_batch_fn(elements):
  """elements: A list of dicts."""
  batched_data = {}
  str_set = {
      "repo_name",
      "docker_image",
      "commit_hash",
      "parsed_commit_content",
      "execution_result_content",
  }
  dict_set = {"modified_files", "relevant_files", "modified_entity_summaries"}
  int_set = {
      "num_non_test_files",
      "num_non_test_func_methods",
      "num_non_test_lines",
      "prompt",
      "problem_statement",
      "expected_output_json",
  }
  keys = elements[0].keys()

  for key in keys:
    if key in str_set or key in dict_set:
      # Keep these as standard Python lists
      batched_data[key] = [item[key] for item in elements]

    elif key in int_set:
      # Convert these to NumPy arrays.
      # np.array() safely handles both single integers and lists of integers.
      batched_data[key] = np.array([item[key] for item in elements])

    else:
      # Fallback for any unexpected keys (defaulting to lists is usually safest)
      batched_data[key] = [item[key] for item in elements]

  return batched_data


train_dataset, _ = data_lib.post_init_dataset(
    grain_dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_batches=NUM_BATCHES,
    max_prompt_length=MAX_PROMPT_LENGTH,
    fraction=TRAIN_FRACTION,
    num_epochs=NUM_EPOCHS,
    prompt_key="problem_statement",
    custom_batch_fn=mixed_type_batch_fn,
)


print("Starting training...")
agentic_grpo_learner.train(train_dataset=train_dataset)


# %%
