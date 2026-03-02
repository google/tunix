# %%
# [WIP] Reproduction of [DeepSWE](https://www.together.ai/blog/deepswe)
# with Multi-turn Agentic framework.

# %%
import os
import sys
import datasets
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import json
from kubernetes import client, config as k8s_config
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix
import transformers
from tunix.cli.utils import data as data_lib
from tunix.utils import compat

Dataset = datasets.Dataset

# %%
# ==========================================
# 1. Path Setup
# ==========================================
# Use the absolute path to the ROOT folder
pathways_root = os.path.expanduser("~/pathways-utils")
r2egym_root = os.path.expanduser("~/r2egym")

for root in [pathways_root, r2egym_root]:
  if root not in sys.path:
    sys.path.insert(0, root)

# Verification
try:
  import pathwaysutils  # type: ignore
  import r2egym  # type: ignore

  print("✅ pathways-utils, r2egym are successfully mapped.")
except ImportError as e:
  pathwaysutils = None
  r2egym = None
  print(f"❌ Still missing a module: {e}")

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
from tunix.rl.experimental import agentic_grpo_learner
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.agentic.rewards.reward_types import RewardOutput
from tunix.oss.examples.deepswe.swe_agent import (
    SWE_SYSTEM_PROMPT,
    SWE_SYSTEM_PROMPT_FN_CALL,
    SWE_USER_PROMPT,
    SWE_USER_PROMPT_FN_CALL,
    SWEAGENT_SYSTEM_PROMPT,
    SWEAGENT_USER_PROMPT,
)

# Assumed custom imports based on usage
from tunix.oss.examples.deepswe.swe_agent import SWEAgent
from tunix.oss.examples.deepswe.swe_env import SWEEnv

# %%
# ==========================================
# 3. Environment Configuration
# ==========================================
DATASET_CACHE = os.getenv(
    "DATASET_CACHE", "/home/sizhi_google_com/dataset_cache"
)
os.makedirs(DATASET_CACHE, exist_ok=True)

os.environ["KUBECONFIG"] = "~/.kube/config"
os.environ["NODE_SELECTOR_KEY"] = "cloud.google.com/gke-nodepool"
os.environ["NODE_SELECTOR_VAL"] = (
    "deepswe-worker-pool"  # NB: change based on your node pool name
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
# MODEL_PATH = "/scratch/models/DeepSeek-R1-Distill-Qwen-1.5B/"
# MODEL_PATH = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
MODEL_PATH = os.path.expanduser("~/models/Qwen3-1.7B/")

# ====== Data ======
TRAIN_FRACTION = 1.0

# ====== Reproducibility ======
SEED = 42

# ====== LoRA ======
RANK = 64
ALPHA = 64.0
TRAIN_WITH_LORA = False

# ====== Sharding ======
# MESH = [(4, 2), ("fsdp", "tp")]


# ====== GRPO ======
# === Generation during GRPO training ===
# MAX_PROMPT_LENGTH = 32768
MAX_PROMPT_LENGTH = 4096
MAX_RESPONSE_LENGTH = 512
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
NUM_GENERATIONS = 2  # This corresponds to `G` in Algorithm 1

# === other GRPO configs ===
NUM_ITERATIONS = 1
BETA = 0.001
EPSILON = 0.2

# ====== Training ======
BATCH_SIZE = 16
MINI_BATCH_SIZE = 16
# ROLLOUT_MICRO_BATCH_SIZE = 8
# LOGPS_MICRO_BATCH_SIZE = 8
NUM_BATCHES = 1
NUM_TEST_BATCHES = 50

EVAL_EVERY_N_STEPS = 10
NUM_EPOCHS = 100

# Number of training steps.
MAX_STEPS = 10

# Max turns in mult-agent interaction (set to 1 for single-turn)
MAX_TURNS = 3

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 1e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = int(0.1 * MAX_STEPS)
MAX_GRAD_NORM = 0.1

# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4
DO_MEM_PROFILING = False

# ====== Inference ======
GENERATION_CONFIGS = {
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# ====== Rollout ======
ROLLOUT_ENGINE = "vanilla"  # one of "vanilla", "vllm" or "sglang_jax"
CKPT_DIR = os.path.join("/tmp/cp", "deepswe_ckpt/00")


# %%
# ==========================================
# 5. JAX Device & Mesh Setup
# ==========================================
import jax
import jax.numpy as jnp

devices = jax.devices()
split = int(len(devices) / 2)
rollout_devices = np.array(devices[:split]).reshape(2, 2)
train_devices = np.array(devices[split:]).reshape(2, 2)

rollout_mesh = Mesh(rollout_devices, axis_names=("fsdp", "tp"))
train_mesh = Mesh(train_devices, axis_names=("fsdp", "tp"))

# %%
# ==========================================
# 6. Model Initialization
# ==========================================
print("Initializing Model...")
config = model_lib.ModelConfig.qwen3_1p7b()


qwen_reference = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, config, mesh=train_mesh, dtype=jnp.bfloat16
)


def get_lora_model(base_model, model_mesh):
  lora_provider = qwix.LoraProvider(
      module_path=(
          ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
          ".*attn_vec_einsum"
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


qwen_actor = get_lora_model(qwen_reference, train_mesh)
sft_utils.show_hbm_usage()

# %%
# ==========================================
# 7. Tokenizer & Parser
# ==========================================
tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True, trust_remote_code=True
)

chat_parser = parser.QwenChatTemplateParser(tokenizer)


# %%
# ==========================================
# 8. Data Loading
# ==========================================
print("Loading Dataset...")

dataset = datasets.load_dataset(
    "R2E-Gym/R2E-Gym-V1", split="train", cache_dir=DATASET_CACHE
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
        train_micro_batch_size=1,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_tokens=[tokenizer.encode("<|im_end|>")[0]],
    ),
)

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
    beta=BETA,
    epsilon=EPSILON,
    system_prompt=SWE_SYSTEM_PROMPT,
    max_concurrency=1,
    epsilon_high=0.28,
    off_policy_steps=0,
)


# Helper for dummy reward function (placeholder)
def dummy_reward_fn(prompts, completions, **kwargs):
  return 0


agentic_grpo_learner = agentic_grpo_learner.GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=dummy_reward_fn,
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
import grain

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
