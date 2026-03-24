# %%
# Simplified version of DeepSWE Training Benchmark
# Using Mock Rollout and GSM8K dataset.

# %%
import argparse
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
from jax.sharding import Mesh
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix
from transformers import AutoTokenizer
from tunix.cli.utils import data as data_lib
from tunix.utils import compat

# ====== Logging Configuration ======
absl_logging.use_python_logging()
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.INFO)
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")

# ==========================================
# 0. Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(
    description="DeepSWE Training with Mock Rollout (Simplified)"
)

# General Config
parser.add_argument("--models_base_dir", type=str, default="models")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_version", type=str, default="Qwen3-0.6B")

# Data & Training Flow
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--mini_batch_size", type=int, default=1)
parser.add_argument("--num_batches", type=int, default=20)
parser.add_argument("--train_fraction", type=float, default=1.0)
parser.add_argument("--max_steps", type=int, default=10)
parser.add_argument("--eval_every_n_steps", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--enable_remat", type=bool, default=True)

# GRPO Config
parser.add_argument("--num_generations", type=int, default=2)
parser.add_argument("--num_iterations", type=int, default=1)
parser.add_argument("--beta", type=float, default=0.001)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--epsilon_high", type=float, default=0.28)

# Rollout Config
parser.add_argument("--max_prompt_length", type=int, default=1024)
parser.add_argument("--max_response_length", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--min_generation_time", type=float, default=0.1)
parser.add_argument("--max_generation_time", type=float, default=1.0)

# Optimizer Config
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--max_grad_norm", type=float, default=0.1)
parser.add_argument("--warmup_ratio", type=float, default=0.1)

# DeepSWE Agentic Specifics
parser.add_argument("--max_turns", type=int, default=5)
parser.add_argument("--per_turn_timeout_secs", type=int, default=60)
parser.add_argument("--max_concurrency", type=int, default=1)

args, _ = parser.parse_known_args()
MODEL_VERSION = args.model_version
MIN_ENV_WAIT = 0.1
MAX_ENV_WAIT = 0.5
MIN_GENERATION_TIME = args.min_generation_time
MAX_GENERATION_TIME = args.max_generation_time

# ==========================================
# 1. Path Setup
# ==========================================
workdir = os.getcwd()
tunix_root = os.path.join(workdir, "tunix")
pathways_root = os.path.join(workdir, "pathways-utils")

for root in [workdir, tunix_root, pathways_root]:
  if root not in sys.path:
    sys.path.insert(0, root)

# ==========================================
# 2. Imports from Custom Modules
# ==========================================
import time
import random
from tunix.models.qwen3 import params as params_lib
from tunix.models.qwen3 import model as model_lib
from tunix.sft import utils as sft_utils
from tunix.sft import metrics_logger
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.parser.chat_template_parser import parser as template_parser
from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult
from tunix.rl.agentic.agents.base_agent import ConversationAgentBase
from tunix.rl.agentic.agents.agent_types import Action, Step
from tunix import PerfMetricsConfig
from tunix.perf.experimental.export import PerfMetricsExport


class MockEnv(BaseTaskEnv):

  def __init__(self, entry: dict[str, str], max_steps: int, **kwargs):
    self.entry = entry
    super().__init__(max_steps=max_steps, **kwargs)

  def _initial_observation(self) -> Any:
    return self.entry.get("question", "Initial prompt.")

  def _step_impl(self, action: Any) -> EnvStepResult:
    time.sleep(random.uniform(MIN_ENV_WAIT, MAX_ENV_WAIT))
    done = self.step_count >= self.max_steps
    reward = 1.0 if not done else 0.0
    return EnvStepResult(
        observation=f"Observation after step {self.step_count}",
        reward=reward,
        done=done,
        info={"max_steps": self.max_steps},
    )


class MockAgent(ConversationAgentBase):

  def __init__(self, system_prompt: str):
    super().__init__(system_prompt=system_prompt)
    self.step = 0

  def _observation_to_messages(self, observation, reward, done, info):
    self._messages.append({"role": "user", "content": observation})
    step = self.get_current_step()
    if step:
      step.observation = observation

  def update_from_model(self, response, **kwargs):
    action_obj = Action(action=f"Model action: {response}")
    step = Step(model_response=response, action=action_obj)
    self._trajectory.steps.append(step)
    self._messages.append({"role": "assistant", "content": response})
    self.step += 1
    return action_obj


# ==========================================
# 4. Model & Training Hyperparameters
# ==========================================
MODELS_BASE_DIR = os.path.join(workdir, args.models_base_dir)
MODEL_PATH = os.path.join(MODELS_BASE_DIR, MODEL_VERSION)

if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
  os.makedirs(MODEL_PATH, exist_ok=True)
  snapshot_download(
      repo_id=f"Qwen/{MODEL_VERSION}",
      local_dir=MODEL_PATH,
      local_dir_use_symlinks=False,
  )

SEED = args.seed
MAX_PROMPT_LENGTH = args.max_prompt_length
MAX_RESPONSE_LENGTH = args.max_response_length
TEMPERATURE = args.temperature
NUM_GENERATIONS = args.num_generations
NUM_ITERATIONS = args.num_iterations
BETA = args.beta
EPSILON = args.epsilon
EPSILON_HIGH = args.epsilon_high

MODEL_DTYPE = jnp.bfloat16
ENABLE_REMAT = args.enable_remat
BATCH_SIZE = args.batch_size
MINI_BATCH_SIZE = args.mini_batch_size
NUM_BATCHES = args.num_batches

MAX_STEPS = args.max_steps
MAX_TURNS = args.max_turns
PER_TURN_TIMEOUT_SECS = args.per_turn_timeout_secs
MAX_CONCURRENCY = args.max_concurrency

KV_CACHE_SIZE = MAX_PROMPT_LENGTH + (MAX_RESPONSE_LENGTH * 2 * MAX_TURNS)

LEARNING_RATE = args.learning_rate
WARMUP_STEPS = int(args.warmup_ratio * MAX_STEPS)
MAX_GRAD_NORM = args.max_grad_norm

# ==========================================
# 5. JAX Device & Mesh Setup
# ==========================================
from tunix.models.automodel import call_model_config

config = call_model_config(MODEL_VERSION)
if ENABLE_REMAT:
  config.remat_config = model_lib.RematConfig.BLOCK

devices = jax.devices()
split = int(len(devices) / 2)

rollout_tp = np.gcd(split, config.num_kv_heads)
rollout_fsdp = split // rollout_tp
rollout_devices = np.array(devices[:split]).reshape(rollout_fsdp, rollout_tp)

train_fsdp = np.gcd(split, BATCH_SIZE * NUM_GENERATIONS)
train_tp = split // train_fsdp
train_devices = np.array(devices[split:]).reshape(train_fsdp, train_tp)

rollout_mesh = Mesh(rollout_devices, axis_names=("fsdp", "tp"))
train_mesh = Mesh(train_devices, axis_names=("fsdp", "tp"))

# ==========================================
# 6. Model Initialization
# ==========================================
qwen_reference = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, config, mesh=train_mesh, dtype=MODEL_DTYPE
)

graph_def, params = nnx.split(qwen_reference)
qwen_actor = nnx.merge(graph_def, jax.tree.map(jnp.copy, params))

# ==========================================
# 7. Tokenizer & Parser
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True, trust_remote_code=True
)
chat_parser = template_parser.QwenChatTemplateParser(tokenizer)

# ==========================================
# 8. Data Loading (GSM8K)
# ==========================================
print("Loading GSM8K Dataset...")
dataset = load_dataset("gsm8k", "main", split="train")

# ==========================================
# 9. Optimizer & Checkpointing
# ==========================================
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=0.9,
    b2=0.99,
    weight_decay=args.weight_decay,
)

# ==========================================
# 10. RL Cluster Setup (Hardcoded to Mock)
# ==========================================
rollout_engine_config = base_rollout.RolloutConfig(
    max_prompt_length=MAX_PROMPT_LENGTH,
    kv_cache_size=KV_CACHE_SIZE,
    temperature=TEMPERATURE,
    eos_tokens=[tokenizer.encode("<|im_end|>")[0]],
    return_logprobs=True,
    max_tokens_to_generate=MAX_RESPONSE_LENGTH,
    rollout_mock_min_generation_time=MIN_GENERATION_TIME,
    rollout_mock_max_generation_time=MAX_GENERATION_TIME,
)

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: train_mesh,
        rl_cluster_lib.Role.REFERENCE: train_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
    },
    rollout_engine="mock",
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        max_steps=MAX_STEPS,
        mini_batch_size=MINI_BATCH_SIZE,
        train_micro_batch_size=1,
        compute_logps_micro_batch_size=1,
        rollout_micro_batch_size=1,
        metrics_logging_options=metrics_logger.MetricsLoggerOptions(
            log_dir="/tmp/tensorboard/grpo"
        ),
    ),
    rollout_config=rollout_engine_config,
)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=qwen_actor,
    reference=qwen_reference,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

# ==========================================
# 11. Learner Setup
# ==========================================
grpo_config = agentic_grpo_learner.GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    max_response_length=MAX_RESPONSE_LENGTH,
    beta=BETA,
    epsilon=EPSILON,
    system_prompt="System prompt.",
    max_concurrency=MAX_CONCURRENCY,
    epsilon_high=EPSILON_HIGH,
    episode_timeout=PER_TURN_TIMEOUT_SECS * MAX_TURNS,
)


def dummy_reward_fn(prompts, completions, **kwargs):
  return [0.0] * len(prompts)


grpo_learner = agentic_grpo_learner.GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=dummy_reward_fn,
    agent_class=MockAgent,
    agent_kwargs={"system_prompt": "System prompt."},
    env_class=MockEnv,
    env_kwargs={"max_steps": MAX_TURNS},
    algo_config=grpo_config,
    chat_parser=chat_parser,
)

# ==========================================
# 12. Start Training
# ==========================================
dataset = dataset.shuffle(seed=SEED)
grain_dataset = grain.MapDataset.source(dataset)

train_dataset, _ = data_lib.post_init_dataset(
    grain_dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_batches=args.num_batches,
    max_prompt_length=MAX_PROMPT_LENGTH,
    prompt_key="question",
)

print("Starting training with Mock Rollout...")
grpo_learner.train(train_dataset=train_dataset)
