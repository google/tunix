# %%

# [WIP] Reproduction of [Deepscaler](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) with Single-turn Agentic framework.

import contextlib
import functools
import json
import os

from flax import nnx
import grain
import jax
from jax import numpy as jnp
import optax
import qwix
from tqdm.auto import tqdm

import optax
from orbax import checkpoint as ocp

try:
  from etils import ecolab
  cm = ecolab.adhoc(
      source=ecolab.FROM_NOTEBOOK_OR_HEAD,
      reload='tunix',
      behavior='preferred',
      cell_autoreload=True,
  )
except:
  import contextlib
  cm = contextlib.nullcontext()

with cm:
  from tunix.models.qwen2 import params as params_lib
  from tunix.models.qwen2 import model as model_lib
  from tunix.generate import sampler as sampler_lib
  from tunix.sft import metrics_logger
  from tunix.rl.agentic.agents import model_agent
  from tunix.rl.agentic.environments import task_environment
  from tunix.rl.agentic.rewards import reward
  from tunix.rl.agentic.trajectory import trajectory_collect_engine
  from tunix.rl.agentic.parser.chat_template_parser import parser
  import jax
  import numpy as np
  from tunix.rl.experimental.agentic_grpo_learner import GRPOConfig, GRPOLearner
  from tunix.rl import rl_cluster as rl_cluster_lib
  from tunix.rl.rollout import base_rollout
  from tunix.sft import metrics_logger
  from tunix.sft import utils as sft_utils
  from tunix.utils import math_rewards
  from tunix.utils import compat

# %%
# ====== Data ======
TRAIN_FRACTION = 1.0

# ====== Reproducibility ======
SEED = 42

# ====== LoRA ======
RANK = 64
ALPHA = 64.0
TRAIN_WITH_LORA = False

# ====== Sharding ======
MESH = [(2, 4), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 2048
TOTAL_GENERATION_STEPS = 8192
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = 2

# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = 0.001
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = 0.2

# ====== Training ======
BATCH_SIZE = 32
MINI_BATCH_SIZE = 32
# ROLLOUT_MICRO_BATCH_SIZE = 8
# LOGPS_MICRO_BATCH_SIZE = 8
NUM_BATCHES = 100
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 50

EVAL_EVERY_N_STEPS = 1000  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 100 # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 1e-6
B1 = 0.9  # Adam beta1
B2 = 0.99  # Adam beta2
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = int(0.1 * MAX_STEPS)
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4
DO_MEM_PROFILING = False

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}
# ====== Rollout ======
ROLLOUT_ENGINE = "vanilla" # one of "vanilla", "vllm" or "sglang-jax"

# %%
try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
  file_open = gfile.Open

  NOTEBOOK_ENV = "g3"
except Exception:
  NOTEBOOK_ENV = "git"

  from google.cloud import storage

  import fsspec

  file_open = fsspec.open

if NOTEBOOK_ENV == "g3":
  DATA_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/rl/data/"
  MODEL_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/"
  CKPT_DIR_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/"
else:
  DATA_PATH_PREFIX = "gs://tunix/rl/data"
  MODEL_PATH_PREFIX = "gs://tunix/rl/models"
  CKPT_DIR_PREFIX = "gs://tunix/rl/checkpoints"

print("NOTEBOOK_ENV: ", NOTEBOOK_ENV)
CKPT_DIR = os.path.join(CKPT_DIR_PREFIX, "deepscaler_ckpt/01")

MODEL_VERSION = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = os.path.join(MODEL_PATH_PREFIX, "DeepSeek-R1-Distill-Qwen-1.5B")

# %%
show_hbm_usage = sft_utils.show_hbm_usage

# %%
import pandas as pd
import datasets as datasets_lib
import transformers

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer


DEEPSCALER_DATA_PATH = os.path.join(DATA_PATH_PREFIX, "DeepScaleR-Preview-Dataset/deepscaler.json")
AIME_2024_DATA_PATH = os.path.join(DATA_PATH_PREFIX, "HuggingFaceH4/aime_2024/train-00000-of-00001.parquet")

def create_datasets(
    train_ds_path: str = DEEPSCALER_DATA_PATH,
    test_ds_path: str = AIME_2024_DATA_PATH
):
  def preprocess_fn(example, index):
    return {
        "question": example["problem"],
        "ground_truth": example["answer"],
        "data_source": "math",
    }

  with file_open(train_ds_path) as train_f, file_open(test_ds_path, 'rb') as test_f:
    train_df = pd.read_json(train_f)
    test_df = pd.read_parquet(test_f)

  train_ds = Dataset.from_pandas(train_df).map(preprocess_fn, with_indices=True)
  test_ds = Dataset.from_pandas(test_df).map(preprocess_fn, with_indices=True)


  def process_item(item):
      question = item["question"]
      answer = item["answer"]

      instruction = "Let's think step by step, and put your final answer within \\boxed{}."
      prompt = f"{question} {instruction}"
      prompt = tokenizer.apply_chat_template(
          [{"role": "user", "content": prompt}],
          tokenize=False, add_generation_prompt=True)

      return {
          "prompts": prompt,
          "question": question,
          "answer": answer,
      }

  train_ds = grain.MapDataset.source(train_ds).map(process_item)
  test_ds = grain.MapDataset.source(test_ds).map(process_item)
  return train_ds, test_ds

# %%

tokenizer_source = MODEL_PATH if NOTEBOOK_ENV == "g3" else MODEL_VERSION
print(tokenizer_source)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

chat_parser = parser.QwenChatTemplateParser(tokenizer)

# %%
train_dataset, test_dataset = create_datasets()

train_dataset = train_dataset.batch(BATCH_SIZE)[:NUM_BATCHES]
if TRAIN_FRACTION == 1.0:
  train_dataset = train_dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = train_dataset[: int(len(train_dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)
  val_dataset = train_dataset[int(len(train_dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)
test_dataset = test_dataset.batch(BATCH_SIZE)[:NUM_TEST_BATCHES]

for s in iter(train_dataset):
  print(s)
  break

for s in iter(test_dataset):
  print(s)
  break

# %%
show_hbm_usage()

# %%
mesh = jax.make_mesh(
    *MESH,
    axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
)
config = model_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b()
print("MODEL_PATH: ", MODEL_PATH)
qwen2_ref = params_lib.create_model_from_safe_tensors(MODEL_PATH, config, mesh, dtype=jnp.float32)
# nnx.display(qwen2_ref)


# %%
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

# %%
if TRAIN_WITH_LORA:
  qwen2_actor = get_lora_model(qwen2_ref, mesh)
else:
  qwen2_actor = params_lib.create_model_from_safe_tensors(MODEL_PATH, config, mesh, dtype=jnp.float32)

# %%
show_hbm_usage()

# %%
ModelAgent = model_agent.ModelAgent
TaskEnvironment = task_environment.TaskEnvironment
TrajectoryCollectEngine = trajectory_collect_engine.TrajectoryCollectEngine
is_two_reward = reward.is_two_reward

# %%
# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/grpo", flush_every_n_steps=20
)

# %%
# # Logs
# if NOTEBOOK_ENV == "g3":
#   %load_ext GOOGLE_INTERNAL_PACKAGE_PATH.learning.brain.tensorboard.notebook.extension
# else:
#   %load_ext tensorboard
# %tensorboard --logdir /tmp/content/tmp/tensorboard/grpo --port=0

# %%
# Optimizer, learning rate scheduler, gradient clipping
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
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )

# %%
# Training config
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine=ROLLOUT_ENGINE,
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=MINI_BATCH_SIZE,
        train_micro_batch_size = 1,  # larger than 1 will cause OOM on HBM
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_tokens=[tokenizer.encode("<|im_end|>")[0]],
    ),
)

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
    system_prompt="",
    max_concurrency=8,
)

# %%
# RL cluster
with compat.set_mesh(mesh):
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=qwen2_actor,
      reference=qwen2_ref,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

# GRPO Trainer
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        math_rewards.math_reward,
    ],
    algo_config=grpo_config,
    chat_parser=chat_parser,
)

# %%
grpo_trainer.train(train_dataset)
