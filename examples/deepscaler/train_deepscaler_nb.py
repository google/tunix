# %%

# [WIP] Reproduction of [Deepscaler](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) with Single-turn Agentic framework.

import logging
import math
import os
import sys

#TODO(linchai):
# exp1: with parser but gspo-token and rloo instead of grpo
# exp2: no parser, directly parse from rl_cluster
# exp3: increase batch size to 128


from absl import logging as absl_logging
# from etils import ecolab
from flax import nnx
import grain
import jax
from jax import numpy as jnp
import numpy as np
import optax
import optax
from orbax import checkpoint as ocp
import qwix
from tqdm.auto import tqdm
import logging
import math
import sys
from absl import logging as absl_logging

import wandb

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

# 4. Set absl verbosity
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")

print("Logging configured at INFO level.")


try:
  import pathwaysutils
  pathwaysutils.initialize()
except:
  pass

print("jax devices: ", jax.devices())
# import os
# os.environ["WANDB_MODE"] = "online"


from tunix.models.qwen2 import params as params_lib
from tunix.models.qwen2 import model as model_lib
from tunix.sft import metrics_logger
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import task_environment
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import utils as sft_utils
from tunix.utils import math_rewards
from tunix.utils import compat
from tunix.cli.utils import data as data_lib
from tunix import PerfMetricsConfig
from tunix.perf.experimental.export import PerfMetricsExport


# %%
import argparse

arg_parser = argparse.ArgumentParser(description="Train DeepScaleR parameters")
arg_parser.add_argument("--batch_size", type=int, default=64)
arg_parser.add_argument("--mini_batch_size", type=int, default=64)
arg_parser.add_argument("--learning_rate", type=float, default=1e-6)
arg_parser.add_argument("--b1", type=float, default=0.9)
arg_parser.add_argument("--b2", type=float, default=0.95)
arg_parser.add_argument("--weight_decay", type=float, default=0.0)
arg_parser.add_argument("--num_batches", type=int, default=600)
arg_parser.add_argument("--num_generations", type=int, default=8)
arg_parser.add_argument("--beta", type=float, default=0.0)
arg_parser.add_argument("--epsilon", type=float, default=0.003)
arg_parser.add_argument("--epsilon_high", type=float, default=0.005)
arg_parser.add_argument("--max_prompt_length", type=int, default=1024)
arg_parser.add_argument("--max_response_length", type=int, default=2048)
arg_parser.add_argument("--temperature", type=float, default=0.6)
arg_parser.add_argument("--top_p", type=float, default=1)
arg_parser.add_argument("--top_k", type=int, default=None)
arg_parser.add_argument("--max_concurrency", type=int, default=512)
arg_parser.add_argument("--shuffle_data", type=bool, default=True)
arg_parser.add_argument("--seed", type=int, default=42)
arg_parser.add_argument(
    "--loss_agg_mode", type=str, default="sequence-mean-token-mean"
)
arg_parser.add_argument(
    "--kl_loss_mode", type=str, default="low_var_kl"
)
# Advantage estimator. "rloo" (leave-one-out baseline) has smaller-magnitude
# advantages than "grpo" (z-score with /std), which interacts gently with very
# tight PPO clip ratios. "grpo" is the registry default; switch via CLI.
arg_parser.add_argument(
    "--advantage_estimator", type=str, default="rloo",
    help="'grpo' (z-score) or 'rloo' (leave-one-out baseline).",
)
arg_parser.add_argument(
    "--loss_algo", type=str, default="gspo-token",
    help="'grpo' (per-token PPO) or 'gspo-token' (sequence-mean IS).",
)
args, _ = arg_parser.parse_known_args()

# ====== Data ======
TRAIN_FRACTION = 1.0

# ====== Reproducibility ======
SEED = args.seed

# ====== Sharding ======
ROLLOUT_MESH = [(2, 1), ("fsdp", "tp")]
TRAINER_MESH = [(2, 1), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = args.max_prompt_length
MAX_RESPONSE_LENGTH = args.max_response_length
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = args.temperature
TOP_P = args.top_p
TOP_K = args.top_k
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = args.num_generations

MAX_CONCURRENCY = args.max_concurrency

# Max number of sequences to be processed in parallel by vllm.
VLLM_MAX_NUM_SEQS = MAX_CONCURRENCY

# Max number of tokens to be processed in parallel by vllm.
# Divide by 8 for on policy, 1 step off divide by 4
VLLM_MAX_BATCHED_TOKENS = VLLM_MAX_NUM_SEQS

# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = args.beta
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = args.epsilon
EPSILON_HIGH = args.epsilon_high

# ====== Training ======
ENABLE_REMAT = True
ENABLE_FLASH_ATTENTION = True
ENABLE_MIX_PRECISION = True
BATCH_SIZE = args.batch_size
MINI_BATCH_SIZE = args.mini_batch_size
NUM_BATCHES = args.num_batches
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 2

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 3  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)


# Max number of off-policy steps. Default to 0 for synchronous training.
OFF_POLICY_STEPS = 0

MODEL_DTYPE = jnp.bfloat16

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = args.learning_rate
B1 = args.b1  # Adam beta1
B2 = args.b2  # Adam beta2
WEIGHT_DECAY = args.weight_decay
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 100.0

# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = 5
MAX_TO_KEEP = 100

# ====== Rollout ======
ROLLOUT_ENGINE = os.getenv(
    "ROLLOUT_ENGINE", "vllm"
)  # one of "vanilla", "vllm" or "sglang_jax"


try:
  wandb.login()
except wandb.errors.UsageError as e:
  # Handle the error, maybe disable W&B logging
  wandb.init(mode="disabled")

MODEL_VERSION = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
from huggingface_hub import snapshot_download

MODEL_PATH = snapshot_download(repo_id=MODEL_VERSION, max_workers=16)


trainer_devices = math.prod(TRAINER_MESH[0])
rollout_devices = math.prod(ROLLOUT_MESH[0])

if trainer_devices + rollout_devices > jax.device_count():
  raise ValueError(
      "Trainer devices must be less than or equal to the number of devices"
      " available."
  )

rollout_device_list = jax._src.mesh_utils.create_device_mesh(
    ROLLOUT_MESH[0], jax.devices()[:rollout_devices]
)

rollout_mesh = jax.sharding.Mesh(
    rollout_device_list,
    axis_names=ROLLOUT_MESH[1],
    axis_types=(jax.sharding.AxisType.Auto,) * len(ROLLOUT_MESH[0]),
)
print(f"YY {rollout_device_list=} {rollout_mesh.devices=}")
trainer_devices_list = jax._src.mesh_utils.create_device_mesh(
    TRAINER_MESH[0], jax.devices()[-trainer_devices:]
)
trainer_mesh = jax.sharding.Mesh(
    trainer_devices_list,
    axis_names=TRAINER_MESH[1],
    axis_types=(jax.sharding.AxisType.Auto,) * len(TRAINER_MESH[0]),
)
print(f"ZZ {trainer_devices_list=} {trainer_mesh.devices=}")

# %%
from google.cloud import storage

import fsspec

file_open = fsspec.open


DATA_PATH_PREFIX = "gs://tunix/data"
CKPT_DIR_PREFIX = "gs://linchai-bucket-dev/rl/checkpoints/"

CKPT_DIR = os.path.join(CKPT_DIR_PREFIX, "deepscaler_ckpt/grpo/03")
print(f"Checkpoint directory: {CKPT_DIR}")


print(f"Hyperparams: BATCH_SIZE={BATCH_SIZE}, NUM_BATCHES={NUM_BATCHES}, NUM_EPOCHS={NUM_EPOCHS}, TRAIN_FRACTION={TRAIN_FRACTION}, MAX_STEPS={MAX_STEPS}, LEARNING_RATE={LEARNING_RATE}, BETA={BETA}, EPSILON={EPSILON}, EPSILON_HIGH={EPSILON_HIGH}, ROLLOUT_ENGINE={ROLLOUT_ENGINE}, TOP_P={TOP_P}, TEMPERATURE={TEMPERATURE}, TOP_K={TOP_K}, NUM_GENERATIONS={NUM_GENERATIONS}")
# %%
show_hbm_usage = sft_utils.show_hbm_usage

# %%
import pandas as pd
import datasets as datasets_lib
import transformers

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer


DEEPSCALER_DATA_PATH = os.path.join(
    DATA_PATH_PREFIX, "DeepScaleR-Preview-Dataset/deepscaler.json"
)
AIME_2024_DATA_PATH = os.path.join(
    DATA_PATH_PREFIX, "HuggingFaceH4/aime_2024/train-00000-of-00001.parquet"
)


def create_datasets(
    train_ds_path: str = DEEPSCALER_DATA_PATH,
    test_ds_path: str = AIME_2024_DATA_PATH,
):
  def preprocess_fn(example, index):
    return {
        "question": example["problem"],
        "ground_truth": example["answer"],
        "data_source": "math",
    }

  with file_open(train_ds_path) as train_f, file_open(
      test_ds_path, "rb"
  ) as test_f:
    train_df = pd.read_json(train_f)
    test_df = pd.read_parquet(test_f)

  train_ds = Dataset.from_pandas(train_df).map(preprocess_fn, with_indices=True)
  test_ds = Dataset.from_pandas(test_df).map(preprocess_fn, with_indices=True)
  if args.shuffle_data:
    train_ds = train_ds.shuffle(SEED)
    test_ds = test_ds.shuffle(SEED)

  def process_item(item):
    question = item["question"]
    answer = item["answer"]

    instruction = (
        "Let's think step by step, and put your final answer within \\boxed{}."
    )
    question = f"{question} {instruction}"

    return {
        "prompts": question,
        "question": question,
        "answer": answer,
    }

  train_ds = grain.MapDataset.source(train_ds).map(process_item)
  test_ds = grain.MapDataset.source(test_ds).map(process_item)
  return train_ds, test_ds


# %%

tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)

chat_parser = parser.DefaultChatTemplateParser(tokenizer)

# %%
train_dataset, test_dataset = create_datasets()

train_dataset, val_dataset = data_lib.post_init_dataset(
    train_dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_batches=NUM_BATCHES,
    max_prompt_length=MAX_PROMPT_LENGTH,
    fraction=TRAIN_FRACTION,
    num_epochs=NUM_EPOCHS,
)

test_dataset, _ = data_lib.post_init_dataset(
    test_dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_batches=NUM_TEST_BATCHES,
    max_prompt_length=MAX_PROMPT_LENGTH,
)
# %%
show_hbm_usage("Done with loading datasets")

# %%
config = model_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b()
if ENABLE_REMAT:
  config.remat_config = model_lib.RematConfig.BLOCK
if ENABLE_FLASH_ATTENTION:
  config.use_flash_attention = True
  config.flash_attention_block_size = 512
if ENABLE_MIX_PRECISION:
  config.dtype = jnp.bfloat16

qwen2_ref = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, config, trainer_mesh, dtype=MODEL_DTYPE
)
show_hbm_usage("after loading qwen2_ref")


qwen2_actor = params_lib.create_model_from_safe_tensors(
  MODEL_PATH, config, trainer_mesh, dtype=jnp.float32
)

# %%
show_hbm_usage("after loading qwen2_actor")

# %%
# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
wandb_config = vars(args)
wandb_config.update({
    "WARMUP_STEPS": WARMUP_STEPS,
    "num_steps": MAX_STEPS,
    "rollout_engine": ROLLOUT_ENGINE,
    "model_id": MODEL_VERSION,
})
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="gs://linchai-bucket-dev/tensorboard/grpo",
    project_name=os.getenv("WANDB_PROJECT", "tunix-deepscaler"),
    flush_every_n_steps=1,
    backend_kwargs={"wandb": {"config": wandb_config, "settings": wandb.Settings(console="off")}},
)

# %%
# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.schedules.inject_hyperparams(optax.adamw)(
  learning_rate=LEARNING_RATE,
  b1=B1,
  b2=B2,
  weight_decay=WEIGHT_DECAY
)

if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )

# %%
# Training config
print("Rollout mesh: ", rollout_mesh)
print("Trainer mesh: ", trainer_mesh)

base_rollout_dict = {
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "kv_cache_size": MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "return_logprobs": True,
    "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
}

vllm_rollout_dict = {
    # vllm-tpu specific configs
    "rollout_vllm_model_version": MODEL_VERSION,
    "rollout_vllm_hbm_utilization": 0.3,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_async_scheduling": False,
    "tensor_parallel_size": ROLLOUT_MESH[0][1],
    "data_parallel_size": ROLLOUT_MESH[0][0],
    "rollout_vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
    "rollout_vllm_max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
        "dtype": "bfloat16",
    },
    # "rollout_vllm_sampling_kwargs": {
        # "skip_special_tokens": False,
    # },
}
rollout_engine_config = base_rollout.RolloutConfig(
    **base_rollout_dict, **vllm_rollout_dict
)

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: trainer_mesh,
        rl_cluster_lib.Role.REFERENCE: trainer_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
    },
    rollout_engine=ROLLOUT_ENGINE,
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=MINI_BATCH_SIZE,
        # deepscaler defaults to using dynamic batch size.
        # with dynamic batch size, the config that matters are: ppo_max_token_len_per_gpu=30000.
        # so 30000 * 8 = 240000 tokens , given that we have total 2k + 8K = 10k tokens per sample,
        # so effective batch size is 240000 / 10240 = 24 samples per micro batch. num_generations = 8,
        # ideally we can try max to 4. Given we use only 4 devices for trainer, we can set it to 2 here.
        train_micro_batch_size=2,
        compute_logps_micro_batch_size=2,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        # checkpoint_root_directory=CKPT_DIR,
        # checkpointing_options=checkpointing_options,
    ),
    rollout_config=rollout_engine_config,
)

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    max_response_length=MAX_RESPONSE_LENGTH,
    beta=BETA,
    epsilon=EPSILON,
    epsilon_high=EPSILON_HIGH,
    system_prompt="",
    max_concurrency=MAX_CONCURRENCY,
    off_policy_steps=OFF_POLICY_STEPS,
    loss_agg_mode=args.loss_agg_mode,
    kl_loss_mode=args.kl_loss_mode,
    sampler_is="token",
    sampler_is_threshold=2.0,
    advantage_estimator=args.advantage_estimator,
    degenerate_group_masking=False,
)
# %%
# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=qwen2_actor,
    reference=qwen2_ref,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

show_hbm_usage("after RLCluster creation")


# %%
_metric_call_idx = 0
def metric_fn(prompts, completions, rewards, advantages, **kwargs):
  del prompts, completions, advantages, kwargs
  global _metric_call_idx
  _metric_call_idx += 1
  solve_all = (rewards > 0.1).all()
  solve_none = (rewards == 0).all()
  solve_partial = (~solve_all) and (~solve_none)
  solve_ratio = (rewards > 0.1).mean()
  reward_mean = float(rewards.mean())
  reward_max = float(rewards.max())
  absl_logging.info(
      "[rollout-metric] call=%d n=%d solve_ratio=%.3f reward_mean=%.3f"
      " reward_max=%.3f solve_all=%d solve_none=%d",
      _metric_call_idx, len(rewards), float(solve_ratio), reward_mean,
      reward_max, int(solve_all), int(solve_none),
  )
  return {
      "rewards/solve_all": (1 if solve_all else 0, np.mean),
      "rewards/solve_none": (1 if solve_none else 0, np.mean),
      "rewards/solve_partial": (1 if solve_partial else 0, np.mean),
      "rewards/solve_ratio": (solve_ratio, np.mean),
  }


# GRPO Trainer
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        math_rewards.math_reward,
    ],
    algo_config=grpo_config,
    chat_parser=chat_parser,
    metric_fns=[metric_fn],
    agent_class=model_agent.ModelAgent,
    env_class=task_environment.TaskEnvironment,
    env_kwargs={"reward_fn": math_rewards.math_reward_env},
)
show_hbm_usage("after GRPOLearner creation")

# %%
grpo_trainer.train(train_dataset, test_dataset)
