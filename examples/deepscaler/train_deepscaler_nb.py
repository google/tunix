# %%

# [WIP] Reproduction of [Deepscaler](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) with Single-turn Agentic framework.

import contextlib
import logging
import math
import os
import sys

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
from tunix.utils import math_utils
from datetime import datetime

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

try:
  from etils import ecolab

  cm = ecolab.adhoc(
      source=ecolab.FROM_NOTEBOOK_OR_HEAD,
      reload="tunix",
      behavior="preferred",
      cell_autoreload=True,
  )
except:
  import contextlib

  cm = contextlib.nullcontext()

with cm:
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
  from tunix.sft import profiler
  from tunix.cli.utils import data as data_lib
  from tunix import PerfMetricsConfig
  from tunix.perf.experimental.export import PerfMetricsExport

try:
  import pathwaysutils

  pathwaysutils.initialize()
except:
  pass

print("jax devices: ", jax.devices())

# %%
import argparse

arg_parser = argparse.ArgumentParser(description="Train DeepScaleR parameters")
arg_parser.add_argument("--batch_size", type=int, default=128)
arg_parser.add_argument("--mini_batch_size", type=int, default=128)
arg_parser.add_argument("--train_micro_batch_size", type=int, default=2)
arg_parser.add_argument("--learning_rate", type=float, default=1e-6)
arg_parser.add_argument("--b1", type=float, default=0.9)
arg_parser.add_argument("--b2", type=float, default=0.99)
arg_parser.add_argument("--weight_decay", type=float, default=0.01)
arg_parser.add_argument("--num_batches", type=int, default=312)
arg_parser.add_argument("--num_generations", type=int, default=8)
arg_parser.add_argument("--beta", type=float, default=0.0)
arg_parser.add_argument("--ent_coef", type=float, default=1e-4)
arg_parser.add_argument("--epsilon", type=float, default=0.2)
arg_parser.add_argument("--epsilon_high", type=float, default=0.28)
arg_parser.add_argument("--max_response_length", type=int, default=8192)
arg_parser.add_argument("--temperature", type=float, default=0.8)
arg_parser.add_argument("--top_p", type=float, default=1.0)
arg_parser.add_argument("--top_k", type=int, default=None)
arg_parser.add_argument("--max_concurrency", type=int, default=768)
arg_parser.add_argument("--shuffle_data", type=bool, default=True)
arg_parser.add_argument("--seed", type=int, default=42)
arg_parser.add_argument("--loss_agg_mode", type=str, default="token-mean")

args, _ = arg_parser.parse_known_args()

# ====== Data ======
TRAIN_FRACTION = 1.0

# ====== Reproducibility ======
SEED = args.seed

# ====== LoRA ======
RANK = 64
ALPHA = 64.0
TRAIN_WITH_LORA = False

# ====== Sharding ======
MESH = [(2, 4), ("fsdp", "tp")]
ROLLOUT_MESH = [(4, 1), ("fsdp", "tp")]
TRAINER_MESH = [(4, 1), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 2048
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

# Max number of sequences to be processed in parallel by vllm.
VLLM_MAX_NUM_SEQS = 768

# Max number of tokens to be processed in parallel by vllm.
# Divide by 8 for on policy, 1 step off divide by 4
VLLM_MAX_BATCHED_TOKENS = VLLM_MAX_NUM_SEQS * 10 * 1024 // 8

# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = args.beta
ENT_COEF = args.ent_coef
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = args.epsilon
EPSILON_HIGH = args.epsilon_high

# ====== Training ======
ENABLE_FLASH_ATTN=True
ENABLE_REMAT = True
BATCH_SIZE = args.batch_size
MINI_BATCH_SIZE = args.mini_batch_size
TRAIN_MICRO_BATCH_SIZE = args.train_micro_batch_size
NUM_BATCHES = args.num_batches
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 50

EVAL_EVERY_N_STEPS = 1000  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 3  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# Max concurrency for parallel processing of trajectories.
MAX_CONCURRENCY = args.max_concurrency

# Max number of off-policy steps. Default to 0 for synchronous training.
OFF_POLICY_STEPS = 0
LOSS_AGG_MODE = args.loss_agg_mode
# MODEL_DTYPE = jnp.float32
# ACTIVATION_DTYPE = jnp.float32
MODEL_DTYPE = jnp.bfloat16
ACTIVATION_DTYPE = jnp.float32


# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = args.learning_rate
B1 = args.b1  # Adam beta1
B2 = args.b2  # Adam beta2
WEIGHT_DECAY = args.weight_decay
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = int(0.1 * MAX_STEPS)
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 1

# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = 20
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
ROLLOUT_ENGINE = os.getenv(
    "ROLLOUT_ENGINE", "vanilla"
)  # one of "vanilla", "vllm" or "sglang_jax"


try:
  wandb.login()
  print("linchai: logged in to W&B")
except wandb.errors.UsageError as e:
  # Handle the error, maybe disable W&B logging
  wandb.init(mode="disabled")


try:

  run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  wandb.init(
    project="tunix",
    name=run_name,
    config={
        "batch_size": BATCH_SIZE,
        "mini_batch_size": MINI_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "B1": B1,
        "B2": B2,
        "WARMUP_STEPS": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "num_steps": MAX_STEPS,
        "num_generations": NUM_GENERATIONS,
        "beta": BETA,
        "ent_coef": ENT_COEF,
        "epsilon": EPSILON,
        "epsilon_high": EPSILON_HIGH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_concurrency": MAX_CONCURRENCY,
        "rollout_engine": ROLLOUT_ENGINE,
    })
  # wandb.init(project="tunix", id="q0djft6p", resume="must",)
except Exception as e:
  print(f"linchai: W&B initialization failed with error: {e}")



# mesh = jax.make_mesh(
#     *MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0])
# )
mesh = None

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
  DATA_PATH_PREFIX = "gs://tunix/data"
  MODEL_PATH_PREFIX = "gs://tunix/models"
  # CKPT_DIR_PREFIX = "gs://linchai-bucket-dev/rl/checkpoints/"
  CKPT_DIR_PREFIX = "gs://lancewang-dev-supercomputer-testing/tunix/deepscaler"

print("NOTEBOOK_ENV: ", NOTEBOOK_ENV)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CKPT_DIR = os.path.join(CKPT_DIR_PREFIX, "deepscaler_ckpt", timestamp, "01")
print(f"Checkpoint directory: {CKPT_DIR}")

MODEL_VERSION = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = os.path.join(MODEL_PATH_PREFIX, "DeepSeek-R1-Distill-Qwen-1.5B")
# MODEL_VERSION = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_PATH = "gs://tunix/models/qwen2_5/torch/1.5b-it"

print(f"Hyperparams: BATCH_SIZE={BATCH_SIZE}, NUM_BATCHES={NUM_BATCHES}, NUM_EPOCHS={NUM_EPOCHS}, TRAIN_FRACTION={TRAIN_FRACTION}, MAX_STEPS={MAX_STEPS}, LEARNING_RATE={LEARNING_RATE}, BETA={BETA}, ENT_COEF={ENT_COEF}, EPSILON={EPSILON}, EPSILON_HIGH={EPSILON_HIGH}, ROLLOUT_ENGINE={ROLLOUT_ENGINE}, TOP_P={TOP_P}, TEMPERATURE={TEMPERATURE}, TOP_K={TOP_K}, NUM_GENERATIONS={NUM_GENERATIONS}")
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
    prompt = f"{question} {instruction}"

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
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

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
for s in train_dataset:
  print(s)
  break
# %%
show_hbm_usage("Done with loading datasets")

# %%
config = model_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b()
config.use_flash_attention = ENABLE_FLASH_ATTN
config.param_dtype = MODEL_DTYPE
config.dtype = ACTIVATION_DTYPE

if ENABLE_REMAT:
  config.remat_config = model_lib.RematConfig.BLOCK
else:
  config.remat_config = model_lib.RematConfig.NONE

print("MODEL_PATH: ", MODEL_PATH)
qwen2_ref = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, config, trainer_mesh, dtype=MODEL_DTYPE
)


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
  qwen2_actor = get_lora_model(qwen2_ref, trainer_mesh)
else:
  qwen2_actor = params_lib.create_model_from_safe_tensors(
      MODEL_PATH, config, trainer_mesh, dtype=MODEL_DTYPE
  )

# %%
show_hbm_usage("after loading qwen2_actor")


# %%
ModelAgent = model_agent.ModelAgent
TaskEnvironment = task_environment.TaskEnvironment
TrajectoryCollectEngine = trajectory_collect_engine.TrajectoryCollectEngine

# %%
# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
  #  log_dir="gs://linchai-bucket-dev/tensorboard/grpo", flush_every_n_steps=20
  log_dir="gs://lancewang-dev-supercomputer-testing/tunix/pw", flush_every_n_steps=1
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

sglang_jax_rollout_dict = {
    # sglang-jax specific configs
    "rollout_sglang_jax_model_version": MODEL_VERSION,
    "rollout_sglang_jax_mem_fraction_static": 0.8,
    "rollout_sglang_jax_init_with_random_weights": True,
    "rollout_sglang_jax_disable_radix_cache": True,
    "rollout_sglang_jax_enable_deterministic_sampling": False,
    "rollout_sglang_jax_chunked_prefill_size": 2048,
    "rollout_sglang_jax_max_running_requests": BATCH_SIZE,
    "rollout_sglang_jax_page_size": 128,
}

vllm_rollout_dict = {
    # vllm-tpu specific configs
    "rollout_vllm_model_version": MODEL_VERSION,
    "rollout_vllm_hbm_utilization": 0.4,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_async_scheduling": True,
    "tensor_parallel_size": ROLLOUT_MESH[0][1],
    "data_parallel_size": ROLLOUT_MESH[0][0],
    "rollout_vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
    "rollout_vllm_max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": True,
        "generation_config": "vllm",
    },
}

print(f"Rollout engine: {ROLLOUT_ENGINE}")
if ROLLOUT_ENGINE == "sglang_jax":
  rollout_engine_config = base_rollout.RolloutConfig(
      **base_rollout_dict, **sglang_jax_rollout_dict
  )
elif ROLLOUT_ENGINE == "vllm":
  rollout_engine_config = base_rollout.RolloutConfig(
      **base_rollout_dict, **vllm_rollout_dict
  )
elif ROLLOUT_ENGINE == "vanilla":
  rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict)
else:
  raise ValueError(f"Unsupported rollout engine: {ROLLOUT_ENGINE}")

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
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
        # profiler
        # profiler_options = profiler.ProfilerOptions(
          # profiler_steps=1,
          # skip_first_n_steps=1,
          # set_profile_options=False,
          # log_dir=PROFILER_PATH,
        # ) if ENABLE_PROFILER else None,
    ),
    rollout_config=rollout_engine_config,
)

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    max_response_length=MAX_RESPONSE_LENGTH,
    beta=BETA,
    ent_coef=ENT_COEF,
    epsilon=EPSILON,
    epsilon_high=EPSILON_HIGH,
    system_prompt="",
    max_concurrency=MAX_CONCURRENCY,
    off_policy_steps=OFF_POLICY_STEPS,
    loss_agg_mode=LOSS_AGG_MODE,
)

# Perf Metrics logging
perf_metrics_config = PerfMetricsConfig(
    custom_export_fn_v2=PerfMetricsExport(
        trace_dir="/tmp/agentic_perf"
    ).export_metrics
)

# %%
# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=qwen2_actor,
    reference=qwen2_ref,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
    perf_config=perf_metrics_config,
)

show_hbm_usage("after RLCluster creation")


# %%
def metric_fn(prompts, completions, rewards, advantages, **kwargs):
  del prompts, advantages
  solve_all = (rewards > 0.1).all()
  solve_none = (rewards == 0).all()

  answers = kwargs.get("answer", [None] * len(completions))
  parsing_hits = 0
  value_hits = 0

  for completion, ground_truths in zip(completions, answers):
    model_response = completion or ""
    if math_rewards.THOUGHT_DELIMITER_END in model_response:
      model_solution = model_response.split(math_rewards.THOUGHT_DELIMITER_END)[1]
    else:
      model_solution = model_response

    model_answer = math_utils.extract_answer(model_solution)
    parsed_ok = model_answer is not None
    if parsed_ok:
      parsing_hits += 1

    if not parsed_ok or ground_truths is None:
      continue

    if isinstance(ground_truths, str | float | int):
      ground_truths = [ground_truths]

    processed_ground_truths = []
    for truth in ground_truths:
      truth = str(truth)
      if "\\boxed" in truth:
        processed_truth = math_utils.extract_answer(truth)
        if processed_truth is not None:
          processed_ground_truths.append(processed_truth)
      else:
        processed_ground_truths.append(truth)

    normalized_model_answer = math_rewards._strip_math_mode_wrappers(
        str(model_answer)
    )
    is_value_correct = False
    for ground_truth in processed_ground_truths:
      normalized_ground_truth = math_rewards._strip_math_mode_wrappers(
          str(ground_truth)
      )
      if (
          math_utils.grade_answer_mathd(
              normalized_model_answer, normalized_ground_truth
          )
          or math_utils.grade_answer_sympy(
              normalized_model_answer, normalized_ground_truth
          )
          or math_rewards._is_repeating_decimal_equivalent(
              normalized_model_answer, normalized_ground_truth
          )
          or math_rewards._is_numeric_close(
              normalized_model_answer, normalized_ground_truth
          )
      ):
        is_value_correct = True
        break

    if is_value_correct:
      value_hits += 1

  denom = max(len(completions), 1)
  solve_parsing_ratio = parsing_hits / denom
  solve_value_ratio = value_hits / denom

  return {
    "rewards/solve_all_ratio": (
          1 if solve_all else 0,
          np.mean,
      ),
    "rewards/solve_none_ratio": (
          1 if solve_none else 0,
          np.mean,
      ),
    "rewards/solve_parsing_ratio": (
          solve_parsing_ratio,
          np.mean,
      ),
    "rewards/solve_value_ratio": (
          solve_value_ratio,
          np.mean,
      ),
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
)
show_hbm_usage("after GRPOLearner creation")

# %%
grpo_trainer.train(train_dataset)

try:
  wandb.finish()
  print("WandB session finished successfully")
except Exception as e:
  print(f"Warning: Failed to finish WandB session: {e}")
