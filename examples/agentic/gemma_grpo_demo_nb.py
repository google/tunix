# %%
"""Agentic GRPO training demo with async-rollout.

This tutorial demonstrates training the Gemma 2 2B-IT model on the GSM8K math
reasoning benchmark using Group Relative Policy Optimization (GRPO). GRPO can
enhance your model's problem-solving skills on mathematical word problems,
coding problems, etc.

We use v5e-8 for this experiment.
"""

# %%
# Imports
import contextlib
import os
from pprint import pprint
import re
import time

# %%
# Environment detection
try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
  from GOOGLE_INTERNAL_PACKAGE_PATH.perftools.accelerators.xprof.api.python import xprof_session

  ENV = 'g3'
except ImportError:
  ENV = 'oss'

# %%
# OSS-specific imports
if ENV == 'oss':
  # In OSS, gfile and xprof_session are not available.
  gfile = None
  xprof_session = None

# %%
import jax
from jax import numpy as jnp
import optax
from orbax import checkpoint as ocp

# %%
if ENV == 'g3':
  from etils import ecolab

  adhoc_context = ecolab.adhoc(
      source=ecolab.FROM_NOTEBOOK_OR_HEAD,
      reload='tunix',
      behavior='preferred',
      cell_autoreload=True,
  )
else:  # oss
  adhoc_context = contextlib.nullcontext()

# %%
with adhoc_context:
  from tunix.rl import rl_cluster as rl_cluster_lib
  from tunix.rl.rollout import base_rollout
  from tunix.sft import metrics_logger
  from tunix.rl.agentic.parser.chat_template_parser import parser
  from tunix.generate import tokenizer_adapter as tokenizer_lib
  from tunix.models.gemma import model as gemma_lib
  from tunix.sft import utils
  from tunix.utils import script_utils
  from tunix.rl.experimental.agentic_grpo_learner import GRPOConfig, GRPOLearner
  from flax import nnx
  from tunix.cli.utils import model as model_utils

# %%
show_hbm_usage = utils.show_hbm_usage
show_hbm_usage()
# %%
# ------------------------------------------------------------------------------
# Section 1: Hyperparameters
# ------------------------------------------------------------------------------
# Here we define all the configurations for our training run.
if ENV == 'g3':
  # ====== Data ======
  TRAIN_DATA_PATH = '/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/rl/grpo/data/gsm8k_train.json'
  TEST_DATA_PATH = '/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/rl/grpo/data/gsm8k_test.json'
  # ====== Base Model ======
  NNX_CKPT_DIR = '/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/gemma2/nnx/'
  # ====== Checkpoint saving ======
  run_name = f'grpo_demo_{int(time.time())}'
  CKPT_DIR = f'/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/rl/grpo/demo/experiments/gemma2/training_runs/{run_name}'
else:  # oss
  # ====== Data ======
  # Data will be downloaded to these local directories.
  TRAIN_DATA_PATH = './data/train'
  TEST_DATA_PATH = './data/test'
  # ====== Base Model ======
  # Model will be downloaded from Kaggle and converted to an intermediate
  # format.
  MODEL_DOWNLOAD_PATH = '/tmp/content/model_download/'
  NNX_CKPT_DIR = '/tmp/content/intermediate_ckpt/'
  # ====== Checkpoint saving ======
  run_name = f'grpo_demo_{int(time.time())}'
  CKPT_DIR = f'/tmp/content/ckpts/{run_name}'

# %%
# --- Data & Model Configs ---
TRAIN_FRACTION = 1.0
MODEL_VERSION = '2b-it'

# %%
# ====== Reproducibility ======
SEED = 42

# %%
# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# %%
# ====== Sharding ======
# Defines how the model and data are distributed across the available devices.
MESH = [(2, 4), ('fsdp', 'tp')]

# %%
# ====== GRPO ======
# --- Generation during GRPO training ---
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0  # implies we don't do nucleus sampling
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = 4

# %%
# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = 0.08
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = 0.2

# %%
# ====== Training ======
BATCH_SIZE = 16
NUM_BATCHES = 100
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 5

EVAL_EVERY_N_STEPS = 1000  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# %%
# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9  # Adam beta1
B2 = 0.99  # Adam beta2
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0.1 * MAX_STEPS
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# %%
# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4
DO_MEM_PROFILING = False

# %%
# --- Inference & Sampling Configurations ---
GENERATION_CONFIGS = {
    # greedy
    'greedy': {'temperature': 1e-4, 'top_k': 1, 'top_p': 1.0},
    # some randomness
    'standard': {'temperature': 0.7, 'top_k': 50, 'top_p': 0.95},
    # liberal
    'liberal': {'temperature': 0.85, 'top_k': 2000, 'top_p': 1.0},
}


# %%
# ------------------------------------------------------------------------------
# Section 2: Utility Functions
# ------------------------------------------------------------------------------
# %%
# Check initial memory usage
show_hbm_usage()

# %%
# ------------------------------------------------------------------------------
# Section 3: Data Preprocessing
# ------------------------------------------------------------------------------
# Data preprocessing
#
# First, let's define some special tokens. We instruct the model to first reason
# between the `<reasoning>` and `</reasoning>` tokens. After
# reasoning, we expect it to provide the answer between the `<answer>` and
# `</answer>` tokens.
reasoning_start = '<reasoning>'
reasoning_end = '</reasoning>'
solution_start = '<answer>'
solution_end = '</answer>'

# %%
# Define the system prompt for the model's desired output format.
SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""


# We use OpenAI's GSM8K dataset. GSM8K comprises grade school math word
# problems.


# %%
def extract_hash_answer(text: str) -> str | None:
  if '####' not in text:
    return None
  return text.split('####')[1].strip()


# %%
# Load and prepare the datasets.
train_dataset, val_dataset = script_utils.get_train_and_eval_datasets(
    data_path=TRAIN_DATA_PATH,
    split='train',
    seed=SEED,
    system_prompt=SYSTEM_PROMPT,
    batch_size=BATCH_SIZE,
    num_batches=NUM_BATCHES,
    train_fraction=TRAIN_FRACTION,
    num_epochs=NUM_EPOCHS,
    answer_extractor=extract_hash_answer,
)

test_dataset = script_utils.get_dataset(
    TEST_DATA_PATH,
    split='test',
    seed=SEED,
    system_prompt=SYSTEM_PROMPT,
    answer_extractor=extract_hash_answer,
).batch(BATCH_SIZE)[:NUM_TEST_BATCHES]

# %%

print((
    len(train_dataset),
    len(val_dataset) if val_dataset is not None else 0,
    len(test_dataset),
))

# %%
# Let's see how one batch of the dataset looks like!
for ele in train_dataset[:1]:
  pprint(ele)

# %%
# ------------------------------------------------------------------------------
# Section 4: Model Loading
# ------------------------------------------------------------------------------
# Load policy model and reference model
#
# The policy model is the model which is actually trained and whose weights are
# updated. The reference model is the model with which we compute KL divergence.
# This is to ensure that the policy updates are not huge and that it does not
# deviate too much from the reference model.
#
# Typically, the reference model is the base model, and the policy model is the
# same base model, but with LoRA parameters. Only the LoRA parameters are
# updated.
#
# Note: We perform full precision (fp32) training. You can, however, leverage
# Qwix for QAT.
MODEL_CONFIG = {
    '2b': gemma_lib.ModelConfig.gemma2_2b,
    '2b-it': gemma_lib.ModelConfig.gemma2_2b,
}


def get_ref_model():
  """Loads the reference model, from CNS in g3 or Kaggle in OSS."""
  mesh = jax.make_mesh(
      *MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0])
  )

  if ENV == 'g3':
    model_config = MODEL_CONFIG[MODEL_VERSION]()
    ckpt_path = os.path.join(NNX_CKPT_DIR, MODEL_VERSION)
    abs_gemma: nnx.Module = nnx.eval_shape(
        lambda: gemma_lib.Gemma(model_config, rngs=nnx.Rngs(params=0))
    )
    abs_state = nnx.state(abs_gemma)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(ckpt_path, target=abs_state)

    graph_def, _ = nnx.split(abs_gemma)
    gemma = nnx.merge(graph_def, restored_params)
    return gemma, mesh, None
  else:  # oss
    model_name = f'gemma2-{MODEL_VERSION}'
    model_config_dict = {
        'model_name': model_name,
        'model_source': 'kaggle',
        'model_id': f'google/gemma-2/flax/{model_name}',
        'model_download_path': MODEL_DOWNLOAD_PATH,
        'intermediate_ckpt_dir': NNX_CKPT_DIR,
        'model_display': False,
    }
    tokenizer_config = {'tokenizer_path': None}
    gemma, tokenizer_path = model_utils.create_model(
        model_config_dict, tokenizer_config, mesh
    )
    return gemma, mesh, tokenizer_path


# %%
# Load the reference model (the base Gemma 2 model).
gemma, mesh, tokenizer_path = get_ref_model()
nnx.display(gemma)

# %%
# Create the policy model by applying LoRA to the reference model.
lora_config = {
    'module_path': '.*attention',
    'rank': RANK,
    'alpha': ALPHA,
}
lora_gemma = model_utils.apply_lora_to_model(
    gemma, mesh=mesh, lora_config=lora_config
)
nnx.display(lora_gemma)

# %%
# Check memory usage after loading models.
show_hbm_usage()
# %%
# ------------------------------------------------------------------------------
# Section 5: Reward Functions
# ------------------------------------------------------------------------------
# This section defines the reward functions used to score the model's generated
# responses. First, we define a RegEx to check if the output format is correct.
match_format = re.compile(
    rf'^[\s]{{0,}}'
    rf'{reasoning_start}.+?{reasoning_end}.*?'
    rf'{solution_start}(.+?){solution_end}'
    rf'[\s]{{0,}}$',
    flags=re.MULTILINE | re.DOTALL,
)

# Test the regex with an example.
print(
    match_format.search(
        f'{reasoning_start}Let me'
        f' think!{reasoning_end}{solution_start}2{solution_end}'
    )
)


# %%
# Give the model a reward of 3 points if the format matches exactly.
def match_format_exactly(prompts, completions, **kargs):
  scores = []
  for completion in completions:
    score = 0
    response = completion
    # Match if format is seen exactly!
    if match_format.search(response) is not None:
      score += 3.0
    scores.append(score)
  return scores


# We also reward the model if the format of the output matches partially.
def match_format_approximately(prompts, completions, **kargs):
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores


# Reward the model if the answer is correct. A reward is also given if the
# answer does not match exactly, i.e., based on how close the answer is to the
# correct value.
def check_answer(prompts, completions, answer, **kargs):
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_format.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets 3 points!
    if guess == true_answer:
      score += 3.0
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += 1.5
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if ratio >= 0.9 and ratio <= 1.1:
          score += 0.5
        elif ratio >= 0.8 and ratio <= 1.2:
          score += 0.25
        else:
          score -= 1.0  # Penalize wrong answers
      except:
        score -= 0.5  # Penalize
    scores.append(score)
  return scores


# Sometimes, the text between `<answer>` and `</answer>` might not be one
# number; it can be a sentence. So, we extract the number and compare the
# answer.
match_numbers = re.compile(
    rf'{solution_start}.*?([\d\.]{{1,}})', flags=re.MULTILINE | re.DOTALL
)

# %%
print(match_numbers.findall(f'{solution_start}  0.34  {solution_end}'))


def check_numbers(prompts, completions, answer, **kargs):
  question = kargs['question']
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  # print("START ============================")
  # print(f"Question: {question[0]}")
  # print(f"Answer: {answer[0]}")
  # print(f"Response: {responses[0]}")
  # print(f"Extracted: {extracted_responses[0]}")
  # print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except:
      scores.append(0)
      continue
  return scores


# %%
# ------------------------------------------------------------------------------
# Section 6: Training Setup
# ------------------------------------------------------------------------------
# Configure the trainer, optimizer, and other components for the GRPO run.

# %%
# The following is a notebook magic and will not work in a .py file.
# It is kept here for reference.
# %load_ext GOOGLE_INTERNAL_PACKAGE_PATH.learning.brain.tensorboard.notebook.extension

# %%
# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# %%
# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir='/tmp/tensorboard/grpo', flush_every_n_steps=20
)

# %%
# The following is a notebook magic and will not work in a .py file.
# It is kept here for reference.
# %tensorboard --logdir /tmp/tensorboard/grpo --port=0

# %%
# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=int(WARMUP_STEPS),
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
    rollout_engine='vanilla',
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
        train_micro_batch_size=1,
        mini_batch_size=4,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    ),
)

# %%
grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
    system_prompt='',
    max_concurrency=8,
)

# %%
if ENV == 'oss':
  tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)
else:
  tokenizer = tokenizer_lib.Tokenizer()
chat_parser = parser.GemmaChatTemplateParser(tokenizer)

# %%
# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_gemma,
    reference=gemma,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

# %%
# GRPO Trainer
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    algo_config=grpo_config,
    chat_parser=chat_parser,
)

# %%
# ------------------------------------------------------------------------------
# Section 7: Execute Training
# ------------------------------------------------------------------------------
with script_utils.profile_and_capture_log(
    'gemma_benchmark', enable_profile=DO_MEM_PROFILING
):
  grpo_trainer.train(train_dataset, eval_dataset=val_dataset)
