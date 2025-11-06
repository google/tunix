# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Demo script for GRPO with Llama3 model.

This script demonstrates how to run GRPO with a Llama3 model. It includes
training, evaluation, and inference.

Example usage:
python3 grpo_demo_llama3_qwen2.py --root-dir=/path/to/root_dir \
--model-version=Qwen/Qwen2.5-0.5B

"""

import argparse
import json
import os
import pprint
import re

from absl import logging
from flax import nnx
import grain
import jax
import optax
from orbax import checkpoint as ocp
import qwix
from tqdm.auto import tqdm
import transformers
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama_params
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen2 import params as qwen2_params
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.sft import utils
from tunix.tests import test_common as tc

logging.set_verbosity(logging.INFO)

show_hbm_usage = utils.show_hbm_usage

print(
    "This script is still WIP and you'll need to download all the data to"
    "local first. Functionality and performance is not guaranteed. Try at "
    "your own discretion"
)

# Disable precompilation for faster iteration, need to toggle it back for
# official run
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

# Parse command line options
parser = argparse.ArgumentParser(description="Arguments for GRPO demo")
parser.add_argument(
    "--root-dir",
    type=str,
    required=False,
    help="The root dir of model, data, etc.",
)
parser.add_argument(
    "--model-version",
    type=str,
    # default="meta-llama/Llama-3.1-8B-Instruct"
    # default="meta-llama/Llama-3.2-3B-Instruct",
    default="meta-llama/Llama-3.2-1B-Instruct",
    required=False,
    help="The model version to use.",
)
parser.add_argument(
    "--num-batches",
    type=int,
    default=1869,
    required=False,
    help=(
        "Number of batches for training. Defaults to total number of samples //"
        " global batch size."
    ),
)
parser.add_argument(
    "--num-test-batches",
    type=int,
    default=50,
    required=False,
    help="Number of test batches for evaluation.",
)
parser.add_argument(
    "--global-batch-size",
    type=int,
    default=4,
    required=False,
    help="Number of global batches for learning.",
)
parser.add_argument(
    "--train-micro-batch-size",
    type=int,
    default=2,
    required=False,
    help="Number of micro batches for training.",
)
parser.add_argument(
    "--train-mini-batch-size",
    type=int,
    default=4,
    required=False,
    help="Number of mini batches for training.",
)
parser.add_argument(
    "--rollout-engine",
    type=str,
    default="vanilla",
    choices=["vanilla", "vllm"],
    required=False,
    help="Rollout engine to use (vanilla or vllm).",
)
parser.add_argument(
    "--rollout-server-mode",
    type=bool,
    default=False,
    required=False,
    help="Rollout engine server model.",
)

# Parse arguments
args = parser.parse_args()

# ====== Data ======
# The data is not available in gcs bucket yet, please manually copy the
# following data to your local TRAIN_DATA_PATH (to avoid leakr error using *):
# /***/gg-d/home/qwix-dev/rl/grpo/data/gsm8k_train.json
# /***/gg-d/home/qwix-dev/rl/grpo/data/gsm8k_test.json

GCS_BUCKET_PREFIX = "gcs://tunix/"
TRAIN_DATA_PATH_SUBDIR = "rl/grpo/data/gsm8k_train.json"
TEST_DATA_PATH_SUBDIR = "rl/grpo/data/gsm8k_test.json"
HF_MODEL_VERSION = args.model_version


TRAIN_FRACTION = 1.0

# Derived Data Path
GCS_TRAIN_DATA_PATH = os.path.join(GCS_BUCKET_PREFIX, TRAIN_DATA_PATH_SUBDIR)
GCS_TEST_DATA_PATH = os.path.join(GCS_BUCKET_PREFIX, TEST_DATA_PATH_SUBDIR)

TRAIN_DATA_PATH = os.path.join(args.root_dir, TRAIN_DATA_PATH_SUBDIR)
TEST_DATA_PATH = os.path.join(args.root_dir, TEST_DATA_PATH_SUBDIR)

VLLM_MODEL_SUBDIR = "rl/grpo/models/"
VLLM_MODEL_VERSION = os.path.join(
    args.root_dir, VLLM_MODEL_SUBDIR, HF_MODEL_VERSION
)

# ====== Base Model ======
NNX_CKPT_DIR = os.path.join(args.root_dir, "rl/grpo/models/", HF_MODEL_VERSION)

# ====== Reproducibility ======
SEED = 42

# ====== LoRA ======
ENABLE_LORA = False
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
if "Qwen2.5-0.5B-Instruct" in args.model_version:
  TOTAL_TPU_TO_USE = 2
elif "Qwen2.5-7B-Instruct" in args.model_version:
  TOTAL_TPU_TO_USE = 4
else:
  TOTAL_TPU_TO_USE = jax.device_count()

MESH = [(1, TOTAL_TPU_TO_USE), ("fsdp", "tp")]  # YY

# ====== GRPO ======
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

# ====== Training ======
# To speed up for quick workflow validation, we can change NUM_BATCHES to e.g. 2
NUM_BATCHES = min(args.num_batches, 7473 // args.global_batch_size)
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
# To speed up for quick workflow validation, we can change it to e.g. 1
NUM_TEST_BATCHES = args.num_test_batches

EVAL_EVERY_N_STEPS = 1000  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

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

# ====== Checkpoint saving ======
CKPT_DIR = os.path.join(
    args.root_dir, "rl/grpo/demo/experiments/llama3/training_runs/2"
)

SAVE_INTERVAL_STEPS = (
    500  # To speed up for quick workflow validation, we can change it to e.g. 2
)
MAX_TO_KEEP = 1
DO_MEM_PROFILING = False
DO_MODEL_DISPLAY = False

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 1e-2, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# ====== Profiler ======
PROFILER_PATH = os.path.join(
    args.root_dir, "rl/grpo/demo/experiments/llama3/profiler"
)

# Delete local checkpoint directory
tc.delete_directory(CKPT_DIR)
tc.clear_jax_arrays()

# Download checkpoints
tc.download_from_huggingface(
    repo_id=HF_MODEL_VERSION, model_path=VLLM_MODEL_VERSION
)


def download_from_gcs(zip_gcs_path, target_path):
  return f"""
    echo "{write_download_from_gcs_sh(zip_gcs_path, target_path)}" > download_from_gcs.sh
    bash download_from_gcs.sh
  """


def write_download_from_gcs_sh(zip_gcs_path, target_path):
  # pylint: disable=anomalous-backslash-in-string
  return f"""GCS_READ_SUCCESS=0
while [ \$GCS_READ_SUCCESS -eq 0 ]
do
  {{ # try
      gsutil cp {zip_gcs_path} {target_path} &&
      echo 'Code download from GCS successful!' && GCS_READ_SUCCESS=1
  }} || {{ # catch
      echo 'Failed to read GCS via gsutil, trying again'
      sleep 10
  }}
done"""


# download_from_gcs(GCS_TRAIN_DATA_PATH, TRAIN_DATA_PATH)
# download_from_gcs(GCS_TEST_DATA_PATH, TEST_DATA_PATH)


def load_json_from_local(path):
  # with gfile.Open(path, "rb") as f:
  with open(path, "rb") as f:
    return json.loads(f.read())


show_hbm_usage()

model_tokenizer = transformers.AutoTokenizer.from_pretrained(VLLM_MODEL_VERSION)

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"


SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""


def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def get_dataset(path: str) -> grain.MapDataset:
  """Loads a JSON dataset from a local path and converts it to a grain dataset.

  Args:
      path: The local path to the JSON file.

  Returns:
      A grain.MapDataset object.
  """

  data = load_json_from_local(path)

  loaded_dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=SEED)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": model_tokenizer.apply_chat_template(
                  [
                      {"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": x["question"]},
                  ],
                  tokenize=False,
                  add_generation_prompt=True,
              ),
              # passed to reward functions
              "question": x["question"],
              # passed to reward functions
              "answer": extract_hash_answer(x["answer"]),
          }
      )
  )
  return loaded_dataset


dataset = get_dataset(TRAIN_DATA_PATH).batch(args.global_batch_size)[
    :NUM_BATCHES
]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_PATH).batch(args.global_batch_size)[
    :NUM_TEST_BATCHES
]

print(
    f"train_dataset size: {len(train_dataset)}, val_dataset size:"
    f"{len(val_dataset) if val_dataset is not None else 0},"
    f"test_dataset size: {len(test_dataset)}"
)

for ele in train_dataset[:1]:
  pprint.pprint(ele)

MODEL_CONFIG = {
    "meta-llama/Llama-3.2-1B-Instruct": llama_lib.ModelConfig.llama3_2_1b,
    "meta-llama/Llama-3.2-3B-Instruct": llama_lib.ModelConfig.llama3_2_3b,
    "meta-llama/Llama-3.1-8B-Instruct": llama_lib.ModelConfig.llama3_1_8b,
    "Qwen/Qwen2.5-0.5B-Instruct": qwen2_lib.ModelConfig.qwen2_5_0_5b,
    "Qwen/Qwen2.5-7B-Instruct": qwen2_lib.ModelConfig.qwen2_5_7b,
}


def get_trainer_model(ckpt_path, model_mesh, ref_model_config):
  if "Llama" in HF_MODEL_VERSION:
    return llama_params.create_model_from_safe_tensors(
        ckpt_path, ref_model_config, model_mesh
    )
  elif "Qwen2.5" in HF_MODEL_VERSION:
    return qwen2_params.create_model_from_safe_tensors(
        ckpt_path, ref_model_config, model_mesh
    )
  raise NotImplementedError(
      f"{HF_MODEL_VERSION} tensor loading not implemented"
  )


def get_ref_model():
  ckpt_path = os.path.join(NNX_CKPT_DIR)
  model_mesh = jax.make_mesh(*MESH, devices=jax.devices()[:TOTAL_TPU_TO_USE])
  ref_model_config = MODEL_CONFIG[HF_MODEL_VERSION]()
  model = get_trainer_model(ckpt_path, model_mesh, ref_model_config)
  return model, model_mesh, ref_model_config


def get_lora_model(base_model, model_mesh=None):
  """Creates a LoRA model from a base model.

  Args:
    base_model: The base model to apply LoRA to.
    model_mesh: The mesh to use for sharding the model.

  Returns:
    A LoRA model.
  """
  if isinstance(base_model, llama_lib.Llama3):
    module_path = (
        ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj"
    )
  else:
    module_path = ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"

  lora_provider = qwix.LoraProvider(
      module_path=(module_path),
      rank=RANK,
      alpha=ALPHA,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  return lora_model


# Reference model
transformer, mesh, model_config = get_ref_model()
if DO_MODEL_DISPLAY:
  nnx.display(transformer)

# Policy model
# TODO(b/434959964): Supports lora in vLLM Jax backend
lora_transformer = (
    get_lora_model(transformer, model_mesh=mesh) if ENABLE_LORA else transformer
)

if DO_MODEL_DISPLAY:
  nnx.display(lora_transformer)

show_hbm_usage("After creating the reference lora model")

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_format.search(
    f"{reasoning_start}Let me"
    f" think!{reasoning_end}{solution_start}2{solution_end}",
)


def match_format_exactly(prompts, completions, **kargs):  # pylint: disable=unused-argument
  scores = []
  for completion in completions:
    score = 0
    response = completion
    # Match if format is seen exactly!
    if match_format.search(response) is not None:
      score += 3.0
    scores.append(score)
  return scores


def match_format_approximately(prompts, completions, **kargs):  # pylint: disable=unused-argument
  """Computes a score based on the approximate match of the format, penalizing if too many keywords are seen.

  Args:
      prompts: A list of prompts.
      completions: A list of completions.
      **kargs: Additional keyword arguments.

  Returns:
      A list of scores.
  """
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


def check_answer(prompts, completions, answer, **kargs):  # pylint: disable=unused-argument
  """Computes a score based on the correctness of the answer.

  Args:
      prompts: A list of prompts.
      completions: A list of completions.
      answer: A list of correct answers.
      **kargs: Additional keyword arguments.

  Returns:
      A list of scores.
  """
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
      except Exception:  # pylint: disable=broad-except
        score -= 0.5  # Penalize
    scores.append(score)
  return scores


match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")


def check_numbers(prompts, completions, answer, **kargs):  # pylint: disable=unused-argument
  """Computes a score based on the correctness of the extracted number.

  Args:
      prompts: A list of prompts.
      completions: A list of completions.
      answer: A list of correct answers.
      **kargs: Additional keyword arguments.

  Returns:
      A list of scores.
  """
  question = kargs["question"]
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  print("START ============================")
  print(f"Question: {question[0]}")
  print(f"Answer: {answer[0]}")
  print(f"Response: {responses[0]}")
  print(f"Extracted: {extracted_responses[0]}")
  print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except Exception:  # pylint: disable=broad-except
      scores.append(0)
      continue
  return scores


def generate(
    question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None
):
  """Given prompt, generates text."""

  if isinstance(question, str):
    input_batch = [
        model_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ),
    ]
  else:
    input_batch = [
        model_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in question
    ]

  out_data = sampler(
      input_strings=input_batch,
      max_generation_steps=TOTAL_GENERATION_STEPS,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=False,
      seed=seed if seed is not None else None,
  )

  output = out_data.text
  if isinstance(question, str):
    return output[0]
  return output


def evaluate(
    eval_dataset,
    sampler,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """Computes accuracy and percentage of outputs matching the format."""

  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0

  for batch in tqdm(eval_dataset):
    answers = batch["answer"]
    questions = batch["question"]

    multiple_call_responses = [[] for _ in range(len(questions))]
    for p in range(num_passes):
      responses = generate(
          questions, sampler, temperature, top_k, top_p, seed=p
      )
      for idx, response in enumerate(responses):
        multiple_call_responses[idx].append(response)

    for question, multiple_call_response, answer in zip(
        questions, multiple_call_responses, answers
    ):
      # check answer
      corr_ctr_per_question = 0
      partially_corr_per_question = 0
      corr_format_per_question = 0
      for response in multiple_call_response:
        extracted_response = (
            guess.group(1)
            if (guess := match_numbers.search(response)) is not None
            else "-1000000"
        )
        try:
          if float(extracted_response.strip()) == float(answer.strip()):
            corr_ctr_per_question += 1

          ratio = float(extracted_response.strip()) / float(answer.strip())
          if ratio >= 0.9 and ratio <= 1.1:
            partially_corr_per_question += 1
        except (ValueError, ZeroDivisionError):
          print("SKIPPED")

        # check format
        if match_format.search(response) is not None:
          corr_format_per_question += 1

        if (
            corr_ctr_per_question > 0
            and partially_corr_per_question > 0
            and corr_format_per_question > 0
        ):
          break

      if corr_ctr_per_question > 0:
        corr += 1
        if corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      else:
        if not corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      if partially_corr_per_question > 0:
        partially_corr += 1
      if corr_format_per_question > 0:
        corr_format += 1

      total += 1
      if total % 10 == 0:
        print(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )

  to_return = (
      corr,
      total,
      corr / total * 100,
      partially_corr / total * 100,
      corr_format / total * 100,
  )
  if make_lst:
    return to_return, response_lst
  return to_return


show_hbm_usage("After creating a raw sampler")

# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/grpo", flush_every_n_steps=20
)


show_hbm_usage("After creating a new rollout worker")
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

# Training config
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine=args.rollout_engine,
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=args.train_mini_batch_size,
        train_micro_batch_size=args.train_micro_batch_size,
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
        rollout_vllm_model_version=VLLM_MODEL_VERSION,
        rollout_vllm_hbm_utilization=0.2,
        rollout_vllm_tpu_backend_type="jax",
        rollout_vllm_server_mode=args.rollout_server_mode,
    ),
)

grpo_config = grpo_learner.GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
)

# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_transformer,
    reference=transformer,
    tokenizer=model_tokenizer,
    cluster_config=cluster_config,
)

# GRPO Trainer
grpo_trainer = grpo_learner.GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    grpo_config=grpo_config,
)

show_hbm_usage("After creating the learner")

rollout_sampler = rl_cluster._rollout._sampler  # pylint: disable=protected-access
(eval_corr, eval_total, eval_accuracy, eval_partial_accuracy, eval_format_accuracy) = evaluate(  # pylint: disable=unbalanced-tuple-unpacking
    test_dataset,
    rollout_sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{eval_corr=}, {eval_total=}, {eval_accuracy=}%,"
    f" {eval_partial_accuracy=}%, {eval_format_accuracy=}%"
)

# for eval_example in QUALITATIVE_EVAL_EXAMPLES:
#   question = eval_example["question"]
#   answer = eval_example["answer"]
#   response = generate(
#       question,
#       rollout_sampler,
#       temperature=INFERENCE_TEMPERATURE,
#       top_k=INFERENCE_TOP_K,
#       top_p=INFERENCE_TOP_P,
#   )

#   print(f"Question:\n{question}")
#   print(f"Answer:\n{answer}")
#   print(f"Response:\n{response}")
#   print("===============")


show_hbm_usage("Right before training")
with mesh:
  if DO_MEM_PROFILING:
    jax.profiler.start_trace(PROFILER_PATH)
    grpo_trainer.train(train_dataset)
    jax.profiler.stop_trace()
  else:
    grpo_trainer.train(train_dataset, eval_ds=val_dataset)

# Load checkpoint first.

show_hbm_usage("After training the reference lora model")

trained_ckpt_path = os.path.join(
    CKPT_DIR, "actor", str(MAX_STEPS), "model_params"
)

filter_type = nnx.LoRAParam if ENABLE_LORA else nnx.Param
abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(lora_transformer, filter_type),
)
checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

nnx.update(
    lora_transformer,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(lora_transformer, filter_type),
        trained_lora_params,
    ),
)

(eval_corr, eval_total, eval_accuracy, eval_partial_accuracy, eval_format_accuracy) = evaluate(  # pylint: disable=unbalanced-tuple-unpacking
    test_dataset,
    rollout_sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{eval_corr=}, {eval_total=}, {eval_accuracy=}%,"
    f" {eval_partial_accuracy=}%, {eval_format_accuracy=}%"
)

# for eval_example in QUALITATIVE_EVAL_EXAMPLES:
#   question = eval_example["question"]
#   answer = eval_example["answer"]
#   response = generate(
#       question,
#       rollout_sampler,
#       temperature=INFERENCE_TEMPERATURE,
#       top_k=INFERENCE_TOP_K,
#       top_p=INFERENCE_TOP_P,
#   )

#   print(f"Question:\n{question}")
#   print(f"Answer:\n{answer}")
#   print(f"Response:\n{response}")
#   print("===============")
